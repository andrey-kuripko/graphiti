"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import os
from collections.abc import Coroutine
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from neo4j import time as neo4j_time
from numpy._typing import NDArray
from typing_extensions import LiteralString

load_dotenv()

DEFAULT_DATABASE = os.getenv('DEFAULT_DATABASE', None)
USE_PARALLEL_RUNTIME = bool(os.getenv('USE_PARALLEL_RUNTIME', False))
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))
MAX_REFLEXION_ITERATIONS = int(os.getenv('MAX_REFLEXION_ITERATIONS', 0))
DEFAULT_PAGE_LIMIT = 20

# Determine the graph database backend. Default to Neo4j but allow Memgraph.
GRAPH_DB = os.getenv('GRAPH_DB', 'neo4j').lower()

# Backend specific configuration for features such as fulltext and vector search.
if GRAPH_DB == 'memgraph':
    # Memgraph exposes similar procedures but under different namespaces.
    COSINE_SIMILARITY_FUNC = 'similarity.cosine'
    NODE_FULLTEXT_SEARCH = 'CALL db.index.fulltext.searchNodes'
    REL_FULLTEXT_SEARCH = 'CALL db.index.fulltext.searchRelationships'
    FULLTEXT_INDEX_TEMPLATES = [
        """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS FOR (e:Episodic) ON (e.content, e.source, e.source_description, e.group_id)""",
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS FOR (n:Entity) ON (n.name, n.summary, n.group_id)""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS FOR (n:Community) ON (n.name, n.group_id)""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)""",
    ]
else:
    # Neo4j default implementation
    COSINE_SIMILARITY_FUNC = 'vector.similarity.cosine'
    NODE_FULLTEXT_SEARCH = 'CALL db.index.fulltext.queryNodes'
    REL_FULLTEXT_SEARCH = 'CALL db.index.fulltext.queryRelationships'
    FULLTEXT_INDEX_TEMPLATES = [
        """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS
        FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS
        FOR (n:Community) ON EACH [n.name, n.group_id]""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
        FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
    ]

RUNTIME_QUERY: LiteralString = (
    'CYPHER runtime = parallel parallelRuntimeSupport=all\n' if USE_PARALLEL_RUNTIME else ''
)


def parse_db_date(neo_date: neo4j_time.DateTime | None) -> datetime | None:
    return neo_date.to_native() if neo_date else None


def lucene_sanitize(query: str) -> str:
    # Escape special characters from a query before passing into Lucene
    # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
    escape_map = str.maketrans(
        {
            '+': r'\+',
            '-': r'\-',
            '&': r'\&',
            '|': r'\|',
            '!': r'\!',
            '(': r'\(',
            ')': r'\)',
            '{': r'\{',
            '}': r'\}',
            '[': r'\[',
            ']': r'\]',
            '^': r'\^',
            '"': r'\"',
            '~': r'\~',
            '*': r'\*',
            '?': r'\?',
            ':': r'\:',
            '\\': r'\\',
            '/': r'\/',
            'O': r'\O',
            'R': r'\R',
            'N': r'\N',
            'T': r'\T',
            'A': r'\A',
            'D': r'\D',
        }
    )

    sanitized = query.translate(escape_map)
    return sanitized


def normalize_l2(embedding: list[float]) -> NDArray:
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array, 2, axis=0, keepdims=True)
    return np.where(norm == 0, embedding_array, embedding_array / norm)


# Use this instead of asyncio.gather() to bound coroutines
async def semaphore_gather(
    *coroutines: Coroutine,
    max_coroutines: int = SEMAPHORE_LIMIT,
):
    semaphore = asyncio.Semaphore(max_coroutines)

    async def _wrap_coroutine(coroutine):
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(_wrap_coroutine(coroutine) for coroutine in coroutines))
