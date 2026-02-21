"""
CocoIndex Codebase Knowledge Graph

Index any codebase into a Neo4j knowledge graph with semantic search.

Usage:
    # Build index
    cocoindex update main -- --repository /path/to/repo

    # Semantic search
    python main.py search --repository /path/to/repo

    # Start CocoInsight server
    cocoindex server -ci main
"""

import argparse
import functools
import os
import sys
from dataclasses import dataclass, field

import cocoindex
import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INCLUDED_PATTERNS: list[str] = [
    # JavaScript / TypeScript
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    # Python
    "*.py",
    # Rust
    "*.rs",
    # C / C++
    "*.c",
    "*.h",
    "*.cpp",
    "*.hpp",
    "*.cc",
    "*.hh",
    "*.cxx",
    "*.hxx",
    # Java
    "*.java",
    # Schemas / Config
    "*.prisma",
    # Documentation
    "*.md",
    "*.mdx",
]

DEFAULT_EXCLUDED_PATTERNS: list[str] = [
    # Version control & editors
    "**/.git/**",
    # JS/TS
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/.next/**",
    "**/.turbo/**",
    "**/coverage/**",
    "**/pnpm-lock.yaml",
    "**/package-lock.json",
    "**/yarn.lock",
    # Python
    "**/__pycache__/**",
    "**/.venv/**",
    "**/.ruff_cache/**",
    # Rust
    "**/target/**",
    # Java
    "**/.gradle/**",
    "**/out/**",
    "**/*.class",
    # C/C++ build artifacts
    "**/*.o",
    "**/*.so",
    "**/*.dylib",
    "**/*.a",
    # OS
    "**/.DS_Store",
]


@dataclass
class RepoConfig:
    """Configuration for a repository to index."""

    path: str
    included_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_INCLUDED_PATTERNS)
    )
    excluded_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_PATTERNS)
    )


# ---------------------------------------------------------------------------
# LLM extraction dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """An entity defined or referenced in a source file."""

    name: str
    """Name of the entity (e.g. 'UserSchema', 'handleAuth', 'SwipeDeck')."""
    kind: str
    """Kind of entity: component, function, class, schema, model, route, config, type, interface, enum, constant, module."""
    description: str
    """What this entity does -- include business logic, not just 'a function'."""


@dataclass
class Relationship:
    """A relationship between two entities."""

    subject: str
    """Source entity name."""
    predicate: str
    """Relationship type: IMPORTS, EXPORTS, DEFINES, DEPENDS_ON, CALLS, EXTENDS, IMPLEMENTS, CONTAINS."""
    object: str
    """Target entity name."""


@dataclass
class FileAnalysis:
    """Complete analysis of a source file."""

    summary: str
    """Detailed summary of what the file does. Include business logic, data flow, key decisions.
    Not just 'this is a router file' but 'Defines 6 oRPC procedures handling auth, couples, swipes, and match detection.'"""
    entities: list[Entity]
    """All entities defined in this file."""
    relationships: list[Relationship]
    """All relationships between entities in or referenced by this file."""


# ---------------------------------------------------------------------------
# Shared transform: text -> embedding
# ---------------------------------------------------------------------------


@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """Embed text using a local SentenceTransformer model (free, no API key)."""
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


# ---------------------------------------------------------------------------
# Neo4j connection
# ---------------------------------------------------------------------------

neo4j_conn = cocoindex.add_auth_entry(
    "neo4j_connection",
    cocoindex.targets.Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="cocoindex",
    ),
)


# ---------------------------------------------------------------------------
# Flow 1: Code Embedding (scan -> chunk -> embed -> Postgres)
# ---------------------------------------------------------------------------


def code_embedding_flow_def(config: RepoConfig):
    """Create a parameterized flow definition for code embeddings."""

    def _flow_def(
        flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
    ) -> None:
        # Source: scan repository files
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=config.path,
                included_patterns=config.included_patterns,
                excluded_patterns=config.excluded_patterns,
            )
        )

        code_embeddings = data_scope.add_collector()

        with data_scope["files"].row() as file:
            # Detect language for Tree-sitter aware chunking
            file["language"] = file["filename"].transform(
                cocoindex.functions.DetectProgrammingLanguage()
            )
            # Split into chunks using Tree-sitter
            file["chunks"] = file["content"].transform(
                cocoindex.functions.SplitRecursively(),
                language=file["language"],
                chunk_size=1000,
                min_chunk_size=300,
                chunk_overlap=300,
            )
            with file["chunks"].row() as chunk:
                chunk["embedding"] = chunk["text"].call(code_to_embedding)
                code_embeddings.collect(
                    filename=file["filename"],
                    location=chunk["location"],
                    code=chunk["text"],
                    embedding=chunk["embedding"],
                    start=chunk["start"],
                    end=chunk["end"],
                )

        # Export to Postgres with vector index for similarity search
        code_embeddings.export(
            "code_embeddings",
            cocoindex.targets.Postgres(),
            primary_key_fields=["filename", "location"],
            vector_indexes=[
                cocoindex.VectorIndexDef(
                    field_name="embedding",
                    metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
                )
            ],
        )

    return _flow_def


# ---------------------------------------------------------------------------
# Flow 2: Knowledge Graph (scan -> LLM extract -> Neo4j)
# ---------------------------------------------------------------------------


def knowledge_graph_flow_def(config: RepoConfig):
    """Create a parameterized flow definition for the knowledge graph."""

    def _flow_def(
        flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
    ) -> None:
        # Source: scan the same repository files
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=config.path,
                included_patterns=config.included_patterns,
                excluded_patterns=config.excluded_patterns,
            )
        )

        # Collectors for graph elements
        file_nodes = data_scope.add_collector()
        entity_relationships = data_scope.add_collector()
        file_entity_mentions = data_scope.add_collector()

        with data_scope["files"].row() as file:
            # Extract entities, relationships, and summary using Claude
            file["analysis"] = file["content"].transform(
                cocoindex.functions.ExtractByLlm(
                    llm_spec=cocoindex.LlmSpec(
                        api_type=cocoindex.LlmApiType.ANTHROPIC,
                        model="claude-sonnet-4-20250514",
                    ),
                    output_type=FileAnalysis,
                    instruction=(
                        "Analyze this source code file. Extract:\n"
                        "1. A detailed summary of what the file does, including business logic, "
                        "data flow, and key architectural decisions. Be specific, not generic.\n"
                        "2. All entities defined (functions, classes, components, schemas, models, "
                        "routes, types, interfaces, constants, configs).\n"
                        "3. All relationships: what this file imports, exports, depends on, calls, "
                        "extends, or implements.\n"
                        "For entity descriptions, explain the business purpose, not just 'a function'."
                    ),
                ),
            )

            # Collect file as a node with its summary
            file_nodes.collect(
                filename=file["filename"],
                summary=file["analysis"]["summary"],
            )

            # Process each extracted relationship
            with file["analysis"]["relationships"].row() as rel:
                entity_relationships.collect(
                    id=cocoindex.GeneratedField.UUID,
                    subject=rel["subject"],
                    predicate=rel["predicate"],
                    object=rel["object"],
                )

            # Process each extracted entity -> create file-to-entity mention
            with file["analysis"]["entities"].row() as entity:
                file_entity_mentions.collect(
                    id=cocoindex.GeneratedField.UUID,
                    filename=file["filename"],
                    entity_name=entity["name"],
                    entity_kind=entity["kind"],
                    entity_description=entity["description"],
                )

        # ------ Neo4j exports ------

        # Export File nodes
        file_nodes.export(
            "file_node",
            cocoindex.targets.Neo4j(
                connection=neo4j_conn,
                mapping=cocoindex.targets.Nodes(label="File"),
            ),
            primary_key_fields=["filename"],
        )

        # Declare Entity nodes (will be referenced by relationships)
        flow_builder.declare(
            cocoindex.targets.Neo4jDeclaration(
                connection=neo4j_conn,
                nodes_label="Entity",
                primary_key_fields=["value"],
            )
        )

        # Export entity-to-entity relationships
        entity_relationships.export(
            "entity_relationship",
            cocoindex.targets.Neo4j(
                connection=neo4j_conn,
                mapping=cocoindex.targets.Relationships(
                    rel_type="RELATED_TO",
                    source=cocoindex.targets.NodeFromFields(
                        label="Entity",
                        fields=[
                            cocoindex.targets.TargetFieldMapping(
                                source="subject", target="value"
                            )
                        ],
                    ),
                    target=cocoindex.targets.NodeFromFields(
                        label="Entity",
                        fields=[
                            cocoindex.targets.TargetFieldMapping(
                                source="object", target="value"
                            )
                        ],
                    ),
                ),
            ),
            primary_key_fields=["id"],
        )

        # Export file -> entity DEFINES relationships
        file_entity_mentions.export(
            "file_defines_entity",
            cocoindex.targets.Neo4j(
                connection=neo4j_conn,
                mapping=cocoindex.targets.Relationships(
                    rel_type="DEFINES",
                    source=cocoindex.targets.NodeFromFields(
                        label="File",
                        fields=[
                            cocoindex.targets.TargetFieldMapping(
                                source="filename", target="filename"
                            )
                        ],
                    ),
                    target=cocoindex.targets.NodeFromFields(
                        label="Entity",
                        fields=[
                            cocoindex.targets.TargetFieldMapping(
                                source="entity_name", target="value"
                            )
                        ],
                    ),
                ),
            ),
            primary_key_fields=["id"],
        )

    return _flow_def


# ---------------------------------------------------------------------------
# Postgres connection pool for search queries
# ---------------------------------------------------------------------------


@functools.cache
def connection_pool() -> ConnectionPool:
    """Get a cached connection pool to the Postgres database."""
    return ConnectionPool(os.environ["COCOINDEX_DATABASE_URL"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TOP_K = 5


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CocoIndex Codebase Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py search --repository ~/_/ring\n"
            "  python main.py search --repository ~/projects/my-app --top-k 10\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Interactive semantic search")
    search_parser.add_argument(
        "--repository",
        "-r",
        required=True,
        help="Path to the repository to search",
    )
    search_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=TOP_K,
        help=f"Number of results to return (default: {TOP_K})",
    )

    return parser.parse_args()


def get_repo_config(repo_path: str) -> RepoConfig:
    """Create a RepoConfig from a repository path, expanding ~ and resolving."""
    expanded = os.path.expanduser(repo_path)
    resolved = os.path.realpath(expanded)
    if not os.path.isdir(resolved):
        print(f"Error: repository path does not exist: {resolved}")
        sys.exit(1)
    return RepoConfig(path=resolved)


def run_search(config: RepoConfig, top_k: int) -> None:
    """Run interactive semantic search loop."""
    # Reuse the module-level flow if already registered, otherwise open a new one
    global embedding_flow
    try:
        _ef = cocoindex.open_flow("CodeEmbedding", code_embedding_flow_def(config))
    except KeyError:
        # Flow already registered at module level
        _ef = embedding_flow
    table_name = cocoindex.utils.get_target_default_name(_ef, "code_embeddings")

    print(f"\nSearching in: {config.path}")
    print(f"Returning top {top_k} results")
    print("Enter a query to search, or press Enter to quit.\n")

    while True:
        query = input("Enter search query (or Enter to quit): ")
        if query.strip() == "":
            break

        # Embed the query using the same model as indexing
        query_vector = code_to_embedding.eval(query)

        with connection_pool().connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT filename, code, embedding <=> %s AS distance, start, "end"
                    FROM {table_name}
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (query_vector, top_k),
                )
                rows = cur.fetchall()

        if not rows:
            print("  No results found.\n")
            continue

        print(f"\nSearch results ({len(rows)} matches):")
        for row in rows:
            filename, code, distance, start, end = row
            score = 1.0 - distance
            start_line = start["line"] if isinstance(start, dict) else start
            end_line = end["line"] if isinstance(end, dict) else end
            print(f"  [{score:.3f}] {filename} (L{start_line}-L{end_line})")
            # Show first 200 chars of code, indented
            preview = code[:200].replace("\n", "\n    ")
            print(f"    {preview}")
            if len(code) > 200:
                print("    ...")
            print("  ---")
        print()


def _main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.command == "search":
        config = get_repo_config(args.repository)
        cocoindex.init()
        run_search(config, args.top_k)
    else:
        print("Usage:")
        print("  # Set the repo path, then index:")
        print("  export COCOINDEX_REPO_PATH=/path/to/repo")
        print("  cocoindex update main")
        print()
        print("  # Semantic search:")
        print("  python main.py search --repository /path/to/repo")
        print()
        print("  # CocoInsight pipeline visualization:")
        print("  export COCOINDEX_REPO_PATH=/path/to/repo")
        print("  cocoindex server -ci main")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Module-level flow registration for `cocoindex update` and `cocoindex server`
#
# When cocoindex CLI runs (e.g. `cocoindex update main`), it imports this module
# and looks for flows. We read COCOINDEX_REPO_PATH from the environment.
# ---------------------------------------------------------------------------

# Load .env early so COCOINDEX_REPO_PATH can be set there too
load_dotenv()

_repo_path = os.environ.get("COCOINDEX_REPO_PATH")

if _repo_path is not None:
    _config = get_repo_config(_repo_path)

    # Register both flows with cocoindex
    embedding_flow = cocoindex.open_flow(
        "CodeEmbedding", code_embedding_flow_def(_config)
    )
    knowledge_graph_flow = cocoindex.open_flow(
        "KnowledgeGraph", knowledge_graph_flow_def(_config)
    )


if __name__ == "__main__":
    _main()
