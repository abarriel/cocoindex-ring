"""
CocoIndex Codebase Knowledge Graph

Index any codebase into a Neo4j knowledge graph with semantic search.

Usage:
    python main.py index --repository /path/to/repo
    python main.py index --repository /path/to/repo --llm gemini
    python main.py search --repository /path/to/repo
    python main.py search --repository /path/to/repo --top-k 10
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

# Load .env early
load_dotenv()


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

# LLM provider configurations
LLM_PROVIDERS = {
    "anthropic": {
        "api_type": cocoindex.LlmApiType.ANTHROPIC,
        "model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "api_type": cocoindex.LlmApiType.GEMINI,
        "model": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
    },
}


@dataclass
class RepoConfig:
    """Configuration for a repository to index."""

    path: str
    llm_provider: str = "anthropic"
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
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=config.path,
                included_patterns=config.included_patterns,
                excluded_patterns=config.excluded_patterns,
            )
        )

        code_embeddings = data_scope.add_collector()

        with data_scope["files"].row() as file:
            file["language"] = file["filename"].transform(
                cocoindex.functions.DetectProgrammingLanguage()
            )
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

    provider = LLM_PROVIDERS[config.llm_provider]

    def _flow_def(
        flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
    ) -> None:
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=config.path,
                included_patterns=config.included_patterns,
                excluded_patterns=config.excluded_patterns,
            )
        )

        file_nodes = data_scope.add_collector()
        entity_relationships = data_scope.add_collector()
        file_entity_mentions = data_scope.add_collector()

        with data_scope["files"].row() as file:
            file["analysis"] = file["content"].transform(
                cocoindex.functions.ExtractByLlm(
                    llm_spec=cocoindex.LlmSpec(
                        api_type=provider["api_type"],
                        model=provider["model"],
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
                        "For entity descriptions, explain the business purpose, not just 'a function'.\n"
                        "IMPORTANT: Always return non-null arrays for entities and relationships. "
                        "If none found, return empty arrays []."
                    ),
                ),
            )

            file_nodes.collect(
                filename=file["filename"],
                summary=file["analysis"]["summary"],
            )

            with file["analysis"]["relationships"].row() as rel:
                entity_relationships.collect(
                    id=cocoindex.GeneratedField.UUID,
                    subject=rel["subject"],
                    predicate=rel["predicate"],
                    object=rel["object"],
                )

            with file["analysis"]["entities"].row() as entity:
                file_entity_mentions.collect(
                    id=cocoindex.GeneratedField.UUID,
                    filename=file["filename"],
                    entity_name=entity["name"],
                    entity_kind=entity["kind"],
                    entity_description=entity["description"],
                )

        # Neo4j exports
        file_nodes.export(
            "file_node",
            cocoindex.targets.Neo4j(
                connection=neo4j_conn,
                mapping=cocoindex.targets.Nodes(label="File"),
            ),
            primary_key_fields=["filename"],
        )

        flow_builder.declare(
            cocoindex.targets.Neo4jDeclaration(
                connection=neo4j_conn,
                nodes_label="Entity",
                primary_key_fields=["value"],
            )
        )

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
# Helpers
# ---------------------------------------------------------------------------


def get_repo_config(repo_path: str, llm: str = "anthropic") -> RepoConfig:
    """Create a RepoConfig from a repository path, expanding ~ and resolving."""
    expanded = os.path.expanduser(repo_path)
    resolved = os.path.realpath(expanded)
    if not os.path.isdir(resolved):
        print(f"Error: repository path does not exist: {resolved}")
        sys.exit(1)
    if llm not in LLM_PROVIDERS:
        print(
            f"Error: unknown LLM provider '{llm}'. Choose from: {', '.join(LLM_PROVIDERS)}"
        )
        sys.exit(1)
    # Check API key
    env_key = LLM_PROVIDERS[llm]["env_key"]
    if not os.environ.get(env_key):
        print(f"Error: {env_key} environment variable is not set.")
        print(f"  export {env_key}=your-key-here")
        sys.exit(1)
    return RepoConfig(path=resolved, llm_provider=llm)


def open_flows(config: RepoConfig) -> tuple:
    """Open both flows for a given config. Returns (embedding_flow, kg_flow)."""
    ef = cocoindex.open_flow("CodeEmbedding", code_embedding_flow_def(config))
    kf = cocoindex.open_flow("KnowledgeGraph", knowledge_graph_flow_def(config))
    return ef, kf


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    """Index a repository: create embeddings + knowledge graph."""
    config = get_repo_config(args.repository, args.llm)
    provider = LLM_PROVIDERS[config.llm_provider]

    print(f"Repository:  {config.path}")
    print(f"LLM:         {provider['model']} ({config.llm_provider})")
    print()

    cocoindex.init()

    ef, kf = open_flows(config)

    # Setup backend (creates tables, constraints, etc.)
    print("Setting up backends...")
    ef.setup(report_to_stdout=True)
    kf.setup(report_to_stdout=True)

    # Run the update
    print("\nIndexing...")
    ef_stats = ef.update()
    print(f"CodeEmbedding: {ef_stats}")

    kf_stats = kf.update()
    print(f"KnowledgeGraph: {kf_stats}")

    print("\nDone! Open http://localhost:7474 to explore the graph.")
    print("  Login: neo4j / cocoindex")
    print("  Query: MATCH p=()-->() RETURN p LIMIT 200")


def cmd_search(args: argparse.Namespace) -> None:
    """Interactive semantic search."""
    config = get_repo_config(args.repository, args.llm)

    cocoindex.init()

    ef = cocoindex.open_flow("CodeEmbedding", code_embedding_flow_def(config))
    table_name = cocoindex.utils.get_target_default_name(ef, "code_embeddings")

    print(f"\nSearching in: {config.path}")
    print(f"Returning top {args.top_k} results")
    print("Type a query, or press Enter to quit.\n")

    while True:
        try:
            query = input("query> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip() == "":
            break

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
                    (query_vector, args.top_k),
                )
                rows = cur.fetchall()

        if not rows:
            print("  No results found.\n")
            continue

        print(f"\n  {len(rows)} results:")
        for row in rows:
            filename, code, distance, start, end = row
            score = 1.0 - distance
            start_line = start["line"] if isinstance(start, dict) else start
            end_line = end["line"] if isinstance(end, dict) else end
            print(f"  [{score:.3f}] {filename} (L{start_line}-L{end_line})")
            preview = code[:200].replace("\n", "\n    ")
            print(f"    {preview}")
            if len(code) > 200:
                print("    ...")
            print("  ---")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CocoIndex Codebase Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py index -r ~/_/ring\n"
            "  python main.py index -r ~/_/ring --llm gemini\n"
            "  python main.py search -r ~/_/ring\n"
            "  python main.py search -r ~/_/ring --top-k 10\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- index ---
    idx = subparsers.add_parser(
        "index", help="Index a repository (embeddings + knowledge graph)"
    )
    idx.add_argument(
        "--repository",
        "-r",
        required=True,
        help="Path to the repository to index",
    )
    idx.add_argument(
        "--llm",
        default="anthropic",
        choices=list(LLM_PROVIDERS.keys()),
        help="LLM provider for entity extraction (default: anthropic)",
    )

    # --- search ---
    srch = subparsers.add_parser("search", help="Interactive semantic search")
    srch.add_argument(
        "--repository",
        "-r",
        required=True,
        help="Path to the repository to search",
    )
    srch.add_argument(
        "--llm",
        default="anthropic",
        choices=list(LLM_PROVIDERS.keys()),
        help="LLM provider (must match what was used for indexing)",
    )
    srch.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
