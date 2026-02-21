"""
CocoIndex Codebase Knowledge Graph

Index any codebase into a Neo4j knowledge graph with semantic search.

Usage:
    python main.py index --repository /path/to/repo
    python main.py index --repository /path/to/repo --llm gemini
    python main.py search --repository /path/to/repo
    python main.py search --repository /path/to/repo --top-k 10
    python main.py explain --repository /path/to/repo --output ./docs
"""

import argparse
import functools
import os
import shutil
import sys
from dataclasses import dataclass, field

import cocoindex
import numpy as np
from cocoindex.op import TargetSpec, target_connector
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


@dataclass
class FileExplanation:
    """Detailed markdown explanation of a source file."""

    purpose: str
    """What this file does and why it exists. 2-3 sentences of business context."""
    key_concepts: str
    """Bullet-point list of the main concepts, patterns, or architectural decisions in the file."""
    code_walkthrough: str
    """A section-by-section walkthrough of the code. Explain each important function, class, or block.
    Use markdown headers (###) for each section. Include relevant code snippets in fenced code blocks."""
    dependencies: str
    """What this file imports/depends on and why. Bullet-point list."""
    usage: str
    """How this file is used by other parts of the codebase. Where it's imported, called, or referenced."""


# ---------------------------------------------------------------------------
# Custom target: write .md files to local directory
# ---------------------------------------------------------------------------


class MarkdownFileTarget(TargetSpec):
    """Target that writes markdown files to a local directory, preserving structure."""

    directory: str


@dataclass
class MarkdownFileValues:
    """Values written to each markdown file."""

    markdown: str


@target_connector(
    spec_cls=MarkdownFileTarget,
    persistent_key_type=str,
    setup_state_cls=MarkdownFileTarget,
)
class MarkdownFileTargetConnector:
    @staticmethod
    def get_persistent_key(spec: MarkdownFileTarget, target_name: str) -> str:
        return spec.directory

    @staticmethod
    def describe(key: str) -> str:
        return f"Markdown directory {key}"

    @staticmethod
    def apply_setup_change(
        key: str,
        previous: MarkdownFileTarget | None,
        current: MarkdownFileTarget | None,
    ) -> None:
        if previous is None and current is not None:
            os.makedirs(current.directory, exist_ok=True)
        if previous is not None and current is None:
            if os.path.isdir(previous.directory):
                shutil.rmtree(previous.directory, ignore_errors=True)

    @staticmethod
    def prepare(spec: MarkdownFileTarget) -> MarkdownFileTarget:
        return spec

    @staticmethod
    def mutate(
        *all_mutations: tuple[MarkdownFileTarget, dict[str, MarkdownFileValues | None]],
    ) -> None:
        for spec, mutations in all_mutations:
            for filename, mutation in mutations.items():
                full_path = os.path.join(spec.directory, filename) + ".md"
                if mutation is None:
                    try:
                        os.remove(full_path)
                    except FileNotFoundError:
                        pass
                else:
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w") as f:
                        f.write(mutation.markdown)


# ---------------------------------------------------------------------------
# Custom function: format FileExplanation into markdown
# ---------------------------------------------------------------------------


@cocoindex.op.function()
def format_explanation(filename: str, explanation: FileExplanation) -> str:
    """Format a FileExplanation dataclass into a full markdown document."""
    return f"""# `{filename}`

## Purpose

{explanation.purpose}

## Key Concepts

{explanation.key_concepts}

## Code Walkthrough

{explanation.code_walkthrough}

## Dependencies

{explanation.dependencies}

## Usage

{explanation.usage}
"""


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
# Flow 3: Explain (scan -> LLM explain -> write .md files)
# ---------------------------------------------------------------------------


def explain_flow_def(config: RepoConfig, output_dir: str):
    """Create a parameterized flow definition for generating markdown explanations."""

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

        md_output = data_scope.add_collector()

        with data_scope["files"].row() as file:
            file["explanation"] = file["content"].transform(
                cocoindex.functions.ExtractByLlm(
                    llm_spec=cocoindex.LlmSpec(
                        api_type=provider["api_type"],
                        model=provider["model"],
                    ),
                    output_type=FileExplanation,
                    instruction=(
                        "You are a senior engineer writing documentation for a codebase.\n"
                        "Analyze this source code file and produce a detailed explanation.\n\n"
                        "For each field:\n"
                        "- purpose: What this file does and why it exists. 2-3 sentences.\n"
                        "- key_concepts: Bullet-point list (use - prefix) of main concepts, "
                        "patterns, or architectural decisions.\n"
                        "- code_walkthrough: Section-by-section explanation of the code. "
                        "Use ### headers for each function/class/block. Include short code "
                        "snippets in fenced code blocks (```lang) to illustrate key points.\n"
                        "- dependencies: Bullet-point list of imports and why each is needed.\n"
                        "- usage: How other files use this file. If unknown, describe the "
                        "likely consumers based on what it exports.\n\n"
                        "Write in clear, concise technical prose. Focus on the WHY, not just the WHAT."
                    ),
                ),
            )

            file["markdown"] = file["filename"].transform(
                format_explanation,
                explanation=file["explanation"],
            )

            md_output.collect(
                filename=file["filename"],
                markdown=file["markdown"],
            )

        md_output.export(
            "markdown_docs",
            MarkdownFileTarget(directory=output_dir),
            primary_key_fields=["filename"],
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


def cmd_explain(args: argparse.Namespace) -> None:
    """Generate markdown explanations for each file in a repository."""
    config = get_repo_config(args.repository, args.llm)
    provider = LLM_PROVIDERS[config.llm_provider]
    output_dir = os.path.realpath(os.path.expanduser(args.output))

    print(f"Repository:  {config.path}")
    print(f"LLM:         {provider['model']} ({config.llm_provider})")
    print(f"Output:      {output_dir}")
    print()

    cocoindex.init()

    flow = cocoindex.open_flow("ExplainDocs", explain_flow_def(config, output_dir))

    print("Setting up...")
    flow.setup(report_to_stdout=True)

    print("\nGenerating explanations...")
    stats = flow.update()
    print(f"ExplainDocs: {stats}")

    print(f"\nDone! Markdown files written to: {output_dir}")


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
            "  python main.py explain -r ~/_/ring -o ./docs\n"
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

    # --- explain ---
    exp = subparsers.add_parser(
        "explain", help="Generate markdown explanation per file"
    )
    exp.add_argument(
        "--repository",
        "-r",
        required=True,
        help="Path to the repository to explain",
    )
    exp.add_argument(
        "--llm",
        default="anthropic",
        choices=list(LLM_PROVIDERS.keys()),
        help="LLM provider (default: anthropic)",
    )
    exp.add_argument(
        "--output",
        "-o",
        default="./docs",
        help="Output directory for markdown files (default: ./docs)",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "explain":
        cmd_explain(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
