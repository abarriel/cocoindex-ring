# CocoIndex Codebase Knowledge Graph

Index any codebase into a **Neo4j knowledge graph** with **semantic search**. Uses [CocoIndex](https://github.com/cocoindex-io/cocoindex) to scan, chunk, embed, and extract entities from source code using Claude.

## What it does

Given a path to any repository, this tool:

1. **Scans** all source files (TS, Python, Rust, C/C++, Java, etc.)
2. **Chunks** code using Tree-sitter (language-aware splitting)
3. **Embeds** chunks locally with SentenceTransformer (free, no API key)
4. **Extracts entities & relationships** using Claude via `ExtractByLlm` -- in a single pass, produces:
   - **Graph data**: entities (functions, components, schemas...) and relationships (imports, calls, depends on...)
   - **Per-file summaries**: business logic description, dependencies, key architectural decisions
5. **Exports** to Neo4j (knowledge graph) + Postgres (vector embeddings)
6. **Enables** visual graph exploration in Neo4j Browser + semantic code search

## Requirements

- **Docker** (for Postgres + Neo4j)
- **Python 3.11+**
- **`ANTHROPIC_API_KEY`** environment variable

## Quick Start

```bash
# Clone
git clone <repo-url>
cd cocoindex-ring

# Start infrastructure
docker compose up -d

# Install
pip install -e .

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Index a repository
export COCOINDEX_REPO_PATH=/path/to/your/repo
cocoindex update main

# Open the knowledge graph
open http://localhost:7474
# Login: neo4j / cocoindex

# Semantic search
python main.py search --repository /path/to/your/repo
```

## Infrastructure

| Service | Port | Credentials |
|---------|------|-------------|
| PostgreSQL 16 (pgvector) | `5433` | `cocoindex` / `cocoindex` |
| Neo4j 5 | `7474` (browser), `7687` (bolt) | `neo4j` / `cocoindex` |

## Architecture

### Two flows, one LLM pass

```
Repository
    |
    v
+-------------------+     +------------------------+
| Flow: CodeEmbedding|     | Flow: KnowledgeGraph   |
|                   |     |                        |
| LocalFile source  |     | LocalFile source       |
|       |           |     |       |                |
| SplitRecursively  |     | ExtractByLlm (Claude)  |
|       |           |     |   -> FileAnalysis       |
| SentenceTransformer|    |   -> summary            |
|       |           |     |   -> entities[]         |
| Export: Postgres   |     |   -> relationships[]    |
| (vector index)    |     |       |                |
+-------------------+     | Export: Neo4j           |
                          | (nodes + edges)         |
                          +------------------------+
```

### What Claude extracts per file

```python
@dataclass
class FileAnalysis:
    summary: str                     # Business logic, data flow, key decisions
    entities: list[Entity]           # Functions, components, classes, schemas...
    relationships: list[Relationship] # IMPORTS, CALLS, DEPENDS_ON, EXTENDS...
```

### Graph structure

- **File nodes** -- each source file with a `summary` property
- **Entity nodes** -- functions, components, classes, schemas, models, routes
- **DEFINES edges** -- File -> Entity
- **RELATED_TO edges** -- Entity -> Entity (imports, calls, depends on)

## Usage

### Index a repository

```bash
export COCOINDEX_REPO_PATH=/path/to/repo
cocoindex update main
```

On first run, CocoIndex creates the database tables and Neo4j constraints. Subsequent runs only process changed files (incremental).

### Explore the graph (Neo4j Browser)

Open `http://localhost:7474`, login with `neo4j` / `cocoindex`.

```cypher
-- Full graph (limit for performance)
MATCH p=()-->() RETURN p LIMIT 200

-- File summaries
MATCH (f:File) RETURN f.filename, f.summary

-- What does a specific file define?
MATCH (f:File {filename: "src/router.ts"})-[:DEFINES]->(e)
RETURN e.value

-- Dependency chain
MATCH p=(f:File)-[:DEFINES]->(e)-[:RELATED_TO*1..2]->(target)
WHERE f.filename CONTAINS "router"
RETURN p LIMIT 50

-- Find entities by keyword
MATCH (e:Entity) WHERE e.value CONTAINS "auth"
RETURN e.value
```

### Semantic search

```bash
python main.py search --repository /path/to/repo
python main.py search --repository /path/to/repo --top-k 10
```

Search uses cosine similarity on the embedded code chunks. Type natural language queries like:
- "how does authentication work?"
- "database schema relationships"
- "push notifications"

### CocoInsight (pipeline visualization)

```bash
export COCOINDEX_REPO_PATH=/path/to/repo
cocoindex server -ci main
# Open https://cocoindex.io/cocoinsight
```

## Supported languages

| Language | Extensions |
|----------|-----------|
| TypeScript / JavaScript | `.ts` `.tsx` `.js` `.jsx` |
| Python | `.py` |
| Rust | `.rs` |
| C / C++ | `.c` `.h` `.cpp` `.hpp` `.cc` `.hh` `.cxx` `.hxx` |
| Java | `.java` |
| Prisma | `.prisma` |
| Markdown | `.md` `.mdx` |

## Configuration

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COCOINDEX_REPO_PATH` | Yes | Path to the repository to index |
| `COCOINDEX_DATABASE_URL` | Yes | Postgres connection string (set in `.env`) |
| `ANTHROPIC_API_KEY` | Yes | For entity extraction via Claude |
| `PYTORCH_ENABLE_MPS_FALLBACK` | No | Set to `1` on Mac for MPS compatibility |

### Excluded patterns

Build artifacts, lock files, and generated code are automatically excluded:

```
node_modules/, dist/, build/, .next/, .turbo/, __pycache__/,
target/, .gradle/, *.class, *.o, *.so, pnpm-lock.yaml, etc.
```

## Cost

- **Embeddings**: Free (local SentenceTransformer model)
- **Entity extraction**: ~$0.50-$3.00 per run depending on codebase size
- **Incremental updates**: Only changed files are re-processed

## Teardown

```bash
docker compose down        # Stop containers (data preserved in volumes)
docker compose down -v     # Stop containers and delete all data
```
