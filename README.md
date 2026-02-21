# CocoIndex Codebase Knowledge Graph

Index any codebase into a **Neo4j knowledge graph** with **semantic search**, powered by [CocoIndex](https://github.com/cocoindex-io/cocoindex).

## Requirements

- **Docker** (Postgres + Neo4j)
- **Python 3.11+**
- **API key** — one of: `ANTHROPIC_API_KEY` (Claude) or `GEMINI_API_KEY` (Gemini)

## Quick Start

```bash
git clone https://github.com/abarriel/cocoindex-ring.git && cd cocoindex-ring

# Infrastructure
docker compose up -d

# Install
pip install -e .

# Configure
cp .env.example .env
export ANTHROPIC_API_KEY=sk-ant-...   # or GEMINI_API_KEY=AIza...

# Index a repository
python main.py index -r /path/to/your/repo

# Explore the graph
open http://localhost:7474   # neo4j / cocoindex

# Semantic search
python main.py search -r /path/to/your/repo
```

## CLI

```bash
# Index (default: Claude)
python main.py index  -r /path/to/repo
python main.py index  -r /path/to/repo --llm gemini

# Search
python main.py search -r /path/to/repo
python main.py search -r /path/to/repo --top-k 10
```

| Command | Flag | Short | Default | Description |
|---------|------|-------|---------|-------------|
| `index` | `--repository` | `-r` | required | Path to the repository |
| `index` | `--llm` | | `anthropic` | LLM provider: `anthropic` or `gemini` |
| `search` | `--repository` | `-r` | required | Path to the repository |
| `search` | `--top-k` | `-k` | `5` | Number of results |

## Configuration

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COCOINDEX_DATABASE_URL` | Yes | Postgres connection (set in `.env`) |
| `ANTHROPIC_API_KEY` | If `--llm anthropic` | Claude API key |
| `GEMINI_API_KEY` | If `--llm gemini` | Gemini API key |

### Infrastructure

| Service | Port | Credentials |
|---------|------|-------------|
| PostgreSQL 16 (pgvector) | `5433` | `cocoindex` / `cocoindex` |
| Neo4j 5 | `7474` (browser), `7687` (bolt) | `neo4j` / `cocoindex` |

### LLM providers

| Provider | Model | Cost (~100 files) |
|----------|-------|--------------------|
| Anthropic | `claude-sonnet-4-20250514` | ~$1-3 |
| Gemini | `gemini-2.0-flash` | ~$0.10-0.50 |

### Supported languages

TypeScript, JavaScript, Python, Rust, C/C++, Java, Prisma, Markdown

## Examples

### Semantic search

```
$ python main.py search -r ~/my-project

Searching in: /Users/you/my-project
Returning top 5 results
Type a query, or press Enter to quit.

query> how does authentication work?
  [0.410] src/router.ts (L12-L45)
    Session token generation, cookie handling, refresh logic...

query> database schema
  [0.390] prisma/schema.prisma (L1-L80)
    User, Ring, Swipe, Couple, Match models...

query> push notifications
  [0.370] src/router.ts (L120-L155)
    Expo push notification on match detection...

query> swipe deck UI
  [0.350] src/components/swipe-deck.tsx (L1-L90)
    Swipeable card component with gesture handling...
```

### Neo4j queries

Open `http://localhost:7474`, login with `neo4j` / `cocoindex`.

```cypher
-- Full graph
MATCH p=()-->() RETURN p LIMIT 200

-- File summaries
MATCH (f:File) RETURN f.filename, f.summary

-- What does a file define?
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

## Teardown

```bash
docker compose down        # Stop (data preserved)
docker compose down -v     # Stop and wipe all data
```
