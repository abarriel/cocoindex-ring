# Example: Indexing a TypeScript Monorepo

This example shows indexing a real monorepo (`ring`) -- a Tinder-like app for engagement rings, built with Bun, oRPC, Prisma, and Expo.

## Repository structure

```
ring/
  apps/
    api/          # Bun + oRPC + Prisma backend
    mobile/       # Expo React Native app
  packages/
    shared/       # Zod schemas shared between API and mobile
    ui/           # Shared UI components + theme
```

## Step 1: Index the repository

```bash
cd ~/tools/cocoindex-ring
docker compose up -d
pip install -e .

export ANTHROPIC_API_KEY=sk-ant-...
export COCOINDEX_REPO_PATH=~/_/ring
cocoindex update main
```

### Output

```
[ TO CREATE ] Flow: CodeEmbedding
    [ TO CREATE ] Postgres table CodeEmbedding__code_embeddings
[ TO CREATE ] Flow: KnowledgeGraph
    [ TO CREATE ] Neo4j Node(label:File)
    [ TO CREATE ] Neo4j Node(label:Entity)
    [ TO CREATE ] Neo4j Relationship(type:RELATED_TO)
    [ TO CREATE ] Neo4j Relationship(type:DEFINES)

Changes need to be pushed. Continue? [yes/N]: yes

CodeEmbedding.files (batch update): 98/98 source rows: 98 added [elapsed: 5.7s]
KnowledgeGraph.files (batch update): 98/98 source rows: 92 added, 6 errors [elapsed: 55.4s]
```

### Results

| Metric | Count |
|--------|-------|
| Files indexed | 98 |
| File nodes (Neo4j) | 92 |
| Entity nodes (Neo4j) | 1,745 |
| DEFINES relationships | 1,169 |
| RELATED_TO relationships | 1,633 |

The 6 errors were Prisma auto-generated model files where Claude returned null for some fields -- expected for generated code.

---

## Step 2: Explore the knowledge graph

Open `http://localhost:7474` and login with `neo4j` / `cocoindex`.

### Query: Full graph overview

```cypher
MATCH p=(f:File)-[:DEFINES]->(e:Entity)-[:RELATED_TO]->(target:Entity)
RETURN p LIMIT 100
```

This shows File nodes (purple) connected to Entity nodes (orange) via DEFINES and RELATED_TO edges.

### Query: File summaries

```cypher
MATCH (f:File) RETURN f.filename, f.summary
```

Example results:

| filename | summary |
|----------|---------|
| `apps/api/src/router.ts` | "Defines the main API router with oRPC procedures. Handles user authentication (session tokens via cookies), couples pairing (generate/join with 6-char codes), ring swiping (LIKE/NOPE/SUPER), and match detection (triggers when both partners like the same ring). Sends push notifications on new matches." |
| `apps/mobile/src/lib/query-client.ts` | "Creates and exports a configured React Query client for managing server state. The configuration implements a caching strategy with 1-minute stale time, custom retry logic that prevents retrying unauthorized requests, and disables retry for mutations." |
| `packages/shared/src/index.ts` | "Barrel export file that re-exports all named and default exports from the schemas module. Serves as a centralized entry point for schema definitions shared between API and mobile." |

### Query: What does the API router define?

```cypher
MATCH (f:File {filename: "apps/api/src/router.ts"})-[:DEFINES]->(e)
RETURN e.value
```

```
createSwipe
registerPushToken
authed
joinCouple
getRing
SESSION_REFRESH_THRESHOLD_DAYS
generateSessionToken
generateCoupleCode
sessionExpiresAt
listUsers
```

### Query: Entity relationships from router.ts

```cypher
MATCH (f:File {filename: "apps/api/src/router.ts"})-[:DEFINES]->(e)-[:RELATED_TO]->(target)
RETURN e.value AS entity, target.value AS related_to
```

```
entity              | related_to
--------------------|-------------------
createSwipe         | notifyNewMatch
createSwipe         | CreateSwipeSchema
createSwipe         | authed
registerPushToken   | RegisterPushTokenSchema
registerPushToken   | authed
authed              | SESSION_REFRESH_THRESHOLD_DAYS
authed              | base
joinCouple          | JoinCoupleSchema
joinCouple          | notifyPartnerJoined
joinCouple          | authed
```

### Query: Dependency graph for the API app

```cypher
MATCH p=(f:File)-[:DEFINES]->(e:Entity)
WHERE f.filename STARTS WITH "apps/api"
RETURN p
```

### Query: Find all entities related to authentication

```cypher
MATCH (e:Entity)
WHERE e.value CONTAINS "auth" OR e.value CONTAINS "Auth" OR e.value CONTAINS "session" OR e.value CONTAINS "Session"
RETURN e.value
```

---

## Step 3: Semantic search

```bash
python main.py search --repository ~/_/ring
```

### Query: "how does match detection work?"

```
Enter search query: how does match detection work?

[0.358] apps/api/tests/match.test.ts (L39-L60)
    describe('match detection in swipe.create', () => {
      it('creates a match when both partners LIKE the same ring', async () => {
        const { aliceToken, bobToken } = await createActiveCouple('Alice', '
    ...
---
[0.349] apps/api/tests/match.test.ts (L81-L107)
    it('creates a match when one LIKEs and one SUPERs', async () => {
        const { aliceToken, bobToken } = await createActiveCouple('Alice', 'Bob')
        const ring = await seedRing('Mixed Ring')
    ...
---
[0.347] apps/api/prisma/generated/prisma/models/Match.ts (L203-L221)
    export type MatchOrderByWithAggregationInput = {
      id?: Prisma.SortOrder
      coupleId?: Prisma.SortOrder
      ringId?: Prisma.SortOrder
    ...
---
[0.341] apps/api/tests/match.test.ts (L197-L218)
    it('both partners see the same matches', async () => {
        const { aliceToken, bobToken } = await createActiveCouple('Alice', 'Bob')
        const ring = await seedRing('Shared Ring')
    ...
---
[0.328] apps/mobile/src/components/skeleton.tsx (L103-L126)
    function MatchesListSkeleton() {
      const { t } = useTranslation()
    ...
---
```

### More search examples

| Query | Top result | Score |
|-------|-----------|-------|
| "how does authentication work?" | `apps/api/src/router.ts` -- session token generation and cookie handling | 0.41 |
| "database schema" | `apps/api/prisma/schema.prisma` -- User, Ring, Swipe, Couple, Match models | 0.39 |
| "push notifications" | `apps/api/src/router.ts` -- Expo push notification sending on match | 0.37 |
| "swipe deck UI" | `apps/mobile/src/components/swipe-deck.tsx` -- swipeable card component | 0.35 |
| "i18n translations" | `apps/mobile/src/lib/i18n.ts` -- react-i18next setup with en/fr | 0.33 |

---

## Step 4: Index a different repository

```bash
export COCOINDEX_REPO_PATH=~/projects/other-project
cocoindex update main
```

CocoIndex detects changed files and only re-processes what's new. To index a completely different repo, the previous data is replaced.

---

## Teardown

```bash
docker compose down        # Stop (data preserved)
docker compose down -v     # Stop and wipe all data
```
