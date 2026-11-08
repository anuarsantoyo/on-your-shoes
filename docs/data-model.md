# On Your Shoes — Data Model

## Two layers

- **Source of truth (JSON, git-tracked, human-reviewable)**
  - `data/journeys/jNNN.json` — one file per person: structured fields + the full
    anonymized prose (`narrative`).
  - `data/nodes.json`, `data/edges.json`, `data/paths.json` — the segmentation
    (phase 2), written as plain JSON so it can be read and corrected as diffs.
- **Graph layer (SQLite, generated)** — `graph.db`, rebuilt from the JSON by
  `scripts/build_db.py`. Never hand-edited; delete and regenerate freely.

## The graph is a multigraph

Multiple edges may connect the same two nodes. They are distinguished by
`action_type` — a categorically different way of getting from A to B.

Rule: **one edge per `(from_node, to_node, action_type)`.** Parallel edges exist when
the *kind of action* differs (train vs walk vs smuggler's taxi). The same action with
different cost or circumstance is ONE edge; the per-person detail lives in the prose
`narrative`, not as a separate edge. This prevents edge explosion.

Different *outcomes* are NOT parallel edges. "Made it to Kos" and "drowned" are edges
to different target nodes (a terminal `death` node), not two flavors of one edge.

## The LLM's role

At each node the LLM interprets the person's free-form decision and selects the real
outgoing edge that matches it. It never invents an edge — no real edge, no action.
Parallel edges give it more real options; the grounding constraint ("every step was
real") holds regardless.

## Schema

`graph.db` (SQLite, single file, no server).

### journeys — one row per person (phase 1)

| column | type | notes |
| --- | --- | --- |
| id | TEXT PK | e.g. `j003` |
| origin_country | TEXT | |
| origin_region | TEXT | e.g. `Damascus` |
| corridor | TEXT | balkan / mediterranean / belarus / calais |
| year | TEXT | coarse, e.g. `2015` |
| age_band | TEXT | child / teen / 20s / 30s / 40s / adult / family |
| gender | TEXT | male / female / family |
| terminal_outcome | TEXT | germany / death / prison / stuck / returned / other |
| source_type | TEXT | news_interview / ngo_report / first_person_essay |
| language | TEXT | en / ar / ... |
| first_person | INTEGER | 0/1 |
| narrative | TEXT | the full anonymized prose (copied from JSON) |

### nodes — canonical states (phase 2)

| column | type | notes |
| --- | --- | --- |
| id | TEXT PK | canonical slug, e.g. `izmir`, `lesbos_arrival` |
| place | TEXT | canonical place name |
| country | TEXT | |
| role | TEXT | origin / transit / sea_crossing / border / detention / camp / arrival / terminal |
| terminal_type | TEXT NULL | arrived_granted / arrived_rejected / death / prison / returned / stuck (terminal nodes only) |
| description | TEXT | synthesized situation |

### edges — canonical actions (phase 2)

| column | type | notes |
| --- | --- | --- |
| id | TEXT PK | canonical slug, e.g. `belgrade_budapest_train` |
| from_node | TEXT NOT NULL | → nodes.id |
| to_node | TEXT NOT NULL | → nodes.id |
| action_type | TEXT NOT NULL | pay_smuggler / board_boat / cross_border / walk / take_train / take_bus / take_ferry / take_taxi / get_arrested / get_returned / get_detained / rescued_at_sea / relocate / flee / ... (open set) |
| description | TEXT | |

`UNIQUE(from_node, to_node, action_type)` — this is what makes the graph a multigraph
(allows parallel edges) while forbidding exact duplicates.

### journey_paths — the ordered sequence per journey (phase 2)

| column | type | notes |
| --- | --- | --- |
| journey_id | TEXT NOT NULL | → journeys.id |
| position | INTEGER NOT NULL | 0-based order in the path |
| node_id | TEXT NOT NULL | → nodes.id (the state at this position) |
| edge_id | TEXT NULL | → edges.id (the edge taken to LEAVE this node; NULL at the terminal) |

`PRIMARY KEY (journey_id, position)`.

`paths.json` is the human-readable form of this table: a list of
`{"journey_id", "steps": [{"node_id", "edge_id"}, ...]}`.

## Provenance

`journey_paths` doubles as provenance. From it you can derive both:
- which journeys passed through a node (JOIN on node_id), and
- which journeys took an edge (consecutive pairs with a matching edge_id).

This is what powers the end-of-run reveal ("these people walked this exact edge") and
the stochastic selection (weight each outgoing edge by how many real journeys took it).

## Controlled vocabularies (summary)

- corridor: balkan, mediterranean, belarus, calais
- terminal_outcome (journeys, coarse): germany, death, prison, stuck, returned, other
- node role: origin, transit, sea_crossing, border, detention, camp, arrival, terminal
- node terminal_type: arrived_granted, arrived_rejected, death, prison, returned, stuck
- action_type: open set, grows during segmentation (see above)

Known reconciliation point: `journeys.terminal_outcome` is coarse (phase 1) and
`nodes.terminal_type` is finer (phase 2). Segmentation will reconcile them (e.g. split
"germany" into arrived_granted vs arrived_rejected, as with the Dublin case j026).

## Build / rebuild

```
python3 scripts/build_db.py
```

Regenerates `graph.db` from the JSON. Run it after any change to the JSON. The DB is
not committed (see `.gitignore`); the JSON and the script are.
