#!/usr/bin/env python3
"""Build graph.db from the JSON source-of-truth files.

Source of truth (git-tracked, human-reviewable):
  data/journeys/jNNN.json   - one file per person (phase 1)
  data/nodes.json           - canonical states (phase 2, optional until segmented)
  data/edges.json           - canonical actions (phase 2, optional)
  data/paths.json           - journey -> ordered steps (phase 2, optional)

This script REGENERATES graph.db from scratch on every run.
Never edit graph.db by hand; fix the JSON and rerun this.

Schema: see docs/data-model.md
"""

import glob
import json
import os
import sqlite3
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
DB_PATH = os.path.join(REPO, "graph.db")

SCHEMA = """
DROP TABLE IF EXISTS journey_paths;
DROP TABLE IF EXISTS edges;
DROP TABLE IF EXISTS nodes;
DROP TABLE IF EXISTS journeys;

CREATE TABLE journeys (
    id TEXT PRIMARY KEY,
    origin_country TEXT,
    origin_region TEXT,
    corridor TEXT,
    year TEXT,
    age_band TEXT,
    gender TEXT,
    terminal_outcome TEXT,
    source_type TEXT,
    language TEXT,
    first_person INTEGER,
    narrative TEXT
);

CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    place TEXT,
    country TEXT,
    role TEXT,
    terminal_type TEXT,
    description TEXT
);

CREATE TABLE edges (
    id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL REFERENCES nodes(id),
    to_node TEXT NOT NULL REFERENCES nodes(id),
    action_type TEXT NOT NULL,
    description TEXT,
    UNIQUE(from_node, to_node, action_type)
);

CREATE TABLE journey_paths (
    journey_id TEXT NOT NULL REFERENCES journeys(id),
    position INTEGER NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    edge_id TEXT REFERENCES edges(id),
    PRIMARY KEY (journey_id, position)
);

CREATE INDEX idx_edges_from ON edges(from_node);
CREATE INDEX idx_edges_to ON edges(to_node);
CREATE INDEX idx_paths_node ON journey_paths(node_id);
CREATE INDEX idx_paths_edge ON journey_paths(edge_id);
"""


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript(SCHEMA)

    # --- journeys (phase 1) ---
    for jf in sorted(glob.glob(os.path.join(DATA, "journeys", "*.json"))):
        d = load_json(jf)
        c.execute(
            """
            INSERT INTO journeys
              (id, origin_country, origin_region, corridor, year, age_band, gender,
               terminal_outcome, source_type, language, first_person, narrative)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                d["id"],
                d["origin"].get("country"),
                d["origin"].get("region"),
                d.get("corridor"),
                d.get("year"),
                d.get("age_band"),
                d.get("gender"),
                d.get("terminal_outcome"),
                d.get("source_type"),
                d.get("language"),
                1 if d.get("first_person") else 0,
                d.get("narrative"),
            ),
        )

    # --- nodes (phase 2, optional) ---
    nodes = load_json(os.path.join(DATA, "nodes.json"))
    if nodes is not None:
        for n in nodes:
            c.execute(
                "INSERT INTO nodes (id, place, country, role, terminal_type, description) VALUES (?,?,?,?,?,?)",
                (n["id"], n.get("place"), n.get("country"), n.get("role"),
                 n.get("terminal_type"), n.get("description")),
            )

    # --- edges (phase 2, optional) ---
    edges = load_json(os.path.join(DATA, "edges.json"))
    if edges is not None:
        for e in edges:
            c.execute(
                "INSERT INTO edges (id, from_node, to_node, action_type, description) VALUES (?,?,?,?,?)",
                (e["id"], e["from_node"], e["to_node"], e["action_type"], e.get("description")),
            )

    # --- paths (phase 2, optional) ---
    paths = load_json(os.path.join(DATA, "paths.json"))
    if paths is not None:
        for p in paths:
            jid = p["journey_id"]
            for i, step in enumerate(p["steps"]):
                c.execute(
                    "INSERT INTO journey_paths (journey_id, position, node_id, edge_id) VALUES (?,?,?,?)",
                    (jid, i, step["node_id"], step.get("edge_id")),
                )

    conn.commit()

    print("graph.db contents:")
    for table in ("journeys", "nodes", "edges", "journey_paths"):
        n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:15} {n} rows")
    conn.close()
    print(f"wrote {DB_PATH}")


if __name__ == "__main__":
    build()
