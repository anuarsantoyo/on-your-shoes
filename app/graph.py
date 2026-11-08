"""Read-only access to graph.db (the graph built in phase 3)."""
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "graph.db"

TERMINAL_PHRASES = {
    "germany": "reached Germany",
    "death": "died on the journey",
    "prison": "was imprisoned",
    "stuck": "got stuck along the way",
    "returned": "was forcibly returned",
    "other": "met an uncertain fate",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def get_node(node_id):
    c = _conn()
    row = c.execute(
        "SELECT id, place, country, role, terminal_type, description FROM nodes WHERE id=?",
        (node_id,),
    ).fetchone()
    c.close()
    if not row:
        return None
    return dict(id=row[0], place=row[1], country=row[2], role=row[3],
                terminal_type=row[4], description=row[5])


def get_edge(edge_id):
    c = _conn()
    row = c.execute(
        "SELECT id, from_node, to_node, action_type, description FROM edges WHERE id=?",
        (edge_id,),
    ).fetchone()
    c.close()
    if not row:
        return None
    return dict(id=row[0], from_node=row[1], to_node=row[2],
                action_type=row[3], description=row[4])


def outgoing_edges(node_id):
    """The real actions available from this node (the ground-truth the LLM must choose among)."""
    c = _conn()
    rows = c.execute(
        "SELECT e.id, e.to_node, e.action_type, n.place "
        "FROM edges e JOIN nodes n ON n.id = e.to_node WHERE e.from_node = ?",
        (node_id,),
    ).fetchall()
    c.close()
    return [dict(id=r[0], to_node=r[1], action_type=r[2], to_place=r[3]) for r in rows]


def random_origin():
    c = _conn()
    rows = c.execute("SELECT id FROM nodes WHERE role = 'origin'").fetchall()
    c.close()
    return random.choice([r[0] for r in rows])


def node_label(node_id):
    n = get_node(node_id)
    if not n:
        return "?"
    return f"{n['place']} — {n['country']}"


def all_nodes():
    c = _conn()
    rows = c.execute("SELECT id, place, country, role, terminal_type FROM nodes").fetchall()
    c.close()
    return [dict(id=r[0], place=r[1], country=r[2], role=r[3], terminal_type=r[4]) for r in rows]


def all_edges():
    c = _conn()
    rows = c.execute("SELECT id, from_node, to_node, action_type FROM edges").fetchall()
    c.close()
    return [dict(id=r[0], from_node=r[1], to_node=r[2], action_type=r[3]) for r in rows]


def reveal(path_node_ids):
    """Which real journeys overlapped the player's path, and where."""
    c = _conn()
    shared = {}
    for nid in path_node_ids:
        rows = c.execute(
            "SELECT jp.journey_id, j.origin_region, j.terminal_outcome, n.place "
            "FROM journey_paths jp "
            "JOIN journeys j ON j.id = jp.journey_id "
            "JOIN nodes n ON n.id = jp.node_id "
            "WHERE jp.node_id = ?",
            (nid,),
        ).fetchall()
        for jid, origin, outcome, place in rows:
            if jid not in shared:
                shared[jid] = dict(id=jid, origin=origin, outcome=outcome, places=set())
            shared[jid]["places"].add(place)
    c.close()

    out = []
    for jid, d in shared.items():
        out.append(dict(
            id=jid,
            origin=d["origin"],
            outcome=TERMINAL_PHRASES.get(d["outcome"], d["outcome"]),
            shared_places=sorted(d["places"]),
        ))
    return out
