"""FastAPI app: the experience backend.

Run:  uvicorn app.main:app --reload   (from the repo root)
Play:  http://localhost:8000
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import graph
from .engine import Session
from .models import MessageIn, NarratorTransition
from .narrator import get_narrator

load_dotenv()  # read .env (OPENAI_API_KEY, OPENAI_BASE_URL, MODEL) before get_narrator()

app = FastAPI(title="On Your Shoes")
SESSIONS = {}
narrator = get_narrator()

STATIC = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/graph")
def get_graph():
    return {"nodes": graph.all_nodes(), "edges": graph.all_edges()}


@app.get("/graph/view")
def graph_view():
    return FileResponse(STATIC / "graph.html")


@app.post("/sessions")
def create_session():
    origin = graph.random_origin()
    s = Session(origin)
    corridor = s.advance_to_decision()
    text = narrator.open(s, origin, corridor)
    SESSIONS[s.session_id] = s
    return {"session_id": s.session_id, "scene": graph.node_label(s.current_node_id),
            "narration": text, "ended": s.ended}


@app.post("/sessions/{sid}/message")
def send_message(sid: str, body: MessageIn):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    s.history.append({"role": "user", "content": body.message})
    s.turns_since_transition += 1
    resp = narrator.decide(s)

    if isinstance(resp, NarratorTransition):
        edge = s.apply_transition(resp.edge_id)
        if edge is None:
            # Hallucinated/invalid edge: never advance, steer instead.
            text = resp.narration or "That path is not open to you right now."
            s.history.append({"role": "assistant", "content": text})
            return {"narration": text, "scene": graph.node_label(s.current_node_id), "ended": s.ended}
        corridor = s.advance_to_decision()
        text = narrator.narrate_move(s, edge, corridor)
        s.history.append({"role": "assistant", "content": text})
        return {"narration": text, "scene": graph.node_label(s.current_node_id), "ended": s.ended}

    text = resp.text
    s.history.append({"role": "assistant", "content": text})
    return {"narration": text, "scene": graph.node_label(s.current_node_id), "ended": s.ended}


@app.get("/sessions/{sid}/state")
def get_state(sid: str):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    return {"session_id": s.session_id, "current_node_id": s.current_node_id,
            "available_actions": s.available_actions(), "path": s.path,
            "turns_since_transition": s.turns_since_transition,
            "history": s.history, "ended": s.ended}


@app.get("/sessions/{sid}/reveal")
def get_reveal(sid: str):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    if not s.ended:
        raise HTTPException(400, "journey not finished")
    return {"your_path": [graph.node_label(n) for n in s.path],
            "journeys": graph.reveal(s.path)}


@app.delete("/sessions/{sid}")
def delete_session(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}
