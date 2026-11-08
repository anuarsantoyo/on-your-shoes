"""Narrators. The engine never talks to an LLM directly — it talks to a Narrator.

DeepSeekNarrator: the real one (OpenAI-compatible API, DeepSeek), using native function
calling so the model always returns a structured action (narrate / commit / steer).
ScriptedNarrator: a deterministic stand-in so the whole loop runs with no API key.

The narrator has three jobs:
- open():          narrate the opening (origin -> corridor -> first decision).
- decide():        at a decision node, narrate and classify the person's intent.
                   After FORCE_COMMIT_AFTER turns it is forced to commit.
- narrate_move():  narrate a committed action + any auto-advanced corridor in one passage.
"""
import json
import os
import sys
from typing import Any

from pydantic import BaseModel, ValidationError

from . import graph
from .models import NarratorMessage, NarratorTransition, NarratorSteer

ACTION_PHRASES = {
    "flee": "flee / leave home",
    "relocate": "relocate to another place",
    "travel": "travel onward",
    "smuggle": "pay a smuggler to be moved",
    "cross_border": "cross a border (often on foot, illegally)",
    "retry": "try again after being stopped or sent back",
    "walk": "walk on foot",
    "take_bus": "take a bus",
    "take_train": "take a train",
    "take_ferry": "take a ferry",
    "take_taxi": "take a taxi / hire a driver",
    "take_truck": "be smuggled in a truck",
    "board_boat": "board a boat (a smuggler's crossing)",
    "sea_crossing": "cross the sea",
    "reach_shore": "reach the shore / the island",
    "rescued": "be rescued at sea",
    "get_caught": "be caught / detained",
    "get_returned": "be returned / pushed back",
    "drown": "the boat capsizes",
    "arrive": "arrive",
}

PREMISE = ('You are the narrator of "On Your Shoes", an interactive experience. You narrate the '
           'journey of a person trying to reach Germany from Syria. You are the world: every wall, '
           'every smuggler, every closed border.')

RULES = (
    "1. Narrate in grounded, sensory, second-person prose. Keep it SHORT — one to three sentences per turn.\n"
    "2. NEVER break character. Never mention nodes, edges, actions, choices, a \"graph\", or that this is an "
    "experience. To the person, this is simply their life.\n"
    "3. The available ways forward are the ONLY real possibilities. Do not invent others.\n"
    "4. The situation is urgent: time, money and safety are running out."
)

FORCE_COMMIT_AFTER = 3  # turns of deliberation before the narrator is forced to commit


# --- Pydantic models for the function-call arguments (structured output) ---
class NarrateArgs(BaseModel):
    text: str = ""


class CommitArgs(BaseModel):
    action_id: str
    narration: str = ""


class SteerArgs(BaseModel):
    text: str = ""


# --- Function/tool definitions ---
NARRATE_TOOL = {
    "type": "function",
    "function": {
        "name": "narrate",
        "description": "Narrate the situation in second-person prose. Do NOT advance the journey.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The narration, 1-3 sentences."}},
            "required": ["text"],
        },
    },
}

COMMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "commit",
        "description": "The person has committed to a course of action. Advance the journey.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {"type": "string", "description": "The id from AVAILABLE WAYS FORWARD."},
                "narration": {"type": "string", "description": "Optional brief narration of the decision."},
            },
            "required": ["action_id"],
        },
    },
}

STEER_TOOL = {
    "type": "function",
    "function": {
        "name": "steer",
        "description": "The person tried something impossible. Steer them back, in-fiction.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Why it is impossible, in-fiction, 1-3 sentences."}},
            "required": ["text"],
        },
    },
}


DECIDE_TEMPLATE = PREMISE + """

Rules:
""" + RULES + """

You are at a moment of decision. Narrate the situation, then respond to what the person says. Push them
toward a decision. Commit the moment their intent is clear ("I'll pay", "let's go", "I'll walk", "we leave"
are all commitments). If they try something NOT among the available ways forward, steer them back in-fiction:
describe concretely why it is impossible. Never say "that's not an option".

CURRENT SITUATION:
{scene}

AVAILABLE WAYS FORWARD (match the id when the person commits):
{actions}

{pressure}
"""

MOVE_TEMPLATE = PREMISE + """

Rules:
""" + RULES + """

The person committed to a course of action: {action_text}. Narrate the consequences in ONE continuous
passage: what happened as they did it, then the road that followed ({corridor_text}), until they arrive at
{scene}. End the passage with a question or an in-fiction pressure that forces them to choose among the
available ways forward.

AVAILABLE WAYS FORWARD (at the new scene):
{actions}
"""

OPEN_TEMPLATE = PREMISE + """

Rules:
""" + RULES + """

Open the story. The person is in {origin_place}. They have decided to leave. Narrate the departure in ONE
continuous passage: leaving {origin_place}, then the road that followed ({corridor_text}), until they arrive
at {scene}. End the passage with a question or an in-fiction pressure that forces them to choose among the
available ways forward.

AVAILABLE WAYS FORWARD (at the new scene):
{actions}
"""

FORCE_TEXT = (
    "The person has had enough time to decide. They must choose NOW. Pick the action that best matches "
    "their situation and the conversation so far, and commit to it."
)


def _humanize(action):
    return ACTION_PHRASES.get(action, action.replace("_", " "))


class Narrator:
    def open(self, session, origin_id, corridor):
        raise NotImplementedError

    def decide(self, session):
        raise NotImplementedError

    def narrate_move(self, session, edge, corridor):
        raise NotImplementedError


class DeepSeekNarrator(Narrator):
    def __init__(self):
        from openai import OpenAI  # lazy import: only needed in real mode

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.environ.get("MODEL", "deepseek-chat")

    def _scene(self, node_id):
        n = graph.get_node(node_id)
        return f"{n['place']}, {n['country']}"

    def _actions_block(self, node_id):
        actions = graph.outgoing_edges(node_id)
        if actions:
            return "\n".join(
                f"- id: {a['id']} | {_humanize(a['action_type'])} | leads to: {a['to_place']}"
                for a in actions
            )
        return "(none — this is the end of the journey)"

    def _corridor_text(self, corridor):
        if not corridor:
            return "directly onward"
        return "; ".join(f"then {_humanize(c['action_type'])} toward {c['to_place']}" for c in corridor)

    def _call_tools(self, messages, tools, tool_choice: Any = "auto"):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.7,
            )
        except Exception as exc:
            print(f"[narrator] LLM call failed: {exc!r}", file=sys.stderr)
            return NarratorMessage(text=f"(The narrator failed: {exc})")
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        if not calls:
            content = msg.content or ""
            print(f"[narrator] no tool call; content={content!r}", file=sys.stderr)
            return NarratorMessage(text=str(content).strip())
        tc = calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        print(f"[narrator] tool={name} args={args!r}", file=sys.stderr)
        if name == "narrate":
            try:
                return NarratorMessage(text=NarrateArgs.model_validate(args).text)
            except ValidationError:
                return NarratorMessage(text="(invalid narrate args)")
        if name == "commit":
            try:
                a = CommitArgs.model_validate(args)
            except ValidationError as exc:
                return NarratorMessage(text=f"(invalid commit args: {exc})")
            return NarratorTransition(edge_id=a.action_id, narration=a.narration)
        if name == "steer":
            try:
                return NarratorSteer(text=SteerArgs.model_validate(args).text)
            except ValidationError:
                return NarratorSteer(text="That path is not open to you right now.")
        return NarratorMessage(text="(unknown tool)")

    @staticmethod
    def _text(resp):
        if isinstance(resp, NarratorTransition):
            return resp.narration or ""
        return resp.text or ""

    def _force(self, name):
        return {"type": "function", "function": {"name": name}}

    def open(self, session, origin_id, corridor):
        origin = graph.get_node(origin_id)
        system = OPEN_TEMPLATE.format(
            origin_place=origin["place"],
            corridor_text=self._corridor_text(corridor),
            scene=self._scene(session.current_node_id),
            actions=self._actions_block(session.current_node_id),
        )
        resp = self._call_tools(
            [{"role": "system", "content": system}, {"role": "user", "content": "Begin."}],
            [NARRATE_TOOL], self._force("narrate"),
        )
        return self._text(resp)

    def decide(self, session):
        forced = session.turns_since_transition >= FORCE_COMMIT_AFTER
        pressure = FORCE_TEXT if forced else ""
        system = DECIDE_TEMPLATE.format(
            scene=self._scene(session.current_node_id),
            actions=self._actions_block(session.current_node_id),
            pressure=pressure,
        )
        messages = [{"role": "system", "content": system}] + session.history[:-1] + [{"role": "user", "content": session.history[-1]["content"]}]
        if forced:
            return self._call_tools(messages, [COMMIT_TOOL], self._force("commit"))
        return self._call_tools(messages, [NARRATE_TOOL, COMMIT_TOOL, STEER_TOOL], "auto")

    def narrate_move(self, session, edge, corridor):
        to_place = graph.get_node(edge["to_node"])["place"]
        system = MOVE_TEMPLATE.format(
            action_text=f"{_humanize(edge['action_type'])} toward {to_place}",
            corridor_text=self._corridor_text(corridor),
            scene=self._scene(session.current_node_id),
            actions=self._actions_block(session.current_node_id),
        )
        resp = self._call_tools(
            [{"role": "system", "content": system}, {"role": "user", "content": "Continue."}],
            [NARRATE_TOOL], self._force("narrate"),
        )
        return self._text(resp)


class ScriptedNarrator(Narrator):
    """Deterministic stand-in (no API key). Advances when the person types COMMIT."""

    def open(self, session, origin_id, corridor):
        origin = graph.get_node(origin_id)
        fin = graph.get_node(session.current_node_id)
        mid = " -> ".join(c["to_place"] for c in corridor[:-1])
        path = (mid + " -> " + fin["place"]) if mid else fin["place"]
        return (f"You are in {origin['place']}. You decide to leave, and the road carries you to {path}. "
                f"(Scripted narrator — add a DeepSeek key to .env for the real thing.) What do you do?")

    def decide(self, session):
        actions = session.available_actions()
        last = session.history[-1]["content"] if session.history else ""
        if "COMMIT" in last.upper():
            if not actions:
                return NarratorMessage(text="There is nowhere left to go. The journey ends here.")
            return NarratorTransition(edge_id=actions[0]["id"], narration="")
        lines = "; ".join(f"{_humanize(a['action_type'])} toward {a['to_place']}" for a in actions)
        return NarratorMessage(text=f"You weigh what is possible: {lines}. (Type COMMIT to take the first path.)")

    def narrate_move(self, session, edge, corridor):
        to_place = graph.get_node(edge["to_node"])["place"]
        fin = graph.get_node(session.current_node_id)
        corr = " -> ".join(c["to_place"] for c in corridor) if corridor else ""
        return (f"You {_humanize(edge['action_type'])} toward {to_place}"
                + (f", then pass through {corr}" if corr else "")
                + f", arriving at {fin['place']}.")


def get_narrator():
    return DeepSeekNarrator() if os.environ.get("OPENAI_API_KEY") else ScriptedNarrator()
