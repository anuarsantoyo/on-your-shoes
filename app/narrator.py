"""Narrators. The engine never talks to an LLM directly — it talks to a Narrator.

DeepSeekNarrator: the real one (OpenAI-compatible API, DeepSeek).
ScriptedNarrator: a deterministic stand-in so the whole loop runs with no API key.

The narrator has three jobs:
- open():          narrate the opening (origin -> corridor -> first decision).
- decide():        at a decision node, narrate and classify the person's intent
                   into a message / a committed transition / a steer.
- narrate_move():  narrate a committed action + any auto-advanced corridor in one passage.
"""
import json
import os
import sys

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

DECIDE_TEMPLATE = PREMISE + """

Rules:
""" + RULES + """

You are at a moment of decision. Narrate the situation, then respond to what the person says. Push them
toward a decision; end your narration with a question or an in-fiction pressure that forces them to choose
among the available ways forward. Commit the moment their intent is clear ("I'll pay", "let's go",
"I'll walk", "we leave" are all commitments) — return a "transition" naming the matching action id. If they
try something NOT among the available ways forward, steer them back IN-FICTION: describe concretely why it
is impossible. Never say "that's not an option".

CURRENT SITUATION:
{scene}

AVAILABLE WAYS FORWARD (match the id when the person commits):
{actions}

{pressure}

Return ONLY a JSON object in exactly one of these shapes, nothing else:
{{"type": "message", "text": "..."}}
{{"type": "transition", "edge_id": "<id>"}}
{{"type": "steer", "text": "..."}}
"""

MOVE_TEMPLATE = PREMISE + """

Rules:
""" + RULES + """

The person committed to a course of action: {action_text}. Narrate the consequences in ONE continuous
passage: what happened as they did it, then the road that followed ({corridor_text}), until they arrive at
{scene}. End the passage at the new scene with a question or an in-fiction pressure that forces them to
choose among the available ways forward.

AVAILABLE WAYS FORWARD (at the new scene):
{actions}

Return ONLY a JSON object in this shape:
{{"type": "message", "text": "..."}}
"""

OPEN_TEMPLATE = PREMISE + """

Rules:
""" + RULES + """

Open the story. The person is in {origin_place}. They have decided to leave. Narrate the departure in ONE
continuous passage: leaving {origin_place}, then the road that followed ({corridor_text}), until they arrive
at {scene}. End the passage at the new scene with a question or an in-fiction pressure that forces them to
choose among the available ways forward.

AVAILABLE WAYS FORWARD (at the new scene):
{actions}

Return ONLY a JSON object in this shape:
{{"type": "message", "text": "..."}}
"""

PRESSURE_TEXT = (
    "The person has been hesitating for several turns. The situation is urgent: "
    "raise the stakes now — a smuggler is about to leave, a patrol is approaching, "
    "money or time is running out — and push them to decide among the available ways "
    "forward this turn."
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

    def _llm(self, messages):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
            )
        except Exception as exc:
            print(f"[narrator] LLM call failed: {exc!r}", file=sys.stderr)
            return NarratorMessage(text=f"(The narrator failed: {exc})")
        content = resp.choices[0].message.content
        print(f"[narrator] raw ({len(content or '')} chars): {content!r}", file=sys.stderr)
        return self._parse(content)

    @staticmethod
    def _parse(content):
        if content is None:
            return NarratorMessage(text="")
        text = str(content).strip()
        if not text:
            return NarratorMessage(text="")
        # DeepSeek sometimes wraps JSON in markdown fences
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
        except Exception:
            print("[narrator] JSON parse failed; using raw text as narration", file=sys.stderr)
            return NarratorMessage(text=text)
        if not isinstance(data, dict):
            return NarratorMessage(text=text)
        t = data.get("type")
        if t == "transition":
            return NarratorTransition(edge_id=data.get("edge_id", ""), narration=data.get("narration", ""))
        if t == "steer":
            return NarratorSteer(text=data.get("text", ""))
        return NarratorMessage(text=data.get("text", ""))

    @staticmethod
    def _text(resp):
        if isinstance(resp, NarratorTransition):
            return resp.narration or ""
        return resp.text or ""

    def open(self, session, origin_id, corridor):
        origin = graph.get_node(origin_id)
        system = OPEN_TEMPLATE.format(
            origin_place=origin["place"],
            corridor_text=self._corridor_text(corridor),
            scene=self._scene(session.current_node_id),
            actions=self._actions_block(session.current_node_id),
        )
        return self._text(self._llm([{"role": "system", "content": system}, {"role": "user", "content": "Begin."}]))

    def decide(self, session):
        pressure = PRESSURE_TEXT if session.turns_since_transition >= 2 else ""
        system = DECIDE_TEMPLATE.format(
            scene=self._scene(session.current_node_id),
            actions=self._actions_block(session.current_node_id),
            pressure=pressure,
        )
        messages = [{"role": "system", "content": system}] + session.history[:-1] + [{"role": "user", "content": session.history[-1]["content"]}]
        return self._llm(messages)

    def narrate_move(self, session, edge, corridor):
        to_place = graph.get_node(edge["to_node"])["place"]
        system = MOVE_TEMPLATE.format(
            action_text=f"{_humanize(edge['action_type'])} toward {to_place}",
            corridor_text=self._corridor_text(corridor),
            scene=self._scene(session.current_node_id),
            actions=self._actions_block(session.current_node_id),
        )
        return self._text(self._llm([{"role": "system", "content": system}, {"role": "user", "content": "Continue."}]))


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
