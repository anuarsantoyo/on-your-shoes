"""Narrators. The engine never talks to an LLM directly — it talks to a Narrator.

DeepSeekNarrator: the real one (OpenAI-compatible API, DeepSeek).
ScriptedNarrator: a deterministic stand-in so the whole loop runs with no API key.
"""
import json
import os

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

SYSTEM_TEMPLATE = """You are the narrator of "On Your Shoes", an interactive experience. You narrate, in vivid second-person prose, the journey of a person trying to reach Germany from Syria. You are the world: every wall, every smuggler, every closed border.

Rules:
1. Narrate in grounded, sensory prose. A few sentences at a time.
2. NEVER break character. Never mention nodes, edges, actions, choices, a "graph", or that this is an experience. To the person, this is simply their life.
3. The person will say what they want to do. Respond in prose, describing what happens.
4. Only when the person has CLEARLY committed to a course of action, return a "transition" naming the matching action id and narrating the move.
5. If the person tries something that is NOT among the available ways forward, steer them back IN-FICTION: describe concretely why it is not possible in their situation (no money, no documents, closed border, too dangerous). Never say "that's not an option".
6. The available ways forward are the ONLY real possibilities. Do not invent others.

CURRENT SITUATION:
{scene}

AVAILABLE WAYS FORWARD (match the id when the person commits):
{actions}

Return ONLY a JSON object in exactly one of these shapes, nothing else:
{{"type": "message", "text": "..."}}
{{"type": "transition", "edge_id": "<id>", "narration": "..."}}
{{"type": "steer", "text": "..."}}
"""


def _humanize(action):
    return ACTION_PHRASES.get(action, action.replace("_", " "))


class Narrator:
    def start(self, node_id):
        raise NotImplementedError

    def respond(self, session):
        raise NotImplementedError


class DeepSeekNarrator(Narrator):
    def __init__(self):
        from openai import OpenAI  # lazy import: only needed in real mode

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.environ.get("MODEL", "deepseek-chat")

    def _scene_and_actions(self, node_id):
        n = graph.get_node(node_id)
        scene = f"You are in {n['place']}, {n['country']}. {n['description'] or ''}".strip()
        actions = graph.outgoing_edges(node_id)
        if actions:
            lines = "\n".join(
                f"- id: {a['id']} | {_humanize(a['action_type'])} | leads to: {a['to_place']}"
                for a in actions
            )
        else:
            lines = "(none — this is the end of the journey)"
        return scene, lines

    def _call(self, node_id, history, user_prompt):
        scene, actions = self._scene_and_actions(node_id)
        system = SYSTEM_TEMPLATE.format(scene=scene, actions=actions)
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        return self._parse(resp.choices[0].message.content)

    @staticmethod
    def _parse(content):
        try:
            data = json.loads(content)
        except Exception:
            return NarratorMessage(text=content)
        t = data.get("type")
        if t == "transition":
            return NarratorTransition(edge_id=data.get("edge_id", ""), narration=data.get("narration", ""))
        if t == "steer":
            return NarratorSteer(text=data.get("text", ""))
        return NarratorMessage(text=data.get("text", ""))

    def start(self, node_id):
        return self._call(node_id, [], "Begin the story. Open the scene in a few vivid, concrete sentences.")

    def respond(self, session):
        return self._call(session.current_node_id, session.history[:-1], session.history[-1]["content"])


class ScriptedNarrator(Narrator):
    """Deterministic stand-in (no API key). Advances when the person types COMMIT."""

    def start(self, node_id):
        n = graph.get_node(node_id)
        return NarratorMessage(
            text=f"You are in {n['place']}, {n['country']}. The war has made staying impossible, and "
                 f"there is nothing left to hold you here. (Scripted narrator — add a DeepSeek key to "
                 f".env for the real thing.) What do you do?"
        )

    def respond(self, session):
        actions = session.available_actions()
        last = session.history[-1]["content"] if session.history else ""
        if "COMMIT" in last.upper():
            if not actions:
                return NarratorMessage(text="There is nowhere left to go. The journey ends here.")
            a = actions[0]
            return NarratorTransition(
                edge_id=a["id"],
                narration=f"You take the step: {_humanize(a['action_type'])} toward {a['to_place']}.",
            )
        lines = "; ".join(f"{_humanize(a['action_type'])} toward {a['to_place']}" for a in actions)
        return NarratorMessage(
            text=f"You weigh what is possible: {lines}. (Type COMMIT to take the first path.)"
        )


def get_narrator():
    return DeepSeekNarrator() if os.environ.get("OPENAI_API_KEY") else ScriptedNarrator()
