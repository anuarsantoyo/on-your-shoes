"""The session state machine — pure logic, no LLM, fully testable."""
import uuid

from . import graph


class Session:
    def __init__(self, node_id):
        self.session_id = str(uuid.uuid4())
        self.current_node_id = node_id
        self.path = [node_id]            # ordered node ids visited
        self.history = []                # list of {"role", "content"}
        self.ended = False
        self.turns_since_transition = 0  # turns elapsed since the last state change

    def available_actions(self):
        return graph.outgoing_edges(self.current_node_id)

    def _apply_edge(self, edge_id):
        edge = graph.get_edge(edge_id)
        self.current_node_id = edge["to_node"]
        self.path.append(edge["to_node"])
        self.turns_since_transition = 0
        node = graph.get_node(self.current_node_id)
        if node and node["role"] == "terminal":
            self.ended = True
        return edge

    def apply_transition(self, edge_id):
        """Advance along an edge if and only if it is real and leaves the current node.

        Returns the edge on success, None on a hallucinated/invalid edge.
        """
        edge = graph.get_edge(edge_id)
        if edge is None or edge["from_node"] != self.current_node_id:
            return None
        return self._apply_edge(edge_id)

    def advance_to_decision(self):
        """Auto-advance through one-way (out-degree 1) nodes until a decision or terminal.

        Returns the corridor as a list of {"action_type", "to_place"} for each
        auto-followed edge (empty if already at a decision/terminal).
        """
        corridor = []
        while True:
            edges = self.available_actions()
            if len(edges) != 1:
                break
            edge = self._apply_edge(edges[0]["id"])
            to_node = graph.get_node(edge["to_node"])
            corridor.append({"action_type": edge["action_type"],
                             "to_place": to_node["place"] if to_node else edge["to_node"]})
        return corridor
