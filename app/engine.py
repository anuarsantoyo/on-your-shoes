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

    def apply_transition(self, edge_id):
        """Advance along an edge if and only if it is real and leaves the current node.

        Returns the edge on success, None on a hallucinated/invalid edge.
        """
        edge = graph.get_edge(edge_id)
        if edge is None or edge["from_node"] != self.current_node_id:
            return None
        self.current_node_id = edge["to_node"]
        self.path.append(edge["to_node"])
        self.turns_since_transition = 0
        node = graph.get_node(self.current_node_id)
        if node and node["role"] == "terminal":
            self.ended = True
        return edge
