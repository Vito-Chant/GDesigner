import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import weave

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    Modified for MAGRPO training:
    1. Removed graph topology learning (GCN, MLP, optimizable structure logits).
    2. Topology is now strictly determined by fixed_spatial/temporal_masks AND Role Connections.
    3. Collects and returns log_probs from agent execution for RL training.
    4. Maintained __init__ signature compatibility.
    """

    def __init__(self,
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 # --- Kept for compatibility but ignored/disabled ---
                 optimized_spatial: bool = False,
                 initial_spatial_probability: float = 0.5,
                 # ---------------------------------------------------
                 fixed_spatial_masks: List[List[int]] = None,
                 # --- Kept for compatibility but ignored/disabled ---
                 optimized_temporal: bool = False,
                 initial_temporal_probability: float = 0.5,
                 # ---------------------------------------------------
                 fixed_temporal_masks: List[List[int]] = None,
                 node_kwargs: List[Dict] = None,
                 tokens=None
                 ):

        # --- Topology Configuration (Fixed) ---
        if fixed_spatial_masks is None:
            # Default: Fully connected (excluding self-loops)
            fixed_spatial_masks = [[1 if i != j else 0 for j in range(len(agent_names))] for i in
                                   range(len(agent_names))]
        if fixed_temporal_masks is None:
            # Default: Fully connected temporal
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]

        self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)

        assert len(self.fixed_spatial_masks) == len(agent_names) * len(
            agent_names), "The fixed_spatial_masks doesn't match the number of agents"
        assert len(self.fixed_temporal_masks) == len(agent_names) * len(
            agent_names), "The fixed_temporal_masks doesn't match the number of agents"

        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        # Kept attributes to avoid attribute errors if referenced externally,
        # but set to False/None as we don't support optimization anymore.
        self.optimized_spatial = False
        self.optimized_temporal = False

        # --- Node Initialization ---
        self.decision_node: Node = AgentRegistry.get(decision_method,
                                                     **{"domain": self.domain, "llm_name": self.llm_name})
        self.nodes: Dict[str, Node] = {}
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.init_nodes()

        # Potential edges list (used for indexing masks)
        self.potential_spatial_edges: List[List[str, str]] = []
        self.potential_temporal_edges: List[List[str, str]] = []
        self.init_potential_edges()

        self.prompt_set = PromptSetRegistry.get(domain)

        # --- NEW: Load Role Connections explicitly as Constraints ---
        # Original code used construct_adj_matrix + GCN to enforce this.
        # Now we must enforce it explicitly during connection construction.
        self.valid_role_connections = set(self.prompt_set.get_role_connection())

    # --- Properties (Kept for compatibility/visualization) ---
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges

    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among {[node.id for node in self.nodes.values()]}")

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_potential_edges(self):
        node_ids = list(self.nodes.keys())
        for node1_id in node_ids:
            for node2_id in node_ids:
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    def clear_spatial_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def clear_temporal_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def is_role_connection_valid(self, source_node: Node, target_node: Node) -> bool:
        """Helper to check if a connection violates role constraints."""
        # If valid_role_connections is empty, it usually means no constraints (all allowed),
        # but if it has content, we strictly enforce it.
        if not self.valid_role_connections:
            return True
        return (source_node.role, target_node.role) in self.valid_role_connections

    def construct_spatial_connection(self):
        """
        Constructs spatial connections based on fixed_spatial_masks AND Role Constraints.
        Removed temperature/threshold params as logic is now deterministic.
        """
        self.clear_spatial_connection()

        for i, potential_connection in enumerate(self.potential_spatial_edges):
            # 1. Check Fixed Mask (Topology Baseline)
            if self.fixed_spatial_masks[i] == 1:
                out_node = self.find_node(potential_connection[0])  # Source
                in_node = self.find_node(potential_connection[1])  # Target

                # 2. Check Role Constraints (Logic from Prompt Set)
                if self.is_role_connection_valid(out_node, in_node):
                    # 3. Check Cycle (Prevent DAG violations if needed)
                    if not self.check_cycle(in_node, {out_node}):
                        out_node.add_successor(in_node, 'spatial')

    def construct_temporal_connection(self, round: int = 0):
        """
        Constructs temporal connections.
        Role constraints are usually spatial, so we stick to masks here unless specified otherwise.
        """
        self.clear_temporal_connection()
        if round == 0:
            return

        for i, potential_connection in enumerate(self.potential_temporal_edges):
            if self.fixed_temporal_masks[i] == 1:
                out_node = self.find_node(potential_connection[0])
                in_node = self.find_node(potential_connection[1])
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')

    def run(self, inputs: Any,
            num_rounds: int = 3,
            max_tries: int = 3,
            max_time: int = 600) -> List[Any]:
        # Standard run method (retained for inference compatibility)
        log_probs = 0  # Placeholder, as we don't have structural log_probs anymore
        for round in range(num_rounds):
            self.construct_spatial_connection()
            self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        return final_answers, log_probs

    @weave.op()
    async def arun(self, input: Dict[str, str],
                   num_rounds: int = 3,
                   max_tries: int = 3,
                   max_time: int = 600) -> Tuple[List[Any], torch.Tensor, int]:
        """
        Async execution used for training.
        Returns:
            final_answers: List of answers
            total_agent_log_prob: Sum of log_probs from all agent actions
        """
        total_agent_log_prob = 0
        collected_action_indices = []

        # Removed GCN inference logic

        for round in range(num_rounds):
            self.construct_spatial_connection()
            self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        result = await asyncio.wait_for(
                            self.nodes[current_node_id].async_execute(input),
                            timeout=max_time
                        )

                        # Accumulate agent log_probs for RL
                        log_prob = result[0][1]
                        assert isinstance(log_prob, torch.Tensor)
                        total_agent_log_prob = total_agent_log_prob + log_prob
                        collected_action_indices.append(result[0][2])

                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                        import traceback
                        traceback.print_exc()
                    tries += 1

                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        return final_answers, total_agent_log_prob, sum(collected_action_indices)

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False
