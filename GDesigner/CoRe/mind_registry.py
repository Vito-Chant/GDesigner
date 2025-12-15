"""
CoRe Framework: Mind Registry Module
Maintains dynamic profiles and beliefs about agents
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AgentProfile:
    """Static capability description of an agent"""
    agent_id: str
    role: str
    capabilities: List[str]
    specializations: List[str]
    limitations: List[str]
    description: str

    def to_text(self) -> str:
        """Convert profile to natural language description"""
        text = f"Agent {self.agent_id} ({self.role}):\n"
        text += f"Description: {self.description}\n"
        text += f"Capabilities: {', '.join(self.capabilities)}\n"
        text += f"Specializations: {', '.join(self.specializations)}\n"
        if self.limitations:
            text += f"Known Limitations: {', '.join(self.limitations)}\n"
        return text


@dataclass
class RelationalBelief:
    """Dynamic belief about agent interactions"""
    from_agent: str
    to_agent: str
    belief_type: str  # 'trust', 'capability_assessment', 'interaction_pattern'
    content: str
    confidence: float  # 0-1
    evidence_count: int  # How many interactions support this
    last_updated: str

    def to_text(self) -> str:
        """Convert belief to natural language"""
        return f"{self.from_agent} believes: {self.content} (confidence: {self.confidence:.2f})"


class MindRegistry:
    """
    Central registry maintaining all agent profiles and beliefs
    This is the semantic state space S in the paper
    """

    def __init__(self, save_path: Optional[Path] = None):
        self.profiles: Dict[str, AgentProfile] = {}
        self.beliefs: List[RelationalBelief] = []
        self.save_path = save_path

    def register_agent(self, profile: AgentProfile):
        """Register a new agent profile"""
        self.profiles[profile.agent_id] = profile

    def add_belief(self, belief: RelationalBelief):
        """Add or update a relational belief"""
        # Check if similar belief exists
        existing = None
        for i, b in enumerate(self.beliefs):
            if (b.from_agent == belief.from_agent and
                    b.to_agent == belief.to_agent and
                    b.belief_type == belief.belief_type):
                existing = i
                break

        if existing is not None:
            # Update existing belief
            self.beliefs[existing] = belief
        else:
            # Add new belief
            self.beliefs.append(belief)

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Retrieve agent profile"""
        return self.profiles.get(agent_id)

    def get_beliefs_about(self, to_agent: str, from_agent: Optional[str] = None) -> List[RelationalBelief]:
        """Get all beliefs about a specific agent"""
        if from_agent:
            return [b for b in self.beliefs
                    if b.to_agent == to_agent and b.from_agent == from_agent]
        return [b for b in self.beliefs if b.to_agent == to_agent]

    def get_beliefs_from(self, from_agent: str) -> List[RelationalBelief]:
        """Get all beliefs held by a specific agent"""
        return [b for b in self.beliefs if b.from_agent == from_agent]

    def get_context_for_routing(self, current_agent: str, candidate_agents: List[str],
                                task_description: str) -> str:
        """
        Generate rich context for LLM-based routing decision
        This is the input to the Semantic Router
        """
        context = f"Current Task: {task_description}\n\n"
        context += f"Current Agent: {current_agent}\n"

        # Add profiles of candidate agents
        context += "\nCandidate Agents:\n"
        for agent_id in candidate_agents:
            profile = self.get_agent_profile(agent_id)
            if profile:
                context += f"\n{profile.to_text()}\n"

        # Add relevant beliefs
        context += "\nRelevant Past Interactions:\n"
        for agent_id in candidate_agents:
            beliefs = self.get_beliefs_about(agent_id, from_agent=current_agent)
            if beliefs:
                for belief in beliefs[-3:]:  # Last 3 beliefs
                    context += f"- {belief.to_text()}\n"

        return context

    def save(self):
        """Save registry to disk"""
        if self.save_path is None:
            return

        data = {
            'profiles': {k: asdict(v) for k, v in self.profiles.items()},
            'beliefs': [asdict(b) for b in self.beliefs]
        }

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load registry from disk"""
        if self.save_path is None or not self.save_path.exists():
            return

        with open(self.save_path, 'r') as f:
            data = json.load(f)

        self.profiles = {k: AgentProfile(**v) for k, v in data['profiles'].items()}
        self.beliefs = [RelationalBelief(**b) for b in data['beliefs']]


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = MindRegistry()

    # Register math solver
    math_profile = AgentProfile(
        agent_id="math_solver_1",
        role="Math Solver",
        capabilities=["arithmetic", "algebra", "geometry"],
        specializations=["step-by-step reasoning", "equation solving"],
        limitations=["struggles with very large numbers", "no symbolic computation"],
        description="Specialized in solving mathematical problems through systematic reasoning"
    )
    registry.register_agent(math_profile)

    # Register code expert
    code_profile = AgentProfile(
        agent_id="code_expert_1",
        role="Programming Expert",
        capabilities=["python", "algorithm design", "debugging"],
        specializations=["numerical computation", "data processing"],
        limitations=["no expertise in web frameworks"],
        description="Expert programmer capable of implementing algorithms and debugging code"
    )
    registry.register_agent(code_profile)

    # Add a belief based on past interaction
    belief = RelationalBelief(
        from_agent="math_solver_1",
        to_agent="code_expert_1",
        belief_type="capability_assessment",
        content="Code expert is excellent at converting mathematical formulas into executable code",
        confidence=0.85,
        evidence_count=5,
        last_updated="2025-01-15"
    )
    registry.add_belief(belief)

    # Generate routing context
    context = registry.get_context_for_routing(
        current_agent="math_solver_1",
        candidate_agents=["code_expert_1"],
        task_description="Implement the quadratic formula to solve ax^2 + bx + c = 0"
    )

    print("=== Routing Context ===")
    print(context)