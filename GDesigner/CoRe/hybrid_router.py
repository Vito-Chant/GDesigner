"""
CoRe Framework: Hybrid Router Module
Implements System 1 (Fast) and System 2 (Slow) routing paths
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_agent: str
    reasoning: str
    confidence: float
    path_used: str  # 'fast' or 'slow'
    alternative_agents: List[Tuple[str, float]]  # [(agent_id, score), ...]
    cost_tokens: int  # Token cost for this decision


class HybridRouter:
    """
    Hybrid routing combining embedding similarity (fast) and LLM reasoning (slow)

    Key insight: Use cheap embeddings for easy cases, expensive LLM for ambiguous cases
    """

    def __init__(
            self,
            llm,  # LLM instance for slow path
            embedding_model,  # Sentence transformer for fast path
            confidence_threshold: float = 0.7,  # Switch to slow path if top-1 < this
            margin_threshold: float = 0.15,  # Switch to slow path if top-2 within this margin
            max_candidates_for_slow: int = 3,  # How many agents to consider in slow path
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.max_candidates_for_slow = max_candidates_for_slow

        # Statistics
        self.stats = {
            'fast_path_count': 0,
            'slow_path_count': 0,
            'total_tokens_used': 0
        }

    async def route(
            self,
            task_description: str,
            current_agent: str,
            candidate_agents: List[str],
            agent_profiles_text: Dict[str, str],  # agent_id -> description
            context_from_registry: str,  # Rich context from MindRegistry
            handoff_note: Optional[str] = None
    ) -> RoutingDecision:
        """
        Main routing function implementing the hybrid strategy

        Returns:
            RoutingDecision with selected agent and metadata
        """

        # FAST PATH (System 1): Embedding-based similarity
        fast_scores = await self._compute_fast_path_scores(
            task_description, candidate_agents, agent_profiles_text
        )

        # Check if we should use fast path
        should_use_fast = self._should_use_fast_path(fast_scores)

        if should_use_fast:
            return await self._execute_fast_path(
                fast_scores, candidate_agents, task_description
            )
        else:
            return await self._execute_slow_path(
                task_description, current_agent, candidate_agents,
                agent_profiles_text, context_from_registry,
                fast_scores, handoff_note
            )

    async def _compute_fast_path_scores(
            self,
            task_description: str,
            candidate_agents: List[str],
            agent_profiles_text: Dict[str, str]
    ) -> Dict[str, float]:
        """
        System 1: Fast embedding-based scoring
        Cost: ~0 tokens (just embedding computation)
        """
        # Get task embedding
        task_emb = self.embedding_model.encode(task_description)
        task_emb = torch.tensor(task_emb).unsqueeze(0)

        scores = {}
        for agent_id in candidate_agents:
            profile_text = agent_profiles_text.get(agent_id, "")
            if not profile_text:
                scores[agent_id] = 0.0
                continue

            # Get agent profile embedding
            agent_emb = self.embedding_model.encode(profile_text)
            agent_emb = torch.tensor(agent_emb).unsqueeze(0)

            # Cosine similarity
            similarity = F.cosine_similarity(task_emb, agent_emb, dim=1).item()
            scores[agent_id] = similarity

        return scores

    def _should_use_fast_path(self, scores: Dict[str, float]) -> bool:
        """
        Decision logic: Use fast path if top-1 is confident enough

        Criteria:
        1. Top-1 score > confidence_threshold (absolute confidence)
        2. Top-1 - Top-2 > margin_threshold (relative margin)
        """
        if not scores:
            return True

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_scores) < 2:
            return True

        top1_score = sorted_scores[0][1]
        top2_score = sorted_scores[1][1]

        # Check both criteria
        has_confidence = top1_score >= self.confidence_threshold
        has_margin = (top1_score - top2_score) >= self.margin_threshold

        return has_confidence and has_margin

    async def _execute_fast_path(
            self,
            scores: Dict[str, float],
            candidate_agents: List[str],
            task_description: str
    ) -> RoutingDecision:
        """Execute fast path routing"""
        self.stats['fast_path_count'] += 1

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_agent = sorted_scores[0][0]
        confidence = sorted_scores[0][1]

        reasoning = (
            f"Fast path: Selected based on embedding similarity. "
            f"Top candidate has clear advantage (score: {confidence:.3f})"
        )

        return RoutingDecision(
            selected_agent=selected_agent,
            reasoning=reasoning,
            confidence=confidence,
            path_used='fast',
            alternative_agents=sorted_scores[1:],
            cost_tokens=0
        )

    async def _execute_slow_path(
            self,
            task_description: str,
            current_agent: str,
            candidate_agents: List[str],
            agent_profiles_text: Dict[str, str],
            context_from_registry: str,
            fast_scores: Dict[str, float],
            handoff_note: Optional[str]
    ) -> RoutingDecision:
        """
        System 2: LLM-based semantic reasoning
        Cost: ~500-1000 tokens per decision

        This is where the magic happens - deep semantic understanding
        """
        self.stats['slow_path_count'] += 1

        # Select top-k candidates from fast path for detailed analysis
        sorted_scores = sorted(fast_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [agent_id for agent_id, _ in sorted_scores[:self.max_candidates_for_slow]]

        # Build prompt for LLM
        prompt = self._build_slow_path_prompt(
            task_description, current_agent, top_candidates,
            agent_profiles_text, context_from_registry, handoff_note
        )

        # Call LLM
        messages = [
            {'role': 'system', 'content': 'You are an expert at task delegation and team coordination.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)

        # Parse response
        selected_agent, reasoning, confidence = self._parse_llm_response(
            response, top_candidates
        )

        # Estimate token cost
        token_cost = len(prompt.split()) + len(response.split())
        self.stats['total_tokens_used'] += token_cost

        return RoutingDecision(
            selected_agent=selected_agent,
            reasoning=f"Slow path: {reasoning}",
            confidence=confidence,
            path_used='slow',
            alternative_agents=sorted_scores,
            cost_tokens=token_cost
        )

    def _build_slow_path_prompt(
            self,
            task_description: str,
            current_agent: str,
            top_candidates: List[str],
            agent_profiles_text: Dict[str, str],
            context_from_registry: str,
            handoff_note: Optional[str]
    ) -> str:
        """Build detailed prompt for LLM reasoning"""

        prompt = f"""You are coordinating a multi-agent system. Your job is to select the BEST agent for the next step.

CURRENT SITUATION:
- Current agent: {current_agent}
- Task to delegate: {task_description}
"""

        if handoff_note:
            prompt += f"\nHandoff note from current agent: {handoff_note}\n"

        prompt += f"\n{context_from_registry}\n"

        prompt += "\nTOP CANDIDATES (based on initial screening):\n"
        for i, agent_id in enumerate(top_candidates, 1):
            profile = agent_profiles_text.get(agent_id, "No profile available")
            prompt += f"\n{i}. {agent_id}:\n{profile}\n"

        prompt += """
INSTRUCTIONS:
1. Analyze the task requirements carefully
2. Consider each candidate's strengths and weaknesses
3. Think about potential failure modes
4. Select the MOST SUITABLE agent

Respond in this exact format:
SELECTED: <agent_id>
REASONING: <your detailed reasoning>
CONFIDENCE: <0.0-1.0>
"""

        return prompt

    def _parse_llm_response(
            self,
            response: str,
            top_candidates: List[str]
    ) -> Tuple[str, str, float]:
        """Parse LLM response to extract decision"""
        lines = response.strip().split('\n')

        selected_agent = None
        reasoning = ""
        confidence = 0.5

        for line in lines:
            line = line.strip()
            if line.startswith('SELECTED:'):
                selected_agent = line.replace('SELECTED:', '').strip()
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    confidence = 0.5

        # Fallback if parsing fails
        if selected_agent not in top_candidates:
            selected_agent = top_candidates[0]
            reasoning = "Failed to parse LLM response, using top candidate"
            confidence = 0.5

        return selected_agent, reasoning, confidence

    def get_statistics(self) -> Dict:
        """Get routing statistics"""
        total = self.stats['fast_path_count'] + self.stats['slow_path_count']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'fast_path_percentage': self.stats['fast_path_count'] / total * 100,
            'avg_tokens_per_decision': self.stats['total_tokens_used'] / total if total > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    from sentence_transformers import SentenceTransformer


    # Mock LLM for testing
    class MockLLM:
        async def agen(self, messages):
            return """SELECTED: code_expert_1
REASONING: The task requires implementing a mathematical formula in code. While the math solver could explain the formula, the code expert is better suited for actual implementation.
CONFIDENCE: 0.85"""


    async def test_router():
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        llm = MockLLM()

        router = HybridRouter(
            llm=llm,
            embedding_model=embedding_model,
            confidence_threshold=0.7,
            margin_threshold=0.15
        )

        # Test case
        decision = await router.route(
            task_description="Implement the quadratic formula to solve equations",
            current_agent="math_solver_1",
            candidate_agents=["code_expert_1", "math_solver_2"],
            agent_profiles_text={
                "code_expert_1": "Expert programmer specialized in numerical algorithms",
                "math_solver_2": "Mathematical reasoning specialist"
            },
            context_from_registry="Previous interactions show code_expert_1 excels at implementation tasks"
        )

        print(f"Decision: {decision.selected_agent}")
        print(f"Path: {decision.path_used}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Cost: {decision.cost_tokens} tokens")

        print("\nStatistics:", router.get_statistics())


    asyncio.run(test_router())