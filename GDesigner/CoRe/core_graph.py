"""
CoRe Framework: Main Graph Implementation
Integrates all components into a complete Cognitive Relay system
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time

from GDesigner.graph.graph import Graph
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from sentence_transformers import SentenceTransformer


@dataclass
class CoReResult:
    """Result of a CoRe execution"""
    final_answer: str
    execution_trace: List[Dict]
    routing_decisions: List[Dict]
    handoff_notes: List[str]
    belief_updates: List[Any]
    total_time: float
    total_cost_tokens: int
    success: bool


class CoReGraph:
    """
    Cognitive Relay Graph - Main orchestrator

    This is the complete system integrating:
    1. Mind Registry (semantic state)
    2. Hybrid Router (fast/slow routing)
    3. Handoff Generator (explicit transfers)
    4. Belief Evolver (online learning)
    """

    def __init__(
            self,
            domain: str,
            llm_name: str,
            available_roles: List[str],
            decision_method: str = "FinalRefer",
            max_routing: int = 10,
            registry_save_path: Optional[Path] = None,
            fast_path_threshold: float = 0.7,
            slow_path_margin: float = 0.15
    ):
        """Initialize CoRe Graph"""

        # Import local modules
        from mind_registry import MindRegistry, AgentProfile
        from hybrid_router import HybridRouter
        from handoff_generator import HandoffGenerator
        from belief_evolver import BeliefEvolver, InteractionTrace

        self.domain = domain
        self.llm_name = llm_name
        self.available_roles = available_roles
        self.max_routing = max_routing

        # Initialize LLM
        self.llm = LLMRegistry.get(llm_name)

        # Initialize embedding model for fast path
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize Mind Registry
        self.mind_registry = MindRegistry(save_path=registry_save_path)
        self._initialize_agent_profiles()

        # Initialize Hybrid Router
        self.hybrid_router = HybridRouter(
            llm=self.llm,
            embedding_model=self.embedding_model,
            confidence_threshold=fast_path_threshold,
            margin_threshold=slow_path_margin
        )

        # Initialize Handoff Generator
        self.handoff_generator = HandoffGenerator(llm=self.llm)

        # Initialize Belief Evolver
        self.belief_evolver = BeliefEvolver(
            llm=self.llm,
            mind_registry=self.mind_registry
        )

        # Initialize decision maker
        self.decision_maker = AgentRegistry.get(
            decision_method,
            domain=domain,
            llm_name=llm_name
        )

        # Execution state
        self.current_trace = []
        self.interaction_traces = []

    def _initialize_agent_profiles(self):
        """Initialize agent profiles in Mind Registry from domain"""

        from mind_registry import AgentProfile
        from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

        prompt_set = PromptSetRegistry.get(self.domain)

        for role in self.available_roles:
            try:
                description = prompt_set.get_description(role)

                # Parse capabilities (simplified)
                capabilities = []
                if "math" in role.lower() or "solver" in role.lower():
                    capabilities = ["mathematical reasoning", "problem solving"]
                elif "code" in role.lower() or "program" in role.lower():
                    capabilities = ["programming", "implementation", "debugging"]
                elif "analyst" in role.lower():
                    capabilities = ["analysis", "planning", "abstraction"]

                profile = AgentProfile(
                    agent_id=f"{role.lower().replace(' ', '_')}",
                    role=role,
                    capabilities=capabilities,
                    specializations=[role],
                    limitations=[],
                    description=description
                )

                self.mind_registry.register_agent(profile)
            except Exception as e:
                print(f"Warning: Could not register profile for {role}: {e}")

    async def run_cognitive_relay(
            self,
            input_dict: Dict[str, str],
            temperature: float = 1.0,
            training: bool = False
    ) -> CoReResult:
        """
        Main execution loop - The Cognitive Relay

        Args:
            input_dict: Task input with 'task' key
            temperature: Sampling temperature
            training: Whether in training mode

        Returns:
            CoReResult with complete execution details
        """

        start_time = time.time()
        task = input_dict['task']

        # Reset state
        self.current_trace = []
        self.interaction_traces = []
        routing_decisions = []
        handoff_notes = []
        total_tokens = 0

        current_agent = None
        current_output = None
        handoff_note = None

        print(f"\n{'=' * 60}")
        print(f"CoRe: Starting Cognitive Relay")
        print(f"Task: {task[:100]}...")
        print(f"{'=' * 60}\n")

        for step in range(self.max_routing):
            print(f"\n--- Step {step + 1}/{self.max_routing} ---")

            # Get candidate agents (all available except current)
            candidate_agents = [
                role.lower().replace(' ', '_')
                for role in self.available_roles
                if role.lower().replace(' ', '_') != current_agent
            ]

            # Add decision maker as final option
            candidate_agents.append(self.decision_maker.id)

            # Build routing context from Mind Registry
            context = self.mind_registry.get_context_for_routing(
                current_agent=current_agent or "system",
                candidate_agents=candidate_agents,
                task_description=task
            )

            # Get agent profile texts
            agent_profiles_text = {}
            for agent_id in candidate_agents:
                profile = self.mind_registry.get_agent_profile(agent_id)
                if profile:
                    agent_profiles_text[agent_id] = profile.to_text()

            # === ROUTING DECISION (System 1 or System 2) ===
            routing_decision = await self.hybrid_router.route(
                task_description=task,
                current_agent=current_agent or "system",
                candidate_agents=candidate_agents,
                agent_profiles_text=agent_profiles_text,
                context_from_registry=context,
                handoff_note=handoff_note
            )

            print(f"Router: {routing_decision.path_used.upper()} path")
            print(f"Selected: {routing_decision.selected_agent}")
            print(f"Confidence: {routing_decision.confidence:.2f}")
            print(f"Reasoning: {routing_decision.reasoning[:100]}...")

            routing_decisions.append({
                'step': step + 1,
                'selected': routing_decision.selected_agent,
                'path': routing_decision.path_used,
                'confidence': routing_decision.confidence,
                'reasoning': routing_decision.reasoning
            })

            total_tokens += routing_decision.cost_tokens

            # Check if decision maker was selected (terminal condition)
            if routing_decision.selected_agent == self.decision_maker.id:
                print("\n🎯 Decision maker selected - reaching consensus...")

                # Execute decision maker
                final_output = await self._execute_decision_maker(
                    input_dict, self.current_trace
                )

                execution_time = time.time() - start_time

                print(f"\n{'=' * 60}")
                print(f"CoRe: Relay Complete")
                print(f"Total Steps: {step + 1}")
                print(f"Time: {execution_time:.2f}s")
                print(f"Tokens: {total_tokens}")
                print(f"{'=' * 60}\n")

                return CoReResult(
                    final_answer=final_output,
                    execution_trace=self.current_trace,
                    routing_decisions=routing_decisions,
                    handoff_notes=handoff_notes,
                    belief_updates=[],
                    total_time=execution_time,
                    total_cost_tokens=total_tokens,
                    success=True
                )

            # === AGENT EXECUTION ===
            next_agent_id = routing_decision.selected_agent
            next_agent = await self._get_agent_instance(next_agent_id)

            # Prepare input with handoff note if available
            agent_input = input_dict.copy()
            if handoff_note:
                agent_input['hints'] = handoff_note

            # Execute agent
            print(f"Executing: {next_agent_id}...")
            agent_output = await self._execute_agent(next_agent, agent_input)
            print(f"Output preview: {agent_output[:100]}...")

            # Record trace
            self.current_trace.append({
                'step': step + 1,
                'agent': next_agent_id,
                'output': agent_output,
                'routing_decision': routing_decision
            })

            # === HANDOFF GENERATION ===
            if current_agent is not None:  # Not first step
                print("Generating handoff note...")

                next_profile = self.mind_registry.get_agent_profile(next_agent_id)

                handoff = await self.handoff_generator.generate_handoff_note(
                    current_agent=current_agent,
                    next_agent=next_agent_id,
                    original_task=task,
                    current_agent_output=current_output or "",
                    full_history=self.current_trace,
                    next_agent_profile=next_profile.to_text() if next_profile else "",
                    routing_reasoning=routing_decision.reasoning
                )

                handoff_note = handoff.to_text()
                handoff_notes.append(handoff_note)

                print(f"Handoff confidence: {handoff.confidence_in_handoff:.2f}")
                if handoff.warnings:
                    print(f"⚠️  Warnings: {len(handoff.warnings)}")

            # Update state for next iteration
            current_agent = next_agent_id
            current_output = agent_output

        # Max routing reached without decision
        print("\n⚠️  Max routing steps reached - forcing decision...")
        final_output = await self._execute_decision_maker(input_dict, self.current_trace)

        return CoReResult(
            final_answer=final_output,
            execution_trace=self.current_trace,
            routing_decisions=routing_decisions,
            handoff_notes=handoff_notes,
            belief_updates=[],
            total_time=time.time() - start_time,
            total_cost_tokens=total_tokens,
            success=False  # Didn't converge naturally
        )

    async def _get_agent_instance(self, agent_id: str):
        """Get or create agent instance"""

        # Map agent_id back to role
        role = agent_id.replace('_', ' ').title()

        # Find matching role
        for available_role in self.available_roles:
            if available_role.lower().replace(' ', '_') == agent_id:
                role = available_role
                break

        # Create agent instance
        # Determine agent type from domain
        if self.domain == "gsm8k":
            agent_class = "MathSolver"
        elif self.domain == "humaneval":
            agent_class = "CodeWriting"
        elif self.domain == "mmlu":
            agent_class = "AnalyzeAgent"
        else:
            agent_class = "MathSolver"  # Default

        agent = AgentRegistry.get(
            agent_class,
            domain=self.domain,
            llm_name=self.llm_name,
            role=role
        )

        return agent

    async def _execute_agent(self, agent, input_dict: Dict) -> str:
        """Execute an agent and return its output"""

        # Use the async_execute_with_hints method
        await agent.async_execute_with_hints(input_dict)

        if agent.outputs:
            return agent.outputs[-1]
        return "No output generated"

    async def _execute_decision_maker(
            self,
            input_dict: Dict,
            trace: List[Dict]
    ) -> str:
        """Execute decision maker with full trace"""

        # Build context from trace
        spatial_info = {}
        for i, step in enumerate(trace):
            spatial_info[f"agent_{i}"] = {
                'role': step['agent'],
                'output': step['output']
            }

        # Execute decision maker
        await self.decision_maker.async_execute(input_dict)

        if self.decision_maker.outputs:
            return self.decision_maker.outputs[-1]
        return "No decision produced"

    def get_statistics(self) -> Dict:
        """Get execution statistics"""

        router_stats = self.hybrid_router.get_statistics()
        evolution_stats = self.belief_evolver.get_evolution_summary()

        return {
            'routing': router_stats,
            'evolution': evolution_stats,
            'total_beliefs': len(self.mind_registry.beliefs),
            'registered_agents': len(self.mind_registry.profiles)
        }


# Example usage
if __name__ == "__main__":
    async def test_core():
        # Initialize CoRe Graph
        core = CoReGraph(
            domain="gsm8k",
            llm_name="Qwen/Qwen3-4B-Instruct-2507",
            available_roles=[
                "Math Solver",
                "Mathematical Analyst",
                "Programming Expert",
                "Inspector"
            ],
            max_routing=5
        )

        # Test task
        input_dict = {
            "task": "Solve the equation 2x^2 + 5x - 3 = 0"
        }

        # Run cognitive relay
        result = await core.run_cognitive_relay(input_dict)

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(f"Answer: {result.final_answer}")
        print(f"Success: {result.success}")
        print(f"Steps: {len(result.execution_trace)}")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"Total Tokens: {result.total_cost_tokens}")

        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        stats = core.get_statistics()
        print(f"Fast path: {stats['routing']['fast_path_percentage']:.1f}%")
        print(f"Slow path: {100 - stats['routing']['fast_path_percentage']:.1f}%")
        print(f"Avg tokens/decision: {stats['routing']['avg_tokens_per_decision']:.0f}")


    # Run test
    asyncio.run(test_core())