"""
CoRe Framework v4.3.1: Main Graph Implementation - 修复路径记录
完整集成：Retrieve(Reranker) -> Execute -> Store(List) -> Route(LLM with Path Awareness)

v4.3.1 关键修复:
- 正确记录冷启动到 routing_history
- 完整的路径追踪（含推理和建议）
- 循环检测和警告
"""

import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

from GDesigner.graph.graph import Graph
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
import weave


@dataclass
class CoReResult:
    """CoRe执行结果"""
    final_answer: str
    execution_trace: List[Dict]
    routing_decisions: List[Dict]
    belief_updates: List[Any]
    total_time: float
    total_cost_tokens: int
    success: bool
    kv_cache_hits: int
    loop_detections: int  # **v4.3.1新增**


class CoReGraph:
    """
    Cognitive Relay Graph v4.3.1 - 主编排器 (修复路径记录)

    核心流程：
    Step 0: 冷启动 (Reranker) - **记录到 routing_history**
    循环 (max_routing次):
        Step 1: Retrieve (Reranker) - RAG检索历史
        Step 2: Execute - Agent执行，返回 (output, messages)
        Step 3: Store - 存储输出
        Step 4: Post-hoc Route (LLM) - **使用完整路径历史进行决策**
    """

    def __init__(
            self,
            domain: str,
            llm_name: str,
            available_roles: List[str],
            decision_method: str = "FinalRefer",
            max_routing: int = 10,
            registry_save_path: Optional[Path] = None,
            reranker_model: str = "BAAI/bge-reranker-v2-m3",
            rag_top_k: int = 3,
            max_loop_count: int = 2  # **v4.3.1新增**
    ):
        """初始化CoRe Graph"""
        from GDesigner.CoRe.mind_registry import MindRegistry
        from GDesigner.CoRe.unified_ranker import UnifiedRanker
        from GDesigner.CoRe.belief_evolver import BeliefEvolver

        self.domain = domain
        self.llm_name = llm_name
        self.available_roles = available_roles
        self.max_routing = max_routing
        self.rag_top_k = rag_top_k

        # 初始化组件
        self.llm = LLMRegistry.get(llm_name)
        self.mind_registry = MindRegistry(save_path=registry_save_path)
        self._initialize_agent_profiles()

        self.decision_maker = AgentRegistry.get(
            decision_method,
            domain=domain,
            llm_name=llm_name,
            id="final_decision"
        )

        # **v4.3.2: 传递 Decision Maker ID 到 Ranker**
        self.unified_ranker = UnifiedRanker(
            llm=self.llm,
            reranker_model_name=reranker_model,
            max_loop_count=max_loop_count,
            decision_maker_id=self.decision_maker.id  # **新增参数**
        )

        self.belief_evolver = BeliefEvolver(
            llm=self.llm,
            mind_registry=self.mind_registry
        )

        # 执行状态
        self.history_trace = []
        self.current_trace = []
        self.interaction_traces = []
        self.kv_cache_hits = 0
        self.loop_detections = 0
        self.termination_attempts = 0  # **v4.3.2新增**

    def _initialize_agent_profiles(self):
        """从domain初始化Agent profiles"""
        from GDesigner.CoRe.mind_registry import AgentProfile
        from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

        prompt_set = PromptSetRegistry.get(self.domain)

        for role in self.available_roles:
            try:
                description = prompt_set.get_description(role)

                capabilities = []
                if "math" in role.lower():
                    capabilities = ["mathematical reasoning", "problem solving"]
                elif "code" in role.lower():
                    capabilities = ["programming", "implementation"]
                elif "analyst" in role.lower():
                    capabilities = ["analysis", "planning"]

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

    @weave.op()
    async def run_cognitive_relay(
            self,
            input_dict: Dict[str, str],
            temperature: float = 1.0,
            training: bool = False
    ) -> CoReResult:
        """
        主执行循环 - Cognitive Relay (v4.3.2 修复终止逻辑)
        """

        start_time = time.time()
        task = input_dict['task']

        # 重置状态
        self.history_trace = []
        self.current_trace = []
        self.interaction_traces = []
        routing_decisions = []
        total_tokens = 0
        self.kv_cache_hits = 0
        self.loop_detections = 0
        self.termination_attempts = 0

        print(f"\n{'=' * 60}")
        print(f"CoRe v4.3.2: Starting Cognitive Relay (Fixed Termination)")
        print(f"Decision Maker ID: {self.decision_maker.id}")
        print(f"Task: {task[:100]}...")
        print(f"{'=' * 60}\n")

        # **Step 0: 冷启动**
        print("=== Step 0: Cold Start (Reranker) ===")
        profiles = {
            agent_id: self.mind_registry.get_agent_profile(agent_id).to_text()
            for agent_id in [role.lower().replace(' ', '_') for role in self.available_roles]
        }

        current_agent = await self.unified_ranker.cold_start(task, profiles)
        print(f"Cold Start Selected: {current_agent}\n")

        routing_decisions.append({
            'step': 0,
            'selected': current_agent,
            'reasoning': 'Cold start selection by Reranker based on task-profile similarity',
            'suggestion': 'Analyze the task and provide your expert perspective',
            'method': 'reranker'
        })

        current_output = None
        insight_instruction = "Analyze the task and provide your expert perspective"

        # **主循环**
        for step in range(self.max_routing):
            print(f"\n--- Step {step + 1}/{self.max_routing} ---")

            # **Step 1: RAG检索**
            print("Step 1: RAG Retrieval (Reranker)")
            retrieved_context = self.unified_ranker.retrieve(
                task=task,
                history_list=self.history_trace,
                top_k=self.rag_top_k
            )
            if retrieved_context:
                print(f"Retrieved {len(retrieved_context.split('---'))} items from history")

            # **Step 2: Execute**
            print(f"Step 2: Executing {current_agent}...")
            agent = await self._get_agent_instance(current_agent)

            agent_input = input_dict.copy()
            agent_input['retrieved_history'] = retrieved_context
            if insight_instruction:
                agent_input['insight'] = insight_instruction

            agent_output, agent_execution_history = await self._execute_agent(agent, agent_input)
            print(f"Output preview: {agent_output[:100]}...")
            print(f"✓ Received execution history with {len(agent_execution_history)} messages")

            # **Step 3: Store**
            self.history_trace.append(agent_output)

            self.current_trace.append({
                'step': step + 1,
                'agent': current_agent,
                'output': agent_output,
                'retrieved_context': retrieved_context,
                'insight': insight_instruction
            })

            # **Step 4: Post-hoc Route**
            print("Step 4: Post-hoc Routing (LLM with Fixed Termination)...")

            candidate_agents = [
                role.lower().replace(' ', '_')
                for role in self.available_roles
                if role.lower().replace(' ', '_') != current_agent
            ]
            candidate_agents.append(self.decision_maker.id)

            context = self.mind_registry.get_context_for_routing(
                current_agent=current_agent,
                candidate_agents=candidate_agents,
                task_description=task
            )

            routing_decision = await self.unified_ranker.route_llm(
                task=task,
                current_output=agent_output,
                current_agent_id=current_agent,
                candidate_agents=candidate_agents,
                context_from_registry=context,
                agent_input_context=agent_input,
                routing_history=routing_decisions,
                agent_execution_history=agent_execution_history
            )

            # **统计**
            if routing_decision.kv_cache_used:
                self.kv_cache_hits += 1
                print(f"✓ KV Cache HIT (Total: {self.kv_cache_hits}/{step + 1})")

            if routing_decision.loop_detected:
                self.loop_detections += 1
                print(f"⚠ Loop detected (Total: {self.loop_detections})")

            print(f"Selected: {routing_decision.selected_agent}")
            print(f"Reasoning: {routing_decision.reasoning[:80]}...")
            print(f"Suggestion: {routing_decision.insight_instruction}")

            routing_decisions.append({
                'step': step + 1,
                'selected': routing_decision.selected_agent,
                'reasoning': routing_decision.reasoning,
                'suggestion': routing_decision.insight_instruction,
                'method': 'llm_with_kv_cache' if routing_decision.kv_cache_used else 'llm_full_prompt',
                'loop_detected': routing_decision.loop_detected
            })

            total_tokens += routing_decision.cost_tokens

            # **检查终止条件（v4.3.2: 更严格的检查）**
            if routing_decision.selected_agent == self.decision_maker.id:
                print("\n🎯 Decision maker selected - reaching consensus...")

                final_output = await self._execute_decision_maker(
                    input_dict, self.history_trace
                )

                execution_time = time.time() - start_time
                cache_hit_rate = self.kv_cache_hits / (step + 1) if step > 0 else 0

                print(f"\n{'=' * 60}")
                print(f"CoRe v4.3.2: Relay Complete")
                print(f"Total Steps: {step + 1} (+ 1 cold start)")
                print(f"Routing Path: {' -> '.join([d['selected'] for d in routing_decisions])}")
                print(f"KV Cache Hit Rate: {cache_hit_rate:.1%}")
                print(f"Loop Detections: {self.loop_detections}")
                print(f"Termination Attempts: {self.unified_ranker.stats['termination_attempts']}")
                print(f"Time: {execution_time:.2f}s")
                print(f"Tokens: {total_tokens}")
                print(f"{'=' * 60}\n")

                self.mind_registry.save()

                result = CoReResult(
                    final_answer=final_output,
                    execution_trace=self.current_trace,
                    routing_decisions=routing_decisions,
                    belief_updates=[],
                    total_time=execution_time,
                    total_cost_tokens=total_tokens,
                    success=True,
                    kv_cache_hits=self.kv_cache_hits,
                    loop_detections=self.loop_detections
                )

                return result

            # 更新状态
            current_agent = routing_decision.selected_agent
            current_output = agent_output
            insight_instruction = routing_decision.insight_instruction

        # 达到最大步数
        print("\n⚠️  Max routing steps reached - forcing decision...")
        print(f"Final Routing Path: {' -> '.join([d['selected'] for d in routing_decisions])}")
        final_output = await self._execute_decision_maker(input_dict, self.history_trace)

        result = CoReResult(
            final_answer=final_output,
            execution_trace=self.current_trace,
            routing_decisions=routing_decisions,
            belief_updates=[],
            total_time=time.time() - start_time,
            total_cost_tokens=total_tokens,
            success=False,
            kv_cache_hits=self.kv_cache_hits,
            loop_detections=self.loop_detections
        )

        return result

    async def _get_agent_instance(self, agent_id: str):
        """获取或创建Agent实例"""
        role = agent_id.replace('_', ' ').title()

        for available_role in self.available_roles:
            if available_role.lower().replace(' ', '_') == agent_id:
                role = available_role
                break

        if self.domain == "gsm8k":
            agent_class = "MathSolver"
        elif self.domain == "humaneval":
            agent_class = "CodeWriting"
        elif self.domain == "mmlu":
            agent_class = "CoReAnalyzeAgent"
        else:
            agent_class = "MathSolver"

        agent = AgentRegistry.get(
            agent_class,
            domain=self.domain,
            llm_name=self.llm_name,
            role=role
        )

        return agent

    @weave.op()
    async def _execute_agent(
            self,
            agent,
            input_dict: Dict
    ) -> Tuple[str, List[Dict[str, str]]]:
        """执行Agent并返回 (输出, 对话历史)"""
        result = await agent.async_execute(input_dict)

        if isinstance(result, tuple) and len(result) == 2:
            outputs_list, messages = result
            if outputs_list:
                output = outputs_list[-1]
            else:
                output = "No output generated"
            return output, messages
        else:
            if result:
                output = result[-1]
            else:
                output = "No output generated"
            return output, []

    async def _execute_decision_maker(
            self,
            input_dict: Dict,
            history: List[str]
    ) -> str:
        """执行Decision Maker"""
        spatial_info = {}
        for i, output in enumerate(history[-5:]):
            spatial_info[f"agent_{i}"] = {
                'role': f"step_{i}",
                'output': output
            }

        await self.decision_maker.async_execute(input_dict)

        if self.decision_maker.outputs:
            return self.decision_maker.outputs[-1]
        return "No decision produced"

    def get_statistics(self) -> Dict:
        """获取执行统计"""
        ranker_stats = self.unified_ranker.get_statistics()
        evolution_stats = self.belief_evolver.get_evolution_summary()

        return {
            'routing': ranker_stats,
            'evolution': evolution_stats,
            'total_beliefs': len(self.mind_registry.beliefs),
            'registered_agents': len(self.mind_registry.profiles),
            'kv_cache_hits': self.kv_cache_hits,
            'loop_detections': self.loop_detections
        }


# 使用示例
if __name__ == "__main__":
    weave.init(
        project_name='vito_chan/G-Designer',
    )


    async def test_core():
        core = CoReGraph(
            domain="mmlu",
            llm_name="Qwen/Qwen3-4B-Instruct-2507",
            available_roles=[
                'Knowlegable Expert',
                'Critic',
                'Mathematician',
                'Psychologist',
                'Historian',
            ],
            max_routing=5
        )

        input_dict = {
            "task": "Solve the equation 2x^2 + 5x - 3 = 0"
        }

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
        print(f"Cold Starts: {stats['routing']['cold_start_count']}")
        print(f"RAG Retrievals: {stats['routing']['rag_retrieval_count']}")
        print(f"LLM Routes: {stats['routing']['post_hoc_route_count']}")


    asyncio.run(test_core())
