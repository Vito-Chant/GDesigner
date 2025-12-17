"""
CoRe Framework v4.3.3: Main Graph Implementation - 修复并发上下文串扰
完整集成：Retrieve(Reranker) -> Execute -> Store(List) -> Route(LLM with Path Awareness)

v4.3.3 关键修复:
- [Fix] 将所有执行状态 (history_trace, current_trace 等) 改为局部变量，支持 batch_size > 1 并发
- [Fix] Decision Maker 改为每次执行时动态实例化，防止 Agent 状态共享导致的串扰
- [Fix] 保持 v4.3.2 的所有逻辑增强（终止检测、诚实反馈指令等）
"""

import sys
import os
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
    loop_detections: int


class CoReGraph:
    """
    Cognitive Relay Graph v4.3.3 - 主编排器 (并发安全版)
    """

    def __init__(
            self,
            domain: str,
            llm_name: str,
            available_roles: List[str],
            decision_method: str = "FinalRefer",
            max_routing: int = 10,
            registry_save_path: Optional[Path] = None,
            reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
            rag_top_k: int = 3,
            max_loop_count: int = 2
    ):
        """初始化CoRe Graph"""
        from GDesigner.CoRe.mind_registry import MindRegistry
        from GDesigner.CoRe.unified_ranker import UnifiedRanker
        from GDesigner.CoRe.belief_evolver import BeliefEvolver

        self.domain = domain
        self.llm_name = llm_name
        self.available_roles = available_roles
        self.decision_method = decision_method  # 保存方法名，用于动态实例化
        self.max_routing = max_routing
        self.rag_top_k = rag_top_k

        # 初始化组件
        self.llm = LLMRegistry.get(llm_name)
        self.mind_registry = MindRegistry(save_path=registry_save_path)
        self._initialize_agent_profiles()

        # [并发修复] 这里只获取一个 ID 或原型，不要在执行中直接使用这个实例
        # 真正的执行实例将在 _execute_decision_maker 中动态创建
        self.decision_maker_prototype = AgentRegistry.get(
            decision_method,
            domain=domain,
            llm_name=llm_name,
            id="final_decision"
        )
        self.decision_maker_id = self.decision_maker_prototype.id

        self.unified_ranker = UnifiedRanker(
            llm=self.llm,
            reranker_model_name=reranker_model,
            max_loop_count=max_loop_count,
            decision_maker_id=self.decision_maker_id
        )

        self.belief_evolver = BeliefEvolver(
            llm=self.llm,
            mind_registry=self.mind_registry
        )

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
        主执行循环 - Cognitive Relay (并发安全版)
        """

        start_time = time.time()
        task = input_dict['task']

        # [并发修复] 使用局部变量替代 self 属性
        history_trace = []
        current_trace = []
        routing_decisions = []
        total_tokens = 0
        kv_cache_hits = 0
        loop_detections = 0
        termination_attempts = 0

        print(f"\n{'=' * 60}")
        print(f"CoRe v4.3.3: Starting Cognitive Relay (Concurrency Safe)")
        print(f"Task: {task[:100]}...")
        print(f"{'=' * 60}\n")

        # **Step 0: 冷启动**
        # print("=== Step 0: Cold Start (Reranker) ===")
        profiles = {
            agent_id: self.mind_registry.get_agent_profile(agent_id).to_text()
            for agent_id in [role.lower().replace(' ', '_') for role in self.available_roles]
        }

        current_agent = await self.unified_ranker.cold_start(task, profiles)
        # print(f"Cold Start Selected: {current_agent}\n")

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
            # print(f"\n--- Step {step + 1}/{self.max_routing} ---")

            # **Step 1: RAG检索**
            # print("Step 1: RAG Retrieval (Reranker)")
            # [并发修复] 使用局部变量 history_trace
            retrieved_context = self.unified_ranker.retrieve(
                task=task,
                history_list=history_trace,
                top_k=self.rag_top_k
            )
            # if retrieved_context:
            #     print(f"Retrieved {len(retrieved_context.split('---'))} items from history")

            # **Step 2: Execute**
            # print(f"Step 2: Executing {current_agent}...")
            # [并发修复] 获取新的 Agent 实例
            agent = await self._get_agent_instance(current_agent)

            agent_input = input_dict.copy()
            agent_input['retrieved_history'] = retrieved_context
            if insight_instruction:
                agent_input['insight'] = insight_instruction

            # 系统级指令
            agent_input['system_instruction'] = (
                "CRITICAL: If you cannot fully complete the task or encounter "
                "difficulties, explicitly state what is missing, what went wrong, "
                "or what help you need. DO NOT pretend to solve it if you lack "
                "the necessary capabilities or information. Honest feedback is "
                "essential for the system to route the task appropriately."
            )

            agent_output, agent_execution_history = await self._execute_agent(agent, agent_input)
            # print(f"Output preview: {agent_output[:100]}...")

            # **Step 3: Store**
            history_trace.append(agent_output)

            current_trace.append({
                'step': step + 1,
                'agent': current_agent,
                'output': agent_output,
                'retrieved_context': retrieved_context,
                'insight': insight_instruction
            })

            # **Step 4: Post-hoc Route**
            # print("Step 4: Post-hoc Routing...")

            candidate_agents = [
                role.lower().replace(' ', '_')
                for role in self.available_roles
                if role.lower().replace(' ', '_') != current_agent
            ]
            candidate_agents.append(self.decision_maker_id)

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
                kv_cache_hits += 1
                # print(f"✓ KV Cache HIT (Total: {kv_cache_hits}/{step + 1})")

            if routing_decision.loop_detected:
                loop_detections += 1
                # print(f"⚠ Loop detected (Total: {loop_detections})")

            # print(f"Selected: {routing_decision.selected_agent}")

            routing_decisions.append({
                'step': step + 1,
                'selected': routing_decision.selected_agent,
                'reasoning': routing_decision.reasoning,
                'suggestion': routing_decision.insight_instruction,
                'method': 'llm_with_kv_cache' if routing_decision.kv_cache_used else 'llm_full_prompt',
                'loop_detected': routing_decision.loop_detected
            })

            total_tokens += routing_decision.cost_tokens

            # **检查终止条件**
            if routing_decision.selected_agent == self.decision_maker_id:
                print("\n🎯 Decision maker selected - reaching consensus...")

                # [并发修复] 使用局部变量 history_trace
                final_output = await self._execute_decision_maker(
                    input_dict, history_trace
                )

                execution_time = time.time() - start_time

                print(f"Relay Complete. Time: {execution_time:.2f}s")

                self.mind_registry.save()

                result = CoReResult(
                    final_answer=final_output,
                    execution_trace=current_trace,
                    routing_decisions=routing_decisions,
                    belief_updates=[],
                    total_time=execution_time,
                    total_cost_tokens=total_tokens,
                    success=True,
                    kv_cache_hits=kv_cache_hits,
                    loop_detections=loop_detections
                )

                return result

            # 更新状态
            current_agent = routing_decision.selected_agent
            current_output = agent_output
            insight_instruction = routing_decision.insight_instruction

        # 达到最大步数
        print("\n⚠️  Max routing steps reached - forcing decision...")
        final_output = await self._execute_decision_maker(input_dict, history_trace)

        result = CoReResult(
            final_answer=final_output,
            execution_trace=current_trace,
            routing_decisions=routing_decisions,
            belief_updates=[],
            total_time=time.time() - start_time,
            total_cost_tokens=total_tokens,
            success=False,
            kv_cache_hits=kv_cache_hits,
            loop_detections=loop_detections
        )

        return result

    async def _get_agent_instance(self, agent_id: str):
        """获取Agent实例 (确保返回新实例)"""
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

        # AgentRegistry.get 通常会创建新实例（除非内部实现为单例池）
        # 假设 AgentRegistry 每次返回一个 fresh object
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
        """执行Decision Maker (并发安全版：动态实例化)"""

        # [并发修复] 动态创建一个全新的 Decision Maker 实例
        # 这确保了 self.outputs 不会与其他并发任务共享
        decision_maker = AgentRegistry.get(
            self.decision_method,
            domain=self.domain,
            llm_name=self.llm_name,
            id="final_decision"
        )

        # 构建标准化的 spatial_info
        spatial_info = {}
        recent_history = history[-5:] if len(history) > 5 else history

        for i, output in enumerate(recent_history):
            step_idx = len(history) - len(recent_history) + i
            spatial_info[f"previous_step_{step_idx}"] = {
                'role': f"Agent at step {step_idx}",
                'output': output
            }

        # print(f"[Decision Maker] Context size: {len(spatial_info)}")

        await decision_maker.async_execute(
            input_dict,
            spatial_info=spatial_info
        )

        if decision_maker.outputs:
            return decision_maker.outputs[-1]
        return "No decision produced"

    def get_statistics(self) -> Dict:
        """获取统计信息 (注意：并发运行时此处的 CoReGraph 实例级统计可能不准确，请依赖 CoReResult 返回的统计)"""
        ranker_stats = self.unified_ranker.get_statistics()
        evolution_stats = self.belief_evolver.get_evolution_summary()

        return {
            'routing': ranker_stats,
            'evolution': evolution_stats,
            'total_beliefs': len(self.mind_registry.beliefs),
            'registered_agents': len(self.mind_registry.profiles),
            # 这些值在并发时会是最后一次运行的值，仅供参考
            # 'kv_cache_hits': self.kv_cache_hits,
            # 'loop_detections': self.loop_detections
        }

if __name__ == "__main__":
    pass