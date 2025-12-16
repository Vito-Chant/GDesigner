"""
CoRe Framework v4.2: Main Graph Implementation
完整集成：Retrieve(Reranker) -> Execute -> Store(List) -> Route(LLM)

v4.2 更新:
- 适配 Unified Ranker 新签名 (agent_input_context, routing_history)
- 移除 Confidence 相关逻辑
- 优化路由决策的上下文传递
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
import weave


@dataclass
class CoReResult:
    """CoRe执行结果 (v4.2: 移除confidence相关字段)"""
    final_answer: str
    execution_trace: List[Dict]
    routing_decisions: List[Dict]
    belief_updates: List[Any]
    total_time: float
    total_cost_tokens: int
    success: bool


class CoReGraph:
    """
    Cognitive Relay Graph v4.2 - 主编排器

    核心流程：
    Step 0: 冷启动 (Reranker)
    循环 (max_routing次):
        Step 1: Retrieve (Reranker) - RAG检索历史
        Step 2: Execute - Agent执行
        Step 3: Store - 存储到历史列表
        Step 4: Post-hoc Route (LLM) - 决策下一棒 + 生成Suggestion
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
            rag_top_k: int = 3
    ):
        """初始化CoRe Graph"""

        # 导入本地模块
        from mind_registry import MindRegistry, AgentProfile
        from unified_ranker import UnifiedRanker
        from belief_evolver import BeliefEvolver, InteractionTrace

        self.domain = domain
        self.llm_name = llm_name
        self.available_roles = available_roles
        self.max_routing = max_routing
        self.rag_top_k = rag_top_k

        # 初始化LLM
        self.llm = LLMRegistry.get(llm_name)

        # 初始化Mind Registry (去中心化互认)
        self.mind_registry = MindRegistry(save_path=registry_save_path)
        self._initialize_agent_profiles()

        # 初始化Unified Ranker (Reranker + LLM)
        self.unified_ranker = UnifiedRanker(
            llm=self.llm,
            reranker_model_name=reranker_model
        )

        # 初始化Belief Evolver
        self.belief_evolver = BeliefEvolver(
            llm=self.llm,
            mind_registry=self.mind_registry
        )

        # 初始化Decision Maker
        self.decision_maker = AgentRegistry.get(
            decision_method,
            domain=domain,
            llm_name=llm_name
        )

        # 执行状态
        self.history_trace = []  # List[str] - 纯文本历史
        self.current_trace = []  # List[Dict] - 详细执行轨迹
        self.interaction_traces = []

    def _initialize_agent_profiles(self):
        """从domain初始化Agent profiles，触发互认初始化"""
        from mind_registry import AgentProfile
        from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

        prompt_set = PromptSetRegistry.get(self.domain)

        for role in self.available_roles:
            try:
                description = prompt_set.get_description(role)

                # 简化的能力解析
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

                # **关键**: register_agent会自动触发互认初始化
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
        主执行循环 - Cognitive Relay (v4.2)
        """

        start_time = time.time()
        task = input_dict['task']

        # 重置状态
        self.history_trace = []
        self.current_trace = []
        self.interaction_traces = []
        routing_decisions = []
        total_tokens = 0

        print(f"\n{'=' * 60}")
        print(f"CoRe v4.2: Starting Cognitive Relay")
        print(f"Task: {task[:100]}...")
        print(f"{'=' * 60}\n")

        # **Step 0: 冷启动 (Reranker)**
        print("=== Step 0: Cold Start (Reranker) ===")
        profiles = {
            agent_id: self.mind_registry.get_agent_profile(agent_id).to_text()
            for agent_id in [role.lower().replace(' ', '_') for role in self.available_roles]
        }

        current_agent = await self.unified_ranker.cold_start(task, profiles)
        print(f"Cold Start Selected: {current_agent}\n")

        current_output = None
        insight_instruction = None

        # **主循环**
        for step in range(self.max_routing):
            print(f"\n--- Step {step + 1}/{self.max_routing} ---")

            # **Step 1: Retrieve (Reranker) - RAG检索**
            print("Step 1: RAG Retrieval (Reranker)")
            retrieved_context = self.unified_ranker.retrieve(
                task=task,
                history_list=self.history_trace,
                top_k=self.rag_top_k
            )
            if retrieved_context:
                print(f"Retrieved {len(retrieved_context.split('---'))} items from history")

            # **Step 2: Execute - Agent执行**
            print(f"Step 2: Executing {current_agent}...")
            agent = await self._get_agent_instance(current_agent)

            # **v4.2关键修改**: 准备 agent_input (包含完整上下文)
            agent_input = input_dict.copy()
            agent_input['retrieved_history'] = retrieved_context
            if insight_instruction:
                agent_input['insight'] = insight_instruction

            agent_output = await self._execute_agent(agent, agent_input)
            print(f"Output preview: {agent_output[:100]}...")

            # **Step 3: Store - 存入历史列表**
            self.history_trace.append(agent_output)

            # 记录trace
            self.current_trace.append({
                'step': step + 1,
                'agent': current_agent,
                'output': agent_output,
                'retrieved_context': retrieved_context,
                'insight': insight_instruction
            })

            # **Step 4: Post-hoc Route (LLM) - 决策下一棒**
            print("Step 4: Post-hoc Routing (LLM)...")

            # 获取候选Agent
            candidate_agents = [
                role.lower().replace(' ', '_')
                for role in self.available_roles
                if role.lower().replace(' ', '_') != current_agent
            ]
            candidate_agents.append(self.decision_maker.id)

            # 获取当前Agent的私有上下文
            context = self.mind_registry.get_context_for_routing(
                current_agent=current_agent,
                candidate_agents=candidate_agents,
                task_description=task
            )

            # **v4.2关键修改**: 调用LLM路由时传入完整上下文
            routing_decision = await self.unified_ranker.route_llm(
                task=task,
                current_output=agent_output,
                current_agent_id=current_agent,
                candidate_agents=candidate_agents,
                context_from_registry=context,
                agent_input_context=agent_input,  # **新增**
                routing_history=routing_decisions  # **新增**
            )

            print(f"Selected: {routing_decision.selected_agent}")
            print(f"Reasoning: {routing_decision.reasoning[:80]}...")
            print(f"Suggestion: {routing_decision.insight_instruction}")

            # **v4.2修改**: 存储决策（移除confidence）
            routing_decisions.append({
                'step': step + 1,
                'selected': routing_decision.selected_agent,
                'reasoning': routing_decision.reasoning,
                'suggestion': routing_decision.insight_instruction,
            })

            total_tokens += routing_decision.cost_tokens

            # **检查是否选择了Decision Maker (终止条件)**
            if routing_decision.selected_agent == self.decision_maker.id:
                print("\n🎯 Decision maker selected - reaching consensus...")

                final_output = await self._execute_decision_maker(
                    input_dict, self.history_trace
                )

                execution_time = time.time() - start_time

                print(f"\n{'=' * 60}")
                print(f"CoRe v4.2: Relay Complete")
                print(f"Total Steps: {step + 1}")
                print(f"Time: {execution_time:.2f}s")
                print(f"Tokens: {total_tokens}")
                print(f"{'=' * 60}\n")

                # 保存记忆和进化
                self.mind_registry.save()

                result = CoReResult(
                    final_answer=final_output,
                    execution_trace=self.current_trace,
                    routing_decisions=routing_decisions,
                    belief_updates=[],
                    total_time=execution_time,
                    total_cost_tokens=total_tokens,
                    success=True
                )

                return result

            # 更新状态用于下一轮
            current_agent = routing_decision.selected_agent
            current_output = agent_output
            insight_instruction = routing_decision.insight_instruction

        # 达到最大步数
        print("\n⚠️  Max routing steps reached - forcing decision...")
        final_output = await self._execute_decision_maker(input_dict, self.history_trace)

        result = CoReResult(
            final_answer=final_output,
            execution_trace=self.current_trace,
            routing_decisions=routing_decisions,
            belief_updates=[],
            total_time=time.time() - start_time,
            total_cost_tokens=total_tokens,
            success=False
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
    async def _execute_agent(self, agent, input_dict: Dict) -> str:
        """执行Agent并返回输出"""
        await agent.async_execute(input_dict)

        if agent.outputs:
            return agent.outputs[-1]
        return "No output generated"

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
            'registered_agents': len(self.mind_registry.profiles)
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
