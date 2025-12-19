"""
MindRegistry v5.0 - 支持泛化信念系统
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from GDesigner.CoRe.generalized_belief import GeneralizedBelief


class MindRegistryV5:
    """
    中央注册表 v5.0

    新增特性:
    - 支持 GeneralizedBelief 的存储和查询
    - 按能力维度检索
    - 信念时间衰减
    - 向后兼容旧版 RelationalBelief
    """

    def __init__(self, save_path: Optional[Path] = None):
        self.profiles: Dict = {}  # Agent profiles (保持不变)
        self.generalized_beliefs: List[GeneralizedBelief] = []  # 新: 泛化信念
        self.legacy_beliefs: List = []  # 旧版信念 (兼容)
        self.save_path = save_path

        if save_path and save_path.exists():
            self.load()

    # ------------------------------------------------------------------------
    # 核心方法: 泛化信念
    # ------------------------------------------------------------------------

    def add_generalized_belief(self, belief: GeneralizedBelief):
        """添加或更新泛化信念"""
        # 查找是否存在相同能力维度的信念
        existing_idx = None
        for i, b in enumerate(self.generalized_beliefs):
            if (b.from_agent == belief.from_agent and
                    b.to_agent == belief.to_agent and
                    b.capability_dimension == belief.capability_dimension):
                existing_idx = i
                break

        if existing_idx is not None:
            # 更新现有信念
            self.generalized_beliefs[existing_idx] = belief
        else:
            # 添加新信念
            self.generalized_beliefs.append(belief)

    def get_beliefs_by_capability(
            self,
            to_agent: str,
            from_agent: str,
            capability: str
    ) -> List[GeneralizedBelief]:
        """按能力维度检索信念 (核心查询方法)"""
        return [
            b for b in self.generalized_beliefs
            if b.to_agent == to_agent
               and b.from_agent == from_agent
               and b.capability_dimension == capability
        ]

    def get_all_beliefs_about(
            self,
            to_agent: str,
            from_agent: Optional[str] = None
    ) -> List[GeneralizedBelief]:
        """获取关于特定 Agent 的所有信念"""
        if from_agent:
            return [
                b for b in self.generalized_beliefs
                if b.to_agent == to_agent and b.from_agent == from_agent
            ]
        return [
            b for b in self.generalized_beliefs
            if b.to_agent == to_agent
        ]

    def get_beliefs_by_context(
            self,
            to_agent: str,
            from_agent: str,
            context: str
    ) -> List[GeneralizedBelief]:
        """获取适用于特定上下文的信念"""
        return [
            b for b in self.generalized_beliefs
            if b.to_agent == to_agent
               and b.from_agent == from_agent
               and b.matches_context(context)
        ]

    # ------------------------------------------------------------------------
    # 路由决策支持
    # ------------------------------------------------------------------------

    def get_context_for_routing(
            self,
            current_agent: str,
            candidate_agents: List[str],
            task_description: str,
            task_capabilities: List[str] = None
    ) -> str:
        """
        为路由决策生成上下文 (v5.0 增强版)

        Args:
            current_agent: 当前 Agent
            candidate_agents: 候选 Agent 列表
            task_description: 任务描述
            task_capabilities: 任务涉及的能力维度 (可选)
        """
        from GDesigner.CoRe.task_abstractor import TaskAbstractor

        # 如果未提供能力维度,自动提取
        if task_capabilities is None:
            abstractor = TaskAbstractor()
            task_capabilities = list(abstractor.extract_task_types(task_description))
            task_complexity = abstractor.extract_complexity(task_description)
        else:
            task_complexity = "moderate"  # 默认

        context = f"**Your Perspective (as {current_agent}):**\n\n"

        # === 1. 任务分析 ===
        context += f"**Task Analysis:**\n"
        context += f"- Required Capabilities: {', '.join(task_capabilities)}\n"
        context += f"- Estimated Complexity: {task_complexity}\n\n"

        # === 2. 候选 Agent 的能力评估 ===
        context += "**Candidate Agent Capabilities:**\n"

        for agent_id in candidate_agents:
            # 获取该 Agent 的 profile
            profile = self.get_agent_profile(agent_id)
            if profile:
                context += f"\n• **{agent_id}** ({profile.role}):\n"
                context += f"  Description: {profile.description[:100]}...\n"
            else:
                context += f"\n• **{agent_id}**:\n"

            # 获取该 Agent 在相关能力维度上的信念
            relevant_beliefs = []
            for capability in task_capabilities:
                beliefs = self.get_beliefs_by_capability(
                    to_agent=agent_id,
                    from_agent=current_agent,
                    capability=capability
                )
                relevant_beliefs.extend(beliefs)

            if relevant_beliefs:
                context += f"  **Your Beliefs about {agent_id}:**\n"
                for belief in relevant_beliefs[:3]:  # 最多3条
                    rate = belief.success_rate
                    total = belief.total_count

                    # 格式化输出
                    context += f"    - {belief.capability_dimension.replace('_', ' ').title()}: "
                    context += f"{belief.general_description}\n"
                    context += f"      Track Record: {belief.success_count}/{total} ({rate:.0%})"

                    # 适用场景
                    if belief.applicable_contexts:
                        contexts_str = ', '.join(belief.applicable_contexts[:2])
                        context += f" | Applicable: {contexts_str}"

                    # 限制
                    if belief.known_limitations:
                        lims_str = ', '.join(belief.known_limitations[:1])
                        context += f" | Caution: {lims_str}"

                    context += "\n"

                    # 可靠性评估
                    if rate >= 0.8 and total >= 3:
                        context += f"      ✓ Highly reliable for this capability\n"
                    elif rate < 0.4 and total >= 3:
                        context += f"      ⚠ Consider alternatives\n"
            else:
                context += f"  No prior interaction history\n"

        return context

    # ------------------------------------------------------------------------
    # 向后兼容
    # ------------------------------------------------------------------------

    def register_agent(self, profile):
        """注册 Agent (兼容旧版)"""
        self.profiles[profile.agent_id] = profile

    def get_agent_profile(self, agent_id: str):
        """获取 Agent profile"""
        return self.profiles.get(agent_id)

    def get_beliefs_about(self, to_agent: str, from_agent: Optional[str] = None):
        """
        兼容旧版的查询方法
        返回泛化信念 (优先) + 旧版信念
        """
        beliefs = self.get_all_beliefs_about(to_agent, from_agent)
        beliefs.extend(self.legacy_beliefs)  # 追加旧版
        return beliefs

    # ------------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------------

    def save(self):
        """保存到磁盘"""
        if self.save_path is None:
            return

        data = {
            'profiles': {k: v.__dict__ if hasattr(v, '__dict__') else v
                         for k, v in self.profiles.items()},
            'generalized_beliefs': [b.to_dict() for b in self.generalized_beliefs],
            'legacy_beliefs': [b.__dict__ if hasattr(b, '__dict__') else b
                               for b in self.legacy_beliefs]
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[MindRegistry] Saved {len(self.generalized_beliefs)} beliefs to {self.save_path}")

    def load(self):
        """从磁盘加载"""
        if self.save_path is None or not self.save_path.exists():
            return

        with open(self.save_path, 'r') as f:
            data = json.load(f)

        # 加载 profiles (保持原逻辑)
        from GDesigner.CoRe.mind_registry import AgentProfile
        self.profiles = {
            k: AgentProfile(**v) if isinstance(v, dict) else v
            for k, v in data.get('profiles', {}).items()
        }

        # 加载泛化信念
        self.generalized_beliefs = [
            GeneralizedBelief.from_dict(b)
            for b in data.get('generalized_beliefs', [])
        ]

        # 加载旧版信念 (兼容)
        from GDesigner.CoRe.mind_registry import RelationalBelief
        self.legacy_beliefs = [
            RelationalBelief(**b) if isinstance(b, dict) else b
            for b in data.get('legacy_beliefs', [])
        ]

        print(f"[MindRegistry] Loaded {len(self.generalized_beliefs)} beliefs from {self.save_path}")