"""
CoRe Framework v4.1: Mind Registry Module
去中心化的社会关系网，支持互认初始化
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class AgentProfile:
    """静态能力描述"""
    agent_id: str
    role: str
    capabilities: List[str]
    specializations: List[str]
    limitations: List[str]
    description: str

    def to_text(self) -> str:
        """转换为自然语言描述"""
        text = f"Agent {self.agent_id} ({self.role}):\n"
        text += f"Description: {self.description}\n"
        text += f"Capabilities: {', '.join(self.capabilities)}\n"
        text += f"Specializations: {', '.join(self.specializations)}\n"
        if self.limitations:
            text += f"Known Limitations: {', '.join(self.limitations)}\n"
        return text


@dataclass
class RelationalBelief:
    """关于Agent交互的动态信念"""
    from_agent: str
    to_agent: str
    belief_type: str  # 'trust', 'capability_assessment', 'interaction_pattern'
    content: str
    confidence: float  # 0-1
    evidence_count: int
    last_updated: str

    def to_text(self) -> str:
        return f"{self.from_agent} believes: {self.content} (confidence: {self.confidence:.2f})"


class MindRegistry:
    """
    中央注册表，维护所有Agent的profile和私有信念
    关键特性：去中心化的"互认"初始化
    """

    def __init__(self, save_path: Optional[Path] = None):
        self.profiles: Dict[str, AgentProfile] = {}
        self.beliefs: List[RelationalBelief] = []
        self.save_path = save_path

        # 如果存在保存路径，尝试加载
        if save_path and save_path.exists():
            self.load()

    def register_agent(self, profile: AgentProfile):
        """
        注册新Agent，并自动进行"互认"初始化
        """
        # 1. 存储新Profile
        self.profiles[profile.agent_id] = profile

        # 2. 互认初始化：为所有现有Agent创建对新Agent的初始信念
        for existing_id, existing_profile in self.profiles.items():
            if existing_id == profile.agent_id:
                continue

            # Existing -> New: 基于New的Profile生成初始信念
            initial_belief_to_new = RelationalBelief(
                from_agent=existing_id,
                to_agent=profile.agent_id,
                belief_type='capability_assessment',
                content=f"New agent with role {profile.role}. Capabilities: {', '.join(profile.capabilities[:2])}",
                confidence=0.5,  # 初始信念置信度较低
                evidence_count=0,
                last_updated=datetime.now().isoformat()
            )
            self.beliefs.append(initial_belief_to_new)

            # New -> Existing: 基于Existing的Profile生成初始信念
            initial_belief_from_new = RelationalBelief(
                from_agent=profile.agent_id,
                to_agent=existing_id,
                belief_type='capability_assessment',
                content=f"Experienced agent with role {existing_profile.role}. May help with {', '.join(existing_profile.specializations[:2])}",
                confidence=0.5,
                evidence_count=0,
                last_updated=datetime.now().isoformat()
            )
            self.beliefs.append(initial_belief_from_new)

    def add_belief(self, belief: RelationalBelief):
        """添加或更新关系信念"""
        # 查找是否存在相似信念
        existing_idx = None
        for i, b in enumerate(self.beliefs):
            if (b.from_agent == belief.from_agent and
                    b.to_agent == belief.to_agent and
                    b.belief_type == belief.belief_type):
                existing_idx = i
                break

        if existing_idx is not None:
            # 更新现有信念
            self.beliefs[existing_idx] = belief
        else:
            # 添加新信念
            self.beliefs.append(belief)

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """检索Agent profile"""
        return self.profiles.get(agent_id)

    def get_beliefs_about(
        self,
        to_agent: str,
        from_agent: Optional[str] = None
    ) -> List[RelationalBelief]:
        """
        获取关于特定Agent的信念
        **重要**：如果指定from_agent，只返回该Agent的主观视角
        """
        if from_agent:
            return [b for b in self.beliefs
                    if b.to_agent == to_agent and b.from_agent == from_agent]
        return [b for b in self.beliefs if b.to_agent == to_agent]

    def get_beliefs_from(self, from_agent: str) -> List[RelationalBelief]:
        """获取特定Agent持有的所有信念"""
        return [b for b in self.beliefs if b.from_agent == from_agent]

    def get_context_for_routing(
            self,
            current_agent: str,
            candidate_agents: List[str],
            task_description: str  # 保留参数以兼容调用，但不使用
    ) -> str:
        """
        为LLM路由决策生成丰富的上下文

        **v4.3.2 关键修改**:
        - 移除 Task 描述（由 Ranker 统一管理）
        - 聚焦于 Agent Profiles 和 Beliefs
        - 减少与 Ranker Prompt 的冗余
        """

        # ✅ 修改前: context = f"Current Task: {task_description}\n\n"
        # ✅ 修改后: 不包含 Task
        context = f"**Your Perspective (as {current_agent}):**\n\n"

        # === 1. 添加候选 Agent 的 Profile（精简格式）===
        context += "**Candidate Agent Profiles:**\n"
        for agent_id in candidate_agents:
            profile = self.get_agent_profile(agent_id)
            if profile:
                # ✅ 精简格式：只列出关键信息
                context += f"\n• **{agent_id}** ({profile.role}):\n"
                context += f"  Capabilities: {', '.join(profile.capabilities[:3])}\n"
                if profile.specializations:
                    context += f"  Specializes in: {', '.join(profile.specializations[:2])}\n"

        # === 2. 添加私有信念（保持详细）===
        context += f"\n\n**Your Private Beliefs about Candidates:**\n"
        for agent_id in candidate_agents:
            beliefs = self.get_beliefs_about(
                agent_id,
                from_agent=current_agent
            )
            if beliefs:
                context += f"\n• About **{agent_id}**:\n"
                for belief in beliefs[-3:]:  # 最近3条
                    context += (
                        f"  - {belief.content} "
                        f"(confidence: {belief.confidence:.2f}, "
                        f"evidence: {belief.evidence_count})\n"
                    )
            else:
                context += f"\n• About **{agent_id}**: No prior interactions\n"

        return context

    def save(self):
        """持久化到磁盘"""
        if self.save_path is None:
            return

        data = {
            'profiles': {k: asdict(v) for k, v in self.profiles.items()},
            'beliefs': [asdict(b) for b in self.beliefs]
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """从磁盘加载"""
        if self.save_path is None or not self.save_path.exists():
            return

        with open(self.save_path, 'r') as f:
            data = json.load(f)

        self.profiles = {k: AgentProfile(**v) for k, v in data['profiles'].items()}
        self.beliefs = [RelationalBelief(**b) for b in data['beliefs']]


# 使用示例
if __name__ == "__main__":
    registry = MindRegistry()

    # 注册第一个Agent
    math_profile = AgentProfile(
        agent_id="math_solver_1",
        role="Math Solver",
        capabilities=["arithmetic", "algebra"],
        specializations=["step-by-step reasoning"],
        limitations=["no symbolic computation"],
        description="Specialized in solving math problems"
    )
    registry.register_agent(math_profile)

    # 注册第二个Agent - 自动触发互认初始化
    code_profile = AgentProfile(
        agent_id="code_expert_1",
        role="Code Expert",
        capabilities=["python", "algorithm design"],
        specializations=["numerical computation"],
        limitations=["no web frameworks"],
        description="Expert programmer for algorithms"
    )
    registry.register_agent(code_profile)

    # 查看互认初始化的结果
    print("=== Math Solver's View of Code Expert ===")
    beliefs = registry.get_beliefs_about("code_expert_1", from_agent="math_solver_1")
    for b in beliefs:
        print(b.to_text())

    print("\n=== Code Expert's View of Math Solver ===")
    beliefs = registry.get_beliefs_about("math_solver_1", from_agent="code_expert_1")
    for b in beliefs:
        print(b.to_text())