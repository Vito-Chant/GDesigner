"""
CoRe Framework v4.3.2: Unified Ranker Module - 修复终止逻辑
System 1.5 (Reranker) + System 2 (LLM with Path Awareness + Proper Termination)

v4.3.2 关键修复:
- 正确处理 LLM 希望终止的情况
- 明确告知 LLM 如何触发 Decision Maker
- 改进错误处理和降级逻辑
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import weave
from sentence_transformers import CrossEncoder
import numpy as np


@dataclass
class RoutingDecision:
    """路由决策结果"""
    selected_agent: str
    reasoning: str
    path_used: str
    insight_instruction: Optional[str]
    alternative_agents: List[Tuple[str, float]]
    cost_tokens: int
    kv_cache_used: bool
    loop_detected: bool


class UnifiedRanker:
    """
    统一的Ranker模块 (v4.3.2 修复终止逻辑版)

    职责划分：
    - System 1.5 (Reranker):
      * 冷启动: 选择第一个Agent
      * RAG检索: 从历史中检索上下文

    - System 2 (LLM with Proper Termination):
      * 事后路由: 复用历史 + 强化路径感知
      * **正确终止**: 明确引导 LLM 选择 Decision Maker
      * 循环检测: 防止 Agent 死循环
      * 见解生成: 生成战略性Suggestion
    """

    def __init__(
            self,
            llm,
            reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
            max_loop_count: int = 2,
            decision_maker_id: str = "final_decision"  # **v4.3.2新增**
    ):
        """初始化Unified Ranker"""
        self.llm = llm
        self.max_loop_count = max_loop_count
        self.decision_maker_id = decision_maker_id  # **v4.3.2: 记录 Decision Maker ID**

        print(f"Loading Cross-Encoder: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)

        self.stats = {
            'cold_start_count': 0,
            'rag_retrieval_count': 0,
            'post_hoc_route_count': 0,
            'kv_cache_hits': 0,
            'loop_detections': 0,
            'termination_attempts': 0,  # **v4.3.2新增**
            'total_tokens_used': 0
        }

    async def cold_start(
            self,
            task: str,
            profiles: Dict[str, str]
    ) -> str:
        """System 1.5: 冷启动"""
        self.stats['cold_start_count'] += 1

        if not profiles:
            raise ValueError("No agent profiles provided for cold start")

        agent_ids = list(profiles.keys())
        pairs = [(task, profiles[agent_id]) for agent_id in agent_ids]
        scores = self.reranker.predict(pairs)
        best_idx = np.argmax(scores)
        selected_agent = agent_ids[best_idx]

        print(f"[Cold Start] Selected {selected_agent} with score {scores[best_idx]:.3f}")

        return selected_agent

    def retrieve(
            self,
            task: str,
            history_list: List[str],
            top_k: int = 3
    ) -> str:
        """System 1.5: RAG检索"""
        if not history_list:
            return ""

        self.stats['rag_retrieval_count'] += 1

        pairs = [(task, history_item) for history_item in history_list]
        scores = self.reranker.predict(pairs)
        top_k = min(top_k, len(history_list))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved_texts = [history_list[i] for i in top_indices]

        print(f"[RAG] Retrieved {top_k} items from {len(history_list)} history entries")

        return "\n\n---\n\n".join(retrieved_texts)

    def _detect_loop(
            self,
            routing_history: List[Dict],
            current_agent: str,
            candidate_agent: str
    ) -> Tuple[bool, str]:
        """检测路由循环"""
        if not routing_history or len(routing_history) < 2:
            return False, ""

        agent_sequence = [current_agent]
        for decision in routing_history:
            agent_sequence.append(decision.get('selected', ''))

        agent_counts = Counter(agent_sequence)

        # 检测短期往复
        recent_path = agent_sequence[-3:] if len(agent_sequence) >= 3 else agent_sequence

        if len(recent_path) >= 2:
            if recent_path[-1] == current_agent and recent_path[-2] == candidate_agent:
                loop_count = agent_counts[current_agent]
                if loop_count >= self.max_loop_count:
                    warning = f"⚠ LOOP DETECTED: {current_agent} ↔ {candidate_agent} (Count: {loop_count})"
                    self.stats['loop_detections'] += 1
                    return True, warning

        # 检测长期停滞
        if agent_counts[candidate_agent] >= 3:
            warning = f"⚠ REPEATED AGENT: {candidate_agent} already used {agent_counts[candidate_agent]} times"
            return True, warning

        return False, ""

    @weave.op()
    async def route_llm(
            self,
            task: str,
            current_output: str,
            current_agent_id: str,
            candidate_agents: List[str],
            context_from_registry: str,
            agent_input_context: Dict,
            routing_history: List[Dict],
            agent_execution_history: List[Dict[str, str]]
    ) -> RoutingDecision:
        """
        System 2: 事后路由 (v4.3.2 修复终止逻辑版)

        **v4.3.2 核心改进**:
        1. 明确告知 LLM Decision Maker 的名称和作用
        2. 改进解析逻辑，正确处理终止意图
        3. 增强错误处理和降级策略
        """
        self.stats['post_hoc_route_count'] += 1

        # **Step 1: 循环预检测**
        loop_warnings = {}
        for candidate in candidate_agents:
            is_loop, warning = self._detect_loop(routing_history, current_agent_id, candidate)
            if is_loop:
                loop_warnings[candidate] = warning
                print(warning)

        # **Step 2: 构建增强的路由指令（v4.3.2: 明确 Decision Maker）**
        routing_instruction = self._build_enhanced_routing_instruction(
            task, current_output, current_agent_id,
            candidate_agents, context_from_registry,
            agent_input_context, routing_history,
            loop_warnings
        )

        # **Step 3: KV Cache 复用或降级**
        kv_cache_used = False
        if agent_execution_history and len(agent_execution_history) >= 2:
            messages = agent_execution_history.copy()
            messages.append({'role': 'user', 'content': routing_instruction})
            kv_cache_used = True
            self.stats['kv_cache_hits'] += 1
            print(f"✓ Using KV Cache: Appending routing instruction to {len(agent_execution_history)} messages")
        else:
            print(f"⚠ KV Cache unavailable, building full prompt")
            full_prompt = self._build_fallback_prompt(
                task, current_output, current_agent_id,
                candidate_agents, context_from_registry,
                agent_input_context, routing_history,
                loop_warnings
            )
            messages = [
                {'role': 'system', 'content': 'You are an expert at task delegation and strategic planning.'},
                {'role': 'user', 'content': full_prompt}
            ]

        # **Step 4: 调用LLM**
        response = await self.llm.agen(messages)

        # **Step 5: 解析响应（v4.3.2: 改进解析逻辑）**
        selected_agent, reasoning, suggestion, termination_requested = self._parse_llm_route_response_v2(
            response, candidate_agents
        )

        # **Step 6: 处理终止请求**
        if termination_requested:
            self.stats['termination_attempts'] += 1
            print(f"[Termination] LLM requested to end chain, routing to Decision Maker")
            selected_agent = self.decision_maker_id
            suggestion = "Synthesize all agent outputs and provide the final answer"

        # **Step 7: 循环后验检测**
        is_loop, _ = self._detect_loop(routing_history, current_agent_id, selected_agent)

        # **Step 8: Token 统计**
        if kv_cache_used:
            token_cost = len(routing_instruction.split()) + len(response.split())
        else:
            token_cost = len(messages[1]['content'].split()) + len(response.split())

        self.stats['total_tokens_used'] += token_cost

        print(f"[LLM Route] {current_agent_id} -> {selected_agent}")
        print(f"[KV Cache] {'HIT' if kv_cache_used else 'MISS'}")
        if is_loop:
            print(f"[Loop] DETECTED but LLM chose it anyway")
        if termination_requested:
            print(f"[Termination] Requested by LLM")

        return RoutingDecision(
            selected_agent=selected_agent,
            reasoning=reasoning,
            path_used='llm_post_hoc',
            insight_instruction=suggestion,
            alternative_agents=[(cand, 0.0) for cand in candidate_agents if cand != selected_agent],
            cost_tokens=token_cost,
            kv_cache_used=kv_cache_used,
            loop_detected=is_loop
        )

    def _build_enhanced_routing_instruction(
            self,
            task: str,
            current_output: str,
            current_agent_id: str,
            candidate_agents: List[str],
            context: str,
            agent_input_context: Dict,
            routing_history: List[Dict],
            loop_warnings: Dict[str, str]
    ) -> str:
        """
        构建增强的路由指令 (v4.3.2: 明确终止机制)
        """

        # === 1. 构建详细的路径历史 ===
        if routing_history:
            path_summary = "\n=== DETAILED ROUTING PATH ===\n"
            path_summary += f"Step 0 (Cold Start): → {routing_history[0].get('selected', 'Unknown')}\n"

            for i, decision in enumerate(routing_history, 1):
                selected = decision.get('selected', 'Unknown')
                reasoning = decision.get('reasoning', 'N/A')
                suggestion = decision.get('suggestion', 'N/A')

                path_summary += f"\nStep {i}: → {selected}\n"
                path_summary += f"  Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}\n"
                path_summary += f"  Suggestion: {suggestion[:80]}{'...' if len(suggestion) > 80 else ''}\n"

            path_summary += f"\nStep {len(routing_history) + 1} (Current): You are {current_agent_id}\n"
            path_summary += "=== END OF PATH ===\n"
        else:
            path_summary = f"\n=== ROUTING PATH ===\nThis is Step 1 right after cold start.\nYou are the first agent: {current_agent_id}\n=== END OF PATH ===\n"

        # === 2. 构建循环警告 ===
        loop_warning_str = ""
        if loop_warnings:
            loop_warning_str = "\n⚠️ === CRITICAL: LOOP WARNINGS === ⚠️\n"
            for agent, warning in loop_warnings.items():
                loop_warning_str += f"- {agent}: {warning}\n"
            loop_warning_str += "Please AVOID selecting agents with loop warnings unless absolutely necessary.\n"
            loop_warning_str += "=== END OF WARNINGS ===\n"

        # === 3. 识别 Decision Maker ===
        decision_maker = None
        regular_agents = []
        for agent in candidate_agents:
            if 'final' in agent.lower() or 'decision' in agent.lower():
                decision_maker = agent
            else:
                regular_agents.append(agent)

        # === 4. 构建候选信息（v4.3.2: 明确 Decision Maker）===
        candidates_info = ""
        if regular_agents:
            candidates_info += f"\n**Regular Agents** (for continued analysis): {', '.join(regular_agents)}\n"
        if decision_maker:
            candidates_info += f"\n**Decision Maker** (to END the chain): `{decision_maker}`\n"
            candidates_info += "  → Select this when the task is fully resolved and you want to produce the final answer.\n"

        # === 5. 组装完整指令（v4.3.2: 强化终止指引）===
        instruction = f"""
=== ROLE SWITCH: YOU ARE NOW THE COORDINATOR ===

Excellent work on the analysis above. Now, please act as the **Coordinator** to decide the next step.

{path_summary}

**IMPORTANT: Path Analysis**
- Review the ENTIRE routing path above
- Identify patterns: Are we making progress or circling?
- Consider what each agent has already contributed
- Avoid selecting agents that would create loops or redundancy

{loop_warning_str}

**Your Beliefs About Candidates**:
{context}

**Available Candidates**:
{candidates_info}

**CRITICAL DECISION CRITERIA**:
1. **If the task is FULLY SOLVED and no more analysis is needed:**
   - Select the Decision Maker: `{decision_maker if decision_maker else 'final_decision'}`
   - This will END the routing chain and produce the final answer

2. **If the task needs MORE work (new perspective, verification, implementation):**
   - Select an appropriate regular agent
   - Provide a clear suggestion for what they should focus on

3. **NEVER select "none" or invalid names** - always choose from the list above

**Decision Format** (MUST follow exactly):
REASONING: <detailed analysis: Is task complete? What's missing? Loop risks?>
SELECTED: <exact agent name from the list above>
SUGGESTION: <brief, actionable advice - under 50 words>

**Remember**: 
- The routing path is CRUCIAL - don't ignore it!
- If the task is solved, SELECT THE DECISION MAKER to end the chain
- Only continue routing if NEW value can be added
- Use the EXACT agent names from the candidate list
"""
        return instruction

    def _build_fallback_prompt(
            self,
            task: str,
            current_output: str,
            current_agent_id: str,
            candidate_agents: List[str],
            context: str,
            agent_input_context: Dict,
            routing_history: List[Dict],
            loop_warnings: Dict[str, str]
    ) -> str:
        """
        降级方案：构建完整 Prompt (v4.3.2: 包含终止指引)
        """

        # 详细路径历史
        if routing_history:
            history_str = "\n=== COMPLETE ROUTING PATH ===\n"
            history_str += f"Step 0 (Cold Start): → {routing_history[0].get('selected', 'Unknown')}\n"

            for i, decision in enumerate(routing_history, 1):
                selected = decision.get('selected', 'Unknown')
                reasoning = decision.get('reasoning', 'N/A')
                suggestion = decision.get('suggestion', 'N/A')

                history_str += f"\nStep {i}: → {selected}\n"
                history_str += f"  Why: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}\n"
                history_str += f"  Suggestion Given: {suggestion[:100]}{'...' if len(suggestion) > 100 else ''}\n"

            history_str += f"\nStep {len(routing_history) + 1} (You): {current_agent_id}\n"
            history_str += "=== END OF PATH ===\n"
        else:
            history_str = f"\n=== ROUTING PATH ===\nStep 1 (First): You are {current_agent_id}\n=== END OF PATH ===\n"

        # 循环警告
        loop_warning_str = ""
        if loop_warnings:
            loop_warning_str = "\n⚠️ === LOOP WARNINGS === ⚠️\n"
            for agent, warning in loop_warnings.items():
                loop_warning_str += f"- {warning}\n"
            loop_warning_str += "Avoid these agents unless truly necessary.\n=== END OF WARNINGS ===\n"

        # 识别 Decision Maker
        decision_maker = None
        for agent in candidate_agents:
            if 'final' in agent.lower() or 'decision' in agent.lower():
                decision_maker = agent
                break

        prompt = f"""You are {current_agent_id}, coordinating a multi-agent system to solve a complex task.

=== TASK ===
{task}

=== YOUR RECENT OUTPUT ===
{current_output}

{history_str}

{loop_warning_str}

=== YOUR PERSPECTIVE (Private Beliefs) ===
{context}

=== CANDIDATE AGENTS ===
{', '.join(candidate_agents)}

**Decision Maker**: `{decision_maker if decision_maker else 'final_decision'}` - Select this to END the chain

=== YOUR DECISION ===
**CRITICAL**: 
1. Review the COMPLETE routing path above
2. Assess: Is the task fully resolved? Or does it need more work?
3. If RESOLVED: Select the Decision Maker ({decision_maker}) to end the chain
4. If NOT: Select an agent that brings NEW value

**DO NOT** select invalid names like "none" - use the exact names from the candidate list.

Respond in this EXACT format:
REASONING: <detailed analysis considering path history, completeness, and loop risks>
SELECTED: <exact agent name from candidates list>
SUGGESTION: <brief, actionable advice - under 50 words>
"""
        return prompt

    def _parse_llm_route_response_v2(
            self,
            response: str,
            candidate_agents: List[str]
    ) -> Tuple[str, str, str, bool]:
        """
        解析LLM路由响应 (v4.3.2: 增强版)

        Returns:
            (selected_agent, reasoning, suggestion, termination_requested)
        """

        lines = response.strip().split('\n')

        selected_agent = None
        reasoning = ""
        suggestion = ""
        termination_requested = False

        # **Step 1: 标准解析**
        for line in lines:
            line = line.strip()
            if line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('SELECTED:'):
                selected_agent = line.replace('SELECTED:', '').strip()
            elif line.startswith('SUGGESTION:'):
                suggestion = line.replace('SUGGESTION:', '').strip()

        # **Step 2: 检测终止意图**
        termination_keywords = ['none', 'end', 'stop', 'finish', 'complete', 'resolved', 'done']
        if selected_agent:
            selected_lower = selected_agent.lower()
            if any(keyword in selected_lower for keyword in termination_keywords):
                termination_requested = True
                print(f"[Parse] Detected termination intent: '{selected_agent}'")

        # **Step 3: 检测推理中的终止意图**
        if reasoning:
            reasoning_lower = reasoning.lower()
            if ('no further' in reasoning_lower or
                'fully resolved' in reasoning_lower or
                'task is complete' in reasoning_lower or
                'end the chain' in reasoning_lower):
                termination_requested = True
                print(f"[Parse] Detected termination intent in reasoning")

        # **Step 4: 容错处理**
        if not selected_agent or selected_agent.lower() not in [a.lower() for a in candidate_agents]:
            print(f"[Warning] Invalid/missing selection: '{selected_agent}'")

            # 如果明确要求终止，找 Decision Maker
            if termination_requested:
                for agent in candidate_agents:
                    if 'final' in agent.lower() or 'decision' in agent.lower():
                        selected_agent = agent
                        print(f"[Fallback] Routing to Decision Maker: {selected_agent}")
                        break

            # 否则使用第一个候选
            if not selected_agent or selected_agent.lower() not in [a.lower() for a in candidate_agents]:
                selected_agent = candidate_agents[0]
                reasoning = reasoning or "Failed to parse LLM response, using fallback"
                suggestion = "Please continue the task using your expertise"
                print(f"[Fallback] Using first candidate: {selected_agent}")

        if not suggestion:
            suggestion = "Build on the previous work and focus on quality"

        if not reasoning:
            reasoning = "Selection based on agent capabilities"

        return selected_agent, reasoning, suggestion, termination_requested

    def get_statistics(self) -> Dict:
        """获取统计信息 (v4.3.2: 新增终止统计)"""
        total_ops = (self.stats['cold_start_count'] +
                     self.stats['rag_retrieval_count'] +
                     self.stats['post_hoc_route_count'])

        kv_cache_hit_rate = 0.0
        if self.stats['post_hoc_route_count'] > 0:
            kv_cache_hit_rate = self.stats['kv_cache_hits'] / self.stats['post_hoc_route_count']

        return {
            **self.stats,
            'total_operations': total_ops,
            'kv_cache_hit_rate': kv_cache_hit_rate,
            'loop_detection_rate': self.stats['loop_detections'] / self.stats['post_hoc_route_count'] if self.stats['post_hoc_route_count'] > 0 else 0,
            'termination_attempt_rate': self.stats['termination_attempts'] / self.stats['post_hoc_route_count'] if self.stats['post_hoc_route_count'] > 0 else 0,
            'avg_tokens_per_llm_route': (
                self.stats['total_tokens_used'] / self.stats['post_hoc_route_count']
                if self.stats['post_hoc_route_count'] > 0 else 0
            )
        }