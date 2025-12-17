"""
CoRe Framework v4.3.2: Unified Ranker Module - 修复终止逻辑 & 增强失败检测
System 1.5 (Reranker) + System 2 (LLM with Path Awareness + Proper Termination)

v4.3.2 关键修复:
- 正确处理 LLM 希望终止的情况
- 明确告知 LLM 如何触发 Decision Maker
- 改进错误处理和降级逻辑
- 增强 Agent 失败/求助信号的检测
- 优化 Prompt 结构与历史格式
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import weave
from sentence_transformers import CrossEncoder
import numpy as np
from transformers import AutoTokenizer
import math
import requests


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


class QwenRerankerClient:
    """
    Qwen3-Reranker vLLM 客户端
    模拟 CrossEncoder 的接口，通过 HTTP 调用 vLLM Server
    """

    def __init__(self, base_url: str, model_name: str = "Qwen/Qwen3-Reranker-0.6B", api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 预计算 token IDs
        self.true_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]

        # 强制跳过思考的后缀
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_length = 16384

        # 默认指令 (Web Search fallback)
        self.default_instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def format_instruction(self, query: str, doc: str, instruction: Optional[str] = None) -> List[Dict]:
        """构建 Prompt，支持自定义 Instruction"""
        instruct_content = instruction if instruction else self.default_instruction

        return [
            {"role": "system",
             "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruct_content}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]

    @weave.op()
    def predict(self, pairs: List[Tuple[str, str]], instruction: Optional[str] = None) -> np.ndarray:
        """
        批量计算 (Query, Document) 对的分数
        Args:
            pairs: List of (query, document) tuples
            instruction: Custom instruction for the task (Optional)
        """
        if not pairs:
            return np.array([])

        # 1. 构建 Prompt (应用 Chat Template 并添加后缀)
        prompts = []
        for query, doc in pairs:
            # 传入自定义 instruction
            messages = self.format_instruction(query, doc, instruction)

            # 使用 tokenizer 应用模板
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
            )
            # 截断并添加后缀 (Force Output)
            final_token_ids = prompt_token_ids[:self.max_length - len(self.suffix_tokens)] + self.suffix_tokens
            prompts.append(final_token_ids)

        # 2. 调用 vLLM Completions API
        payload = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 20,
            "echo": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(f"{self.base_url}/v1/completions", json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()['choices']

            results.sort(key=lambda x: x['index'])

            scores = []
            for res in results:
                logprobs = res['logprobs']['top_logprobs'][0]

                true_logit = -10.0
                false_logit = -10.0

                for token_str, logprob in logprobs.items():
                    if token_str.strip().lower() == "yes":
                        true_logit = logprob
                    elif token_str.strip().lower() == "no":
                        false_logit = logprob

                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)

                if true_score + false_score == 0:
                    score = 0.0
                else:
                    score = true_score / (true_score + false_score)

                scores.append(score)

            return np.array(scores)

        except Exception as e:
            print(f"Error calling vLLM Reranker: {e}")
            return np.zeros(len(pairs))


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
      * **失败检测**: 识别 Agent 的求助信号，防止错误终止
    """

    def __init__(
            self,
            llm,
            # reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
            reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B",
            max_loop_count: int = 2,
            decision_maker_id: str = "final_decision",
            reranker_api_url: str = "http://localhost:8001"
    ):
        """初始化Unified Ranker"""
        self.llm = llm
        self.max_loop_count = max_loop_count
        self.decision_maker_id = decision_maker_id
        self.reranker_model_name = reranker_model_name

        # print(f"Loading Cross-Encoder: {reranker_model_name}")
        if "Qwen" in reranker_model_name:
            self.reranker = QwenRerankerClient(
                base_url=reranker_api_url,
                model_name=reranker_model_name
            )
        else:
            self.reranker = CrossEncoder(reranker_model_name)

        self.stats = {
            'cold_start_count': 0,
            'rag_retrieval_count': 0,
            'post_hoc_route_count': 0,
            'kv_cache_hits': 0,
            'loop_detections': 0,
            'termination_attempts': 0,
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

        if "Qwen" in self.reranker_model_name:
            cold_start_instruction = "Given a task description, retrieve the agent profile that has the most relevant capabilities and expertise to solve it."
            scores = self.reranker.predict(pairs, instruction=cold_start_instruction)
        else:
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

        if "Qwen" in self.reranker_model_name:
            rag_instruction = "Given a problem-solving task, retrieve relevant historical outputs or intermediate results that provide useful context for the next step."
            scores = self.reranker.predict(pairs, instruction=rag_instruction)
        else:
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
        """检测路由循环 (v4.3.5: 支持 ReAct 自循环)"""
        if not routing_history:
            return False, ""

        # 获取历史路径序列
        agent_sequence = [d.get('selected', '') for d in routing_history]

        # 1. ✅ 检测连续自循环 (Consecutive Self-Loop)
        # 如果 candidate 是自己，检查是否已经连续太多次了
        if candidate_agent == current_agent:
            # 倒序计算当前已经连续了多少次
            consecutive_count = 1  # 加上当前这次
            for agent in reversed(agent_sequence):
                if agent == current_agent:
                    consecutive_count += 1
                else:
                    break

            # 允许最多连续 3 次 (Step 1 -> Step 2 -> Step 3)
            MAX_CONSECUTIVE = 3
            if consecutive_count > MAX_CONSECUTIVE:
                warning = f"⚠ EXCESSIVE SELF-LOOP: You have selected yourself {consecutive_count} times in a row. Please handover to another agent."
                self.stats['loop_detections'] += 1
                return True, warning

            # 如果没超限，允许自选
            return False, ""

        # 2. 检测 A -> B -> A 的乒乓循环 (Ping-Pong)
        if len(agent_sequence) >= 2:
            if agent_sequence[-1] == candidate_agent:
                # 上一步选的就是 candidate (这其实被上面的自循环逻辑覆盖了，但保留作为保险)
                pass
            elif len(agent_sequence) >= 2 and agent_sequence[-2] == candidate_agent:
                # 这种 A->B->A 通常是不好的，除非 B 是 Critic
                # 可以根据需要放宽，或者保持警告
                agent_counts = Counter(agent_sequence)
                if agent_counts[candidate_agent] >= self.max_loop_count:
                    warning = f"⚠ LOOP DETECTED: {current_agent} ↔ {candidate_agent} (Ping-Pong)"
                    self.stats['loop_detections'] += 1
                    return True, warning

        # 3. 总次数限制 (放宽)
        # ReAct 模式下，主力 Agent (如 MathSolver) 可能会被多次使用
        agent_counts = Counter(agent_sequence)
        total_limit = 6  # 提高总上限
        if agent_counts[candidate_agent] >= total_limit:
            warning = f"⚠ REPEATED AGENT: {candidate_agent} used too many times ({agent_counts[candidate_agent]})"
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
        """
        self.stats['post_hoc_route_count'] += 1

        # **Step 1: 循环预检测**
        loop_warnings = {}
        for candidate in candidate_agents:
            is_loop, warning = self._detect_loop(routing_history, current_agent_id, candidate)
            if is_loop:
                loop_warnings[candidate] = warning
                print(warning)

        # **Step 2: 构建增强的路由指令**
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

        # **Step 5: 解析响应（v4.3.2: 改进解析逻辑，包含失败检测）**
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
        构建增强的路由指令 (v4.3.2: 明确终止机制与Prompt优化)
        """

        # === 1. 构建详细的路径历史（统一格式）===
        path_summary = "\n=== ROUTING HISTORY ===\n"
        if not routing_history:
            path_summary += (
                f"Step 0 (Cold Start): Selected {current_agent_id}\n"
                f"  Reason: Initial selection by system\n"
                f"  Suggestion: Analyze task\n"
                f"\nYou are at Step 1 (Current): {current_agent_id}\n"
            )
        else:
            path_summary += f"Step 0 (Cold Start): Selected {routing_history[0].get('selected', 'Unknown')}\n"
            for i, decision in enumerate(routing_history, 1):
                selected = decision.get('selected', 'Unknown')
                reasoning = decision.get('reasoning', 'N/A')
                suggestion = decision.get('suggestion', 'N/A')
                path_summary += f"\nStep {i}: → {selected}\n"
                path_summary += f"  Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}\n"
                path_summary += f"  Suggestion: {suggestion[:80]}{'...' if len(suggestion) > 80 else ''}\n"

            path_summary += f"\nCurrent Step {len(routing_history) + 1}: You are {current_agent_id}\n"
        path_summary += "=== END OF HISTORY ===\n"

        # === 2. 构建循环警告 ===
        loop_warning_str = ""
        if loop_warnings:
            loop_warning_str = "\n⚠️ === CRITICAL: LOOP WARNINGS === ⚠️\n"
            for agent, warning in loop_warnings.items():
                loop_warning_str += f"- {agent}: {warning}\n"
            loop_warning_str += "Please AVOID selecting agents with loop warnings unless absolutely necessary.\n"
            loop_warning_str += "=== END OF WARNINGS ===\n"

        # === 3. 识别与构建候选列表（精简格式）===
        decision_maker = None
        regular_agents = []
        for agent in candidate_agents:
            if 'final' in agent.lower() or 'decision' in agent.lower():
                decision_maker = agent
            else:
                regular_agents.append(agent)

        candidates_info = "\n**Available Candidates**:\n"
        if regular_agents:
            candidates_info += "  Regular Agents (for continued work):\n"
            for agent in regular_agents:
                candidates_info += f"    - {agent}\n"

        if decision_maker:
            candidates_info += f"\n  **Decision Maker** (to END the chain):\n"
            candidates_info += f"    - `{decision_maker}` → Select ONLY when task is FULLY resolved\n"

        candidates_info += "\n  (Note: Detailed profiles for these agents are in 'YOUR BELIEFS' above)\n"

        # === 4. 组装完整指令（优化结构）===
        instruction = f"""
    === ROLE SWITCH: YOU ARE NOW THE COORDINATOR ===
    
    Excellent work on the analysis above. Now, please act as the **Coordinator** to decide the next step.
    
    {path_summary}
    
    **IMPORTANT: Path Analysis**
    - Review the ENTIRE routing path above
    - Identify patterns: Are we making progress or circling?
    - Consider what each agent has already contributed
    
    {loop_warning_str}
    
    === YOUR BELIEFS & KNOWLEDGE (from Mind Registry) ===
    {context}
    
    {candidates_info}
    
    **CRITICAL DECISION CRITERIA**:
    
    1. **CHECK FOR FAILURE/PARTIAL SUCCESS (HIGHEST PRIORITY):**
       ⚠️ If the current agent's output indicates ANY of the following:
       - Explicitly stated inability to complete the task
       - Mentioned missing information or capabilities
       - Asked for help or suggested another agent should handle it
       - Only completed part of the work
       - Contains phrases like "I cannot...", "I need...", "This requires...", "Unable to..."
       
       → **DO NOT select the Decision Maker**
       → **SELECT a different agent** who can address the specific issue
       → Provide clear guidance on what that agent should focus on
    
    2. **If the task is FULLY SOLVED and no more analysis is needed:**
       - All requirements are met
       - No agent has flagged issues or requested help
       - Output is complete and validated
       - Select the Decision Maker: `{decision_maker if decision_maker else 'final_decision'}`
       - This will END the routing chain
    
    3. **If the task needs MORE work (new perspective, verification, implementation):**
       - Select an appropriate regular agent
       - Provide a clear suggestion for what they should focus on
    
    4. **SELF-CORRECTION & MULTI-STEP REASONING:**
           - You can SELECT YOURSELF ({current_agent_id}) if:
             * You need to verify your own code/calculation.
             * You need to perform the next step of a complex task.
             * You realized you made a mistake and want to fix it.
    
    5. **NEVER select "none" or invalid names** - always choose from the list above
    
    **Decision Format** (MUST follow exactly):
    REASONING: <detailed analysis: Is task complete? Any failure signals? Loop risks?>
    SELECTED: <exact agent name from the list above>
    SUGGESTION: <brief, actionable advice - under 50 words>
    
    **Remember**: 
    - **PRIORITY 1**: Check for failure/partial completion signals
    - The routing path is CRUCIAL - don't ignore it!
    - Only select Decision Maker when task is FULLY resolved
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
        降级方案：构建完整 Prompt (v4.3.2: 结构同步更新)
        """
        # 简化版逻辑复用 instruction 构建的思路

        history_str = "\n=== ROUTING HISTORY ===\n"
        if routing_history:
            history_str += f"Step 0 (Cold Start): {routing_history[0].get('selected', 'Unknown')}\n"
            for i, decision in enumerate(routing_history, 1):
                history_str += f"Step {i}: {decision.get('selected', 'Unknown')} (Why: {decision.get('reasoning', '')[:50]}...)\n"
            history_str += f"Current Step: You are {current_agent_id}\n"
        else:
            history_str += f"Step 1: You are {current_agent_id}\n"
        history_str += "=== END OF HISTORY ===\n"

        decision_maker = None
        for agent in candidate_agents:
            if 'final' in agent.lower() or 'decision' in agent.lower():
                decision_maker = agent
                break

        prompt = f"""You are {current_agent_id}, coordinating a multi-agent system.
    
    {history_str}
    
    === AGENT BELIEFS ===
    {context}
    
    === CANDIDATE AGENTS ===
    {', '.join(candidate_agents)}
    (Decision Maker: {decision_maker if decision_maker else 'final_decision'})
    
    === YOUR DECISION ===
    **CRITICAL**: 
    1. **Did you FAIL or PARTIALLY complete the task?**
       - If YES -> Select a helper agent. DO NOT END.
    2. **Is task FULLY resolved?**
       - If YES -> Select {decision_maker} to END.
    3. **Loop Risk?**
       - Avoid agents used recently.
    
    Respond in this EXACT format:
    REASONING: <analysis>
    SELECTED: <agent name>
    SUGGESTION: <advice>
    """
        return prompt


    def _parse_llm_route_response_v2(
            self,
            response: str,
            candidate_agents: List[str]
    ) -> Tuple[str, str, str, bool]:
        """
        解析LLM路由响应 (v4.3.2: 增强版 - 包含失败检测)

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

        # **Step 2: ✅ 检测失败信号 (新增)**
        # 防止 LLM 虽然嘴上说失败了，手却选了 Decision Maker
        # failure_keywords = [
        #     'cannot', 'unable to', 'failed to', 'missing',
        #     'incomplete', 'need help', 'requires', 'should ask',
        #     'i need', 'not enough', 'lack', 'insufficient'
        # ]

        has_failure_signal = False
        # if reasoning:
        #     reasoning_lower = reasoning.lower()
        #     has_failure_signal = any(kw in reasoning_lower for kw in failure_keywords)
        #
        # if has_failure_signal:
        #     print(f"[Parse] Detected FAILURE signal in reasoning - preventing termination")
        #     # 如果 LLM 选择了 Decision Maker，强制改为选择其他 Agent
        #     if selected_agent:
        #         selected_lower = selected_agent.lower()
        #         if 'final' in selected_lower or 'decision' in selected_lower:
        #             print(f"[Parse] Overriding Decision Maker selection due to failure signal")
        #             # 选择第一个非 Decision Maker 作为 fallback
        #             for agent in candidate_agents:
        #                 if 'final' not in agent.lower() and 'decision' not in agent.lower():
        #                     selected_agent = agent
        #                     suggestion = f"Previous agent reported issues: {suggestion or 'Please help address the failure'}"
        #                     break

        # **Step 3: 检测终止意图**
        termination_keywords = ['none', 'end', 'stop', 'finish', 'complete', 'resolved', 'done']

        # 只有在没有失败信号时，才允许检测终止
        if selected_agent and not has_failure_signal:
            selected_lower = selected_agent.lower()
            if any(keyword in selected_lower for keyword in termination_keywords):
                termination_requested = True
                print(f"[Parse] Detected termination intent: '{selected_agent}'")

        # **Step 4: 检测推理中的终止意图**
        if reasoning and not has_failure_signal:
            reasoning_lower = reasoning.lower()
            if ('no further' in reasoning_lower or
                    'fully resolved' in reasoning_lower or
                    'task is complete' in reasoning_lower or
                    'end the chain' in reasoning_lower):
                termination_requested = True
                print(f"[Parse] Detected termination intent in reasoning")

        # **Step 5: 容错处理**
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
            'loop_detection_rate': self.stats['loop_detections'] / self.stats['post_hoc_route_count'] if self.stats[
                                                                                                             'post_hoc_route_count'] > 0 else 0,
            'termination_attempt_rate': self.stats['termination_attempts'] / self.stats['post_hoc_route_count'] if
            self.stats['post_hoc_route_count'] > 0 else 0,
            'avg_tokens_per_llm_route': (
                self.stats['total_tokens_used'] / self.stats['post_hoc_route_count']
                if self.stats['post_hoc_route_count'] > 0 else 0
            )
        }
