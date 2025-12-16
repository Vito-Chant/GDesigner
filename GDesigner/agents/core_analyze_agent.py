from typing import List, Any, Dict, Tuple
import re
import weave

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.search.wiki import search_wiki_main


def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)


def extract_content(output: Any) -> str:
    if isinstance(output, tuple) and len(output) >= 1:
        return output[0]
    return output


@AgentRegistry.register('CoReAnalyzeAgent')
class AnalyzeAgent(Node):
    """
    CoRe v4.2 AnalyzeAgent (Optimized)

    优化点:
    1. 结构化 Prompt：清晰区分 Task, Context (RAG/Wiki), Peer Outputs, Suggestion。
    2. 上下文融合：Wiki 知识直接融入 Prompt 上下文，而非生成后追加。
    3. 逻辑流：Task (定义目标) -> Context (提供支持) -> Suggestion (提供策略) -> Execution (执行指令)。
    """

    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = "", tokens=None):
        super().__init__(id, "CoReAnalyzeAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.tokens = tokens

    async def _process_inputs(
            self,
            raw_inputs: Dict[str, str],
            spatial_info: Dict[str, Dict],
            temporal_info: Dict[str, Dict],
            **kwargs
    ) -> Tuple[str, str]:
        """
        处理输入并构建结构化的 Prompt
        """
        # === 1. System Prompt (角色设定与约束) ===
        system_prompt = f"{self.constraint}"

        # === 2. 构建各个 Prompt 板块 ===
        prompt_sections = []

        # [Section 1: 核心任务]
        task_content = raw_inputs.get('task', '')
        if self.role == 'Fake':
            task_content = self.prompt_set.get_adversarial_answer_prompt(task_content)

        prompt_sections.append(f"### 1. CORE TASK\n{task_content}")

        # [Section 2: 历史上下文 (RAG)]
        if 'retrieved_history' in raw_inputs and raw_inputs['retrieved_history']:
            prompt_sections.append(
                f"### 2. RELEVANT HISTORY (RAG)\nThe following context is retrieved from previous steps:\n{raw_inputs['retrieved_history']}")

        # [Section 3: 外部知识 (Wiki) & 同伴输出 (Spatial/Temporal)]
        # 处理 Wiki 搜索 (逻辑保持不变，但结果直接放入 Prompt)
        wiki_context = ""
        peer_outputs = []

        for id, info in spatial_info.items():
            output_content = extract_content(info['output'])

            # Wiki 搜索触发逻辑
            if self.role == 'Wiki Searcher' and info['role'] == 'Knowlegable Expert':
                queries = find_strings_between_pluses(output_content)
                wiki_results = await search_wiki_main(queries)
                if wiki_results:
                    wiki_text = ".\n".join(wiki_results)
                    wiki_context = f"**Wikipedia Knowledge**:\n{wiki_text}\n"

            peer_outputs.append(f"- Agent {id} ({info['role']}): {output_content}")

        for id, info in temporal_info.items():
            output_content = extract_content(info['output'])
            peer_outputs.append(f"- Agent {id} ({info['role']}): {output_content}")

        # 组合 Section 3
        context_str = ""
        if wiki_context:
            context_str += f"{wiki_context}\n"
        if peer_outputs:
            context_str += "**Peer Outputs**:\n" + "\n\n".join(peer_outputs)

        if context_str:
            prompt_sections.append(f"### 3. CURRENT CONTEXT & PEER INFO\n{context_str}")

        # [Section 4: 战略建议 (Suggestion/Insight) - 高优先级]
        if 'insight' in raw_inputs and raw_inputs['insight']:
            prompt_sections.append(
                f"### 4. STRATEGIC SUGGESTION (HIGH PRIORITY)\n{raw_inputs['insight']}\n\n*Please strictly follow the above suggestion during execution.*")

        # === 3. 组装最终 User Prompt ===
        # 添加具体的执行指令作为结尾
        prompt_sections.append(
            "### 5. EXECUTION INSTRUCTION\nBased on the task, history, and suggestions above, please provide your response.")

        user_prompt = "\n\n".join(prompt_sections)

        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """
        同步执行入口
        注意: 由于 search_wiki_main 是异步的，_process_inputs 必须是 async。
        如果必须在同步环境调用，这里需要 wrap 一层，或者接受 Wiki 功能在同步模式下不可用。
        此处为了兼容代码结构，暂且保留，但实际运行中应当主要使用 _async_execute。
        """
        # 警告：如果在 Event Loop 中直接调用此同步方法可能会报错，因为 _process_inputs 是 async
        # 这里假设外部已经处理了 async 环境，或者这只是个 placeholder
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        system_prompt, user_prompt = loop.run_until_complete(
            self._process_inputs(input, spatial_info, temporal_info)
        )

        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    @weave.op()
    async def _async_execute(
            self,
            input: Dict[str, str],
            spatial_info: Dict[str, Dict],
            temporal_info: Dict[str, Dict],
            **kwargs
    ):
        """
        异步执行 (v4.2 极简版)
        """
        # 1. 获取结构化 Prompt (包含 Wiki, RAG, Suggestion 所有信息)
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)

        # 2. 构建消息
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        # 3. 调用 LLM
        # Wiki 内容已经在 _process_inputs 中处理并放入 user_prompt，此处无需再追加
        response = await self.llm.agen(message)

        return response