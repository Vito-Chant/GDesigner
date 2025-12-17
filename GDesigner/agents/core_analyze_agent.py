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
    CoRe v4.3 AnalyzeAgent - KV Cache 优化版

    **关键特性**:
    - _async_execute 返回 (response, messages) 元组
    - messages 包含完整对话历史（System + User + Assistant）
    - 为 KV Cache 复用提供物理基础
    """

    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = "", tokens=None):
        super().__init__(id, "CoReAnalyzeAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""
        self.tokens = tokens

    async def _process_inputs(
            self,
            raw_inputs: Dict[str, str],
            spatial_info: Dict[str, Dict],
            temporal_info: Dict[str, Dict],
            **kwargs
    ) -> Tuple[str, str]:
        """
        处理输入并构建完整的 Prompt

        职责：
        1. 构建基础 system_prompt 和 user_prompt
        2. 处理 Wiki 搜索（如果角色需要）
        3. 拼接 RAG Context
        4. 拼接 Suggestion
        5. 返回最终的完整 Prompt

        Returns:
            (system_prompt, user_prompt)
        """
        # === 1. 基础 Prompt ===
        system_prompt = f"{self.constraint}"
        if 'system_instruction' in raw_inputs:
            system_prompt += f"\n\n**SYSTEM INSTRUCTION**:\n{raw_inputs['system_instruction']}"


        # 基础任务描述
        if self.role != 'Fake':
            user_prompt = f"The task is: {raw_inputs['task']}\n"
        else:
            user_prompt = self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])

        # === 2. 处理空间和时间信息 ===
        spatial_str = ""
        temporal_str = ""

        for id, info in spatial_info.items():
            output_content = extract_content(info['output'])

            # Wiki搜索特殊处理
            if self.role == 'Wiki Searcher' and info['role'] == 'Knowlegable Expert':
                queries = find_strings_between_pluses(output_content)
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"\nThe key entities of the problem are explained in Wikipedia as follows:\n{self.wiki_summary}\n"

            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {output_content}\n\n"

        for id, info in temporal_info.items():
            output_content = extract_content(info['output'])
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {output_content}\n\n"

        # === 3. 拼接 RAG Context ===
        if 'retrieved_history' in raw_inputs and raw_inputs['retrieved_history']:
            rag_context = raw_inputs['retrieved_history']
            user_prompt += f"""
=== RELEVANT CONTEXT FROM HISTORY ===
{rag_context}
=== END OF CONTEXT ===

"""

        # === 4. 拼接 Suggestion ===
        if 'insight' in raw_inputs and raw_inputs['insight']:
            suggestion = raw_inputs['insight']
            user_prompt += f"""
=== STRATEGIC SUGGESTION (HIGH PRIORITY) ===
{suggestion}

Please consider this guidance when formulating your response.
=== END OF SUGGESTION ===

"""

        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """同步执行 (保留兼容性)"""
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
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
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        异步执行 (v4.3 KV Cache 优化版)

        **关键变更**:
        返回值从 `str` 改为 `Tuple[str, List[Dict]]`

        Returns:
            (response, messages) 其中：
            - response: 生成的文本
            - messages: 完整对话历史 [System, User, Assistant]

        **KV Cache 基础**:
        返回的 messages 可被下游直接复用，追加新指令后继续生成
        """
        # 1. 获取完整 Prompt
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)

        # 2. 构建消息历史
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        # 3. 调用 LLM 生成
        response = await self.llm.agen(messages)

        # 4. 附加 Wiki 摘要（如果有）
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""

        # **5. 关键步骤：将 Assistant 的响应加入历史**
        messages.append({'role': 'assistant', 'content': response})

        # **6. 返回元组：(文本, 完整历史)**
        return response, messages