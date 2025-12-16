from typing import List, Any, Dict
import re
import weave
import torch

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
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = "", tokens=None):
        super().__init__(id, "CoReAnalyzeAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""
        self.tokens = tokens

    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict],
                              temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(
            raw_inputs['task'])
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            output_content = extract_content(info['output'])
            if self.role == 'Wiki Searcher' and info['role'] == 'Knowlegable Expert':
                queries = find_strings_between_pluses(output_content)
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {output_content}\n\n"
        for id, info in temporal_info.items():
            output_content = extract_content(info['output'])
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {output_content}\n\n"

        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    @weave.op()
    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)

        if 'retrieved_history' in input and input['retrieved_history']:
            rag_context = input['retrieved_history']

            # 在用户提示词前添加上下文
            user_prompt = f"""=== RELEVANT CONTEXT FROM HISTORY ===
        {rag_context}
        === END OF CONTEXT ===

        {user_prompt}
        """

            # 3. **注入Insight指令** (最高优先级)
        if 'insight' in input and input['insight']:
            insight = input['insight']

            # 在提示词末尾添加战略指令
            user_prompt = f"""{user_prompt}

        === STRATEGIC INSTRUCTION (HIGH PRIORITY) ===
        {insight}
        Please pay special attention to this guidance when formulating your response.
        === END OF INSTRUCTION ===
        """

        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

        response = await self.llm.agen(message)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""

        return response
