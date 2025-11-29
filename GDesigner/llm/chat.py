import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional, Any

import weave
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
import async_timeout
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import weave

# 保持你原有的包引用路径
from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

# 加载环境变量
load_dotenv()

# 配置变量
MINE_BASE_URL = os.getenv("MINE_BASE_URL", "")
MINE_API_KEYS = os.getenv("MINE_API_KEYS", "")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_BASE_URL_CHAT = os.getenv("VLLM_BASE_URL_CHAT", "http://localhost:8001/v1")
VLLM_BASE_URL_EMBED = os.getenv("VLLM_BASE_URL_EMBED", "http://localhost:8000/v1")


# --- 2. 核心通用请求逻辑 (私有函数) ---

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
@weave.op()
async def _generic_achat(
        model: str,
        messages: List[Dict],
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        **kwargs
) -> str:
    """
    通用的异步聊天请求函数，底层使用 AsyncOpenAI。
    vLLM 和 GPT 类模型均可复用此逻辑。
    """
    # 如果 api_key 为空（如本地 vLLM），OpenAI SDK 需要一个非空占位符
    if not api_key:
        api_key = "EMPTY"
    return_prompt_embedding = kwargs.pop("return_prompt_embedding", False)
    extra_body = kwargs.pop("extra_body", {})
    if return_prompt_embedding:
        extra_body["return_prompt_embedding"] = True
        max_tokens = 1

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        # 300秒超时，防止请求无限挂起
        async with async_timeout.timeout(300):
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **kwargs  # 传递其他可能的参数，如 top_p
            )

        if return_prompt_embedding:
            # 尝试从 model_extra 获取，如果 API Server 修改正确，它应该在这里
            prompt_embeds = completion.prompt_embeds

            # 返回内容和向量
            return prompt_embeds

        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            # 简单的 token 统计与计费 (保留原有逻辑)
            prompt_text = "".join([str(m.get('content', '')) for m in messages])
            cost_count(prompt_text, response_message, model)

            return response_message
        else:
            raise RuntimeError("Response content is not a string.")

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request for model {model}: {e}")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
@weave.op()
async def _generic_acompletion(
        model: str,
        prompt: Union[str, List[str]],
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        **kwargs
) -> Union[str, Any]:
    """
    通用的异步文本补全请求函数，底层使用 AsyncOpenAI.completions.create。
    适用于 Base Model 或需要直接发送 Prompt 的场景。
    """
    # 如果 api_key 为空（如本地 vLLM），OpenAI SDK 需要一个非空占位符
    if not api_key:
        api_key = "EMPTY"

    # 处理 return_prompt_embedding 参数
    return_prompt_embedding = kwargs.pop("return_prompt_embedding", False)
    extra_body = kwargs.pop("extra_body", {})
    if return_prompt_embedding:
        extra_body["return_prompt_embedding"] = True
        max_tokens = 1

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        # 300秒超时，防止请求无限挂起
        async with async_timeout.timeout(300):
            completion = await client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **kwargs  # 传递其他可能的参数，如 top_p, logprobs 等
            )

        if return_prompt_embedding:
            prompt_embeds = getattr(completion, "prompt_embeds", None)
            return prompt_embeds

        # Completions API 的返回结果在 choices[0].text 中
        response_text = completion.choices[0].text

        if isinstance(response_text, str):
            prompt_content = prompt if isinstance(prompt, str) else "\n".join(prompt)
            cost_count(prompt_content, response_text, model)

            return response_text
        else:
            raise RuntimeError("Response content is not a string.")

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async completion request for model {model}: {e}")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def _generic_aembed(
        model: str,
        input: Union[str, List[str]],
        api_key: str,
        base_url: str,
        **kwargs
) -> Union[List[float], List[List[float]]]:
    """
    通用的异步 Embedding 请求函数，底层使用 AsyncOpenAI.embeddings.create
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        # 设置超时，防止请求无限挂起
        async with async_timeout.timeout(300):
            response = await client.embeddings.create(
                model=model,
                input=input,
                **kwargs
            )

            # 如果输入是单个字符串，返回单个向量 List[float]
            if isinstance(input, str):
                return response.data[0].embedding

            # 如果输入是列表，返回向量列表 List[List[float]]
            return [item.embedding for item in response.data]

    except Exception as e:
        # 这里可以添加特定的日志记录
        print(f"Embedding request failed: {e}")
        raise e


# --- 4. 具体实现类 ---

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    """
    用于调用通用/商业 API (如 OpenAI, DeepSeek 等)
    """

    def __init__(self, model_name: str, max_tokens: int = 2048, temperature: float = 1.0):
        # 将 model_name 传给基类的 llm_name
        super().__init__(llm_name=model_name, max_tokens=max_tokens, temperature=temperature)

    async def agen(self, messages: Union[List[Message], List[Dict], str], **kwargs) -> str:
        # 1. 预处理消息
        formatted_msgs = self._preprocess_messages(messages)

        # 2. 参数合并：优先使用 kwargs 中的参数，否则使用实例默认值
        req_temp = kwargs.get('temperature', self.temperature)
        req_max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # 3. 移除 kwargs 中已经处理过的参数，避免重复传给 API
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}

        # 4. 发起请求
        return await _generic_achat(
            model=self.llm_name,
            messages=formatted_msgs,
            api_key=MINE_API_KEYS,
            base_url=MINE_BASE_URL,
            temperature=req_temp,
            max_tokens=req_max_tokens,
            **api_kwargs
        )


@LLMRegistry.register('VLLMChat')
class VLLMChat(LLM):
    """
    用于调用本地部署的 vLLM 模型
    """

    def __init__(self, model_name: str, max_tokens: int = 2048, temperature: float = 1.0):
        super().__init__(llm_name=model_name, max_tokens=max_tokens, temperature=temperature)
        self._tokenizer = None

    @property
    def tokenizer(self):
        """
        懒加载 Tokenizer，避免初始化时消耗过多时间
        """
        if self._tokenizer is None:
            try:
                # 注意：这里的 model_name 必须是本地路径或 HF 上的模型 ID
                # 且必须与 vLLM 服务端加载的模型一致
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_name,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Error loading tokenizer for {self.llm_name}: {e}")
                raise e
        return self._tokenizer

    async def agen(self, messages: Union[List[Message], List[Dict], str], **kwargs) -> str:
        # 1. 预处理消息
        formatted_msgs = self._preprocess_messages(messages)

        # 2. 参数合并
        req_temp = kwargs.get('temperature', self.temperature)
        req_max_tokens = kwargs.get('max_tokens', self.max_tokens)
        postfix = kwargs.get('postfix', None)

        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'postfix']}

        if postfix is not None:
            prompt_input = self.tokenizer.apply_chat_template(
                formatted_msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_input = prompt_input + postfix
            return await _generic_acompletion(
                model=self.llm_name,
                prompt=prompt_input,
                api_key="EMPTY",
                base_url=VLLM_BASE_URL,
                temperature=req_temp,
                max_tokens=req_max_tokens,
                **api_kwargs
            )
        # else:
        #     prompt_input = self.tokenizer.apply_chat_template(
        #         formatted_msgs,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        #     return await _generic_acompletion(
        #         model=self.llm_name,
        #         prompt=prompt_input,
        #         api_key="EMPTY",
        #         base_url=VLLM_BASE_URL_CHAT,
        #         temperature=req_temp,
        #         max_tokens=req_max_tokens,
        #         **api_kwargs
        #     )

        import random
        url1 = "http://localhost:8000/v1"
        url2 = "http://localhost:8001/v1"
        url = random.choice([url1, url2])

        # 3. 发起请求 (使用 VLLM 的地址和空 Key)
        return await _generic_achat(
            model=self.llm_name,
            messages=formatted_msgs,
            api_key="EMPTY",
            # base_url=VLLM_BASE_URL,
            base_url=url,
            temperature=req_temp,
            max_tokens=req_max_tokens,
            **api_kwargs
        )

    async def aembed(self, messages: Union[List[Message], List[Dict], str], **kwargs) -> List[float]:
        """
        异步获取 Embedding，接口与 agen 保持一致。
        会自动应用 Chat Template，确保生成的文本与 agen 在服务端看到的完全一致。

        Args:
            messages: 可以是消息列表 [{'role':..., 'content':...}]，也可以是纯文本字符串
            **kwargs: 其他参数

        Returns:
            List[float]: 对应的 Embedding 向量
        """
        prompt_input = ""

        # 1. 如果输入是字符串，直接使用（兼容纯文本模式）
        if isinstance(messages, str):
            prompt_input = messages

        # 2. 如果输入是消息列表，手动应用 Chat Template
        else:
            # 预处理：统一转换为 Dict 格式
            formatted_msgs = self._preprocess_messages(messages)

            # 关键点：应用模板
            # tokenize=False: 我们只需要转换后的字符串，不需要 token ids
            # add_generation_prompt=True: 加上最后的 <|im_start|>assistant\n，这对生成任务的 KV Cache 复用至关重要
            prompt_input = self.tokenizer.apply_chat_template(
                formatted_msgs,
                tokenize=False,
                add_generation_prompt=True
            )

        # 3. 发送请求
        # 注意：我们把处理好的 prompt_input 传给 embedding 接口
        return await _generic_aembed(
            model=self.llm_name,
            input=prompt_input,
            api_key="EMPTY",
            base_url=VLLM_BASE_URL_EMBED,
            **kwargs
        )
