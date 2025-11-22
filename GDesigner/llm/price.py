import tiktoken
from transformers import AutoTokenizer
from typing import Optional, Tuple

# 保持原有的全局变量引用，但移除了 Tokenizer 的导入，改为在本地定义
try:
    from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
except ImportError:
    # 提供简单的 Mock 类以防脱离项目单独运行时报错
    class MockSingleton:
        _instance = None

        def __init__(self): self.value = 0

        @classmethod
        def instance(cls):
            if not cls._instance: cls._instance = cls()
            return cls._instance


    Cost = PromptTokens = CompletionTokens = MockSingleton


# --- 1. Tokenizer 单例管理机制 ---

class TokenizerManager:
    """
    管理 HuggingFace Tokenizer 的单例/多例池。
    避免对同一个模型重复加载 tokenizer。
    """
    _instances = {}

    @classmethod
    def get_tokenizer(cls, model_name: str):
        if model_name not in cls._instances:
            # print(f"Loading tokenizer for {model_name}...")
            try:
                cls._instances[model_name] = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception as e:
                print(f"Warning: Failed to load tokenizer for {model_name}: {e}. Falling back to token estimation.")
                return None
        return cls._instances[model_name]


# --- 2. 核心计算函数 ---

def cal_token_openai(model: str, text: str) -> int:
    """使用 tiktoken 计算 OpenAI 模型的 token"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def cal_token_hf(model: str, text: str) -> int:
    """使用 transformers AutoTokenizer 计算 HuggingFace 模型的 token"""
    tokenizer = TokenizerManager.get_tokenizer(model)
    if tokenizer:
        return len(tokenizer(text, return_tensors='pt')['input_ids'][0])
    else:
        # 如果加载失败，按 1 token ≈ 4 chars 估算
        # return len(text) // 4
        raise "tokenizer not found"


# --- 3. 统一计费入口 ---

def cost_count(prompt: str, response: str, model_name: str) -> Tuple[float, int, int]:
    """
    统一计费函数，自动根据 model_name 选择计算逻辑。

    Returns:
        (price, prompt_tokens, completion_tokens)
    """
    price = 0.0
    prompt_len = 0
    completion_len = 0

    # 规范化模型名称
    model_lower = model_name.lower()

    # === 策略 A: OpenAI 模型 ===
    if any(k in model_lower for k in ["gpt-3.5", "gpt-4", "dall-e"]):
        prompt_len = cal_token_openai(model_name, prompt)
        completion_len = cal_token_openai(model_name, response)

        # 计算价格
        if "gpt-4" in model_lower:
            # 默认使用 gpt-4-preview 的价格兜底，具体可细化
            info = OPENAI_MODEL_INFO["gpt-4"].get(model_name, OPENAI_MODEL_INFO["gpt-4"]["gpt-4-1106-preview"])
            price = (prompt_len * info["input"] + completion_len * info["output"]) / 1000

        elif "gpt-3.5" in model_lower:
            info = OPENAI_MODEL_INFO["gpt-3.5"].get(model_name, OPENAI_MODEL_INFO["gpt-3.5"]["gpt-3.5-turbo-0125"])
            price = (prompt_len * info["input"] + completion_len * info["output"]) / 1000

        elif "dall-e" in model_lower:
            # DALL-E 通常按张收费，这里逻辑可能需要根据具体返回调整，暂时置 0 或保留你原有的逻辑
            pass

    # === 策略 B: DeepSeek 模型 (API) ===
    elif "deepseek" in model_lower:
        # DeepSeek 通常兼容 Llama tokenizer 或有自己的，这里尝试用 AutoTokenizer
        # 注意：如果是在线 API，可能不需要本地 load tokenizer，直接用估算或 cl100k
        # 这里假设你需要本地精确计算：
        prompt_len = cal_token_hf(model_name, prompt)
        completion_len = cal_token_hf(model_name, response)

        # DeepSeek V2 API 价格 (示例: 输入 1元/百万, 输出 2元/百万，请根据实际更新)
        # 你代码里写的是 2/1M 和 8/1M
        price = (prompt_len * 2 + completion_len * 8) / 1_000_000

    # === 策略 C: Llama / 本地模型 ===
    else:
        # 对于本地模型，通常默认免费，但需要统计 token 数
        prompt_len = cal_token_hf(model_name, prompt)
        completion_len = cal_token_hf(model_name, response)
        price = 0.0

    # === 更新全局统计 ===
    try:
        Cost.instance().value += price
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len
    except Exception:
        pass  # 忽略统计错误

    return price, prompt_len, completion_len


# --- 4. 价格配置表 (保持原样) ---
OPENAI_MODEL_INFO = {
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {
            "context window": 128000,
            "training": "Jan 2024",
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-preview": {
            "context window": 128000,
            "training": "Apr 2023",
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-vision-preview": {
            "context window": 128000,
            "training": "Apr 2023",
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4": {
            "context window": 8192,
            "training": "Sep 2021",
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4o": {
            "context window": 128000,
            "training": "Jan 2024",
            "input": 0.005,
            "output": 0.015
        },
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125": {
            "context window": 16385,
            "training": "Jan 2024",
            "input": 0.0010,
            "output": 0.0020
        },
        "gpt-3.5-turbo-1106": {
            "context window": 16385,
            "training": "Sep 2021",
            "input": 0.0010,
            "output": 0.0020
        },
        "gpt-3.5-turbo-instruct": {
            "context window": 4096,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo": {
            "context window": 4096,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020
        },
    },
    "dall-e": {
        "current_recommended": "dall-e-3",
        "dall-e-3": {
            "standard": {"1024×1024": 0.040},
            "hd": {"1024×1024": 0.080}
        }
    }
}
