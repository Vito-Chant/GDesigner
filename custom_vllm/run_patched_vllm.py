import sys
import runpy

# --- 步骤 1: 尝试修复 LMCache 的 Bug ---
try:
    # 导入 LMCache 的适配器 (根据报错堆栈定位到的模块)
    from lmcache.integration.vllm import vllm_v1_adapter

    print("[Patch] 正在修复 LMCache 对 Embedding 请求的兼容性...")

    # 保存原函数引用
    _original_extract = vllm_v1_adapter.extract_request_configs


    # 定义一个新的“安全”函数
    def _safe_extract_request_configs(sampling_params):
        # 如果 sampling_params 为空 (Embedding 模式)，直接返回空配置
        # 这会告诉 LMCache 使用默认策略（通常是存储所有 KV）
        if sampling_params is None:
            return {}
        return _original_extract(sampling_params)


    # 替换掉库里的坏函数
    vllm_v1_adapter.extract_request_configs = _safe_extract_request_configs
    print("[Patch] 修复成功！现在可以支持 Embedding 任务了。")

except ImportError:
    print("[Patch] 警告: 未找到 lmcache 模块，如果是纯 vLLM 运行请忽略。")
except Exception as e:
    print(f"[Patch] 修复失败: {e}")

# --- 步骤 2: 启动 vLLM API Server ---
# 这相当于运行 python -m vllm.entrypoints.openai.api_server
if __name__ == "__main__":
    # 伪装进程名称，让 vLLM 以为自己是正常启动的
    sys.argv[0] = "python -m vllm.entrypoints.openai.api_server"
    print(f"[Launcher] 正在启动 vLLM: {' '.join(sys.argv)}")

    # 执行 vLLM 的入口点
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")