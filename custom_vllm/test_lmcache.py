import time
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# 配置
EMBED_URL = "http://localhost:8000/v1"  # 实例 A
CHAT_URL = "http://localhost:8001/v1"  # 实例 B
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"  # 确保与启动参数一致

# 构造一个足够长的前缀，以便体现出传输的优势 (约 2000 token)
long_prefix = "人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。 "
question = "请简要总结上述内容。"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def run_chat(client, prompt, label):
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,  # 只生成少量 token，专注于测试 Prefill 速度
            temperature=0,
            stream=True  # 使用流式传输以测量首字延迟
        )

        first_token = True
        ttft = 0
        for chunk in response:
            if first_token:
                ttft = time.time() - start_time
                first_token = False
                print(f"[{label}] 首字延迟 (TTFT): {ttft * 1000:.2f} ms")

        return ttft
    except Exception as e:
        print(f"[{label}] 请求失败: {e}")
        return float('inf')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def run_embedding(client, text):
    print("[Embedding] 正在向实例 A 发送 Embedding 请求以预热 Cache...")
    start = time.time()
    try:
        embedding=client.embeddings.create(
            model=MODEL_NAME,
            input=text
        )
        print(len(embedding.data[0].embedding))
        print(f"[Embedding] 完成，耗时: {(time.time() - start) * 1000:.2f} ms")
    except Exception as e:
        print(f"[Embedding] 失败: {e}")


# 初始化客户端
client_emb = OpenAI(api_key="EMPTY", base_url=EMBED_URL)
client_chat = OpenAI(api_key="EMPTY", base_url=CHAT_URL)

print("=" * 50)
print("开始 LMCache 测试")
print("=" * 50)

# 1. 构造两个略有不同的 Prompt 以避免测试干扰
# Prompt A 用于测试无缓存情况
prompt_cold = "这是独特的冷启动前缀。" + long_prefix + question
# Prompt B 用于测试 LMCache 流程
prompt_warm = "这是用于缓存测试的前缀。" + long_prefix + question

# --- 测试 1: 冷启动 (Cold Start) ---
# 直接发给 Chat 实例，没有任何预热
print("\n--- 测试 1: 冷启动 (无缓存) ---")
ttft_cold = run_chat(client_chat, prompt_cold, "Cold-Chat")

# --- 测试 2: 热启动 (Warm Start with LMCache) ---
print("\n--- 测试 2: LMCache 流程 ---")
# 步骤 A: 发送给 Embedding 实例 (生成并上传 KV)
run_embedding(client_emb, prompt_warm)

# 等待一小会儿确保数据写入后端 (如果是 Redis 可能会有微小延迟)
time.sleep(0.5)

# 步骤 B: 发送给 Chat 实例 (应该复用 KV)
ttft_warm = run_chat(client_chat, prompt_warm, "Warm-Chat")

# --- 结果分析 ---
print("\n" + "=" * 50)
print("测试结果分析:")
if ttft_warm < ttft_cold * 0.8:  # 简单的阈值判断
    print(f"✅ LMCache 工作正常！")
    print(f"   加速比: {ttft_cold / ttft_warm:.2f}x")
    print(f"   (Cold: {ttft_cold * 1000:.0f}ms vs Warm: {ttft_warm * 1000:.0f}ms)")
else:
    print(f"⚠️ 未检测到显著加速，请检查日志。")
    print(f"   (Cold: {ttft_cold * 1000:.0f}ms vs Warm: {ttft_warm * 1000:.0f}ms)")
    print("   可能原因：Prompt 太短、网络延迟过高、或者 APC/KV-Transfer 未正确配置。")