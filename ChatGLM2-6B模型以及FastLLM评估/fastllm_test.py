import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import time

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b", trust_remote_code=True, device='cuda')
model = model.eval()
from fastllm_pytools import llm
model = llm.from_hf(model, tokenizer, dtype = "float16")

# 输入示例
queries = [
    "标题:“涉卡”罪名汇总 问题: “涉卡”罪名有哪些",
    "标题:关于应用分身提取。 问题: 应用分发怎么提取",
    "标题:优酷 问题: 优酷的地址是哪里？",
    "标题:网宿科技调证 问题: 网宿科技如何调证",
    "标题:对公账户如何查询？ 问题: 对公账户怎么查询",
]

# For循环时间测试
start_time = time.time()
for query in queries:
    response = model.response(query)
end_time = time.time()

elapsed_time_ms = (end_time - start_time) * 1000
print("For循环时间测试:")
print("总体执行时间:", elapsed_time_ms, "毫秒")

# 多并发测试
def run_inference(query):
    response = model.response(query)
    return response

executor = ThreadPoolExecutor()

start_time = time.time()
future_results = [executor.submit(run_inference, query) for query in queries]
results = [future.result() for future in future_results]
end_time = time.time()

elapsed_time_ms = (end_time - start_time) * 1000
print("多并发测试:")
print("总体执行时间:", elapsed_time_ms, "毫秒")

executor.shutdown()

# 显存资源占用评估
allocated_memory = torch.cuda.memory_allocated()
max_allocated_memory = torch.cuda.max_memory_allocated()

print("显存资源占用:")
print("当前显存占用:", allocated_memory / 1024 ** 2, "MB")
print("峰值显存占用:", max_allocated_memory / 1024 ** 2, "MB")