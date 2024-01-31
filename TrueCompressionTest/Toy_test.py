from models import ToyLlama
import torch
# model = ToyLlama(4096,4096,8192,32).to("cuda")
# input = torch.randn(4,1000,4096).to("cuda")
model = ToyLlama(4096,4096,8192,32)
input = [torch.randn(2,1000,4096),torch.randn(2,1000,4096)]
# output, kv_cache = model(input,None)
# print(kv_cache[0][0].shape)
# output1, kv_cache = model(input[:,-1,:].reshape(1,1,4096),kv_cache)
# print(kv_cache[0][0].shape)
with torch.no_grad():
#     output,kv_cache = model.zig_zag_forward(input,None,2,"cuda")
#     print(len(kv_cache))
#     print(len(kv_cache[0]))
#     print(len(kv_cache[0][0]))
#     print(kv_cache[0][0][0].shape)
#     for i in range(len(output)):
#         bsz,seqlen,model_dim = output[i].shape
#         output[i] = output[i][:,-1,:].reshape(bsz,1,model_dim)
#     output,kv_cache = model.zig_zag_forward(output,kv_cache,2,"cuda")
# print(len(kv_cache))
# print(len(kv_cache[0]))
# print(len(kv_cache[0][0]))
# print(kv_cache[0][0][0].shape)
# print(output[0].shape)
    output, kv_cache = model.zig_zag_generate(input,2,10,"cuda")
peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # 转换为MB单位
# while True:
#     pass
print(f"Peak memory usage on GPU: {peak_memory} MB")