import torch
from CompressUtils import CompressUnion
from CompressUtils.TrueCompressFunction import true_gear_tokenwiseQ_compress,true_gear_tokenwiseQ_decompress,true_gear_compress_batchwise,true_gear_tokenwiseQ_compress_nopq,true_gear_tokenwiseQ_decompress_nopq
bsz = 10
numhead = 32
seqlen = 1000
dim = 128
rank = 0.02
loop = 3
rand_input = torch.rand(bsz, numhead, seqlen, dim).to(torch.float16).to("cuda:0")
import time
start = time.time()

quantized_input, shape, min, step, p_base, q_base,shape_p,shape_q,min_p,min_q,scale_p,scale_q = true_gear_tokenwiseQ_compress(rand_input,4,rank,loop)
torch.cuda.synchronize()
end = time.time()
print("tokenwiseQ-gearQ compress time:",end-start)
rand_input = torch.rand(bsz, numhead, seqlen, dim).to(torch.float16).to("cuda:0")
start = time.time()
quantized_input = true_gear_compress_batchwise(rand_input,4,0.02,rank,loop)
torch.cuda.synchronize()
end = time.time()
print("gear compress time:",end-start)
rand_input = torch.rand(bsz, numhead, seqlen, dim).to(torch.float16).to("cuda:0")
start = time.time()
quantized_input, shape, min, step, p_base, q_base = true_gear_tokenwiseQ_compress_nopq(rand_input,4,rank,loop)
torch.cuda.synchronize()
end = time.time()
print("tokenwiseQ-gear compress time:",end-start)
start = time.time()
dequantized_input = true_gear_tokenwiseQ_decompress_nopq(quantized_input,4, shape, min, step, p_base, q_base,torch.float16)
torch.cuda.synchronize()
end = time.time()
print("tokenwiseQ-gear decompress time:",end-start)
# if __name__ == '__main__':
#     bsz = 2
#     numhead = 32
#     seqlen = 200
#     dim = 4096
#     compress_kwargs = {}
#     compress_kwargs["quantize_bit"] = 4
#     compress_kwargs["compress_mode"] = "outlier_batch"
#     compress_kwargs["left"] = 0.02
#     compress_kwargs["loop"] = 3
#     compress_kwargs["rank"] = 0.02
#     compress_kwargs["stream"] = True
#     compress_kwargs["streaming_gap"] = 10
#     kv_cache = torch.rand(bsz, numhead, seqlen, dim).to(torch.float16).to("cuda:0")
#     print(kv_cache)
#     union = CompressUnion(compress_kwargs=compress_kwargs)
#     union.compress_function(kv_cache)
#     kv_cache = union.decompress()
#     print(kv_cache)
