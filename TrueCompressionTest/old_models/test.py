import torch
from CompressUtils import CompressUnion
from CompressUtils.TrueCompressFunction import tokenwise_quantization_compress_with_error,tokenwise_dequantization
bsz = 1
numhead = 1
seqlen = 10
dim = 10
rand_input = torch.rand(bsz, numhead, seqlen, dim).to(torch.float32).to("cuda:0")
quantized_input, error, min, step, shape = tokenwise_quantization_compress_with_error(rand_input,4)
print(error)
print(error.min(),error.max())
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
