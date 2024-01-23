from compress_function import (
    ptcompress,
    ptdecompress,
    PiCache,
    fake_uniformquantization,
    fake_poweriteration,
    fake_svd_lowrank,
    fake_groupwise_asymmetric_quantization,
    fake_sort_quantization,
)
import torch
import time

randinput = torch.rand(1, 12, 1023, 64).to(0)  # 2 * 2
p_buffer = [torch.rand(768, 3)]
q_buffer = [torch.rand(12, 3)]
randinput = (randinput - 0.5) * 12
# loop = 3
# output = ptcompress(randinput, p_buffer, q_buffer, loop)
# print(output)
# test fake group quantization
output = fake_groupwise_asymmetric_quantization(randinput, 4)


compress_cache = PiCache(randinput.shape, 10, 10, 0, 50)
# compress_cache.fill_cache_buffer(randinput)
start = time.time()
compress_cache.compress_cache(randinput)
# compress_cache.compress(randinput)
end = time.time() - start
print("compress time:", end)
start = time.time()
decompressed = compress_cache.decompress()
end = time.time() - start
print("decompress time:", end)
print(torch.any(torch.isnan(decompressed)))
output = fake_svd_lowrank(randinput, 16)
start = time.time()
print(randinput[0, 0, 0, :])
output = fake_sort_quantization(randinput, 4, 8)
print("compress decompress time for sort quantization:", end)
print(output[0, 0, 0, :])
# fake_output = fake_uniformquantization(randinput,16)
# print(randinput[0,0,0,:])
# print(torch.any(torch.isnan(fake_output)))
