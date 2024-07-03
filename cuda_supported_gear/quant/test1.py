from new_pack import triton_quantize_and_pack_along_last_dim_witherror, triton_quantize_and_pack_along_last_dim
import torch 
import time
tensor = torch.rand([1,32,100,4096]).cuda()
group = 128
start = time.time()
# result,scale,mn,error = triton_quantize_and_pack_along_last_dim_witherror(tensor,group,4) # 0.073s
result,scale,mn, error = triton_quantize_and_pack_along_last_dim_witherror(tensor,group,4) # 0.067s
end = time.time() - start
error = error.reshape(1,32,100,4096)
print(error.shape)
