# from h2o_llama_self_written import H2OCache
# import torch
# cache = H2OCache(10,2)
# hidden_states = torch.rand(2,32,20,20)
# key = torch.rand(2,32,20,1024)
# value = torch.rand(2,32,20,1024)
# returned_key,returned_value = cache.update(hidden_states,key,value)
# print("return",returned_key.shape, returned_value.shape)
# hidden_states = torch.rand(2,32,1,13)
# key = torch.rand(2,32,13,1024)
# value = torch.rand(2,32,13,1024)
# returned_key,returned_value = cache.update(hidden_states,key,value)
# print(returned_key.shape, returned_value.shape)
from compress_function import fake_group_channel_quant_with_lrap, fake_group_token_quant_with_lrap


