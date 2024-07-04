import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def fake_groupwise_token_asymmetric_quantization( ####
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ).float()
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2**quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_channel_asymmetric_quantization_new(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size

    fixed_input = input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
    
    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_poweriteration_group(input: torch.Tensor, loop, rank, device, p_base, q_base):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    dtype = input.dtype
    batch, dim1, dim2, dim3 = input.shape


    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(batch,dim1,dim3, rank).to(input.device)]
        q_base = [torch.rand(batch,dim1,dim2, rank).to(input.device)]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 2, 3) @ q_base[0]
    input = q_base[0] @ torch.transpose(p_base[0], 2, 3)
    input = input.view(batch, dim1, dim2, dim3)

    input = input.type(dtype)

    return input

def fake_groupwise_channel_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size
    fixed_length = int(group_num * group_size)
    fixed_input = input[:,:fixed_length,:]
    residual_input = input[:,fixed_length:,:]
    fixed_input = fixed_input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    scale = (mx - mn) / cluster_num
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head * sep_dim)
    concat_input = torch.cat((dequantized_input,residual_input),dim=1)
    dequantized_input = concat_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_token_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / cluster_num
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input


def gearslkivi_channelQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    quantized_output = gearlkivi_channelQ(input, quantize_bit, group_size,rank,loop)
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    input.scatter_(-1, smallest_indices, smallest_value)
    input.scatter_(-1, largest_indices, largest_value)
    

    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    input = input.half()
    quantized_output = quantized_output.half()

    
    return quantized_output



def gearslkivi_tokenQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1): ####
    input = input.float()
    cloned_input = input.clone()
    output = gears_tokenQ(input, quantize_bit, group_size,sparsity)

    error = cloned_input - output
    error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    return output + error_lr

def gearslkivi_channelQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1): ####
    input = input.float()
    cloned_input = input.clone()
    output = gears_channelQ(input, quantize_bit, group_size,sparsity)

    error = cloned_input - output
    error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    return output + error_lr

def gearslkivi_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()

    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    input = input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3) 
    quantized_output = gearlkivi_tokenQ(input, quantize_bit, group_size,rank,loop)
    # Restore the original values at the smallest and largest k indices
    quantized_output = quantized_output = (
        quantized_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    quantized_output.scatter_(-1, smallest_indices, smallest_value)
    quantized_output.scatter_(-1, largest_indices, largest_value)
    

    quantized_output = quantized_output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    quantized_output = quantized_output.half()
    return quantized_output

     
def gears_channelQ(input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = fake_groupwise_channel_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
    

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = output.half()
    return output
def gears_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = fake_groupwise_token_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
    

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()
    return output
def tokenwise_gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = cloned_input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,

                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr

def gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr
def gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr
def tokenwise_gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = cloned_input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
 
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr


def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    layer_idx,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    prefill=None,
):
    batch, num_head, seq_len, sep_dim = previous_key.shape
    if compress_config.token_preserving[layer_idx] == True:
        starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)
    
    if compress_config.compress_method[layer_idx] == "KCVT":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            seq_len,
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                int(num_head * sep_dim),
            )

    if compress_config.compress_method[layer_idx] == "KIVI":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
        previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
            previous_value[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )

    if compress_config.compress_method[layer_idx] == "GEAR":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEAR-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEARL":

        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rank_used,
            compress_config.loop[layer_idx],

            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx],
 
        )
    if compress_config.compress_method[layer_idx] == "GEARL-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            rank_used,
            compress_config.loop[layer_idx],
            
            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            rankv_used,
            compress_config.loop[layer_idx],
            
        )

    return previous_key, previous_value




