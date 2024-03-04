import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class H2OCache:
    def __init__(self, hh_size):
        self.hh_size = hh_size
        self.h2o_score = torch.zeros(hh_size, hh_size)

    def selection(self, attn_score, key, value, query):
        batch, num_head, seq_len, sep_dim = key.shape
        batch, num_head, inp_len, cache_len = attn_score.shape

        if inp_len == cache_len:
            # prefill get the attn score
            self.h2o_score = attn_score.clone()
        else:
            # update attn_score
            new_score_cache = torch.zeros(batch, num_head, cache_len, cache_len)
            new_score_cache[:, :, :-1, :-1] = self.h2o_score.clone()
            new_score_cache[:, :, :, :-1] = attn_score.clone()
            self.h2o_score = new_score_cache
        if seq_len <= self.hh_size:
            return key, value
        else:
            _, _, cache_len, cache_len = self.h2o_score.shape
            # TODO is this about the last dim or should set dim=-2
            summing_score = torch.sum(self.h2o_score, dim=-1)
            # batch,num_head,cache_len
            summing_score = summing_score.reshape(-1, cache_len)
            choosing_idx = torch.topk(summing_score, self.hh_size, dim=-1).indices
            # pruning the h2o_score
            self.h2o_score = self.h2o_score.view(-1, cache_len, cache_len)
            choosing_idx_flat = choosing_idx.view(-1, self.hh_size).clone()

            choosing_idx_flat1 = choosing_idx_flat.unsqueeze(-1).expand(
                -1, -1, cache_len
            )

            choosing_idx_flat2 = choosing_idx_flat.unsqueeze(-2).expand(
                -1, self.hh_size, -1
            )
            selected_h2o_score1 = torch.gather(self.h2o_score, 1, choosing_idx_flat1)

            selected_h2o_score2 = torch.gather(
                selected_h2o_score1, 2, choosing_idx_flat2
            )
            self.h2o_score = selected_h2o_score2.view(
                batch, num_head, self.hh_size, self.hh_size
            )
            # prun the kv cache
            key = key.reshape(-1, seq_len, sep_dim)
            value = value.reshape(-1, seq_len, sep_dim)
            selected_key = torch.gather(
                key, 1, choosing_idx_flat.unsqueeze(-1).expand(-1, -1, sep_dim)
            )
            selected_value = torch.gather(
                value, 1, choosing_idx_flat.unsqueeze(-1).expand(-1, -1, sep_dim)
            )
            print(
                "selected_key:",
                selected_key.shape,
                "selected_value:",
                selected_value.shape,
                "self.h2o_score:",
                self.h2o_score.shape,
            )
            selected_key = selected_key.reshape(batch, num_head, self.hh_size, sep_dim)
            selected_value = selected_value.reshape(
                batch, num_head, self.hh_size, sep_dim
            )
            batch, num_head, q_seq_len, sep_dim = query.shape

            if q_seq_len > 1:
                # prefill part
                print("prefill")
                query = query.reshape(-1, q_seq_len, sep_dim)
                selected_query = torch.gather(
                    query, 1, choosing_idx_flat.unsqueeze(-1).expand(-1, -1, sep_dim)
                )
                query = query.reshape(batch, num_head, q_seq_len, sep_dim)
            else:
                selected_query = query
            print("query shape:", query.shape)
            return selected_key, selected_value, selected_query

        # else:
        #     # select key value
        #     if inp_len == cache_len:
        #         # prefill get the attn score
        #         self.h2o_score = attn_score.clone()
        #     else:
        # return key, value


def fake_groupwise_asymmetric_quantization(input: torch.Tensor, quantize_bit):
    # what flexgen uses
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # 64 groups, group alone sep_dim
    input = input.float()
    input = input.permute(3, 0, 1, 2)
    input = input.reshape(sep_dim, batch * num_head * seq_len)
    min, max = input.min(1).values, input.max(1).values

    quantized_input = input.clone().detach()
    for i in range(len(min)):
        quantized_input[i] = torch.round(
            (input[i] - min[i]) / (max[i] - min[i]) * (pow(2, quantize_bit) - 1)
        )
    for i in range(len(min)):
        input[i] = (
            quantized_input[i] * (max[i] - min[i]) / (pow(2, quantize_bit) - 1) + min[i]
        )
    input = input.reshape(sep_dim, batch, num_head, seq_len)
    input = input.permute(1, 2, 3, 0)
    input = input.type(dtype)
    return input

def fake_groupwise_token_asymmetric_quantization(input: torch.Tensor, quantize_bit, group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ) 
    num_groups = (sep_dim*num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2 ** quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    return dequantized_input


def fake_groupwise_channel_asymmetric_quantization(input: torch.Tensor, quantize_bit, group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ) 

    mx, mn = input.max(dim=-2)[0], input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    scale = (mx - mn) / (2 ** quantize_bit - 1)
    input = (input - mn) / scale
    input = F.relu(input)
    rounded_input = input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    return dequantized_input

def fake_smoothquatization(input, quantize_bit):
    pass


def fake_svd_lowrank2d(input: torch.Tensor, q):
    batch, seq_len, model_dim = input.shape

    input = input.reshape(seq_len, model_dim)
    U, S, V = torch.svd_lowrank(input, q=q)
    V = V.transpose(-1, -2)
    S = torch.diag_embed(S)
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, model_dim)
    return output


def fake_svd_lowrank(input: torch.Tensor, q):
    batch, num_head, seq_len, sep_dim = input.shape

    input = input.permute(0, 2, 1, 3).reshape(batch * seq_len, num_head * sep_dim)
    U, S, V = torch.svd_lowrank(input, q=q)
    V = V.transpose(-1, -2)
    S = torch.diag_embed(S)
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output


def fake_uniformquantization(input: torch.Tensor, quantize_bit):
    shape = input.shape
    dtype = input.dtype
    input = input.reshape(-1)
    input = input.float()  # convert to 32bits to avoid max - min = inf
    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, quantize_bit) - 1)
    # print("before min max:",min,max,step)
    quantized_input = torch.round((input - min) / step)
    # print("after min max:",quantized_input.min(),quantized_input.max())
    # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
    dequantized_input = (quantized_input * step) + min
    returning_input = dequantized_input.reshape(shape)
    returning_input = returning_input.type(dtype)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return returning_input


# def fake_uniformquantization_for_clusterquant(input: torch.Tensor, cluster_num):
#     shape = input.shape
#     input = input.reshape(-1)
#     input = input.float()  # convert to 32bits to avoid max - min = inf
#     min, max = input.min(), input.max()
#     # print("before min max:",min,max)
#     step = (max - min) / (cluster_num - 1)
#     quantized_input = torch.floor((input - min) / step)
#     # print("after min max:",quantized_input.min(),quantized_input.max())
#     # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
#     dequantized_input = (quantized_input * step) + min
#     returning_input = dequantized_input.reshape(shape)
#     returning_input = returning_input.half()
#     # print("isnan:",torch.any(torch.isnan(returning_input)))
#     # while(True):
#     #     pass
#     return returning_input


def fake_top_k_pruning(input_tensor, topk):
    # Flatten the input tensor
    flat_input = input_tensor.reshape(-1)

    # Calculate the absolute values of the flattened input tensor
    abs_values = torch.abs(flat_input)
    k = int(len(abs_values) * topk)
    # Find the top-k largest absolute values and their indices
    _, topk_indices = torch.topk(abs_values, k)

    # Create a mask to keep the top-k values
    mask = torch.zeros_like(flat_input)
    mask[topk_indices] = 1

    # Apply the mask to prune the tensor
    pruned_tensor = flat_input * mask

    # Reshape the pruned tensor to match the original shape
    pruned_tensor = pruned_tensor.view(input_tensor.size())

    return pruned_tensor


def fake_sort_quantization(input_tensor, group_num, quantize_bit):
    shape = input_tensor.shape
    input_tensor = input_tensor.reshape(-1)
    input_tensor = input_tensor.float()  # convert to 32bits to avoid max - min = inf
    sortedtensor, indices = torch.sort(input_tensor)
    group_size = len(sortedtensor) // group_num
    left_values = len(sortedtensor) % group_num

    group_bits = quantize_bit // group_num
    # print(indices)
    for i in range(group_num):
        if i == 0:
            sortedtensor[0 : group_size + left_values] = fake_uniformquantization(
                sortedtensor[0 : group_size + left_values], group_bits
            )
            # print(sortedtensor[0 : group_size + left_values].min(),sortedtensor[0 : group_size + left_values].max())
        else:
            sortedtensor[
                i * group_size + left_values : (i + 1) * group_size + left_values
            ] = fake_uniformquantization(
                sortedtensor[
                    i * group_size + left_values : (i + 1) * group_size + left_values
                ],
                group_bits,
            )
            # print(sortedtensor[i * group_size + left_values : (i + 1) * group_size + left_values].min(),sortedtensor[i * group_size + left_values : (i + 1) * group_size + left_values].max())

    input_tensor[indices] = sortedtensor
    input_tensor = input_tensor.reshape(shape)
    return input_tensor


def fake_dense_sparse_uniformquantization(input: torch.Tensor, quantize_bit, left):
    shape = input.shape
    dtype = input.dtype
    input = input.reshape(-1)
    input = input.float()  # convert to 32bits to avoid max - min = inf
    sortedtensor, indices = torch.sort(input)
    left_num = int(len(sortedtensor) * left / 2)
    sortedtensor[left_num:-left_num] = fake_uniformquantization(
        sortedtensor[left_num:-left_num], quantize_bit
    )
    input[indices] = sortedtensor
    input = input.reshape(shape)
    input = input.type(dtype)
    return input


def fake_dense_sparse_sortquantization(
    input: torch.Tensor, quantize_bit, group_num, left
):
    shape = input.shape
    input = input.reshape(-1)
    input = input.float()  # convert to 32bits to avoid max - min = inf
    sortedtensor, indices = torch.sort(input)
    # this means we will left 10% of the data and not compress them
    left_num = int(len(sortedtensor) * left / 2)

    quantization_list = sortedtensor[left_num:-left_num]
    group_size = len(quantization_list) // group_num
    # these are values that going to be compressed.
    left_values = len(quantization_list) % group_num
    group_bits = quantize_bit // group_num
    print("left values", left_values)
    for i in range(group_num):
        if i == 0:
            quantization_list[0 : group_size + left_values] = fake_uniformquantization(
                quantization_list[0 : group_size + left_values], group_bits
            )
            print(
                quantization_list[0 : group_size + left_values].min(),
                sortedtensor[0 : group_size + left_values].max(),
            )
        else:
            quantization_list[
                i * group_size + left_values : (i + 1) * group_size + left_values
            ] = fake_uniformquantization(
                quantization_list[
                    i * group_size + left_values : (i + 1) * group_size + left_values
                ],
                group_bits,
            )
            print(
                quantization_list[
                    i * group_size + left_values : (i + 1) * group_size + left_values
                ].min(),
                sortedtensor[
                    i * group_size + left_values : (i + 1) * group_size + left_values
                ].max(),
            )
    sortedtensor[left_num:-left_num] = quantization_list

    input[indices] = sortedtensor

    input = input.reshape(shape)
    return input


def fake_poweriteration(input: torch.Tensor, loop, rank, device, p_base, q_base):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    dtype = input.dtype
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    input = input.view(seq_len * batch, sep_dim * num_head)
    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(sep_dim * num_head, rank).to(input.device)]
        q_base = [torch.rand(batch * seq_len, rank).to(input.device)]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 0, 1) @ q_base[0]
    input = q_base[0] @ torch.transpose(p_base[0], 0, 1)
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3)
    input = input.type(dtype)
    p_base[0] = p_base[0].half()
    q_base[0] = q_base[0].half()
    return input


def fake_poweriteration_with_outlierquant(
    input: torch.Tensor, loop, rank, device, p_base, q_base, quantize_bit, left
):
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.float()
    # convert to 32bits for qr decomposition
    input = input.view(seq_len * batch, sep_dim * num_head)
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 0, 1) @ q_base[0]
    # outlier quantization
    q_base[0] = fake_dense_sparse_uniformquantization(q_base[0], quantize_bit, left)
    p_base[0] = fake_dense_sparse_uniformquantization(p_base[0], quantize_bit, left)
    input = q_base[0] @ torch.transpose(p_base[0], 0, 1)
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3)
    input = input.half()
    return input


def fake_stage_poweriteration(
    input: torch.Tensor, loop, rank, stage, device, p_base, q_base
):
    pass
    # Lets say input is of size x,n,m x= stage, p would be [m,rank] q would be [n,rank], but p is separate to [m,rank/x] ad q is [n1,rank/x], [n2,rank/x], [n3,rank/x]
    batch, num_head, seq_len, sep_dim = input.shape
    input = input.float()
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    # input = input.view(stage,int(batch/stage), seq_len, sep_dim * num_head)
    input = input.view(seq_len * batch, sep_dim * num_head)
    stage_dim = seq_len * batch // stage
    stage_rank = rank // stage
    for i in range(stage):
        if i != stage - 1:
            p_slice = p_base[0][:, i * stage_rank : (i + 1) * stage_rank]
            q_slice = q_base[0][
                i * stage_dim : (i + 1) * stage_dim,
                i * stage_rank : (i + 1) * stage_rank,
            ]
            input_slice = input[i * stage_dim : (i + 1) * stage_dim, :]
        else:
            p_slice = p_base[0][:, i * stage_rank :]
            q_slice = q_base[0][i * stage_dim :, i * stage_rank :]
            input_slice = input[i * stage_dim :, :]
        # print(p_slice.shape,q_slice.shape,input_slice.shape)
        for j in range(loop):
            if j == loop - 1:
                p_slice = torch.linalg.qr(p_slice).Q
            q_slice = input_slice @ p_slice
            if j == loop - 1:
                q_slice = torch.linalg.qr(q_slice).Q
            p_slice = torch.transpose(input_slice, 0, 1) @ q_slice
        input_slice = q_slice @ torch.transpose(p_slice, 0, 1)
        if i != stage - 1:
            input[i * stage_dim : (i + 1) * stage_dim, :] = input_slice
        else:
            input[i * stage_dim :, :] = input_slice
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3)
    input = input.half()
    return input


def ptcompress(cache: torch.Tensor, p_buffer, q_buffer, loop):
    # cache size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_buffer of size [model_dim, rank], q_buffer of size [batch * seq_len,rank]
    batch, num_head, seq_len, sep_dim = cache.shape
    cache = (
        cache.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    p_buffer[0] = p_buffer[0]
    q_buffer[0] = q_buffer[0]
    cache = cache.view(seq_len * batch, sep_dim * num_head)
    for i in range(loop):
        if i == loop - 1:
            p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
        q_buffer[0] = cache @ p_buffer[0]
        if i == loop - 1:
            q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
        p_buffer[0] = torch.transpose(cache, 0, 1) @ q_buffer[0]
    p_buffer[0] = p_buffer[0]
    q_buffer[0] = q_buffer[0]
    cache = cache.view(batch, num_head, seq_len, sep_dim)
    return p_buffer, q_buffer, batch, num_head, seq_len, sep_dim


def ptdecompress(p_buffer, q_buffer, batch, num_head, seq_len, sep_dim):
    # p_buffer of size [model_dim, rank], q_buffer of size [batch * seq_len,rank]
    matrix = q_buffer[0] @ torch.transpose(p_buffer[0], 0, 1)
    matrix = matrix.view(batch, num_head, seq_len, sep_dim)
    return matrix


def fake_picache_compress(
    input: torch.Tensor, loop, rank, device, p_base, q_base, quantize_bit, left
):
    input = input.float()
    p_base[0] = p_base[0].float()
    q_base[0] = q_base[0].float()
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    input = input.view(seq_len * batch, sep_dim * num_head)
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 0, 1) @ q_base[0]
    # outlier quantization
    q_base[0] = fake_dense_sparse_uniformquantization(q_base[0], quantize_bit, left)
    p_base[0] = fake_dense_sparse_uniformquantization(p_base[0], quantize_bit, left)
    input = q_base[0] @ torch.transpose(p_base[0], 0, 1)
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3)
    input = input.half()
    p_base[0] = p_base[0].half()
    q_base[0] = q_base[0].half()
    return input


def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    layer_idx,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    attn_weights=None,
):
    batch, num_head, seq_len, sep_dim = previous_key.shape
    if compress_config.token_preserving[layer_idx] == True:
        starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)
    if compress_config.compress_method[layer_idx] == "Picache":
        # TODO
        batch, num_head, seq_len, sep_dim = previous_key.shape

        if seq_len > compress_config.rank[layer_idx]:
            previous_key = fake_picache_compress(
                previous_key,
                compress_config.loop[layer_idx],
                compress_config.rank[layer_idx],
                compress_config.device_num[layer_idx],
                pbase1,
                qbase1,
                compress_config.quantize_bit[layer_idx],
                compress_config.left[layer_idx],
            )
        else:
            previous_key = fake_dense_sparse_uniformquantization(
                previous_key,
                compress_config.quantize_bit[layer_idx],
                compress_config.left[layer_idx],
            )
        if seq_len > compress_config.rankv[layer_idx]:
            previous_value = fake_picache_compress(
                previous_value,
                compress_config.loop[layer_idx],
                compress_config.rankv[layer_idx],
                compress_config.device_num[layer_idx],
                pbase2,
                qbase2,
                compress_config.quantize_bit[layer_idx],
                compress_config.left[layer_idx],
            )
        else:
            previous_value = fake_dense_sparse_uniformquantization(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                compress_config.left[layer_idx],
            )

    if compress_config.compress_method[layer_idx] == "groupquantization":
        previous_key[
            :, :, starting_idx:-locality_idx, :
        ] = fake_groupwise_asymmetric_quantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
        )
        if previous_value is not None:
            previous_value[
                :, :, starting_idx:-locality_idx, :
            ] = fake_groupwise_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
            )
            
    #SK: TODO take a look at this: idea merged from KIVI, token grouping does reasonably well on GSM8k-CoT at 4-bit, around ~13%      
    if compress_config.compress_method[layer_idx] == "groupquantization_token":
        previous_key[
            :, :, starting_idx:-locality_idx, :
        ] = fake_groupwise_token_asymmetric_quantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            128
        )
        if previous_value is not None:
            previous_value[
                :, :, starting_idx:-locality_idx, :
            ] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                128
            )
            
    #SK: TODO take a look at this: idea merged from KIVI, this channel grouping does better tha what was implemented earlier, Please check the code!
    if compress_config.compress_method[layer_idx] == "groupquantization_channel":
        previous_key[
            :, :, starting_idx:-locality_idx, :
        ] = fake_groupwise_channel_asymmetric_quantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            128
        )
        if previous_value is not None:
            previous_value[
                :, :, starting_idx:-locality_idx, :
            ] = fake_groupwise_channel_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                128
            )
            
    #SK: TODO take a look at this: idea merged from KIVI
    if compress_config.compress_method[layer_idx] == "groupquantization_kc_vt":
        previous_key[
            :, :, starting_idx:-locality_idx, :
        ] = fake_groupwise_channel_asymmetric_quantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            128
        )
        if previous_value is not None:
            previous_value[
                :, :, starting_idx:-locality_idx, :
            ] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                128
            )
    
    if compress_config.compress_method[layer_idx] == "uniformquantization":
        # print("begin uniquant",starting_idx,locality_idx,compress_config.quantize_bit[layer_idx])
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_uniformquantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
        )
        if previous_value is not None:
            previous_value[
                :, :, starting_idx:-locality_idx, :
            ] = fake_uniformquantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "poweriteration":
        previous_key = fake_poweriteration(
            previous_key,
            compress_config.loop[layer_idx],
            compress_config.rank[layer_idx],
            compress_config.device_num[layer_idx],
            pbase1,
            qbase1,
        )
        if previous_value is not None:
            previous_value = fake_poweriteration(
                previous_value,
                compress_config.loop[layer_idx],
                compress_config.rank[layer_idx],
                compress_config.device_num[layer_idx],
                pbase2,
                qbase2,
            )
        pass
    if compress_config.compress_method[layer_idx] == "dynamicpoweriteration":
        batch, num_head, seq_len, sep_dim = previous_key.shape

        if seq_len % compress_config.stage[layer_idx] == 0:
            smaller_dim = (
                seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
            )
            smaller_dim = int(smaller_dim)
            rank_k = int(smaller_dim * compress_config.rank[layer_idx])
            rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
            previous_key = fake_poweriteration(
                previous_key,
                compress_config.loop[layer_idx],
                rank_k,
                compress_config.device_num[layer_idx],
                pbase1,
                qbase1,
            )
            if previous_value is not None and compress_config.rankv[layer_idx] != 0.0:
                previous_value = fake_poweriteration(
                    previous_value,
                    compress_config.loop[layer_idx],
                    rank_v,
                    compress_config.device_num[layer_idx],
                    pbase2,
                    qbase2,
                )
    if (
        compress_config.compress_method[layer_idx]
        == "dynamicpoweriteration_intermediate"
    ):
        batch, num_head, seq_len, sep_dim = previous_key.shape

        if seq_len % compress_config.stage[layer_idx] == 0:
            starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
            locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
            compress_length = seq_len - starting_idx - locality_idx
            smaller_dim = (
                compress_length
                if compress_length <= num_head * sep_dim
                else num_head * sep_dim
            )
            smaller_dim = int(smaller_dim)

            # print("seq_len:",seq_len,"starting_idx:",starting_idx,"locality_idx:",locality_idx,"smaller_dim:",smaller_dim)
            rank_k = int(smaller_dim * compress_config.rank[layer_idx])
            rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
            previous_key[:, :, starting_idx:-locality_idx, :] = fake_poweriteration(
                previous_key[:, :, starting_idx:-locality_idx, :],
                compress_config.loop[layer_idx],
                rank_k,
                compress_config.device_num[layer_idx],
                pbase1,
                qbase1,
            )
            if previous_value is not None and compress_config.rankv[layer_idx] != 0.0:
                previous_value[
                    :, :, starting_idx:-locality_idx, :
                ] = fake_poweriteration(
                    previous_value[:, :, starting_idx:-locality_idx, :],
                    compress_config.loop[layer_idx],
                    rank_v,
                    compress_config.device_num[layer_idx],
                    pbase2,
                    qbase2,
                )
    if compress_config.compress_method[layer_idx] == "pt+outlier":
        previous_key = fake_poweriteration_with_outlierquant(
            previous_key,
            compress_config.loop[layer_idx],
            compress_config.rank[layer_idx],
            compress_config.device_num[layer_idx],
            pbase1,
            qbase1,
            compress_config.quantize_bit[layer_idx],
            compress_config.left[layer_idx],
        )

        previous_value = fake_poweriteration_with_outlierquant(
            previous_value,
            compress_config.loop[layer_idx],
            compress_config.rank[layer_idx],
            compress_config.device_num[layer_idx],
            pbase2,
            qbase2,
            compress_config.quantize_bit[layer_idx],
            compress_config.left[layer_idx],
        )
    if compress_config.compress_method[layer_idx] == "stagept":
        previous_key = fake_stage_poweriteration(
            previous_key,
            compress_config.loop[layer_idx],
            compress_config.rank[layer_idx],
            compress_config.stage[layer_idx],
            compress_config.device_num[layer_idx],
            pbase1,
            qbase1,
        )
        if previous_value is not None:
            previous_value = fake_stage_poweriteration(
                previous_value,
                compress_config.loop[layer_idx],
                compress_config.rank[layer_idx],
                compress_config.stage[layer_idx],
                compress_config.device_num[layer_idx],
                pbase2,
                qbase2,
            )
    if compress_config.compress_method[layer_idx] == "sortquantization":
        previous_key = fake_sort_quantization(
            previous_key,
            compress_config.group_num[layer_idx],
            compress_config.quantize_bit[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_sort_quantization(
                previous_value,
                compress_config.group_num[layer_idx],
                compress_config.quantize_bit[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "pruning":
        previous_key = fake_top_k_pruning(
            previous_key, compress_config.top_k[layer_idx]
        )
        if previous_value is not None:
            previous_value = fake_top_k_pruning(
                previous_value, compress_config.top_k[layer_idx]
            )
    if compress_config.compress_method[layer_idx] == "densesparseuniformquantization":
        # print("seqlen:",seq_len,"starting_idx:",starting_idx,"locality_idx:",locality_idx)
        previous_key[
            :, :, starting_idx:-locality_idx, :
        ] = fake_dense_sparse_uniformquantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.left[layer_idx],
        )
        if previous_value is not None:
            previous_value[
                :, :, starting_idx:-locality_idx, :
            ] = fake_dense_sparse_uniformquantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                compress_config.left[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "densesparsesortquantization":
        previous_key = fake_dense_sparse_sortquantization(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_num[layer_idx],
            compress_config.left[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_dense_sparse_sortquantization(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                compress_config.group_num[layer_idx],
                compress_config.left[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "quantize_with_lrap":
        smaller_dim = seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
        smaller_dim = int(smaller_dim)
        rank_k = int(smaller_dim * compress_config.rank[layer_idx])
        rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
        previous_key = fake_quant_with_lrap(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            rank_k,
            compress_config.loop[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_quant_with_lrap(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                rank_v,
                compress_config.loop[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "outquantize_with_lrap":
        smaller_dim = seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
        smaller_dim = int(smaller_dim)
        rank_k = int(smaller_dim * compress_config.rank[layer_idx])
        rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
        previous_key = fake_outquant_with_lrap(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            rank_k,
            compress_config.loop[layer_idx],
            compress_config.left[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_outquant_with_lrap(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                rank_v,
                compress_config.loop[layer_idx],
                compress_config.left[layer_idx],
            )
    # TODO take a look at this
    if compress_config.compress_method[layer_idx] == "outquantize_with_lrap_iter":
        smaller_dim = seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
        smaller_dim = int(smaller_dim)
        rank_k = int(smaller_dim * compress_config.rank[layer_idx])
        rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
        previous_key = fake_outquant_with_lrap_iter(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            rank_k,
            compress_config.loop[layer_idx],
            compress_config.left[layer_idx],
            compress_config.iter[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_outquant_with_lrap_iter(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                rank_v,
                compress_config.loop[layer_idx],
                compress_config.left[layer_idx],
                compress_config.iter[layer_idx],
            )    
    
    # SK: TODO take a look at this: idea merged from KIVI
    if compress_config.compress_method[layer_idx] == "group_channel_with_lrap":
        smaller_dim = seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
        smaller_dim = int(smaller_dim)
        rank_k = int(smaller_dim * compress_config.rank[layer_idx])
        rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
        previous_key = fake_group_channel_quant_with_lrap(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            rank_k,
            compress_config.loop[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_group_channel_quant_with_lrap(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                rank_v,
                compress_config.loop[layer_idx],
            )

    # SK: TODO take a look at this: idea merged from KIVI
    if compress_config.compress_method[layer_idx] == "group_token_with_lrap":
        smaller_dim = seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
        smaller_dim = int(smaller_dim)
        rank_k = int(smaller_dim * compress_config.rank[layer_idx])
        rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
        previous_key = fake_group_token_quant_with_lrap(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            rank_k,
            compress_config.loop[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_group_token_quant_with_lrap(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                rank_v,
                compress_config.loop[layer_idx],
            )

    # SK: TODO take a look at this: idea merged from KIVI
    if compress_config.compress_method[layer_idx] == "group_kc_vt_with_lrap":
        smaller_dim = seq_len if seq_len <= num_head * sep_dim else num_head * sep_dim
        smaller_dim = int(smaller_dim)
        rank_k = int(smaller_dim * compress_config.rank[layer_idx])
        rank_v = int(smaller_dim * compress_config.rankv[layer_idx])
        previous_key = fake_group_channel_quant_with_lrap(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            rank_k,
            compress_config.loop[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_group_token_quant_with_lrap(
                previous_value,
                compress_config.quantize_bit[layer_idx],
                rank_v,
                compress_config.loop[layer_idx],
            )
    return previous_key, previous_value


# iteratively compress the input tensor and simluate the error by low rank approximation
def fake_outquant_with_lrap_iter(tensor, quantize_bit, rank, loop, left, iter):
    lrap_error = tensor
    batch, num_head, seq_len, sep_dim = tensor.shape
    p_base = [torch.rand(sep_dim * num_head, rank).to(tensor.device)]
    q_base = [torch.rand(batch * seq_len, rank).to(tensor.device)]
    for i in range(iter):
        tensor_quantized = fake_dense_sparse_uniformquantization(
            lrap_error, quantize_bit, left
        )
        tensor_error = tensor - tensor_quantized
        tensor_error_lrap = fake_poweriteration(
            tensor_error, loop, rank, tensor_quantized.device, p_base, q_base
        )
        lrap_error = tensor - tensor_error_lrap
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


def fake_outquant_with_lrap(tensor, quantize_bit, rank, loop, left):
    tensor_quantized = fake_dense_sparse_uniformquantization(tensor, quantize_bit, left)
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


def fake_quant_with_lrap(tensor, quantize_bit, rank, loop):
    tensor_quantized = fake_uniformquantization(tensor, quantize_bit)
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return

def fake_group_channel_quant_with_lrap(tensor, quantize_bit, rank, loop):
    tensor_quantized = fake_groupwise_channel_asymmetric_quantization(tensor, quantize_bit, 128)
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return

def fake_group_token_quant_with_lrap(tensor, quantize_bit, rank, loop):
    tensor_quantized = fake_groupwise_token_asymmetric_quantization(tensor, quantize_bit, 128)
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


# def fake_top_k_pruning(input_tensor, rate):
#     shape = input_tensor.shape
#     input_tensor = input_tensor.reshape(-1)
#     map_tensor = input_tensor.abs()
#     map_tensor = map_tensor.float()  # convert to 32bits to avoid max - min = inf
#     sortedtensor, indices = torch.sort(map_tensor)
#     k = int(len(sortedtensor) * rate)
#     threshold = sortedtensor[-k]
#     input_tensor[input_tensor < threshold] = 0
#     input_tensor = input_tensor.reshape(shape)
#     return input_tensor
