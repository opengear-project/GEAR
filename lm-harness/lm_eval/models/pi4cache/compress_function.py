import torch
import time


def fake_groupwise_asymmetric_quantization(input: torch.Tensor, quantize_bit):
    # what flexgen uses
    batch, num_head, seq_len, sep_dim = input.shape
    # 64 groups, group alone sep_dim
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
    return input


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
    returning_input = returning_input.half()
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
    input = input.reshape(-1)
    input = input.float()  # convert to 32bits to avoid max - min = inf
    sortedtensor, indices = torch.sort(input)
    left_num = int(len(sortedtensor) * left / 2)
    sortedtensor[left_num:-left_num] = fake_uniformquantization(
        sortedtensor[left_num:-left_num], quantize_bit
    )
    input[indices] = sortedtensor
    input = input.reshape(shape)
    input = input.half()
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
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    input = input.view(seq_len * batch, sep_dim * num_head)
    input = input.float()
    # print(p_base)
    p_base[0] = p_base[0].float()
    q_base[0] = q_base[0].float()
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
    input = input.half()
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
):
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
        previous_key = fake_groupwise_asymmetric_quantization(
            previous_key, compress_config.quantize_bit[layer_idx]
        )
        if previous_value is not None:
            previous_value = fake_groupwise_asymmetric_quantization(
                previous_value,
                compress_config.quantize_bit[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "uniformquantization":
        previous_key = fake_uniformquantization(
            previous_key, compress_config.quantize_bit[layer_idx]
        )
        if previous_value is not None:
            previous_value = fake_uniformquantization(
                previous_value,
                compress_config.quantize_bit[layer_idx],
            )
    if compress_config.compress_method[layer_idx] == "poweriteration":
        if pbase1 is not None:
            previous_key = fake_poweriteration(
                previous_key,
                compress_config.loop[layer_idx],
                compress_config.rank[layer_idx],
                compress_config.device_num[layer_idx],
                pbase1,
                qbase1,
            )

        if pbase2 is not None:
            previous_value = fake_poweriteration(
                previous_value,
                compress_config.loop[layer_idx],
                compress_config.rank[layer_idx],
                compress_config.device_num[layer_idx],
                pbase2,
                qbase2,
            )
        pass
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
        previous_key = fake_dense_sparse_uniformquantization(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.left[layer_idx],
        )
        if previous_value is not None:
            previous_value = fake_dense_sparse_uniformquantization(
                previous_value,
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
    return previous_key, previous_value


class PiCache:
    def __init__(
        self, cache_shape: tuple, rank, loop, device_num, compress_dim
    ) -> None:
        self.batch, self.num_head, self.seq_len, self.sep_dim = cache_shape
        print(self.batch, self.num_head, self.seq_len, self.sep_dim)
        self.p_buffers = (
            []
        )  # torch.rand(self.sep_dim * self.num_head, rank).to(device_num)
        self.q_buffers = (
            []
        )  # torch.rand(self.batch * self.seq_len, rank).to(device_num)
        self.p_base = [torch.rand(self.sep_dim * self.num_head, rank).to(device_num)]
        self.q_base = [torch.rand(self.batch * self.seq_len, rank).to(device_num)]
        self.cache_buffer = None
        self.loop = loop
        self.compress_dim = compress_dim
        self.device_num = device_num
        self.rank = rank

    def update(self):
        self.seq_len = self.seq_len + 1

    def decompress(self):
        # print("len of p_buffers:",len(self.p_buffers))
        for i in range(len(self.p_buffers)):
            # print("i:",i)
            p_buffer = self.p_buffers[i]
            q_buffer = self.q_buffers[i]
            # print("p_buffer shape:",p_buffer[0].shape)
            # print("q_buffer shape:",q_buffer[0].shape)
            if i == 0:
                matrix0 = ptdecompress(
                    p_buffer,
                    q_buffer,
                    self.batch,
                    self.num_head,
                    self.compress_dim,
                    self.sep_dim,
                )
            else:
                matrix = ptdecompress(
                    p_buffer,
                    q_buffer,
                    self.batch,
                    self.num_head,
                    self.compress_dim,
                    self.sep_dim,
                )
                matrix0 = torch.cat((matrix0, matrix), 2)
        if self.cache_buffer is not None:
            matrix0 = torch.cat((matrix0, self.cache_buffer), 2)
            # print(matrix0.shape, self.cache_buffer.shape)
        matrix0 = matrix0.view(self.batch, self.num_head, self.seq_len, self.sep_dim)
        return matrix0

    def fill_cache_buffer(self, cache):
        self.cache_buffer = cache

    def compress(self, cache):
        self.batch, self.num_head, self.seq_len, self.sep_dim = cache.shape
        if self.seq_len % self.compress_dim != 0:
            cache_dim = self.seq_len % self.compress_dim
            self.cache_buffer = cache[:, :, -cache_dim:, :]
            return
        elif self.seq_len % self.compress_dim == 0:
            p_buffer, q_buffer, _, _, _, _ = ptcompress(
                self.cache_buffer, self.p_base, self.q_base, self.loop
            )
            self.cache_buffer = None
            # p_buffer = p_buffer
            # q_buffer = q_buffer
            self.p_buffers.append(p_buffer)
            self.q_buffers.append(q_buffer)
            return

    def compress_cache(self, cache):
        self.batch, self.num_head, self.seq_len, self.sep_dim = cache.shape
        cache_dim = self.seq_len % self.compress_dim
        # print("cache_dim:",cache_dim)
        # print("cache_num:",self.seq_len // self.compress_dim)
        for i in range(self.seq_len // self.compress_dim):
            cache_buffer = cache[
                :, :, i * self.compress_dim : (i + 1) * self.compress_dim, :
            ]
            p_buffer, q_buffer, _, _, _, _ = ptcompress(
                cache_buffer, self.p_base, self.q_base, self.loop
            )
            # print("p_buffer shape:",p_buffer[0].shape)
            # print("q_buffer shape:",q_buffer[0].shape)
            # print("p_buffer_nan:",torch.any(torch.isnan(p_buffer[0])))
            # print("q_buffer_nan:",torch.any(torch.isnan(q_buffer[0])))
            self.p_buffers.append(p_buffer)
            self.q_buffers.append(q_buffer)
        if cache_dim != 0:
            self.cache_buffer = cache[:, :, -cache_dim:, :]
        # print("len of p_buffers:",len(self.p_buffers))
        # print("len of q_buffers:",len(self.q_buffers))
        return

    def clean(self):
        for i in range(len(self.p_buffers)):
            self.p_buffers[i] = None
            self.q_buffers[i] = None
            self.cache_buffer = None
        self.p_buffers = []
        self.q_buffers = []


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
