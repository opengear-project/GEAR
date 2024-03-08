import torch.nn as nn
import torch


class DecodlerLayer(nn.Module):
    def __init__(self, input_size, hidden_size, ffsize, layer_idx):
        super(DecodlerLayer, self).__init__()
        self.qproj = nn.Linear(input_size, hidden_size)
        self.kproj = nn.Linear(input_size, hidden_size)
        self.vproj = nn.Linear(input_size, hidden_size)
        self.ff = nn.Linear(hidden_size, ffsize)
        self.output = nn.Linear(ffsize, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.layer_idx = layer_idx

    def forward(self, input, kv_cache, chunk=None, device="0"):
        q = self.qproj(input)
        k = self.kproj(input)
        v = self.vproj(input)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        attn = self.softmax(q @ k.transpose(1, 2))
        attn = attn @ v
        attn = self.ff(attn)
        attn = self.output(attn)
        return attn, (k, v)


class ToyLlama(nn.Module):
    def __init__(self, input_size, hidden_size, ffsize, layer_num=0):
        super(ToyLlama, self).__init__()
        self.decoder_list = nn.ModuleList(
            [
                DecodlerLayer(input_size, hidden_size, ffsize, i)
                for i in range(layer_num)
            ]
        )

    def forward(self, input, kv_cache):
        if kv_cache is None:
            kv_cache = []
        for i, layer in enumerate(self.decoder_list):
            if len(kv_cache) < i + 1:
                layer_kv_cache = None
            else:
                layer_kv_cache = kv_cache[i]
            input, layer_kv_cache = layer(input, layer_kv_cache)
            if len(kv_cache) < i + 1:
                kv_cache.append(layer_kv_cache)
            else:
                kv_cache[i] = layer_kv_cache

        return input, kv_cache

    def zig_zag_forward(self, tensor_list, kv_cache, chunk_num, device):
        bsz = len(tensor_list)
        assert bsz == chunk_num
        if kv_cache is None:
            kv_cache = []
        for i, layer in enumerate(self.decoder_list):
            if len(kv_cache) < i + 1:
                layer_kv_cache = []
            else:
                layer_kv_cache = kv_cache[i]
            for j in range(chunk_num):
                if len(layer_kv_cache) < j + 1:
                    chunk_kv_cache = None
                else:
                    chunk_kv_cache = layer_kv_cache[j]
                # to gpu
                chunk_input = tensor_list[j].to(device)

                if chunk_kv_cache is not None:
                    chunk_kv_cache = (
                        chunk_kv_cache[0].to(device),
                        chunk_kv_cache[1].to(device),
                    )
                layer = layer.to(device)
                tensor_list[j], chunk_kv_cache = layer(chunk_input, chunk_kv_cache, j)
                # to cpu
                tensor_list[j].to("cpu")
                layer.to("cpu")
                chunk_kv_cache = (
                    chunk_kv_cache[0].to("cpu"),
                    chunk_kv_cache[1].to("cpu"),
                )
                if len(layer_kv_cache) < j + 1:
                    layer_kv_cache.append(chunk_kv_cache)

                else:
                    layer_kv_cache[j] = chunk_kv_cache
            if len(kv_cache) < i + 1:
                kv_cache.append(layer_kv_cache)
        return tensor_list, kv_cache

    def zig_zag_generate(self, input, chunk_num, generation_length, device):
        for i in range(generation_length):
            if i == 0:
                kvcache = None
            input, kvcache = self.zig_zag_forward(input, kvcache, chunk_num, device)
            if i == 0:
                for j in range(len(input)):
                    bsz, seq_len, model_dim = input[j].shape
                    input[j] = input[j][:, -1, :].reshape(bsz, 1, model_dim)
        return input, kvcache
