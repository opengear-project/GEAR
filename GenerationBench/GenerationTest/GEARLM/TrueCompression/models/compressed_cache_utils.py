from typing import Any, Dict, List, Optional, Tuple
from .cache_utils import Cache
import torch
from .TrueCompressFunction import (
    true_uniform_quantization_compress,
    true_uniform_quantization_decompress,
    true_outlier_quantization_compress,
    true_outlier_quantization_decompress,
    true_gear_compress,
    true_gear_decompress,
)
from .TrueCompressFunction import (
    true_uniform_quantization_compress_batchwise,
    true_uniform_quantization_decompress_batchwise,
    true_outlier_quantization_compress_batchwise,
    true_outlier_quantization_decompress_batchwise,
    true_gear_compress,
    true_gear_decompress_batchwise,
    true_gear_compress_batchwise,
)

compress_function = {
    "uniform": true_uniform_quantization_compress,
    "outlier": true_outlier_quantization_compress,
    "gear": true_gear_compress,
    "uniform_batch": true_uniform_quantization_compress_batchwise,
    "outlier_batch": true_outlier_quantization_compress_batchwise,
    "gear_batch": true_gear_compress_batchwise,
    "gear_tokenwiseQ": None,
}
decompress_function = {
    "uniform": true_uniform_quantization_decompress,
    "outlier": true_outlier_quantization_decompress,
    "gear": true_gear_decompress,
    "uniform_batch": true_uniform_quantization_decompress_batchwise,
    "outlier_batch": true_outlier_quantization_decompress_batchwise,
    "gear_batch": true_gear_decompress_batchwise,
    "gear_tokenwiseQ": None,
}


class FPBuffer:
    """
    Three-zone full-precision buffer for KV cache tokens.

    Zones (contiguous in sequence order):
      [sink_tokens | recency_tokens | buffer_len]

    - sink   : first N tokens of the prefill, always FP, never compressed.
    - recency: rolling FP window; oldest buffer_len slice is flushed to
               compress each time the buffer zone fills.
    - buffer : accumulation zone; triggers a flush when it reaches buffer_len.

    The caller is responsible for passing the tensor returned by append()
    (when non-None) into the CompressedUnion compression pipeline.
    """

    def __init__(self, sink_tokens: int, recency_tokens: int, buffer_len: int):
        self.sink_tokens = sink_tokens
        self.recency_tokens = recency_tokens
        self.buffer_len = buffer_len

        self._sink: Optional[torch.Tensor] = None
        self._recency: Optional[torch.Tensor] = None
        self._buffer: Optional[torch.Tensor] = None

        self._initialised = False

        # if recency_tokens == 0 and buffer_len > 0:
        #     raise ValueError(
        #         "recency_tokens=0 is invalid when buffer_len > 0. "
        #         "The flush window would always be empty."
        #     )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sink_len(self) -> int:
        return self._sink.shape[-2] if self._sink is not None else 0

    def _recency_len(self) -> int:
        return self._recency.shape[-2] if self._recency is not None else 0

    def _buffer_len_cur(self) -> int:
        return self._buffer.shape[-2] if self._buffer is not None else 0

    def total_len(self) -> int:
        return self._sink_len() + self._recency_len() + self._buffer_len_cur()

    def get_fp_view(self) -> Optional[torch.Tensor]:
        """Return [sink | recency | buffer] concatenated for attention."""
        parts = [p for p in (self._sink, self._recency, self._buffer) if p is not None]
        if not parts:
            return None
        return torch.cat(parts, dim=-2)

    # ── main API ──────────────────────────────────────────────────────────────

    def append(self, new_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Ingest new_tokens (shape [B, H, T, D]) into the buffer.

        Returns a tensor of tokens that must be sent to compress, or None
        if no flush is needed yet.

        Cases handled:
          PREFILL OK   : first call, fits within floor+buffer — no compression.
          PREFILL SPLIT: first call, overshoots capacity — compress middle.
          FILLING      : warm-up floor not yet reached — keep accumulating.
          FLUSH        : buffer zone full — compress recency[:buffer_len].
        """
        T = new_tokens.shape[-2]

        # ── First call: prefill ───────────────────────────────────────────────
        if not self._initialised:
            total_capacity = self.sink_tokens + self.recency_tokens + self.buffer_len
            floor = self.sink_tokens + self.recency_tokens

            if T > total_capacity:
                # PREFILL SPLIT: compress the middle, keep sink + last recency_tokens
                to_compress = new_tokens[..., self.sink_tokens: T - self.recency_tokens, :]
                self._sink = new_tokens[..., :self.sink_tokens, :]
                self._recency = new_tokens[..., T - self.recency_tokens:, :]
                self._buffer = None
                self._initialised = True
                # print(
                #     f"[FPBuffer] PREFILL SPLIT | prefill_len={T} | "
                #     f"sink={self.sink_tokens} | recency={self.recency_tokens} | "
                #     f"buffer=0 | compressed={to_compress.shape[-2]}"
                # )
                return to_compress

            # PREFILL OK: fits entirely in FP — carve into zones, no compression
            self._sink = new_tokens[..., :self.sink_tokens, :]
            remainder = new_tokens[..., self.sink_tokens:, :]
            recency_part = min(self.recency_tokens, remainder.shape[-2])
            self._recency = remainder[..., :recency_part, :]
            buf_part = remainder[..., recency_part:, :]
            self._buffer = buf_part if buf_part.shape[-2] > 0 else None
            self._initialised = True
            # print(
            #     f"[FPBuffer] PREFILL OK | prefill_len={T} | "
            #     f"sink={self._sink_len()} | recency={self._recency_len()} | "
            #     f"buffer_len={self._buffer_len_cur()} | no compression needed"
            # )
            return None

        # ── Subsequent calls: decoding ────────────────────────────────────────

        # Warm-up: fill recency zone to its target size before using buffer zone
        if self._recency_len() < self.recency_tokens:
            self._recency = (
                torch.cat([self._recency, new_tokens], dim=-2)
                if self._recency is not None else new_tokens
            )
            floor = self.sink_tokens + self.recency_tokens
            # print(
            #     f"[FPBuffer] FILLING | total_fp={self.total_len()} | "
            #     f"floor={floor} (sink={self.sink_tokens} + recency={self.recency_tokens}) | "
            #     f"need={floor - self.total_len()} more"
            # )
            return None

        # Accumulate into buffer zone
        self._buffer = (
            torch.cat([self._buffer, new_tokens], dim=-2)
            if self._buffer is not None else new_tokens
        )

        # Check flush threshold
        if self._buffer_len_cur() < self.buffer_len:
            floor = self.sink_tokens + self.recency_tokens
            # print(
            #     f"[FPBuffer] FILLING | total_fp={self.total_len()} | "
            #     f"floor={floor} (sink={self.sink_tokens} + recency={self.recency_tokens}) | "
            #     f"need={self.buffer_len - self._buffer_len_cur()} more in buffer"
            # )
            return None

        # FLUSH: compress recency[:buffer_len], slide recency window forward
        to_compress = self._recency[..., :self.buffer_len, :]
        total_before_flush = self.total_len()
        self._recency = torch.cat(
            [self._recency[..., self.buffer_len:, :], self._buffer], dim=-2
        )
        self._buffer = None
        fp_after = self.total_len()
        # print(
        #     f"[FPBuffer] FLUSH | total_fp={total_before_flush} | "
        #     f"sink={self._sink_len()} | recency={self._recency_len()} | "
        #     f"buffer={self._buffer_len_cur()} | "
        #     f"tokens_to_compress={to_compress.shape[-2]} | fp_after={fp_after}"
        # )
        return to_compress


class CompressedUnion:
    def __init__(self, compress_kwargs: Optional[Dict[str, Any]] = None):
        self.quantize_bit = compress_kwargs["quantize_bit"]
        self.compress_mode = compress_kwargs["compress_mode"]
        self.min = None
        self.step = None
        self.left = compress_kwargs["left"]
        self.rank = compress_kwargs["rank"]
        self.loop = compress_kwargs["loop"]
        self.dtype = None
        self.shape = None
        self.is_compressed = False
        self.cache = None
        self.values = None
        self.indices = None
        self.p_base = None
        self.q_base = None
        self.counter = 0
        # self.kvcache_shape = None

    def set_cache(self, input: torch.Tensor):
        self.counter += 1
        # has_inf = torch.isinf(input)
        # has_nan = torch.isnan(input)
        # print(self.counter,has_inf.any(),has_nan.any())
        self.cache = input
        self.kvcache_shape = input.shape

    def get_cache(self):
        return self.cache

    def compress(self):
        input = self.cache
        self.dtype = input.dtype
        self.is_compressed = True
        if self.compress_mode == "uniform":
            output, shape, min, step = compress_function[self.compress_mode](
                input, self.quantize_bit
            )
            self.cache = output
            self.min = min
            self.step = step
            self.shape = shape
        elif self.compress_mode == "outlier":
            output, shape, min, step, values, indices = compress_function[
                self.compress_mode
            ](input, self.quantize_bit, self.left)
            self.cache = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
        elif self.compress_mode == "gear":
            output, shape, min, step, values, indices, p_base, q_base = (
                compress_function[self.compress_mode](
                    input, self.quantize_bit, self.left, self.rank, self.loop
                )
            )
            self.cache = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base
        elif self.compress_mode == "uniform_batch":
            output, shape, min, step = compress_function[self.compress_mode](
                input, self.quantize_bit
            )
            self.cache = output
            self.min = min
            self.step = step
            self.shape = shape
        elif self.compress_mode == "outlier_batch":
            output, shape, min, step, values, indices = compress_function[
                self.compress_mode
            ](input, self.quantize_bit, self.left)
            self.cache = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
        elif self.compress_mode == "gear_batch":
            output, shape, min, step, values, indices, p_base, q_base = (
                compress_function[self.compress_mode](
                    input, self.quantize_bit, self.left, self.rank, self.loop
                )
            )
            self.cache = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base

    # for new method: shared kernel used by decompress() and decompress_readonly()
    def _reconstruct(self) -> torch.Tensor:
        """Run the decompression kernel and return the FP tensor without mutating any state."""
        if self.compress_mode == "uniform":
            return decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier":
            return decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear":
            return decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
            )
        elif self.compress_mode == "uniform_batch":
            return decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier_batch":
            return decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear_batch":
            return decompress_function[self.compress_mode](
                self.cache,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
            )

    # for new method: destructive read — reconstructs FP then clears packed state
    def decompress(self):
        output = self._reconstruct()
        self.clean_cache()
        return output

    # for new method: read-only reconstruct — packed state untouched, safe to call every attention step
    def decompress_readonly(self) -> torch.Tensor:
        """Return the reconstructed FP tensor while keeping the packed state intact."""
        return self._reconstruct()

    def clean_cache(self):
        self.is_compressed = False
        self.cache = None
        self.values = None
        self.indices = None
        self.p_base = None
        self.q_base = None
        self.min = None
        self.step = None


class CompressedCache(Cache):
    def __init__(self) -> None:
        # per-layer compressed prefix (CompressedUnion or None before first flush)
        self.key_cache: List[Optional[Any]] = []
        self.value_cache: List[Optional[Any]] = []
        # per-layer FPBuffer instances replacing the old key_tail / value_tail flat tensors
        self.fp_buffers_key: List[Optional[FPBuffer]] = []
        self.fp_buffers_value: List[Optional[FPBuffer]] = []
        # token count of the compressed prefix for each layer
        self._prefix_seq_lens: List[int] = []
        # compress kwargs snapshot per layer, reused when building a new union at flush
        self._compress_kwargs_store: List[Optional[Dict[str, Any]]] = []
        # FPBuffer construction params, latched once from the first compress_kwargs
        self._sink_tokens: Optional[int] = None
        self._recency_tokens: Optional[int] = None
        self._buffer_len: Optional[int] = None
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __setitem__(
        self, layer_idx: int, key_value_states: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Support for backwards-compatible `past_key_value` assignment, e.g. `past_key_value[0] = (key_states,
        value_states)` to update the cache for the first layer.
        """
        key_states, value_states = key_value_states
        self.key_cache[layer_idx], self.value_cache[layer_idx] = (
            key_states,
            value_states,
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def _flush_to_prefix(self, layer_idx: int, to_compress_k: torch.Tensor, to_compress_v: torch.Tensor):
        """Merge to_compress tensors into the compressed prefix for layer_idx."""
        prefix_k = self.key_cache[layer_idx]
        prefix_v = self.value_cache[layer_idx]

        if prefix_k is not None and prefix_k.is_compressed:
            prefix_fp_k = prefix_k.decompress_readonly()
            prefix_fp_v = prefix_v.decompress_readonly()
            merged_k = torch.cat([prefix_fp_k, to_compress_k], dim=-2)
            merged_v = torch.cat([prefix_fp_v, to_compress_v], dim=-2)
            prefix_k.clean_cache()
            prefix_v.clean_cache()
        else:
            merged_k = to_compress_k
            merged_v = to_compress_v

        kwargs = self._compress_kwargs_store[layer_idx]
        new_key_union = CompressedUnion(kwargs)
        new_value_union = CompressedUnion(kwargs)
        new_key_union.set_cache(merged_k)
        new_value_union.set_cache(merged_v)
        new_key_union.compress()
        new_value_union.compress()

        self.key_cache[layer_idx] = new_key_union
        self.value_cache[layer_idx] = new_value_union
        self._prefix_seq_lens[layer_idx] = merged_k.shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        compress_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            compress_kwargs (`Dict[str, Any]`, `optional`):
                Must contain standard compress keys plus ``"sink_tokens"``,
                ``"recency_tokens"``, and ``"buffer_len"`` (int) to enable the
                three-zone FPBuffer path.

        Return:
            A tuple ``(full_key, full_value)`` shaped ``[B, H, prefix_len+fp_len, D]``
            that the attention kernel should use.
        """
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        if compress_kwargs is not None:
            # Latch FPBuffer construction params once from the first call
            if self._sink_tokens is None:
                self._sink_tokens = compress_kwargs.get("sink_tokens", 4) #to add default values to escape egde cases whihc are immposible 
                self._recency_tokens = compress_kwargs.get("recency_tokens", 40)
                self._buffer_len = compress_kwargs.get("buffer_len", 20)

            if len(self.fp_buffers_key) <= layer_idx:
                # First call for this layer — initialise prefix slot and FPBuffers
                self.key_cache.append(None)
                self.value_cache.append(None)
                self._prefix_seq_lens.append(0)
                self._compress_kwargs_store.append(compress_kwargs)
                self.fp_buffers_key.append(
                    FPBuffer(self._sink_tokens, self._recency_tokens, self._buffer_len)
                )
                self.fp_buffers_value.append(
                    FPBuffer(self._sink_tokens, self._recency_tokens, self._buffer_len)
                )

            buf_k = self.fp_buffers_key[layer_idx]
            buf_v = self.fp_buffers_value[layer_idx]

            # Append to FPBuffer; returns tokens to compress on flush, else None
            to_compress_k = buf_k.append(key_states)
            to_compress_v = buf_v.append(value_states)

            if to_compress_k is not None:
                self._flush_to_prefix(layer_idx, to_compress_k, to_compress_v)

            # Build full FP view for attention: decompress(prefix) ‖ fp_buffer
            prefix_k = self.key_cache[layer_idx]
            prefix_v = self.value_cache[layer_idx]
            fp_view_k = buf_k.get_fp_view()
            fp_view_v = buf_v.get_fp_view()

            if prefix_k is not None and prefix_k.is_compressed:
                prefix_fp_k = prefix_k.decompress_readonly()
                prefix_fp_v = prefix_v.decompress_readonly()
                sink_len = self._sink_tokens
                sink_k    = fp_view_k[..., :sink_len, :] #the order was messed up for it 
                sink_v    = fp_view_v[..., :sink_len, :]
                tail_k    = fp_view_k[..., sink_len:, :]   # recency + buffer
                tail_v    = fp_view_v[..., sink_len:, :]
                full_key   = torch.cat([sink_k, prefix_fp_k, tail_k], dim=-2)
                full_value = torch.cat([sink_v, prefix_fp_v, tail_v], dim=-2)
                # full_key = torch.cat([prefix_fp_k, fp_view_k], dim=-2)
                # full_value = torch.cat([prefix_fp_v, fp_view_v], dim=-2)
            else:
                full_key = fp_view_k
                full_value = fp_view_v

            return full_key, full_value

        # ── fallback: no compress_kwargs, plain tensor cache ──────────────────
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def compress(self, layer_idx: int):
        """No-op: flushing is now driven entirely inside FPBuffer.append() during update().

        Kept for call-site compatibility with the model forward pass.
        Legacy plain-tensor path (no FPBuffer) still handled below.
        """
        if layer_idx < len(self.fp_buffers_key):
            # FPBuffer path: flush is triggered inside update() — nothing to do here
            return

        # Legacy path (no FPBuffer initialised for this layer)
        if len(self.key_cache) <= layer_idx:
            return
        key_union = self.key_cache[layer_idx]
        value_union = self.value_cache[layer_idx]
        if (
            key_union is not None
            and value_union is not None
            and not key_union.is_compressed
            and not value_union.is_compressed
        ):
            key_union.compress()
            value_union.compress()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the total sequence length (compressed prefix + FP buffer) for the given layer."""
        return self.get_cur_seq_len(layer_idx)

    def get_cur_seq_len(self, layer_idx: int = 0) -> int:
        """Returns prefix_len + fp_buffer total_len for the given layer."""
        if layer_idx >= len(self._prefix_seq_lens):
            return 0
        prefix_len = self._prefix_seq_lens[layer_idx]
        buf = self.fp_buffers_key[layer_idx] if layer_idx < len(self.fp_buffers_key) else None
        buf_len = buf.total_len() if buf is not None else 0
        return prefix_len + buf_len

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                device = k.device
                self.key_cache[layer_idx] = k.index_select(0, beam_idx.to(device))
                device = v.device
                self.value_cache[layer_idx] = v.index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "CompressedCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
