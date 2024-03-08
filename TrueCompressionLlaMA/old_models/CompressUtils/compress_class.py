from typing import Any, Dict, List, Optional, Tuple
import torch
from .TrueCompressFunction import (
    true_uniform_quantization_compress,
    true_uniform_quantization_decompress,
    true_outlier_quantization_compress,
    true_outlier_quantization_decompress,
    true_gear_compress,
    true_gear_decompress,
    true_gear_tokenwiseQ_compress,
    true_gear_tokenwiseQ_decompress,
    true_gear_tokenwiseQ_compress_nopq,
    true_gear_tokenwiseQ_decompress_nopq,
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
    "gear_tokenwiseQ": true_gear_tokenwiseQ_compress,
    "gear_tokenwiseQ_nopq": true_gear_tokenwiseQ_compress_nopq,
}
decompress_function = {
    "uniform": true_uniform_quantization_decompress,
    "outlier": true_outlier_quantization_decompress,
    "gear": true_gear_decompress,
    "uniform_batch": true_uniform_quantization_decompress_batchwise,
    "outlier_batch": true_outlier_quantization_decompress_batchwise,
    "gear_batch": true_gear_decompress_batchwise,
    "gear_tokenwiseQ": true_gear_tokenwiseQ_decompress,
    "gear_tokenwiseQ_nopq": true_gear_tokenwiseQ_decompress_nopq,
}


def detect_infnan(input_tensor, string):
    if torch.isnan(input_tensor).any():
        print(string + "has nan")
        while True:
            pass
    if torch.isinf(input_tensor).any():
        print(string + "has inf")
        while True:
            pass


class CompressUnion:
    def __init__(self, compress_kwargs: Optional[Dict[str, Any]] = None):
        self.quantize_bit = compress_kwargs["quantize_bit"]
        self.compress_mode = compress_kwargs["compress_mode"]
        self.min = None
        self.step = None
        self.min_p = None
        self.min_q = None
        self.step_p = None
        self.step_q = None
        self.left = compress_kwargs["left"]
        self.rank = compress_kwargs["rank"]
        self.loop = compress_kwargs["loop"]
        self.dtype = None
        self.shape = None
        self.shape_p = None
        self.shape_q = None
        self.quantize_part = None
        self.values = None
        self.indices = None
        self.p_base = None
        self.q_base = None
        self.counter = 0
        self.streaming_gap = compress_kwargs["streaming_gap"]
        self.buffer = None
        self.streaming = compress_kwargs["stream"]
        self.seq_length = 0
        self.input_shape = 0

    def compress_function(self, input_tensor: torch.Tensor):
        self.dtype = input_tensor.dtype
        # detect_infnan(input_tensor,"compress input tensor")
        if self.compress_mode == "uniform":
            output, shape, min, step = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
        elif self.compress_mode == "outlier":
            output, shape, min, step, values, indices = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.left)
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
        elif self.compress_mode == "gear":
            output, shape, min, step, values, indices, p_base, q_base = (
                compress_function[self.compress_mode](
                    input_tensor, self.quantize_bit, self.left, self.rank, self.loop
                )
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base
        elif self.compress_mode == "uniform_batch":
            output, shape, min, step = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
        elif self.compress_mode == "outlier_batch":
            output, shape, min, step, values, indices = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.left)
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
        elif self.compress_mode == "gear_batch":
            output, shape, min, step, values, indices, p_base, q_base = (
                compress_function[self.compress_mode](
                    input_tensor, self.quantize_bit, self.left, self.rank, self.loop
                )
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base
        elif self.compress_mode == "gear_tokenwiseQ":

            (
                quantized_input,
                shape,
                min,
                step,
                p_base,
                q_base,
                shape_p,
                shape_q,
                min_p,
                min_q,
                scale_p,
                scale_q,
            ) = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit, self.rank, self.loop
            )
            self.quantize_part = quantized_input
            self.min = min
            self.step = step
            self.shape = shape
            self.p_base = p_base
            self.q_base = q_base
            self.shape_p = shape_p
            self.shape_q = shape_q
            self.min_p = min_p
            self.min_q = min_q
            self.step_p = scale_p
            self.step_q = scale_q
        elif self.compress_mode == "gear_tokenwiseQ_nopq":
            quantized_input, shape, min, step, p_base, q_base = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.rank, self.loop)
            self.quantize_part = quantized_input
            self.min = min
            self.step = step
            self.shape = shape
            self.p_base = p_base
            self.q_base = q_base
        # print("quantized_part_min_max:",self.quantize_part.min(),self.quantize_part.max(),"p_base_min_max:",self.min_p.min(),self.p_base[0].max(),"q_base_min_max:",self.min_q.min(),self.q_base[0].max())
        # detect_infnan(quantized_input,"compress quantized_input tensor")
        # detect_infnan(self.p_base[0],"compress p_base tensor")
        # detect_infnan(self.q_base[0],"compress q_base tensor")

    def decompress_function(self):
        if self.compress_mode == "uniform":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
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
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier_batch":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear_batch":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
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
        elif self.compress_mode == "gear_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.p_base,
                self.q_base,
                self.shape_p,
                self.shape_q,
                self.min_p,
                self.min_q,
                self.step_p,
                self.step_q,
                self.dtype,
            )
        elif self.compress_mode == "gear_tokenwiseQ_nopq":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.p_base,
                self.q_base,
                self.dtype,
            )
        # detect_infnan(output,"decompress")
        return output

    def compress(self, input_tensor):
        self.seq_length = input_tensor.shape[-2]
        # print("compress",self.counter)
        self.input_shape = input_tensor.shape
        if self.streaming is True:
            if self.counter % self.streaming_gap == 0:
                self.buffer = None
                self.compress_function(input_tensor)
            else:
                extract_id = self.counter % self.streaming_gap
                self.buffer = input_tensor[:, :, -extract_id:, :].clone()

        else:
            self.compress_function(input_tensor)

    def decompress(self):
        # print("decompress",self.counter)
        if self.streaming is True:
            if self.counter % self.streaming_gap == 0:
                output = self.decompress_function()
                if self.buffer is not None:
                    output = torch.cat([output, self.buffer], dim=-2)

            else:
                output = self.decompress_function()

                output = torch.cat([output, self.buffer], dim=-2)

            self.counter += 1

        else:

            output = self.decompress_function()
        # detect_infnan(output,"decompress output")
        return output
