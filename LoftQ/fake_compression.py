import torch


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
