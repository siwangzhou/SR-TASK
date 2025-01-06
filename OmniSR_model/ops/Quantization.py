import torch
import torch.nn as nn
import math

class Quant_RS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        input = input / 2 + 0.5

        input = torch.clamp(input, 0, 1)
        ctx.save_for_backward(input)
        input = (input * 255.).round() / 255.

        output = (input.cuda() - 0.5) * 2

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad = grad_output * (1 - 0.5 * torch.cos(2 * math.pi * input))

        return grad


class Quantization_RS(nn.Module):
    def __init__(self):
        super(Quantization_RS, self).__init__()

    def forward(self, input):
        return Quant_RS.apply(input)
