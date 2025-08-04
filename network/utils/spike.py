import torch
import torch.nn as nn
import torch.autograd as autograd

class QuantConvBatchNorm(nn.Module):
    """Conv-BN block with 1x1 kernel"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # Batch normalization layer
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Forward propagation
        return self.norm(self.conv(x))


def _make_Quant(
    in_channels: int,
    out_channels: int,
    nb_Conv: int = 1,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    """Create sequence of Conv-BN blocks.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        nb_Conv (int): Number of Conv-BN blocks.
        kernel_size (int, optional): Size of the convolving kernel. Default: 1.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (int, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.

    Returns:
        nn.Sequential: Sequence of Conv-BN blocks.
    """
    layers = [
        QuantConvBatchNorm(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    ]
    for _ in range(nb_Conv - 1):
        layers.append(
            QuantConvBatchNorm(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
        )
    # Convolution - normalization sequence
    return nn.Sequential(*layers)


class Normal_ISF(nn.Module):
    class QuantSpikeFunction(autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, x, threshold, threshold_up):
            ctx.threshold = threshold
            ctx.threshold_up = threshold_up
            ctx.save_for_backward(x)
            
            x_f = torch.clamp(x, min=threshold, max=threshold_up)
            out = torch.round(x_f)
            return out
        
        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            x, = ctx.saved_tensors
            
            grad_input[x < ctx.threshold] = 0
            grad_input[x > ctx.threshold_up] = 0
            
            return grad_input, None, None

    def __init__(
        self,
        mode: str = 'hard',
        threshold: float = 0,
        threshold_up: float = 4,
        t: int = 4,
        in_channels: int = 3,
        nb_Conv: int = 1,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.threshold_up = threshold_up
        self.t = t
        self.mode = mode
        
        if mode != 'hard':
            self.quant = _make_Quant(
                in_channels=in_channels,
                out_channels=in_channels,
                nb_Conv=nb_Conv,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
    
    def __repr__(self) -> str:
        return f"MultiSpike_norm(Norm={self.t})"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the custom autograd function
        out = self.QuantSpikeFunction.apply(x, self.threshold, self.threshold_up) / self.t
        
        if self.mode != 'hard':
            out = self.quant(out)
        
        return out