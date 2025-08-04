
import torch
import torch.nn as nn

class ConvBatchNorm(nn.Module):
    """Conv-BN block with customizable parameters.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int, optional): Size of the convolving kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default: 1.
        dilation (int, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.norm(self.conv(x))


def _make_nConv(
    in_channels: int,
    out_channels: int,
    nb_Conv: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    """Create sequence of Conv-BN blocks.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        nb_Conv (int): Number of Conv-BN blocks.
        kernel_size (int, optional): Size of the convolving kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default: 1.
        dilation (int, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.

    Returns:
        nn.Sequential: Sequence of Conv-BN blocks.
    """
    layers = [
        ConvBatchNorm(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    ]
    for _ in range(nb_Conv - 1):
        layers.append(
            ConvBatchNorm(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
    return nn.Sequential(*layers)