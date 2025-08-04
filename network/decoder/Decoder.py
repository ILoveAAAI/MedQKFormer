import torch
import torch.nn as nn
from utils import Normal_ISF,_make_nConv



class Mask_Spike_Conv(nn.Module):
    """Spike-enhanced convolutional block with residual connections.

    This module combines convolutional operations with spiking neural network (SNN) elements.
    It uses quantized spike functions to introduce sparsity and biological plausibility into the network.
    The block includes residual connections to ease training and improve gradient flow.

    Args:
        in_dim (int): Number of channels in the input tensor.
        num_convs (int, optional): Number of convolutional layers in each convolution block. Default: 1.
        kernel_size (int, optional): Size of the convolving kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default: 1.
        dilation (int, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        mode (str, optional): Mode of the spike function. Can be 'hard' or other modes supporting learnable parameters. Default: 'hard'.
        threshold (float, optional): Lower threshold for spike generation. Default: 0.0.
        threshold_up (float, optional): Upper threshold for spike generation. Default: 4.0.
        t (int, optional): Time steps for spike integration. Default: 4.
    """
    
    def __init__(
        self,
        in_dim: int,
        num_convs: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        mode: str = 'hard',
        threshold: float = 0.0,
        threshold_up: float = 4.0,
        t: int = 4,
    ) -> None:
        super().__init__()
        self.lif1 = Normal_ISF(
            mode=mode,
            threshold=threshold,
            threshold_up=threshold_up,
            t=t,
            in_channels=in_dim,
            nb_Conv=num_convs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv1 = _make_nConv(
            in_channels=in_dim,
            out_channels=in_dim,
            nb_Conv=num_convs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.lif2 = Normal_ISF(
            mode=mode,
            threshold=threshold,
            threshold_up=threshold_up,
            t=t,
            in_channels=in_dim,
            nb_Conv=num_convs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv2 = _make_nConv(
            in_channels=in_dim,
            out_channels=in_dim,
            nb_Conv=num_convs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=in_dim)
        
    def __repr__(self) -> str:
        return f"Mask_Spike_Conv(in_dim={self.conv1[0].conv.in_channels}, num_convs={len(self.conv1)})"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function with residual connections.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying spike-convolutional blocks and residual connections.
        """
        x_ori = x  # Preserve original input for residual connection
        
        # First spike-convolutional block with residual connection
        x = self.lif1(x)
        x = self.bn(self.conv1(x) + x_ori)
        
        # Second spike-convolutional block with residual connection
        x = self.lif2(x)
        x = self.bn(self.conv2(x) + x_ori)
        
        return x

class Decoder(nn.Module):
    """
    Decoder module for feature fusion and upsampling
    
    Args:
        in_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        num_convs (int): Number of convolutional blocks after upsampling (default: 1)
        kernel_size (int): Size of the convolving kernel (default: 3)
        stride (int): Stride of the convolution (default: 1)
        padding (int): Zero-padding added to both sides of the input (default: 1)
        dilation (int): Spacing between kernel elements (default: 1)
        groups (int): Number of blocked connections from input channels to output channels (default: 1)
        bias (bool): If True, adds a learnable bias to the output (default: True)
        mode (str): Mode of the spike function (default: 'hard')
        threshold (float): Lower threshold for spike generation (default: 0.0)
        threshold_up (float): Upper threshold for spike generation (default: 4.0)
        t (int): Time steps for spike integration (default: 4)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_convs: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        mode: str = 'hard',
        threshold: float = 0.0,
        threshold_up: float = 4.0,
        t: int = 4,
    ) -> None:
        super().__init__()
        # Spike convolution enhancement with all parameters passed
        self.sc = Mask_Spike_Conv(
            in_dim=in_dim,
            num_convs=num_convs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            mode=mode,
            threshold=threshold,
            threshold_up=threshold_up,
            t=t,
        )
        
        # Upsampling layer
        self.upsample = nn.Upsample(
            scale_factor=2, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Convolutional sequence with all parameters passed
        self.nConv = _make_nConv(
            in_channels=in_dim, 
            out_channels=out_dim,  
            nb_Conv=num_convs,   
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    
    def __repr__(self) -> str:
        return f"Decoder(in_dim={self.sc.conv1[0].conv.in_channels}, out_dim={self.nConv[-1].conv.out_channels})"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feature enhancement, upsampling, and convolution refinement.
        """
        identity = x
        x = self.sc(x)
        x = x + identity
        
        x = self.upsample(x)
        
        return self.nConv(x)
