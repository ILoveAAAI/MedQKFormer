import torch
import torch.nn as nn
from network.decoder import Decoder, Mask_Spike_Conv
from utils import Normal_ISF,_make_nConv,Patch_Embedding


class SDQK(nn.Module):
    """Spike-Decomposed Q-K Attention Module"""
    def __init__(self, in_dim):
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_dim, in_dim//2, kernel_size=3, 
                     stride=1, padding=1, groups=in_dim//2),
            nn.BatchNorm2d(in_dim//2)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(in_dim, in_dim//2, kernel_size=3, 
                     stride=1, padding=1, groups=in_dim//2),
            nn.BatchNorm2d(in_dim//2)
        )
        self.spike_layers = nn.ModuleList(
            [Normal_ISF() for _ in range(4)]
        )
        self.scale = in_dim**(-0.5)
        self.msc = Mask_Spike_Conv(in_dim)
        self.num_head = 4
        self.fpn = Mask_Spike_Conv(in_dim) 
        self.bn = nn.BatchNorm2d(in_dim)

    def forward_decomposing(self, x):
        B, C, H, W = x.shape
        q = self.spike_layers[0](self.conv_q(x))
        k = self.spike_layers[1](self.conv_k(x))
        return q, 1.0 - q, k, 1.0 - k

    def forward_mm_qk(self, q_p, q_n, k_p, k_n):
        B, C, H, W = q_p.shape
        N = H * W
        
        # Reshape for head-wise processing
        q_p = q_p.view(B, self.num_head, C//self.num_head, N)
        q_n = q_n.view(B, self.num_head, C//self.num_head, N)
        k_p = k_p.view(B, self.num_head, C//self.num_head, N)
        k_n = k_n.view(B, self.num_head, C//self.num_head, N)
        
        # Compute attention maps
        q_p_s = self.spike_layers[2](q_p.sum(dim=-2, keepdim=True))
        q_n_s = self.spike_layers[3](q_n.sum(dim=-2, keepdim=True))

        qk_s = q_p_s * k_p + q_n_s * k_n
        qk_o = q_p_s * k_n + q_n_s * k_p
        
        return (qk_s.view(B, C, H, W), 
                qk_o.view(B, C, H, W))

    def forward(self, x):
        q_p, q_n, k_p, k_n = self.forward_decomposing(x)
        attn_pp_nn, attn_pn_np = self.forward_mm_qk(q_p, q_n, k_p, k_n)
        attn = torch.cat([attn_pp_nn, attn_pn_np], dim=1)
        out = self.bn(self.msc(attn) + x)
        return self.fpn(out) + out

class SDQK_A(nn.Module):
    """SDQK Attention Block with Downsampling"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.patch = Patch_Embedding(in_dim, out_dim)
        self.SDQK = SDQK(out_dim)
        self.pool_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_dim)
        )
    
    def forward(self, x):
        x_patch = self.patch(x)
        out = self.SDQK(x_patch)
        return self.pool_conv(out)

class Net(nn.Module):
    """MedQKFormer Main Network"""
    def __init__(self, in_dim=3, embedding_dim=[32, 64, 128, 256], T=4):
        super().__init__()
        self.encoder_stages = nn.ModuleList()
        dims = [in_dim] + embedding_dim
        
        # Build encoder stages
        for i in range(len(embedding_dim)):
            self.encoder_stages.append(SDQK_A(dims[i], dims[i+1]))
        
        # Build decoder stages
        self.decoder_stages = nn.ModuleList()
        rev_dims = list(reversed(embedding_dim))
        
        for i in range(len(rev_dims)-1):
            self.decoder_stages.append(Decoder(rev_dims[i], rev_dims[i+1]))
        
        self.decoder_stages.append(Decoder(embedding_dim[0], embedding_dim[0]))
        
        # Prediction head
        self.head1 = nn.Sequential(
            nn.Conv2d(embedding_dim[0], 1, kernel_size=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # Feature extraction
        encoder_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)
        
        # Feature fusion
        decoder_features = []
        for i, decoder in enumerate(self.decoder_stages):
            if i == 0:
                d = decoder(encoder_features[-1])
            else:
                fused = encoder_features[len(encoder_features)-i-1] + decoder_features[-1]
                d = decoder(fused)
            decoder_features.append(d)
        
        return self.head1(decoder_features[-1])
