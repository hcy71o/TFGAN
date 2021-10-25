import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d, Conv2d

'''
input: Feature extracted from condition (B, cond_channel, cond_length)
'''
class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions'''
    def __init__(
            self,
            cond_channels,
            conv_in_channels,
            conv_out_channels,
            conv_layers,
            conv_kernel_size=3,
        ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int): number of layers
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers   # l_w
        kpnet_bias_channels = conv_out_channels * conv_layers                                           # l_b

        self.kernel_conv = nn.utils.weight_norm(
            nn.Conv1d(cond_channels, kpnet_kernel_channels, 1, bias=True))
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(cond_channels, kpnet_bias_channels, 1, bias=True))
        
    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        '''
        batch, _, cond_length = c.shape

        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length,
        )
        bias = b.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_out_channels,
            cond_length,
        )

        return kernels, bias


'''
Assume) It is applied to time-D.
input waveform x: (B, 64, 8192)
LVC weights: (B, 64, 64, ker_size, 128)
LVC bias: (B, 64, 128)
'''
class LVC_ConvBlock(nn.Module):
    def __init__(self, dilation, conv_size, hidden_dim, lReLU_slope = 0.2):
        super(LVC_ConvBlock, self).__init__()
        '''
        dilations, conv_size : for dilated convolution
        ker_size: for LVC
        '''
        self.conv_layer = weight_norm(Conv1d(hidden_dim, hidden_dim, conv_size, dilation = dilation, padding =int((conv_size*dilation - dilation)/2)))
        self.act = nn.LeakyReLU(lReLU_slope)


    def forward(self, x, weight, bias):

        hop_size = x.shape[-1]//weight.shape[-1] #* 8192//128 = 64

        x = self.location_variable_convolution(x, weight, bias, hop_size = hop_size)
        x = self.act(x)
        x = self.conv_layer(x)
        x = self.act(x)
        return x

    
    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        ''' 
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length). 
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length) 
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length) 
            dilation (int): the dilation of convolution. 
            hop_size (int): the hop_size of the conditioning sequence. 
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)     # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)   # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation, dilation)     # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]          
        x = x.transpose(3, 4)                   # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)  
        x = x.unfold(4, kernel_size, 1)         # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o


class TDiscriminator(nn.Module):
    def __init__(self, hp):
        super(TDiscriminator, self).__init__()


        hidden_dim = hp.tdisc.hidden_dim
        dilations = hp.tdisc.dilations
          
        self.kernel_predictor = KernelPredictor(512, hidden_dim, hidden_dim, len(dilations), hp.tdisc.ker_size)
        self.preconv = weight_norm(Conv1d(1, hidden_dim, 1, padding = 0))
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            self.blocks.append(LVC_ConvBlock(dilation, hp.tdisc.conv_size, hp.tdisc.hidden_dim, hp.tdisc.ker_size))
        
        self.act = nn.LeakyReLU(hp.tdisc.lReLU_slope)
        self.postconv = weight_norm(Conv1d(hidden_dim, 1, 1, padding = 0))

        #TODO ParallelWaveGAN 출력 부분 어떤 활성화 함수 썼나 확인
        #TODO 1x1 convolution 맞게 구현했나 확인!
    
    def forward(self, x, c):
        '''
        x: (batch, in_channels, in_length). 
        c: Feature extracted from condition (B, cond_channel, cond_length)
        kernel : (batch, in_channel, out_channels, kernel_size, kernel_length) 
        bias:(batch, out_channels, kernel_length) 
        '''
        kernels, bias = self.kernel_predictor(c)
        x = self.preconv(x)
        output = self.act(x)

        for i, conv in enumerate(self.blocks):

            k = kernels[:, i, :, :, :, :]   # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]            # (B, 2 * c_g, cond_length)

            output = conv(output, k, b)
        
        output = self.postconv(output)

        return output

class FDiscriminator(nn.Module):
    def __init__(self, hp):
        super(FDiscriminator, self).__init__()
        hidden_dim = hp.fdisc.hidden_dim
        dilations = hp.fdisc.dilations
        
        self.kernel_predictor = KernelPredictor(hp.audio.n_mel_channels, hidden_dim, hidden_dim, len(dilations), hp.fdisc.ker_size)
        self.preconv = weight_norm(Conv1d(1, hidden_dim, 1, padding = 0))
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            self.blocks.append(LVC_ConvBlock(dilation, hp.fdisc.conv_size, hp.fdisc.hidden_dim, hp.fdisc.ker_size))
        
        self.act = nn.LeakyReLU(hp.fdisc.lReLU_slope)
        self.postconv = weight_norm(Conv1d(hidden_dim, 1, 1, padding = 0))
        
        #TODO ParallelWaveGAN 활성화 함수 썼나 확인 (카카오 기반 보코더 들도 확인)
        #TODO 1x1 convolution 맞게 구현했나 확인, 
        #TODO remove_weight_norm 코드 구현
    
    def forward(self, x, c):
        '''
        x: (batch, in_channels, in_length). 
        c: Feature extracted from condition (B, cond_channel, cond_length)
        kernel : (batch, in_channel, out_channels, kernel_size, kernel_length) 
        bias:(batch, out_channels, kernel_length) 
        '''
        kernels, bias = self.kernel_predictor(c)

        x = self.preconv(x)
        output = self.act(x)

        for i, conv in enumerate(self.blocks):

            k = kernels[:, i, :, :, :, :]   # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]            # (B, 2 * c_g, cond_length)

            output = conv(output, k, b)
        
        output = self.postconv(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.TD = TDiscriminator(hp)
        self.FD = FDiscriminator(hp)

    def forward(self, x, t_c, f_c):
        return self.TD(x,t_c), self.FD(x,f_c)



if __name__ == '__main__':
    
    hp = OmegaConf.load('../config/default_c32.yaml')
    model = Discriminator(hp)
    
    t_numlayers = len(hp.tdisc.dilations)
    t_hid = hp.tdisc.hidden_dim
    f_numlayers = len(hp.fdisc.dilations)
    f_hid = hp.fdisc.hidden_dim

    x = torch.randn(3, 1, 8192)
    print(x.shape)

    # t_kernels = torch.randn(3,t_numlayers,t_hid,t_hid,3,128)
    # t_bias = torch.randn(3,t_numlayers,t_hid,128)
    # f_kernels = torch.randn(3,f_numlayers,f_hid,f_hid,3,32)
    # f_bias = torch.randn(3,f_numlayers,f_hid,32)
    t_c = torch.randn(3,512,128) #* (b_s, time feature dim, time_feature_length)
    f_c = torch.randn(3,80,32) #* (b_s, freq feature dim, frequency_feature_length)
    
    t_output, f_output = model(x, t_c, f_c)
    print(t_output.shape, f_output.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
