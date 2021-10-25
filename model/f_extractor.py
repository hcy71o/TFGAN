import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from omegaconf import OmegaConf
from torch.nn import Conv1d, Conv2d


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

'''
Input: Wideband spectrogram (B, 80, N/64)
Output: (B, 64, N/64)
Output: Weight-(B, N/64, ch*ch*ker_size*num_strides) (B, N/64, 32*32*3*3) / Bias - (B, N/64, out_ch*num_strides) (B, N/64, 32*3) (ex. stride = [8,4,2])
'''
class TExtractor(torch.nn.Module):
    def __init__(self, hp):
        super(TExtractor, self).__init__()

        hidden_channels = hp.td.hidden_channels
        self.lReLU_slope = hp.td.lReLU_slope
        self.num_layers = hp.td.num_layers
        self.conv_size = hp.td.conv_size
        self.input_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(hp.audio.n_mel_channels, hidden_channels[0], 5, padding=2, bias=True)),
            nn.ReLU(self.lReLU_slope)
        )

        #* Consists of 3 residual blocks that each block has two conv+lReLU
        self.conv_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0: 
                self.conv_blocks.append(
                    nn.Sequential(
                        nn.utils.weight_norm(Conv1d(hidden_channels[i], hidden_channels[i], self.conv_size, padding=(self.conv_size - 1) // 2)),
                        nn.LeakyReLU(self.lReLU_slope),
                        nn.utils.weight_norm(Conv1d(hidden_channels[i], hidden_channels[i], self.conv_size, padding=(self.conv_size - 1) // 2)),
                        nn.LeakyReLU(self.lReLU_slope)
                    ))
            else:
                self.conv_blocks.append(
                    nn.Sequential(
                        nn.utils.weight_norm(Conv1d(hidden_channels[i-1], hidden_channels[i], self.conv_size, padding=(self.conv_size - 1) // 2)),
                        nn.LeakyReLU(self.lReLU_slope),
                        nn.utils.weight_norm(Conv1d(hidden_channels[i], hidden_channels[i], self.conv_size, padding=(self.conv_size - 1) // 2)),
                        nn.LeakyReLU(self.lReLU_slope)
                    )
            )
        '''
        #* 3: number of downsampling
        TEx_weight_channels = self.hidden_channels*self.hidden_channels*self.ker_size*len(hp.td.strides)
        TEx_bias_channels = self.hidden_channels * len(hp.td.strides)

        #* for LVC kernels
        self.weight_conv = nn.utils.weight_norm(
            nn.Conv1d(self.hidden_channels, TEx_weight_channels, self.conv_size, padding=(self.conv_size - 1) // 2, bias=True))
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, self.conv_size, padding=(self.conv_size - 1) // 2, bias=True))

        #* for conditioning final downsampled sequence
        self.post_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1, padding=0, bias=True)),
            nn.ReLU(self.LReLU_slope)
        )
        '''
        
    #TODO: 네트워크 설계는 다 해놨고, 그냥 Spec을 받을지 Mel spec을 받을지 확인, (코드는 멜 받는걸로 되어 있음, Conv hyperparams를 mel 차원으로 해놨음) 아마 그냥 Spec 쓰는 게 나을 듯?.
    #TODO: 아래부터 계속하기
    def forward(self, c):
        '''
        c: Wideband spec. (batch, cond_channels, cond_length)
        '''

        c = self.input_conv(c)

        for i, l in enumerate(self.conv_blocks):
            l.to(c.device)
            if i == 0: 
                c = c + l(c)
            else:
                c = l(c)
        '''
        weight = self.weight_conv(c)
        bias = self.bias_conv(c)
        proj_c = self.post_conv(c)
        weight = weight
        '''
        return c

#! receptive field: 
class FExtractor(torch.nn.Module):
    def __init__(self, hp):
        super(FExtractor, self).__init__()

        dilations = hp.fd.dilations
        LReLU_slope = hp.fd.lReLU_slope
        conv_size = hp.fd.conv_size
        
        self.convs = nn.ModuleList([
            nn.utils.spectral_norm(Conv2d(1, 32, (conv_size, 1), dilation=(1, 1), padding=(get_padding(conv_size, 1), 0))),
            nn.utils.weight_norm(Conv2d(32, 64, (conv_size, 1), dilation=(2, 1), padding=(get_padding(conv_size, 2), 0))),
            nn.utils.weight_norm(Conv2d(64, 128, (conv_size, 1), dilation=(4, 1), padding=(get_padding(conv_size, 4), 0))),
            nn.utils.weight_norm(Conv2d(128, 256, (conv_size, 1), dilation=(8, 1), padding=(get_padding(conv_size, 8), 0))),
            nn.utils.weight_norm(Conv2d(256, 512, (conv_size, 1), dilation=(16, 1), padding=(get_padding(conv_size, 16), 0))),
            nn.utils.weight_norm(Conv2d(512, 512, (11, 1), padding=(get_padding(11, 1), 0)))
        ])
        self.post = nn.utils.weight_norm(Conv2d(512,1, (11, 1), padding=(get_padding(11,1),0)))

        self.act = nn.LeakyReLU(LReLU_slope)
   
    def forward(self, c):
        
        b, d, n = c.shape
        c = c.unsqueeze(1) # (b, 1, 80, N/256)

        for l in self.convs:
            c = l(c)
            c = self.act(c)
        c = self.post(c)
        c = c.squeeze(1)

        return c

if __name__ == '__main__':
    hp = OmegaConf.load('../config/default_c32.yaml')
    t_ex = TExtractor(hp)
    f_ex = FExtractor(hp)

    narrow = torch.randn(1, 80, 32)
    wide = torch.randn(1, 80, 128)
    print("TExtractor output:", t_ex(wide).shape)
    print("FExtractor output:", f_ex(narrow).shape)
    # assert y.shape == torch.Size([3, 1, 2560])

    T_total_params = sum(p.numel() for p in t_ex.parameters() if p.requires_grad)
    F_total_params = sum(p.numel() for p in f_ex.parameters() if p.requires_grad)
    print(T_total_params, F_total_params)

    