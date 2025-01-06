import torch
import torch.nn as nn
from ops.OSAG import OSAG
from ops.pixelshuffle import pixelshuffle_block
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torchvision.transforms.functional as T
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
# from    ops.layernorm import LayerNorm2d
from ops.OSA import Conv_PreNormResidual, Gated_Conv_FeedForward


class rec_block(nn.Module):
    def __init__(self):
        super(rec_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.layer1(x)


class deep_rec_dq(nn.Module):
    def __init__(self):
        super(deep_rec_dq, self).__init__()
        self.rec_block1 = rec_block()
        self.rec_block2 = rec_block()
        self.rec_block3 = rec_block()
        self.rec_block4 = rec_block()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)

    def segment1(self, x, lr, down, up):
        lr_temp = down(x)
        x = self.rec_block1(up(lr - lr_temp) + x)
        lr_temp = down(x)
        x = self.rec_block2(up(lr - lr_temp) + x)
        # lr_temp = down(x)
        # x = self.rec_block3(up(lr - lr_temp) + x)
        # lr_temp = down(x)
        # x = self.rec_block4(up(lr - lr_temp) + x)
        return x

    def forward(self, x, lr, down, up):
        resdual = x
        lr_temp = down(x)
        x = self.rec_block1(up(lr - lr_temp) + x)
        lr_temp = down(x)
        x = self.rec_block2(up(lr - lr_temp) + x)
        lr_temp = down(x)
        x = self.rec_block3(up(lr - lr_temp) + x)
        lr_temp = down(x)
        x = self.rec_block4(up(lr - lr_temp) + x)
        x = self.conv1(x + resdual)
        return x


class deep_rec_INV_Omni_dq(nn.Module):
    def __init__(self):
        super(deep_rec_INV_Omni_dq, self).__init__()
        self.rec_block1 = self.rec_block2 = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            # nn.PReLU(),
        )
        self.rec_block2 = nn.Sequential(
            nn.Conv2d(9, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            # nn.PReLU(),
        )
        self.rec_block3 = nn.Sequential(
            nn.Conv2d(12, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            # nn.PReLU(),
        )
        self.rec_block4 = nn.Sequential(
            nn.Conv2d(15, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            # nn.PReLU(),
        )
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x, lr, net):
        resdual = x
        lr_temp = net(x)
        lr_temp, LR_feature = (lr_temp.narrow(1, 0, 3), lr_temp.narrow(1, 3, lr_temp.shape[1] - 3))
        rand_noise = torch.randn(lr_temp.shape[0], LR_feature.shape[1], lr_temp.shape[2], lr_temp.shape[3]).cuda()
        x1 = self.rec_block1(torch.cat((net(torch.cat((lr - lr_temp, rand_noise), 1), rev=True), x), 1))

        lr_temp = net(x1)
        lr_temp, LR_feature = (lr_temp.narrow(1, 0, 3), lr_temp.narrow(1, 3, lr_temp.shape[1] - 3))
        x2 = self.rec_block2(torch.cat((net(torch.cat((lr - lr_temp, rand_noise), 1), rev=True), x, x1), 1))

        lr_temp = net(x2)
        lr_temp, LR_feature = (lr_temp.narrow(1, 0, 3), lr_temp.narrow(1, 3, lr_temp.shape[1] - 3))
        x3 = self.rec_block3(torch.cat((net(torch.cat((lr - lr_temp, rand_noise), 1), rev=True), x, x1, x2), 1))

        lr_temp = net(x3)
        lr_temp, LR_feature = (lr_temp.narrow(1, 0, 3), lr_temp.narrow(1, 3, lr_temp.shape[1] - 3))
        x4 = self.rec_block4(torch.cat((net(torch.cat((lr - lr_temp, rand_noise), 1), rev=True), x, x1, x2, x3), 1))

        # x = self.conv1(x + resdual)
        return x1,x2,x3,x4


class deep_rec(nn.Module):
    def __init__(self):
        super(deep_rec, self).__init__()
        self.rec_block1 = rec_block()
        self.rec_block2 = rec_block()
        self.rec_block3 = rec_block()
        self.rec_block4 = rec_block()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)

    def segment1(self, x):
        x = self.rec_block1(x)
        x = self.rec_block2(x)
        return x

    def forward(self, x):
        resdual = x
        x.requires_grad_(True)
        x = checkpoint(self.segment1, x, use_reentrant=True)
        # x = checkpoint(self.segment2, x, use_reentrant=True)
        # x = checkpoint(self.segment3, x, use_reentrant=True)
        # x = checkpoint(self.segment4, x, use_reentrant=True)
        # x = self.rec_block1(x)
        # x = self.rec_block2(x)
        x = self.rec_block3(x)
        x = self.rec_block4(x)
        x = self.conv1(x + resdual)
        return x


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.PReLU(),
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.layer1(x) + x)


class denoise(nn.Module):
    def __init__(self):
        super(denoise, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.PReLU(),
        )
        self.resblock1 = res_block()
        self.resblock2 = res_block()
        self.resblock3 = res_block()
        self.resblock4 = res_block()
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        resdual = x
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.conv2(x + resdual)
        return x


class dense_block(nn.Module):
    def __init__(self):
        super(dense_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.layer1(x) + x


class deep_rec_reuse(nn.Module):
    def __init__(self):
        super(deep_rec_reuse, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 3, 1, 1)
        self.rec_block1 = res_block()
        self.rec_block2 = res_block()
        self.rec_block3 = res_block()
        self.rec_block4 = res_block()
        # self.denoise=denoise()
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv1(x)
        resdual = x
        # lr_temp = down(x)
        # lr=self.denoise(lr)
        x = self.rec_block1(x)
        x = self.rec_block2(x)
        x = self.rec_block3(x)
        x = self.rec_block4(x)
        x = self.conv2(x + resdual)
        return x


class Channel_Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, padding=1 // 2, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps,
                                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h // self.ps, w=w // self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out


# class myformer(nn.Module):
#     def __init__(self,dim,heads,bias=False,dropout=0.,window_size=7):
#         super(myformer,self).__init__()
#         self.channel_attention=Channel_Attention(dim,heads,bias=False,dropout=0.,window_size=7)


class deep_rec_dense(nn.Module):
    def __init__(self):
        super(deep_rec_dense, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, 3, 1, 1)
        self.rec_block1 = nn.Sequential(
            dense_block()
        )
        self.rec_block2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.rec_block3 = nn.Sequential(
            nn.Conv2d(192, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.rec_block4 = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x1, x2, x3):
        x = self.conv1(torch.cat((x1, x2, x3), 1))
        resdual = x
        x1 = self.rec_block1(x)
        x2 = self.rec_block2(torch.cat((resdual, x1), 1))
        x3 = self.rec_block3(torch.cat((resdual, x1, x2), 1))
        x = self.rec_block4(torch.cat((resdual, x1, x2, x3), 1))
        x = self.conv2(x + resdual)
        return x


class deep_rec_dense_attention_3in(nn.Module):
    def __init__(self):
        super(deep_rec_dense_attention_3in, self).__init__()
        self.up_scale = 4
        self.window_size = 4
        self.dim = 9
        self.channel_attention1 = nn.Sequential(
            Conv_PreNormResidual(self.dim,
                                 Channel_Attention(dim=self.dim, heads=1, dropout=False, window_size=self.window_size)),
            Conv_PreNormResidual(self.dim, Gated_Conv_FeedForward(dim=self.dim, dropout=False)),

            Conv_PreNormResidual(self.dim,
                                 Channel_Attention(dim=self.dim, heads=1, dropout=False, window_size=self.window_size)),
            Conv_PreNormResidual(self.dim, Gated_Conv_FeedForward(dim=self.dim, dropout=False)),
        )
        self.conv1 = nn.Conv2d(9, 64, 3, 1, 1)
        self.rec_block1 = nn.Sequential(
            dense_block()
        )
        self.rec_block2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.rec_block3 = nn.Sequential(
            nn.Conv2d(192, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.rec_block4 = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x1, x2, x3):
        # H, W = x1.shape[2:]
        # x=self.check_image_size(torch.cat((x1, x2, x3), 1))
        x = self.channel_attention1(torch.cat((x1, x2, x3), 1))
        x = self.conv1(x)
        # x=self.prelu(x)
        resdual = x
        x1 = self.rec_block1(x)
        x2 = self.rec_block2(torch.cat((resdual, x1), 1))
        x3 = self.rec_block3(torch.cat((resdual, x1, x2), 1))
        x = self.rec_block4(torch.cat((resdual, x1, x2, x3), 1))
        x = self.conv2(x + resdual)
        # x = x[:, :, :H, :W]
        return x


class deep_rec_dense_4in(nn.Module):
    def __init__(self):
        super(deep_rec_dense_4in, self).__init__()
        self.window_size = 4
        self.dim = 12
        self.channel_attention1 = nn.Sequential(
            Conv_PreNormResidual(12,
                                 Channel_Attention(dim=self.dim, heads=2, dropout=False, window_size=self.window_size)),
            Conv_PreNormResidual(12, Gated_Conv_FeedForward(dim=self.dim, dropout=False)),
        )
        self.channel_attention2 = nn.Sequential(
            Conv_PreNormResidual(12,
                                 Channel_Attention(dim=self.dim, heads=2, dropout=False, window_size=self.window_size)),
            Conv_PreNormResidual(12, Gated_Conv_FeedForward(dim=self.dim, dropout=False)),
        )
        self.conv1 = nn.Conv2d(12, 64, 3, 1, 1)

        self.rec_block1 = nn.Sequential(
            dense_block()
        )
        self.rec_block2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.rec_block3 = nn.Sequential(
            nn.Conv2d(192, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.rec_block4 = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, 0),
            nn.PReLU(),
            dense_block(),
        )
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.channel_attention1(x)
        x = self.channel_attention2(x)
        x = self.conv1(x)
        resdual = x
        x1 = self.rec_block1(x)
        x2 = self.rec_block2(torch.cat((resdual, x1), 1))
        x3 = self.rec_block3(torch.cat((resdual, x1, x2), 1))
        x = self.rec_block4(torch.cat((resdual, x1, x2, x3), 1))
        x = self.conv2(x + resdual)
        return x
