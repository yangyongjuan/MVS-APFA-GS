import torch.nn as nn
from .utils import *
import copy
import math
import torch
import torch.nn.functional as F
import einops

##############-------------APFA-----##############################
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return
class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

from torchvision.ops import DeformConv2d, deform_conv2d
from torch.nn.modules.utils import _pair
class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1,bias=True):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        if bias==False:
            self.bias = None
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask=mask
        )
    
class FeatureNet(nn.Module):
    def __init__(self, base_channels=8):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
                Conv2d(3, base_channels, 3, 1, padding=1),
                Conv2d(base_channels, base_channels, 3, 1, padding=1))

        self.conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1))

        self.out1 = nn.Sequential(
                Conv2d(base_channels * 4, base_channels * 4, 1),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3,stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3,stride=1, padding=1))


        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Sequential(
                Conv2d(final_chs, final_chs, 3,1,padding=1),
                DCN(in_channels=final_chs, out_channels=final_chs,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=final_chs,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=base_channels * 2,kernel_size=3, stride=1, padding=1),
                                  )
        self.out3 = nn.Sequential(
                Conv2d(final_chs, final_chs, 3, 1, padding=1),
                DCN(in_channels=final_chs, out_channels=final_chs, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=final_chs, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=base_channels, kernel_size=3,stride=1, padding=1))

        self.out_channels = [4 * base_channels, base_channels * 2, base_channels]

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]
        :return outputs: stage1 [B, 32， 128， 160], stage2 [B, 16, 256, 320], stage3 [B, 8, 512, 640]
        """
        conv0 = self.conv0(x)      # torch.Size([4, 8, 512, 640])  x= torch.Size([4, 3, 512, 640])
        conv1 = self.conv1(conv0)  # torch.Size([4, 16, 256, 320])
        conv2 = self.conv2(conv1)  # torch.Size([4, 32, 128, 160])

        intra_feat = conv2         # torch.Size([4, 32, 128, 160])
        # outputs = {}
        feat2 = self.out1(intra_feat)   # torch.Size([4, 32, 128, 160])
        #outputs["stage1"] = out
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)    # torch.Size([4, 32, 256, 320])
        feat1 = self.out2(intra_feat)   # torch.Size([4, 16, 256, 320])
        #outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)   # torch.Size([4, 32, 512, 640])
        feat0 = self.out3(intra_feat)   # torch.Size([4, 8, 512, 640])
        #outputs["stage3"] = out

        return feat2,feat1,feat0

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(600, 600), temp_bug_fix=True):
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]
        # self.register_buffer('pe11', pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class FMT(nn.Module):
    def __init__(self, config):
        super(FMT, self).__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

        # self.pos_encoding = PositionEncodingSuperGule(config['d_model'])
        self.pos_encoding = PositionEncodingSine(config['d_model'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref"):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref": # only self attention layer

            assert self.d_model == ref_feature.size(1)
            _, _, H, _ = ref_feature.shape

            ref_feature = einops.rearrange(self.pos_encoding(ref_feature), 'n c h w -> n (h w) c')

            ref_feature_list = []
            for layer, name in zip(self.layers, self.layer_names): # every self attention layer
                if name == 'self':
                    ref_feature = layer(ref_feature, ref_feature)
                    ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
            return ref_feature_list

        elif feat == "src":

            assert self.d_model == ref_feature[0].size(1)
            _, _, H, _ = ref_feature[0].shape

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]

            src_feature = einops.rearrange(self.pos_encoding(src_feature), 'n c h w -> n (h w) c')

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    src_feature = layer(src_feature, src_feature)
                elif name == 'cross':
                    src_feature = layer(src_feature, ref_feature[i // 2])
                else:
                    raise KeyError
            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
        else:
            raise ValueError("Wrong feature name")

class FMT_with_pathway(nn.Module):
    def __init__(self,
            base_channels=8,
            FMT_config={
                'd_model': 32,
                'nhead': 8,
                'layer_names': ['self', 'cross'] * 4}):

        super(FMT_with_pathway, self).__init__()

        self.FMT = FMT(FMT_config)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


    def forward(self, features):
        """forward.

        :param features: multiple views and multiple stages features
        # 0=torch.Size([3, 32, 128, 160])  1=torch.Size([3, 16, 256, 320])   2=torch.Size([3, 8, 512, 640])
        """
        if len(features) !=3:
            raise ValueError("The features list must contain exactly 3 elements")
        feat1, feat2, feat3 = features
        ref_fea_t_list = self.FMT(feat1.clone(), feat="ref")
        feat1 = ref_fea_t_list[-1]
        feat1 = self.FMT([_.clone() for _ in ref_fea_t_list], feat1.clone(), feat="src")
        feat2 = self.smooth_1(self._upsample_add(self.dim_reduction_1(feat1), feat2))
        feat3 = self.smooth_2(self._upsample_add(self.dim_reduction_2(feat2), feat3))
        return feat1, feat2, feat3
    
#############----------------------end------------------------#####################



class CNNRender(nn.Module):
    def __init__(self, norm_act=nn.BatchNorm2d):
        super(CNNRender, self).__init__()
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act)
        self.conv1 = ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act)
        self.conv2 = nn.Conv2d(8, 16, 1)
        self.conv3 = nn.Conv2d(16, 3, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self._upsample_add(conv1, self.conv2(conv0))
        conv3 = self.conv3(conv2)
        return torch.clamp(conv3+x, 0., 1.)


class Unet(nn.Module):
    def __init__(self, in_channels, base_channels, norm_act=nn.BatchNorm2d):
        super(Unet, self).__init__()
        self.conv0 = nn.Sequential(
                        ConvBnReLU(in_channels, base_channels, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(base_channels, base_channels, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(base_channels, base_channels*2, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(base_channels*2, base_channels*2, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(base_channels*2, base_channels*4, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(base_channels*4, base_channels*4, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(base_channels*4, base_channels*4, 1)
        self.lat1 = nn.Conv2d(base_channels*2, base_channels*4, 1)
        self.lat0 = nn.Conv2d(base_channels, base_channels*4, 1)

        self.smooth0 = nn.Conv2d(base_channels*4, in_channels, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat0 = self.smooth0(feat0)
        return feat0
