"""UNeXt architectures.

Adapted from https://github.com/jeya-maria-jose/UNeXt-pytorch (MICCAI 2022).

Changes from upstream:
  * Dropped the unused `mmcv`, `torchvision`, `matplotlib` and `pdb` imports
    (upstream imports mmcv but never calls it; it is a painful build on Windows).
  * `timm.models.layers` -> `timm.layers` (upstream path is deprecated in timm >= 0.9).
  * Removed the dead module-level `shift()` function, which referenced undefined
    names (`xs`, `self`) and would raise if it were ever called.
  * `F.interpolate(..., mode='bilinear')` now passes `align_corners=False`
    explicitly to silence the torch warning and pin the behaviour.
  * Added `UNext_SE`: UNeXt + Squeeze-and-Excitation channel attention (the
    modification described in project_notes.md section 4.1).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from timm.layers import DropPath, to_2tuple, trunc_normal_

__all__ = ['UNext', 'UNext_S', 'UNext_SE', 'UNext_SkipFusion',
           'UNext_Wave', 'UNext_Boundary']


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def _init_weights_common(m):
    """Weight init shared by every block in the network (upstream behaviour)."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Squeeze: global average pool collapses each channel's HxW map to one scalar,
    giving a C-vector of global per-channel context.
    Excite: a bottleneck MLP (C -> C/r -> C) + sigmoid turns that into per-channel
    gates in [0, 1], which rescale the corresponding feature maps.

    Cost is negligible: the MLP acts on a C-vector, not on the feature map, so it
    adds 2*C*C/r parameters and effectively no FLOPs relative to the conv it follows.
    """

    def __init__(self, channels, reduction=16, identity_init=False):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

        if identity_init:
            # With default init the gates sit at sigmoid(~0) ~ 0.5, so inserting an SE
            # block into a *trained* network halves every activation and destroys it.
            # A high bias starts the gates near 1.0, making the block ~identity at
            # insertion; fine-tuning then moves it away from identity.
            #
            # Two failure modes to avoid, both measured:
            #   * bias too high (8.0 -> gate 0.9997): the sigmoid saturates and
            #     d(gate)/d(fc2.weight) collapses to ~1e-4, 172x smaller than at bias 4.
            #   * fc2.weight zeroed: the per-channel path has no gradient at all, so the
            #     block can only ever learn a *uniform* scale -- it cannot do the one
            #     thing SE exists for, which is to differentiate between channels.
            # bias 4.0 (gate 0.982) keeps insertion nearly lossless while leaving the
            # weights in a trainable regime.
            nn.init.normal_(self.fc2.weight, std=1e-3)
            nn.init.constant_(self.fc2.bias, 4.0)   # sigmoid(4) = 0.982

    def forward(self, x):
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))              # squeeze: (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))      # excite: gates in [0, 1]
        return x * w.view(b, c, 1, 1)       # channel-wise rescale


class SkipFusion(nn.Module):
    """Learned gate on a skip connection, replacing UNeXt's bare `torch.add`.

    Baseline UNeXt fuses encoder and decoder features with `out = out + t`, passing the
    encoder feature through untransformed. But encoder features at a given depth are
    local/low-level (edges, texture) while decoder features at the same resolution have
    already been shaped by the bottleneck's global context -- the "semantic gap". A raw
    add forces the decoder to accept all encoder detail, including noise, at fixed weight.

    This module predicts a spatial gate from the *concatenated* pair and uses it to
    modulate the encoder feature before adding:

        g   = sigmoid(conv1x1([decoder, encoder]))     # per-pixel, per-channel in [0,1]
        out = decoder + 2*g * encoder

    so the decoder decides, per location, how much encoder detail to admit. The factor 2
    keeps g=0.5 equivalent to the original add, which is what `identity_init` targets:
    zero weights + zero bias put every gate at exactly 0.5, reproducing the baseline
    bit-for-bit at insertion while leaving the conv in a trainable (unsaturated) regime.

    Cost is one 1x1 conv over 2C channels per skip -- a few hundred to a few thousand
    parameters, versus the 1.47M backbone.
    """

    def __init__(self, channels, identity_init=False):
        super().__init__()
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        if identity_init:
            # sigmoid(0) = 0.5, and 2 * 0.5 = 1.0 -> exactly the original `out + t`.
            # Unlike a saturated init this sits at the sigmoid's steepest point, so
            # gradients flow from the first step.
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, decoder_feat, encoder_feat):
        g = torch.sigmoid(self.gate(torch.cat([decoder_feat, encoder_feat], dim=1)))
        return decoder_feat + 2.0 * g * encoder_feat


class shiftmlp(nn.Module):
    """Shifted MLP: axial cyclic shift -> Linear -> DWConv -> GELU -> axial shift -> Linear.

    The two `torch.roll` passes (one along H, one along W) are pure indexing: they
    cost no parameters and no FLOPs, but they force each token to mix with its
    spatial neighbours before the Linear projection, giving the otherwise
    spatially-blind MLP a locality bias.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(_init_weights_common)

    def _shift(self, x, B, C, H, W, dim):
        """Pad, split into `shift_size` chunks, roll each by a different offset, recrop."""
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, dim)
                   for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        return x_s.reshape(B, C, H * W).contiguous().transpose(1, 2)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(self._shift(x, B, C, H, W, dim=2))   # shift along H
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        # after fc1 the channel count is hidden_features, not C
        C_hidden = x.shape[2]
        x = self.fc2(self._shift(x, B, C_hidden, H, W, dim=3))   # shift along W
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim,
                            act_layer=act_layer, drop=drop)
        self.apply(_init_weights_common)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    """Depth-wise 3x3 conv used inside the shifted MLP as a positional encoding."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to patch embedding via a strided overlapping convolution."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(_init_weights_common)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class HaarDWT(nn.Module):
    """Parameter-free 2D Haar discrete wavelet transform, as a stride-2 depthwise conv.

    Splits a C-channel feature map into the four classical sub-bands, each at half
    resolution:

        LL  local average        -- anatomical context, what the MLP should reason over
        LH  horizontal detail    -- top/bottom edges
        HL  vertical detail      -- left/right edges
        HH  diagonal detail      -- corners and speckle

    Implemented as a fixed (non-learnable) grouped convolution rather than via `pywt`
    so the transform stays on the GPU, inside autograd, and inside AMP. The kernels are
    registered as a buffer, so they carry no parameters and appear in no optimiser.

    The 0.5 scaling is the orthonormal Haar normalisation: it keeps the sub-band
    magnitudes on the same scale as the input, which matters because the LL band feeds
    a LayerNorm-free path where a factor-2 drift would compound across the two blocks.
    """

    def __init__(self):
        super().__init__()
        # [LL, LH, HL, HH] as 2x2 kernels.
        k = torch.tensor([
            [[1., 1.], [1., 1.]],      # LL
            [[1., 1.], [-1., -1.]],    # LH
            [[1., -1.], [1., -1.]],    # HL
            [[1., -1.], [-1., 1.]],    # HH
        ]) * 0.5
        self.register_buffer('kernel', k.unsqueeze(1))   # (4, 1, 2, 2)
        self._cache = {}          # (C, dtype) -> expanded bank; not part of state_dict

    def _bank(self, C, dtype):
        key = (C, dtype)
        w = self._cache.get(key)
        if w is None or w.device != self.kernel.device:
            w = self.kernel.repeat(C, 1, 1, 1).to(dtype)
            self._cache[key] = w
        return w

    def forward(self, x):
        B, C, H, W = x.shape
        # Odd sizes would silently drop the last row/column; pad first so the inverse
        # can restore the original extent exactly.
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        # One shared 2x2 filter bank applied to every channel independently. Cached:
        # rebuilding it per call measured slower than the convolution itself.
        w = self._bank(C, x.dtype)                       # (4C, 1, 2, 2)
        y = F.conv2d(x, w, stride=2, groups=C)           # (B, 4C, H/2, W/2)
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class HaarIWT(nn.Module):
    """Inverse of `HaarDWT` -- a stride-2 transposed conv with the same fixed kernels.

    Haar is orthonormal, so the synthesis filters are the analysis filters transposed;
    reconstruction is exact up to the padding `HaarDWT` may have added.
    """

    def __init__(self):
        super().__init__()
        k = torch.tensor([
            [[1., 1.], [1., 1.]],
            [[1., 1.], [-1., -1.]],
            [[1., -1.], [1., -1.]],
            [[1., -1.], [-1., 1.]],
        ]) * 0.5
        self.register_buffer('kernel', k.unsqueeze(1))
        self._cache = {}

    def _bank(self, C, dtype):
        key = (C, dtype)
        w = self._cache.get(key)
        if w is None or w.device != self.kernel.device:
            w = self.kernel.repeat(C, 1, 1, 1).to(dtype)
            self._cache[key] = w
        return w

    def forward(self, ll, lh, hl, hh, H, W):
        B, C = ll.shape[:2]
        y = torch.stack([ll, lh, hl, hh], dim=2).view(B, 4 * C, *ll.shape[-2:])
        w = self._bank(C, y.dtype)
        x = F.conv_transpose2d(y, w, stride=2, groups=C)
        return x[:, :, :H, :W]                          # undo any DWT padding


class wavemlp(nn.Module):
    """Wavelet-decomposed token mixer: a drop-in replacement for `shiftmlp`.

    UNeXt mixes tokens by cyclically shifting channel groups (`torch.roll`) before each
    Linear. That is cheap but frequency-blind: one rigid operator handles both the smooth
    interior of a lesion and its margin, so sharp boundary evidence is averaged together
    with low-frequency texture.

    Here the token mixing is done in the frequency domain instead:

        LL  -> the MLP (fc1 -> DWConv -> GELU -> fc2), at half resolution
        LH/HL/HH -> one dilated depthwise 3x3 each, dilation=2

    Routing the low band through the MLP keeps the global-context path intact while
    running it on 1/4 of the tokens; the detail bands get an operator whose only job is
    to sharpen margins, with a dilated kernel so a 3x3 still spans a 5x5 neighbourhood.
    The bands are then recombined by the inverse transform, so the block is shape- and
    dtype-preserving and slots in wherever `shiftmlp` did.

    Cost: the DWT/IWT are parameter-free, and the three detail convs are depthwise, so
    the added parameters are ~3*9*C -- negligible beside the 1.47M backbone, though not
    literally zero. The MLP itself now sees 4x fewer tokens, which buys back most of the
    added FLOPs.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., detail_dilation=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        self.dwt = HaarDWT()
        self.iwt = HaarIWT()

        # Low band: unchanged UNeXt MLP path, minus the roll.
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # Detail bands: one dilated depthwise conv each, kept separate because the
        # three orientations carry different evidence and should not share a filter.
        pad = detail_dilation
        self.detail = nn.ModuleList([
            nn.Conv2d(in_features, out_features, 3, padding=pad,
                      dilation=detail_dilation, groups=in_features, bias=True)
            for _ in range(3)
        ])

        self.apply(_init_weights_common)

        # Start each detail branch at zero so the block begins as "LL through the MLP,
        # details passed straight through", i.e. close to a plain low-pass MLP. This is
        # the same reasoning as SkipFusion's identity_init: begin near a known-good
        # configuration and let the detail path earn its contribution.
        for conv in self.detail:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x, H, W):
        B, N, C = x.shape
        xs = x.transpose(1, 2).view(B, C, H, W).contiguous()

        ll, lh, hl, hh = self.dwt(xs)
        h2, w2 = ll.shape[-2:]

        # --- low band through the MLP, at half resolution ---
        y = ll.flatten(2).transpose(1, 2)                # (B, h2*w2, C)
        y = self.fc1(y)
        y = self.dwconv(y, h2, w2)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        ll_out = y.transpose(1, 2).view(B, -1, h2, w2)

        # --- detail bands: sharpen, residually ---
        lh_out = lh + self.detail[0](lh)
        hl_out = hl + self.detail[1](hl)
        hh_out = hh + self.detail[2](hh)

        out = self.iwt(ll_out, lh_out, hl_out, hh_out, H, W)
        return out.flatten(2).transpose(1, 2)


class waveBlock(nn.Module):
    """`shiftedBlock` with the shifted MLP swapped for the wavelet mixer."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # NOTE: no self.apply(_init_weights_common) here. `shiftedBlock` does that,
        # but wavemlp zeroes its detail convs in its own __init__ and a blanket apply()
        # would re-randomise them, silently discarding the identity-style start. wavemlp
        # and norm2 already initialise themselves.
        self.mlp = wavemlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class BoundaryGate(nn.Module):
    """Skip gate conditioned on local spatial gradient -- a learned Laplacian.

    `SkipFusion` already lets the decoder decide, per pixel, how much encoder detail to
    admit; it predicts that gate from the *values* of the two feature maps. But the thing
    that makes a lesion margin worth admitting is not the feature value, it is the local
    *change* in feature value. A gate driven by values has to rediscover that from scratch.

    This module hands it the derivative directly. A fixed 3x3 Laplacian, applied
    depthwise to the encoder feature, gives a per-channel edge response; that response is
    concatenated with the usual (decoder, encoder) pair and drives the gate:

        e   = |laplacian(encoder)|                       # parameter-free, fixed kernel
        g   = sigmoid(conv1x1([decoder, encoder, e]))
        out = decoder + 2*g * encoder

    So the gate can open on anatomical boundaries and close on flat interior, which is
    exactly the failure mode UNeXt has on infiltrative margins. The Laplacian is fixed
    rather than learned because its job is to *supply evidence*, not to be tuned; the
    1x1 conv is free to ignore or invert it.

    `identity_init` keeps the same contract as `SkipFusion`: zero weights and bias put
    every gate at sigmoid(0)=0.5, and 2*0.5=1.0 reproduces the baseline `out + t`
    exactly, at the sigmoid's steepest point so gradients flow from step one.

    Cost is one 1x1 conv over 3C channels per skip -- around 1.5x SkipFusion's gate,
    still a few thousand parameters against a 1.47M backbone.
    """

    def __init__(self, channels, identity_init=False):
        super().__init__()
        self.gate = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=True)

        # 4-neighbour Laplacian, shared across channels, held as a buffer so it has no
        # parameters and never appears in the optimiser.
        lap = torch.tensor([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]])
        self.register_buffer('lap', lap.view(1, 1, 3, 3))

        if identity_init:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def _edges(self, x):
        C = x.shape[1]
        w = self.lap.repeat(C, 1, 1, 1).to(x.dtype)
        # replicate padding: zero padding would manufacture a strong artificial edge
        # around the whole border, which is precisely what the gate must not key on.
        xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(xp, w, groups=C).abs()

    def forward(self, decoder_feat, encoder_feat):
        e = self._edges(encoder_feat)
        g = torch.sigmoid(self.gate(torch.cat([decoder_feat, encoder_feat, e], dim=1)))
        return decoder_feat + 2.0 * g * encoder_feat


class UNext(nn.Module):
    """Conv stage (3 levels) + tokenized shifted-MLP stage (2 levels)."""

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 img_size=224, patch_size=16, in_chans=3, embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],
                 **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        ### Encoder — conv stage
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Encoder — tokenized MLP stage
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))

        return self.final(out)


class UNext_SE(UNext):
    """UNeXt + Squeeze-and-Excitation channel attention (project_notes.md section 4.1).

    An SE block is inserted after every conv block in both the encoder conv stage and
    the decoder, gating each stage's channels on globally-pooled context before the
    features are pooled/upsampled or added into a skip connection.

    Everything else — U-shape, channel widths, Tok-MLP internals, skip topology —
    is identical to `UNext`, so an ablation against the baseline isolates the effect
    of channel attention alone.
    """

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 se_reduction=16, se_identity_init=False, **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, **kwargs)

        ii = se_identity_init
        # encoder conv stage
        self.se1 = SEBlock(16, se_reduction, ii)
        self.se2 = SEBlock(32, se_reduction, ii)
        self.se3 = SEBlock(128, se_reduction, ii)

        # decoder conv stage
        self.dse1 = SEBlock(160, se_reduction, ii)
        self.dse2 = SEBlock(128, se_reduction, ii)
        self.dse3 = SEBlock(32, se_reduction, ii)
        self.dse4 = SEBlock(16, se_reduction, ii)

    def forward(self, x):
        B = x.shape[0]

        ### Encoder — conv stage, each block gated by SE before pooling
        out = F.relu(F.max_pool2d(self.se1(self.ebn1(self.encoder1(x))), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.se2(self.ebn2(self.encoder2(out))), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.se3(self.ebn3(self.encoder3(out))), 2, 2))
        t3 = out

        ### Encoder — tokenized MLP stage (unchanged)
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck (unchanged)
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder — each conv block gated by SE before the skip add
        out = F.relu(F.interpolate(self.dse1(self.dbn1(self.decoder1(out))),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dse2(self.dbn2(self.decoder2(out))),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dse3(self.dbn3(self.decoder3(out))),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dse4(self.dbn4(self.decoder4(out))),
                                   scale_factor=(2, 2), mode='bilinear', align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))

        return self.final(out)


class UNext_SkipFusion(UNext):
    """UNeXt with learned gated skip fusion (project_notes.md section 4.5).

    Identical to `UNext` except the four `torch.add(out, t)` skip merges are replaced by
    `SkipFusion` gates. With `skip_identity_init=True` the gates start at 0.5, which
    reproduces the baseline exactly -- so this can be grafted onto a trained UNeXt
    checkpoint and fine-tuned with only the gates unfrozen.
    """

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 skip_identity_init=False, **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, **kwargs)
        ii = skip_identity_init
        self.fuse4 = SkipFusion(160, ii)   # t4
        self.fuse3 = SkipFusion(128, ii)   # t3
        self.fuse2 = SkipFusion(32, ii)    # t2
        self.fuse1 = SkipFusion(16, ii)    # t1

    def forward(self, x):
        B = x.shape[0]

        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = self.fuse4(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = self.fuse3(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = self.fuse2(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = self.fuse1(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))

        return self.final(out)


class UNext_Wave(UNext):
    """UNeXt with the shifted-MLP token mixer replaced by the wavelet mixer.

    Only the four tokenized-MLP blocks change (block1/block2 in the encoder,
    dblock1/dblock2 in the decoder); the conv stage, the patch embeddings and the whole
    decoder path are inherited unchanged, so `forward` is exactly `UNext.forward`.
    """

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 embed_dims=[128, 160, 256], drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision,
                         embed_dims=embed_dims, drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)

        depths = kwargs.get('depths', [1, 1, 1])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        def mk(dim, dp):
            return nn.ModuleList([waveBlock(dim=dim, num_heads=1, mlp_ratio=1,
                                            drop=drop_rate, drop_path=dp,
                                            norm_layer=norm_layer)])

        self.block1 = mk(embed_dims[1], dpr[0])
        self.block2 = mk(embed_dims[2], dpr[1])
        self.dblock1 = mk(embed_dims[1], dpr[0])
        self.dblock2 = mk(embed_dims[0], dpr[1])


class UNext_Boundary(UNext):
    """UNeXt with boundary-conditioned gated skips (idea 4).

    Structurally identical to `UNext_SkipFusion` -- same four merge points, same
    identity-init contract -- but each gate additionally sees a fixed Laplacian edge
    response computed from the encoder feature. The ablation against `UNext_SkipFusion`
    is therefore clean: same gate topology, one extra input signal.
    """

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 skip_identity_init=False, **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, **kwargs)
        ii = skip_identity_init
        self.fuse4 = BoundaryGate(160, ii)   # t4
        self.fuse3 = BoundaryGate(128, ii)   # t3
        self.fuse2 = BoundaryGate(32, ii)    # t2
        self.fuse1 = BoundaryGate(16, ii)    # t1

    # The merge points are the only difference from UNext, and UNext_SkipFusion already
    # spells out that forward pass with `self.fuseN(out, tN)` in place of torch.add.
    forward = UNext_SkipFusion.forward


class UNext_S(nn.Module):
    """Narrower UNeXt variant (fewer channels throughout)."""

    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 img_size=224, patch_size=16, in_chans=3, embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],
                 **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(input_channels, 8, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2),
                                   mode='bilinear', align_corners=False))

        return self.final(out)
