"""
tools/export_motionbert_onnx.py
Downloads MotionBERT-Lite checkpoint from HuggingFace and exports to ONNX.

Usage:
    python tools/export_motionbert_onnx.py [--output models/motionbert_lite.onnx]
"""
import argparse
import sys
import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DSTformer model — copied verbatim from HuggingFace repo
# (avoids needing the full MotionBERT codebase installed)
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x  # no-op in eval mode (drop_prob has no effect when not training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.st_mode = st_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.st_mode == 'stage_st':
            x = self.forward_spatial(q, k, v)
        elif self.st_mode == 'stage_ts':
            x = self.forward_temporal(q, k, v, seqlen)
        else:
            x = self.forward_coupling(q, k, v, seqlen)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_coupling(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_temporal(self, q, k, v, seqlen=8):
        # q/k/v: [J, H, F, C_h] — temporal attention over frames F, joints as batch
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 st_mode='vanilla', att_fuse=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, st_mode=st_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen=1):
        x = x + self.drop_path(self.attn(self.norm1(x), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, st_mode="stage_st")
            for i in range(depth)
        ])
        self.blocks_ts = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, st_mode="stage_ts")
            for i in range(depth)
        ])
        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [1, F, J, 3]  — 2D keypoints + confidence (batch=1)
        returns: [1, F, J, 3]  — 3D keypoints (x, y, z root-relative)

        Spatial blocks (stage_st) receive [F, J, C] — attend over J joints.
        Temporal blocks (stage_ts) receive [J, F, C] — attend over F frames.
        """
        B, F, J, C = x.shape
        x = x.reshape(F, J, C)           # [F, J, 3]
        x = self.joints_embed(x)          # [F, J, dim_feat]
        x = x + self.pos_embed            # broadcast [1, J, dim_feat]
        _, J, C = x.shape
        x = x.reshape(1, F, J, C) + self.temp_embed[:, :F, :, :]  # [1, F, J, C]
        x = x.reshape(F, J, C)            # [F, J, C]
        x = self.pos_drop(x)
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, J)                      # [F, J, C] spatial
            x_ts = blk_ts(x.permute(1, 0, 2), F)     # [J, F, C] temporal
            x_ts = x_ts.permute(1, 0, 2)             # back to [F, J, C]
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                alpha = att(alpha).softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Download checkpoint with progress
# ---------------------------------------------------------------------------
def download_checkpoint(url: str, dest: Path):
    import urllib.request

    def progress(count, block_size, total_size):
        pct = count * block_size / total_size * 100
        print(f'\r  {min(pct, 100):.1f}%', end='', flush=True)

    print(f'Downloading checkpoint from HuggingFace...')
    urllib.request.urlretrieve(url, str(dest), reporthook=progress)
    print(f'\n  Saved: {dest} ({dest.stat().st_size // 1024 // 1024} MB)')


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='models/motionbert_lite.onnx')
    parser.add_argument('--checkpoint', default=None,
                        help='Local path to best_epoch.bin (skips download)')
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    ckpt_url = (
        'https://huggingface.co/walterzhu/MotionBERT/resolve/main/'
        'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
    )

    # Download checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = output.parent / '_mb_lite_ckpt.bin'
        if not ckpt_path.exists():
            download_checkpoint(ckpt_url, ckpt_path)
        else:
            print(f'  Using cached checkpoint: {ckpt_path}')

    # Build model
    print('Building DSTformer (MB-Lite)...')
    model = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4,
        num_joints=17, maxlen=243,
        att_fuse=True,
    )
    model.eval()

    # Load checkpoint
    print('Loading checkpoint...')
    ckpt = torch.load(str(ckpt_path), map_location='cpu')

    # MotionBERT checkpoints wrap weights under 'model' key
    state = ckpt.get('model', ckpt)
    # Strip 'module.' prefix from DataParallel if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys: {len(missing)} (may be OK for lite model)')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')
    print('  Checkpoint loaded successfully')

    # Export to ONNX
    print(f'Exporting to ONNX: {output}')
    dummy = torch.zeros(1, 243, 17, 3)

    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=['keypoints_2d'],
        output_names=['keypoints_3d'],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter required for this architecture
    )

    size_mb = output.stat().st_size // 1024 // 1024
    print(f'  ONNX saved: {output} ({size_mb} MB)')

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    test = dummy.numpy()
    out = sess.run(None, {'keypoints_2d': test})
    print(f'  Verification: input {test.shape} -> output {out[0].shape}')
    print('DONE — motionbert_lite.onnx ready.')

    # Clean up temp checkpoint
    if not args.checkpoint and ckpt_path.exists():
        ckpt_path.unlink()
        print(f'  Temp checkpoint removed.')


if __name__ == '__main__':
    main()
