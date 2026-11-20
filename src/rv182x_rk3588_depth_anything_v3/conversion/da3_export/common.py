#!/usr/bin/env python3
"""Shared helpers for exporting the DA3-BASE deployment partitions."""

from pathlib import Path

import onnx
import torch


PATCH_SIZE = 14


def aligned_image_size(image_size):
    return ((image_size + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_da3_model(model_path, device):
    import json
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object
    path = Path(model_path)
    config = json.loads((path / 'config.json').read_text())
    model = create_object(OmegaConf.create(config['config']))
    state = load_file(str(path / 'model.safetensors'))
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible
    assert all(k.startswith('head.scratch.output_conv2_aux.') for k in incompatible.missing_keys), incompatible
    print('Unused auxiliary ray-head weights absent from official checkpoint:', incompatible.missing_keys)
    return model.eval().to(device)


def patch_position_getter_for_export():
    from depth_anything_3.model.dinov2.layers.rope import PositionGetter

    def position_getter(self, batch_size, height, width, device):
        del self
        y = torch.arange(height, device=device)
        x = torch.arange(width, device=device)
        yy = y.view(-1, 1).expand(height, width).reshape(-1)
        xx = x.view(1, -1).expand(height, width).reshape(-1)
        positions = torch.stack([yy, xx], dim=1)
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    PositionGetter.__call__ = position_getter


def patch_head_pos_embed_float32():
    import depth_anything_3.model.dualdpt as dualdpt
    import depth_anything_3.model.utils.head_utils as head_utils

    def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
        pos = pos.to(torch.float32).reshape(-1)
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega = 1.0 / (float(omega_0) ** (omega / (embed_dim / 2.0)))
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def position_grid_to_embed(pos_grid, embed_dim, omega_0=100):
        pos_grid = pos_grid.to(torch.float32)
        height, width, grid_dim = pos_grid.shape
        if grid_dim != 2:
            raise RuntimeError(f"unexpected position grid: {tuple(pos_grid.shape)}")
        pos_flat = pos_grid.reshape(-1, grid_dim)
        emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0)
        emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0)
        return torch.cat([emb_x, emb_y], dim=-1).view(height, width, embed_dim)

    head_utils.make_sincos_pos_embed = make_sincos_pos_embed
    head_utils.position_grid_to_embed = position_grid_to_embed
    dualdpt.position_grid_to_embed = position_grid_to_embed


def check_onnx(path):
    path = Path(path)
    onnx.checker.check_model(onnx.load(str(path), load_external_data=True))
