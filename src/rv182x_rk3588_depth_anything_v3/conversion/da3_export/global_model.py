#!/usr/bin/env python3
"""Export the DA3-BASE global backbone stage for a fixed view count."""

from pathlib import Path

import torch

from .common import (
    PATCH_SIZE,
    aligned_image_size,
    check_onnx,
    patch_position_getter_for_export,
)
from .global_bcad import convert_global_to_bcad


EMBED_DIM = 768


class DA3BackboneGlobalWrapper(torch.nn.Module):
    def __init__(self, model, views, image_size, start_block=4, out_layers=(5, 7, 9, 11)):
        super().__init__()
        self.backbone = model.backbone.pretrained
        self.views = views
        self.image_size = image_size
        self.start_block = start_block
        self.out_layers = tuple(out_layers)

    def _inject_camera_token(self, tokens):
        batch, views = tokens.shape[:2]
        if self.views == 1:
            camera_token = self.backbone.camera_token[:, :1].expand(batch, 1, -1).to(tokens.dtype)
            return torch.cat([camera_token.unsqueeze(2), tokens[:, :, 1:]], dim=2)
        ref_token = self.backbone.camera_token[:, :1].expand(batch, -1, -1)
        src_token = self.backbone.camera_token[:, 1:].expand(batch, views - 1, -1)
        camera_token = torch.cat([ref_token, src_token], dim=1).to(tokens.dtype)
        return torch.cat([camera_token.unsqueeze(2), tokens[:, :, 1:]], dim=2)

    def forward(self, local_tokens):
        x = local_tokens
        local_x = local_tokens
        pos, pos_nodiff = self.backbone._prepare_rope(
            1, self.views, self.image_size, self.image_size, x.device
        )
        outputs = []

        for index, block in enumerate(self.backbone.blocks):
            if index < self.start_block:
                continue
            if index < self.backbone.rope_start or self.backbone.rope is None:
                global_pos = local_pos = None
            else:
                global_pos, local_pos = pos_nodiff, pos

            if self.backbone.alt_start != -1 and index == self.backbone.alt_start:
                x = self._inject_camera_token(x)

            if (
                self.backbone.alt_start != -1
                and index >= self.backbone.alt_start
                and index % 2 == 1
            ):
                x = self.backbone.process_attention(x, block, "global", pos=global_pos)
            else:
                x = self.backbone.process_attention(x, block, "local", pos=local_pos)
                local_x = x

            if index not in self.out_layers:
                continue
            out_x = torch.cat([local_x, x], dim=-1) if self.backbone.cat_token else x
            if out_x.shape[-1] == self.backbone.embed_dim:
                out_x = self.backbone.norm(out_x)
            else:
                first = out_x[..., : self.backbone.embed_dim]
                second = self.backbone.norm(out_x[..., self.backbone.embed_dim :])
                out_x = torch.cat([first, second], dim=-1)
            outputs.append(out_x[..., 1 + self.backbone.num_register_tokens :, :])

        return tuple(outputs)


def export_global(model, views, image_size, output, device):
    patch_position_getter_for_export()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_output = output.with_name(f".{output.stem}_base.onnx")
    backbone_size = aligned_image_size(image_size)
    if backbone_size != image_size:
        print(f"aligning backbone image size: {image_size} -> {backbone_size}")
    wrapper = DA3BackboneGlobalWrapper(model, views, backbone_size).eval().to(device)
    tokens = (backbone_size // PATCH_SIZE) ** 2 + 1
    dummy = torch.rand(1, views, tokens, EMBED_DIM, device=device)

    try:
        with torch.no_grad():
            outputs = wrapper(dummy)
            print("PyTorch outputs:", [tuple(value.shape) for value in outputs])
            torch.onnx.export(
                wrapper,
                dummy,
                str(base_output),
                input_names=["local_tokens"],
                output_names=["feat_5", "feat_7", "feat_9", "feat_11"],
                opset_version=16,
                do_constant_folding=True,
                dynamic_axes=None,
                dynamo=False,
            )
        convert_global_to_bcad(base_output, output)
    finally:
        base_output.unlink(missing_ok=True)

    check_onnx(output)
    print(f"exported: {output}")
    return output
