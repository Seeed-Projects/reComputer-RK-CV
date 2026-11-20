#!/usr/bin/env python3
"""Export the fixed-resolution, single-view DA3-BASE local backbone stage."""

from pathlib import Path

import torch
import torch.nn.functional as F

from .common import (
    PATCH_SIZE,
    aligned_image_size,
    check_onnx,
    patch_position_getter_for_export,
)


class DA3BackboneLocalWrapper(torch.nn.Module):
    def __init__(self, model, image_size, stop_block=4):
        super().__init__()
        self.backbone = model.backbone.pretrained
        self.stop_block = stop_block
        self.pad_size = aligned_image_size(image_size) - image_size

    def forward(self, images):
        if self.pad_size:
            images = F.pad(images, (0, self.pad_size, 0, self.pad_size))
        x = images.unsqueeze(0)
        _, _, _, height, width = x.shape
        tokens = self.backbone.prepare_tokens_with_masks(x)
        pos, _ = self.backbone._prepare_rope(1, 1, height, width, x.device)

        for index, block in enumerate(self.backbone.blocks):
            if index >= self.stop_block:
                break
            local_pos = None
            if index >= self.backbone.rope_start and self.backbone.rope is not None:
                local_pos = pos
            tokens = self.backbone.process_attention(tokens, block, "local", pos=local_pos)

        return tokens[:, 0]


def export_local(model, image_size, output, device):
    patch_position_getter_for_export()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DA3BackboneLocalWrapper(model, image_size).eval().to(device)
    dummy = torch.rand(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        result = wrapper(dummy)
        expected_tokens = (aligned_image_size(image_size) // PATCH_SIZE) ** 2 + 1
        if result.shape[1] != expected_tokens:
            raise RuntimeError(
                f"unexpected token count: got {result.shape[1]}, expected {expected_tokens}"
            )
        print("PyTorch output:", tuple(result.shape))
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["images"],
            output_names=["local_tokens"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    check_onnx(output)
    print(f"exported: {output}")
    return output
