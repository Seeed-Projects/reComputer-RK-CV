#!/usr/bin/env python3
"""Export the fixed-resolution, fixed-view DA3-BASE DualDPT head."""

from pathlib import Path

import torch

from .common import (
    PATCH_SIZE,
    aligned_image_size,
    check_onnx,
    patch_head_pos_embed_float32,
)


HEAD_DIM = 1536


class DA3HeadWrapper(torch.nn.Module):
    def __init__(self, model, image_size, chunk_size=None):
        super().__init__()
        self.head = model.head
        self.image_size = image_size
        self.internal_size = aligned_image_size(image_size)
        self.chunk_size = chunk_size

    def forward(self, feat_5, feat_7, feat_9, feat_11):
        feat_5 = feat_5.unsqueeze(0)
        feat_7 = feat_7.unsqueeze(0)
        feat_9 = feat_9.unsqueeze(0)
        feat_11 = feat_11.unsqueeze(0)
        feats = ((feat_5, None), (feat_7, None), (feat_9, None), (feat_11, None))
        kwargs = {}
        if self.chunk_size is not None:
            kwargs["chunk_size"] = self.chunk_size
        output = self.head(
            feats,
            self.internal_size,
            self.internal_size,
            patch_start_idx=0,
            **kwargs,
        )
        return (
            output.depth[..., : self.image_size, : self.image_size],
            output.depth_conf[..., : self.image_size, : self.image_size],
        )


def configure_resize_half_pixel(head):
    import depth_anything_3.model.dualdpt as dualdpt
    from depth_anything_3.model.utils import head_utils

    def interpolate_half_pixel(
        x, size=None, scale_factor=None, mode="bilinear", align_corners=True
    ):
        del align_corners
        return head_utils.custom_interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )

    dualdpt.custom_interpolate = interpolate_half_pixel
    for module in head.modules():
        if hasattr(module, "align_corners"):
            module.align_corners = False


def export_head(model, views, image_size, output, device):
    patch_head_pos_embed_float32()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DA3HeadWrapper(model, image_size, chunk_size=views).eval().to(device)
    configure_resize_half_pixel(wrapper.head)
    patch_tokens = (aligned_image_size(image_size) // PATCH_SIZE) ** 2
    dummy = tuple(
        torch.rand(views, patch_tokens, HEAD_DIM, device=device)
        for _ in range(4)
    )

    with torch.no_grad():
        results = wrapper(*dummy)
        print("PyTorch outputs:", [tuple(value.shape) for value in results])
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["feat_5", "feat_7", "feat_9", "feat_11"],
            output_names=["depth", "depth_conf"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    check_onnx(output)
    print(f"exported: {output}")
    return output
