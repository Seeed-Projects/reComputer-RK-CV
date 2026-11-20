#!/usr/bin/env python3
"""Change Global ONNX tensors from [1,V,T,C] to [V,T,1,C]."""

from pathlib import Path

import onnx
from onnx import helper


def _shape(value_info):
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def convert_global_to_bcad(input_path, output_path):
    model = onnx.load(str(input_path), load_external_data=True)
    graph = model.graph
    if len(graph.input) != 1:
        raise ValueError(f"expected one Global input, got {len(graph.input)}")

    graph_input = graph.input[0]
    input_shape = _shape(graph_input)
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(
            f"expected {graph_input.name} to be [1,V,T,C], got {input_shape}"
        )

    input_name = graph_input.name
    restored_input = f"{input_name}_nvtc"
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == input_name:
                node.input[index] = restored_input
    graph_input.type.tensor_type.shape.ClearField("dim")
    for value in (input_shape[1], input_shape[2], input_shape[0], input_shape[3]):
        graph_input.type.tensor_type.shape.dim.add().dim_value = value

    input_transpose = helper.make_node(
        "Transpose",
        [input_name],
        [restored_input],
        perm=[2, 0, 1, 3],
        name=f"{input_name}_restore_nvtc",
    )

    output_transposes = []
    for graph_output in graph.output:
        output_shape = _shape(graph_output)
        if len(output_shape) != 4 or output_shape[0] != 1:
            raise ValueError(
                f"expected {graph_output.name} to be [1,V,T,C], got {output_shape}"
            )
        output_name = graph_output.name
        original_output = f"{output_name}_nvtc"
        for node in graph.node:
            for index, name in enumerate(node.output):
                if name == output_name:
                    node.output[index] = original_output
        graph_output.type.tensor_type.shape.ClearField("dim")
        for value in (output_shape[1], output_shape[2], output_shape[0], output_shape[3]):
            graph_output.type.tensor_type.shape.dim.add().dim_value = value
        output_transposes.append(
            helper.make_node(
                "Transpose",
                [original_output],
                [output_name],
                perm=[1, 2, 0, 3],
                name=f"{output_name}_to_vtbc",
            )
        )

    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.append(input_transpose)
    graph.node.extend(existing_nodes)
    graph.node.extend(output_transposes)

    onnx.checker.check_model(model)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path
