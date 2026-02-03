import torch
import torch.fx as fx
import operator
from typing import List, Tuple, Callable


def get_trivial_ops_ranges(
    gm: fx.GraphModule, is_trivial_op: Callable[[fx.Node], bool]
) -> List[Tuple[int, int]]:
    nodes = list(gm.graph.nodes)
    ranges = []
    start_idx = None

    for i, node in enumerate(nodes):
        if is_trivial_op(node):
            if start_idx is None:
                start_idx = i
            continue

        if start_idx is not None:
            ranges.append((start_idx, i))
            start_idx = None

    if start_idx is not None:
        ranges.append((start_idx, len(nodes)))

    return ranges


def is_trivial_op(node: fx.Node) -> bool:
    """bool <- fx.Node # if node.target in torch.add, torch.relu"""
    trivial_targets = {
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        torch.relu,
        torch.nn.functional.relu,
    }
    return node.op == "call_function" and node.target in trivial_targets
