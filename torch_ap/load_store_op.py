import torch
import torch.fx as fx


@fx.wrap
def load(x, placeholder_name: str):
    return x


@fx.wrap
def store(x, output_idx: int | None):
    return x
