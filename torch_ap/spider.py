import torch
import torch.fx as fx


@fx.wrap
def down_spider(x):
    return x


@fx.wrap
def up_spider(x, y):
    return ()
