import torch
import torch.fx as fx

def down_spider(x):
    return x

def up_spider(x, y):
    return ()

fx.wrap(down_spider)
fx.wrap(up_spider)
