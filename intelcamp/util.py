import sys
import time
from datetime import datetime
from functools import reduce
import torch
import numpy as np

import os
from pathlib import Path


start = time.time()
lasttime = time.time()


def prtime(*args, **kwargs):
    global lasttime
    print(
        " ".join(map(str, args)),
        "|time:",
        str(datetime.now()),
        "|",
        time.time() - start,
        "secs from start",
        time.time() - lasttime,
        "secs from last",
        **kwargs
    )
    lasttime = time.time()
    sys.stdout.flush()


def factors(n):
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(a, dim, order_index)
