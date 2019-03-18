import torch

import sv2p.criteria as criteria


def test_rot90():
    a = torch.arange(9).reshape(1, 3, 3).float()
    b = torch.tensor([
        [2, 5, 8],
        [1, 4, 7],
        [0, 3, 6],
    ]).unsqueeze(0).float()
    b_ = criteria.rot90(a)
    assert (b == b_).all()

def test_rot90_batch():
    a = torch.rand(10, 5, 5)
    # assuming that single batch works
    b = torch.stack(list(map(criteria.rot90, a)))
    b_ = criteria.rot90(a)
    assert (b == b_).all()
