import torch

import train.schesample as schesample


def test_sample_targets():
    for _ in range(100):
        gt = torch.rand(16, 3, 64, 64)
        pv = torch.rand(16, 3, 64, 64)
        decider = torch.randint(2, size=(16,))
        ta = schesample.sample_targets(pv, gt, decider)
        assert ta.size() == (16, 3, 64, 64)
        for b in range(16):
            if decider[b]:
                assert (ta[b] == pv[b]).all()
            else:
                assert (ta[b] == gt[b]).all()
