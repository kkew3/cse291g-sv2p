# pylint: disable=no-self-use
import torch
from train.train_cdna import decide_targets
from train.train_cdna import CDNATrainer


def test_decide_targets():
    gt = torch.rand(16, 3, 64, 64)
    pv = torch.rand(16, 3, 64, 64)
    decider = torch.randint(2, size=(16,))
    ta = decide_targets(gt, pv, decider)
    assert ta.size() == (16, 3, 64, 64)
    for b in range(16):
        if decider[b]:
            assert (ta[b] == pv[b]).all()
        else:
            assert (ta[b] == gt[b]).all()


class TestCDNATrainer:
    def test_basic_run(self, tmpdir):
        tmpdir = str(tmpdir)
        trainer = CDNATrainer(
            1, 0, 5, 'MovingMNIST', (range(3), range(3, 6), range(6, 9)),
            2, 0.001, 2, 20, 'DSSIM')
        trainer.basedir = tmpdir
        _ = trainer.run()

    def test_basic_run_ss(self, tmpdir):
        tmpdir = str(tmpdir)
        trainer = CDNATrainer(
            1, 0, 5, 'MovingMNIST', (range(3), range(3, 6), range(6, 9)),
            2, 0.001, 2, 20, 'DSSIM', scheduled_sampling='by_epoch')
        trainer.basedir = tmpdir
        _ = trainer.run()
