# pylint: disable=no-self-use
from train.train_cdna import CDNATrainer


class TestCDNATrainer:
    def test_basic_run(self, tmpdir):
        tmpdir = str(tmpdir)
        trainer = CDNATrainer(
            1, 0, 5, 'MovingMNIST', (range(3), range(3, 6), range(6, 9)),
            2, 0.001, 2, 20, 'DSSIM', 1.0, 1.0)
        trainer.basedir = tmpdir
        _ = trainer.run()

    def test_basic_run_ss(self, tmpdir):
        tmpdir = str(tmpdir)
        trainer = CDNATrainer(
            1, 0, 5, 'MovingMNIST', (range(3), range(3, 6), range(6, 9)),
            2, 0.001, 2, 20, 'DSSIM', 1.0, 1.0, scheduled_sampling_k=900.0)
        trainer.basedir = tmpdir
        _ = trainer.run()
