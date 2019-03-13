SV2P reimplementation in pytorch
================================

This project reimplements SV2P network and its depended sub-networks in `PyTorch-1.0.0`.


How to run
----------

Denote the project home directory as `$PROJ_HOME`.

First of all, create an virtual environment named `rt`:

```bash
cd "$PROJ_HOME"
python3 -m virtualenv rt
. rt/bin/activate
pip install -r requirements.txt
```

Next, if you have [direnv](https://direnv.net/) installed, `cd` to `$PROJ_HOME` should automatically brings out prompt to `direnv allow` the `.envrc` file, which prepares necessary environment variables.
Otherwise, do `. "$PROJ_HOME/.envrc"`.

Under `$PROJ_HOME/experiments/cdna` there's a `main.py` as a launcher for CDNA-Net training.
By `python main.py --help`, it accepts an `ini` configuration file and an action.
For example,

```ini
[dataset]
dataset_name=MovingMNIST
indices=(range(8000), range(8000, 9950), range(9950, 10000))
in_channels=1
cond_channels=0

[train]
n_masks=2
batch_size=16
lr=0.001
max_epoch=10
seqlen=20
criterion_name=DSSIM

[train_device]
device=cuda
```

The key names are one-to-one corresponding to the positional/keyword argument names of `sv2p.cdna.CDNA.__init__`.
It should be fairly straightforward what each key means.

An example run:

```bash
python main.py my-run.ini train
```

will produces `my-run.ini.log` as log file and `runs-${TODAY_DATETIME}/` as checkpoint/statistics/visualization base directory.


Dataset
-------

To use dataset:

- MovingMNIST: download [MovingMNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/unsup_video_lstm.tar.gz) and put it under `$PROJ_HOME/data/MovingMNIST/` (make directory if not exists)
