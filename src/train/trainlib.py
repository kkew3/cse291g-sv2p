"""
Common training utilities
"""
import inspect
import configparser
import contextlib
import os
import logging
from datetime import datetime
import collections
import typing

import numpy as np
import torch
import torch.nn as nn


T = typing.TypeVar('T')


def loggername(module_name, *args):
    """
    Get hierarchical logger name.

    Usage::

        .. code-block:: python

            loggername(__name__)
            loggername(__name__, self)
            loggername(__name__, 'function_name')
            loggername(__name__, self, 'method_name')
    """
    tokens = [module_name]
    if len(args) > 0:  # pylint: disable=len-as-condition
        if isinstance(args[0], str):
            tokens.append(args[0])
        else:
            tokens.append(type(args[0]).__name__)
    if len(args) > 1:
        tokens.append(args[1])
    return '.'.join(tokens)


def _l(*args):
    return logging.getLogger(loggername(__name__, *args))


_Predicate = typing.Callable[[typing.Any], bool]
_EBProgress = typing.Tuple[int, int]

def action_fired(fired: typing.Union[int, _Predicate]) -> _Predicate:
    """
    Returns a callable that returns a bool indicating whether an action should
    be performed given the progress of an ongoing task.

    :param fired: an int or the callable to be returned. If
           ``fired`` is an int, then ``fired`` will be initialized to
           ``lambda x: x % fired == 0``
    """
    logger = _l('action_fired')
    if isinstance(fired, int):
        if fired < 1:
            logger.warning('Expect `fired` at least 1 if int but got {}; '
                           'converted to 1'.format(fired))
            fired = 1

        def _fired(progress: int):
            return progress % fired == 0

        return _fired
    return fired


def fired_always(_) -> bool:
    """
    Always fire whatever the progress.
    """
    return True


def fired_batch(batch: int, progress: _EBProgress) -> bool:
    """
    Fire every other ``batch`` given progress in (epoch, batch) tuple.
    """
    return progress[1] % max(1, batch) == 0


class CheckpointSaver:
    """
    Save checkpoint periodically.
    """

    def __init__(self, net: nn.Module, savedir: str,
                 checkpoint_tmpl: str = 'checkpoint_{0}.pth',
                 fired: typing.Union[int, _Predicate] = 10):
        """
        :param net: the network to save states; the network should already been
               in the target device
        :param savedir: the directory under which to write checkpoints; if not
               exists, it will be created automatically
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        :param checkpoint_tmpl: the checkpoint file template
               (by ``str.format``), should accept exactly one positional
               argument the same as that passed to ``fired``
        """
        if not isinstance(net, nn.Module):
            raise TypeError('Expected torch.nn.Module of `net` but got: {}'
                            .format(type(net)))

        savedir = os.path.normpath(savedir)
        os.makedirs(savedir, exist_ok=True)

        self.net = net
        self.fired = action_fired(fired)
        self.checkpoint_tmpl = checkpoint_tmpl
        self.savedir = savedir

    def __call__(self, progress) -> typing.Optional[str]:
        """
        Save checkpoint as needed.

        :param progress: see ``help(type(self).__init__)``
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            _logger = _l(self)
            name = self.checkpoint_tmpl.format(*progress)
            tofile = os.path.join(self.savedir, name)
            torch.save(self.net.state_dict(), tofile)
            _logger.debug('{} written at progress {}'
                          .format(tofile, progress))
        else:
            tofile = None
        return tofile


def load_checkpoint(net: nn.Module, savedir: str, checkpoint_tmpl: str,
                    progress: tuple, map_location: str = 'cpu') -> None:
    """
    Load checkpoint from file.

    :param net: network to load checkpoint
    :param savedir: directory under which checkpoints are saved
    :param checkpoint_tmpl: basename template of the checkpoint files
    :param progress: which checkpoint file to load
    :param map_location: where to load the weights
    """
    logger = _l('load_checkpoint')
    basename = checkpoint_tmpl.format(*progress)
    fromfile = os.path.join(savedir, basename)
    state_dict = torch.load(fromfile, map_location=map_location)
    net.to(map_location)
    net.load_state_dict(state_dict)
    logger.debug('progress {} loaded from {}'
                 .format(progress, fromfile))


class StatSaver:
    """
    Save (scalar) statistics periodically as npz file.
    """

    def __init__(self, statdir: str,
                 statname_tmpl='stats_{0}.npz',
                 fired: typing.Union[int, _Predicate] = 10):
        """
        :param statdir: the directory under which to write statistics npz
               files; if not exists, it will be created automatically
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        :param statname_tmpl: the stat npz file basename template
               (by ``str.format``), should accept exactly one positional
               argument the same as that passed to ``fired``
        """
        statdir = os.path.normpath(statdir)
        os.makedirs(statdir, exist_ok=True)

        self.fired = action_fired(fired)
        self.statname_tmpl = statname_tmpl
        self.statdir = statdir

    def __call__(self, progress, **stat_dict):
        """
        Save statistics, which are wrapped into numpy arrays, as needed.

        :param progress: see ``help(type(self).__init__)``
        :param stat_dict: dict of floats or lists of floats, i.e. the
               non-cumulative statistics at the progress
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            _logger = _l(self)
            name = self.statname_tmpl.format(*progress)
            tofile = os.path.join(self.statdir, name)
            _stat_dict = {k: np.array(stat_dict[k]) for k in stat_dict}
            np.savez(tofile, **_stat_dict)
            _logger.debug('{} written at progress "{}"'
                          .format(tofile, progress))
        else:
            tofile = None
        return tofile


class FieldChangedError(BaseException):
    def __init__(self, original: typing.Sequence[str],
                 now: typing.Sequence[str]):
        super().__init__('Fields changed from {} to {}'
                         .format(original, now))


class CsvStatSaver:
    """
    Save scalar statistics as CSV files. To simplify design and
    implementation, the ``progress`` is assumed to be 2-tuples of integers
    ``(epoch, batch)``. The naming template (``str.format``) of the CSV files
    should contain one argument for the epoch. Within each file, the first
    column will be the batch id. The rest columns will be the scalar
    statistics at each batch.
    """

    def __init__(self, statdir: str,
                 statname_tmpl: str = 'stats_{0}.csv',
                 fired: typing.Union[int, _Predicate] = 10):
        """
        :param statdir: the directory under which to write statistics npz
               files; if not exists, it will be created automatically
        :param statname_tmpl: the base filename of the CSV file
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        """
        statdir = os.path.normpath(statdir)
        os.makedirs(statdir, exist_ok=True)

        self.fired = action_fired(fired)
        self.statdir = statdir
        self.statename_tmpl = statname_tmpl
        self.fields = None

    def __call__(self, progress, **stat_dict):
        """
        Save statistics.
        Once instantiated, the keys of ``stat_dict`` must not be changed,
        otherwise raising ``trainlib.FieldChangedError``

        :param progress: see ``help(type(self).__init__)``
        :param stat_dict: dict of floats i.e. the non-cumulative
               *scalar* statistics at the progress
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            _logger = _l(self)
            epoch, batch = progress
            name = self.statename_tmpl.format(epoch)
            tofile = os.path.join(self.statdir, name)
            if self.fields is None:
                self.fields = sorted(stat_dict)
            elif self.fields != sorted(stat_dict):
                raise FieldChangedError(self.fields, sorted(stat_dict))
            values = [stat_dict[k] for k in self.fields]
            try:
                open(tofile).close()
            except FileNotFoundError:
                with open(tofile, 'w') as outfile:
                    # write the CSV header
                    outfile.write(','.join([''] + self.fields) + '\n')
                    outfile.write(','.join(map(str, [batch] + values)) + '\n')
                    _logger.debug('{} written at progress "{}"'
                                  .format(tofile, progress))
            else:
                with open(tofile, 'a') as outfile:
                    outfile.write(','.join(map(str, [batch] + values)) + '\n')
                    _logger.debug('{} appended at progress "{}"'
                                  .format(tofile, progress))
        else:
            tofile = None
        return tofile


def load_stat(statdir: str, statname_tmpl: str, progress: tuple,
              key: str = None) \
        -> typing.Union[typing.Dict[str, np.ndarray], np.ndarray]:
    """
    Load statistics dumped by ``StatSaver``.

    :param statdir: directory under which the statistics are saved
    :param statname_tmpl: stat file basename template
    :param progress: which stat file to load
    :param key: which field to load
    :return: the npz content dict if ``key`` is not specified, otherwise the
             corresponding field data
    """
    logger = _l('load_stat')
    basename = statname_tmpl.format(*progress)
    fromfile = os.path.join(statdir, basename)
    data = np.load(fromfile)
    logger.debug('progress {} loaded from "{}"'
                 .format(progress, fromfile))
    if key is None:
        data = {k: data[k] for k in data.keys()}
    else:
        data = data[key]
    return data


def freeze_model(model: nn.Module) -> typing.Dict[str, bool]:
    """
    Freeze the model parameters.

    :param model:
    :return: the original ``requires_grad`` attributes of the model parameters
    """
    origrg = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    return origrg


def melt_model(model: nn.Module, origrg: typing.Dict[str, bool],
               empty_after_melting=False) -> None:
    """
    Melt the model parameters into their original ``requires_grad`` states.

    :param model:
    :param origrg: dict mapping ``parameter_name`` to
           ``original_requires_grad``
    :param empty_after_melting: if ``True``, make ``origrg`` empty at the
           end of this function
    """
    for n, p in model.named_parameters():
        p.requires_grad_(origrg[n])
    if empty_after_melting:
        origrg.clear()


class BasicTrainer:
    """
    Abstract class of a basic trainer that codes the general framework of
    training a network. The class defines yet to be implemented callbacks
    intended to be overridden in subclass. ``BasicTrainer`` is integrated
    with the following functions:

        - logging
        - dump checkpoints (pth files)
        - dump runtime statistics (csv file or npz files)
        - resume training from epoch or minibatch
        - evaluation after training (sharing parameters of the original
          trainer)

    The ``BasicTrainer`` is executed for ``max_epoch`` epochs; for each epoch,
    it runs ``len(run_stages)`` stages of loops. This scheme can be visualized
    as the following pseudocode::

        .. code-block:: python

            for epoch in range(self.max_epoch):
                for name in self.run_stages:
                    ...

    For simplicity in implementation while maintaining flexibility for
    subclassing. Some methods to override are based on convention rather than
    structural inheritance. The following table lists these methods.
    Without implementing mandatory methods leads to ``AttributeError`` when
    called from ``run``, the start entry of the execution.

    +-----+--------------------+----------------------------------------------+
    | Man |      Function      |                 Description                  |
    +=====+====================+==============================================+
    | T   | STAGE_once         | Minibatch in loop STAGE                      |
    | T   | get_STAGEloader    | Dataloader of (inputs,targets) in loop STAGE |
    | F   | before_batch_STAGE | Launched before STAGE_once                   |
    | F   | after_batch_STAGE  | Launched after STAGE_once                    |
    +-----+--------------------+----------------------------------------------+

    where "Man" denotes "Mandatory". This table shows the signature. For
    example, if a function has signature ``(x, y) -> z``, then it accepts two
    positional arguments ``x`` and ``y``, and returns ``z``.

    +--------------------+-----------------------------------+
    |      Function      |             Signature             |
    +====================+===================================+
    | STAGE_once         | (inputs, targets) -> stats        |
    | get_STAGEloader    | () -> Iterator[(inputs, targets)] |
    | before_batch_STAGE | () -> None                        |
    | after_batch_STAGE  | () -> None                        |
    +--------------------+-----------------------------------+

    Instance variables need to be specified before ``init_monitors`` is
    called (by default ``init_monitors`` is called within ``setup``, which in
    turn is called at the beginning of ``run``):

    +----------------+----------------------+------------------------------+
    |    Variable    |       Default        |         Description          |
    +================+======================+==============================+
    | basedir        | runs-$TODAY          | Directory to hold            |
    |                |                      | $statdir* and $savedir       |
    | statdir        | $basedir/stat        | Directory to hold            |
    |                |                      | statistics produced in loop  |
    |                |                      | train                        |
    | savedir        | $basedir/save        | Directory to hold            |
    |                |                      | checkpoints produced in loop |
    |                |                      | train                        |
    | fired          | fired_always         | The firing policy of         |
    |                |                      | CheckpointSaver and          |
    |                |                      | StatSaver in loop train      |
    | statdir_$STAGE | $basedir/stat_$STAGE | Directory to hold statistics |
    |                |                      | produced in loop STAGE other |
    |                |                      | than train                   |
    | fired_$STAGE   | fired_always         | The firing policy of         |
    |                |                      | StatSaver in loop STAGE      |
    |                |                      | other than train             |
    +----------------+----------------------+------------------------------+

    Mandatory instance variables need to be specified before ``run``:

    +------------+------------------------------------------------+
    |  Variable  |                  Description                   |
    +============+================================================+
    | run_stages | The loops to run in each epoch                 |
    | stat_names | The names of statistics returned by STAGE_once |
    +------------+------------------------------------------------+
    """

    timestamp_format = '%Y%m%d%H%M'  # e.g. 201811051438
    """
    Used to name the default ``basedir``.
    """

    checkpoint_tmpl = 'checkpoint_{0}_{1}.pth'
    """
    Checkpoint pth file name template, accepting progress tuple ``(epoch_id,
    minibatch_id)``.
    """

    statname_tmpl = 'stats_{0}_{1}.npz'
    """
    Statistics npz file name template, accepting progress tuple ``(epoch_id,
    minibatch_id)``.
    """

    def __init__(self, net: nn.Module, max_epoch: int = 1,
                 device: str = 'cpu', progress: _EBProgress = None,
                 freeze_net_when_necessary=False):
        r"""
        :param net: the network to train
        :param max_epoch: maximum epoch to train, where an epoch is defined as
               a complete traversal of the underlying dataset
        :param device: where to train the network, choices:
               { cpu, cuda(:\d+)? }
        :param progress: where to continue training, default to train from
               scratch. When ``progress`` is not ``None``, denote it as
               ``(E, B)``. The first checkpoint to be dumped by the trainer
               would be ``(E+1, 0)``, and will overwrite existing npy
               statistics and pth checkpoint files.
        :param freeze_net_when_necessary: if True, freeze the network
               parameters whenever ``self.__stage`` is not 'train', and
               always freeze the network if there's no 'train' in
               ``self.run_stages``. Do not specify as ``True`` if in non-train
               stages the network weights are used for backpropagation
        """
        self.device = device
        self.net = net
        self.max_epoch = max_epoch
        self.run_stages = ('train', 'eval')
        self.progress = progress
        self.freeze_net_when_necessary = freeze_net_when_necessary

        self._trained_once = False
        """Used to offset epoch"""
        self._origrg = {}
        """Used to freeze network when necessary"""
        self._frozen_always = False

        # expose (readonly) current training progress
        self.__stage: str = None  # current training stage
        self.__epoch: int = None
        self.__batch: int = None

    @property
    def stage(self):
        return self.__stage

    @property
    def epoch(self):
        return self.__epoch

    @property
    def batch(self):
        return self.__batch

    @property
    def default_basedir(self):
        """
        The default base directory of checkpoints and statistics, in form of
        "runs-{timestamp}", with ``timestamp`` of datetime format
        ``type(self).timestamp_format``.
        """
        return 'runs-{}'.format(datetime.today().strftime(
            type(self).timestamp_format))

    def init_monitors(self):
        """
        Initialize CheckpointSaver and StatSaver as per settings in
        ``__init__``.
        """
        if not hasattr(self, 'basedir'):
            setattr(self, 'basedir', self.default_basedir)
        defaults_train = {
            'statdir': os.path.join(getattr(self, 'basedir'), 'stat'),
            'savedir': os.path.join(getattr(self, 'basedir'), 'save'),
            'fired'  : fired_always,
        }
        for stage in self.run_stages:
            if stage == 'train':
                for k, v in defaults_train.items():
                    if not hasattr(self, k):
                        setattr(self, k, v)
            else:
                if not hasattr(self, 'statdir_{}'.format(stage)):
                    setattr(self, 'statdir_{}'.format(stage),
                            os.path.join(getattr(self, 'basedir'),
                                         'stat_{}'.format(stage)))
                if not hasattr(self, 'fired_{}'.format(stage)):
                    setattr(self, 'fired_{}'.format(stage), fired_always)

    def prepare_net(self, ext_savedir: str = None) -> None:
        """
        Load checkpoint if ``progress`` is not ``None``, and move network to
        train to the specified device.

        :param ext_savedir: external savedir; if not set, use ``self.savedir``
        """
        savedir = ext_savedir if ext_savedir else getattr(self, 'savedir')
        if self.progress is not None:
            load_checkpoint(self.net, savedir, type(self).checkpoint_tmpl,
                            self.progress, map_location=self.device)
        self.net.to(self.device)

    def freeze_net(self):
        self._origrg = freeze_model(self.net)

    def melt_net(self):
        melt_model(self.net, self._origrg, empty_after_melting=True)

    def __statsaver(self, stage):
        """Deferred instantiation of ``(Csv)StatSaver``'s."""
        if stage == 'train':
            try:
                saver = getattr(self, 'statsaver')
            except AttributeError:
                saver = StatSaver(getattr(self, 'statdir'),
                                  fired=getattr(self, 'fired'),
                                  statname_tmpl=type(self).statname_tmpl)
                setattr(self, 'statsaver', saver)
        else:
            try:
                saver = getattr(self, 'statsaver_{}'.format(stage))
            except AttributeError:
                saver = StatSaver(getattr(self, 'statdir_{}'.format(stage)),
                                  fired=getattr(self, 'fired_{}'.format(stage)),
                                  statname_tmpl=type(self).statname_tmpl)
                setattr(self, 'statsaver_{}'.format(stage), saver)
        return saver

    def __checkpointsaver(self):
        """Deferred instantiation of the ``CheckpointSaver``."""
        try:
            saver = getattr(self, 'checkpointsaver')
        except AttributeError:
            saver = CheckpointSaver(self.net, getattr(self, 'savedir'),
                                    fired=getattr(self, 'fired'),
                                    checkpoint_tmpl=type(self).checkpoint_tmpl)
            setattr(self, 'checkpointsaver', saver)
        return saver

    def __before_batch(self, stage):
        """Call ``before_batch_STAGE``."""
        with contextlib.suppress(AttributeError):
            getattr(self, 'before_batch_{}'.format(stage))()

    def __after_batch(self, stage):
        """Call ``after_batch_STAGE``."""
        with contextlib.suppress(AttributeError):
            getattr(self, 'after_batch_{}'.format(stage))()

    def __get_loader(self, stage):
        """Call ``get_STAGEloader``."""
        return getattr(self, 'get_{}loader'.format(stage))()

    def __once(self, stage, inputs, targets):
        """Call ``STAGE_once``."""
        return getattr(self, '{}_once'.format(stage))(inputs, targets)

    def before_epoch(self):
        """
        Callback before each epoch.
        """
        _l(self, 'before_epoch').debug('')

    def after_epoch(self):
        """
        Callback after each epoch.
        """
        _l(self, 'after_epoch').debug('')

    def setup(self):
        """
        Callback before ``run`` and after ``__init__``. Default to:

            - initializing checkpoing and statistics savers (``init_monitors``)
            - loading the checkpoint (``prepare_net``)
            - freeze the network if necessary (``freeze_net``)
        """
        self.init_monitors()
        self.prepare_net()
        if self.freeze_net_when_necessary and 'train' not in self.run_stages:
            self._frozen_always = True
        if self._frozen_always:
            self.freeze_net()

    def teardown(self, error: BaseException = None):
        """
        Callback before the return of ``run``, whether or not a successful
        return.

            - melt the network if necessary (``melt_net``)

        :param error: the cause of the return, or ``None`` if there's no
               error. Note that when not None, it's not necessarily of exactly
               type ``BaseException`` -- might be exception subclass of it
        """
        if self.freeze_net_when_necessary:
            self.melt_net()

    def run(self):
        logger = _l(self, 'run')
        logger.debug('Initializing')
        self.setup()

        # Since it's uneasy and not necessary to train from exact batch of the
        # loaded checkpoint, it will start training from the next epoch of the
        # checkpoint epoch
        if self.progress is not None:
            cpepoch, _ = self.progress
            epoch0 = cpepoch + 1
            if epoch0 >= self.max_epoch:
                logger.warning('No epoch left to run: current_epoch(1st)={} '
                               'max_epoch={}'.format(epoch0, self.max_epoch))
                self.teardown()
                return
        else:
            epoch0 = 0

        try:
            for self.__epoch in range(epoch0, self.max_epoch):
                self.before_epoch()
                for self.__stage in self.run_stages:
                    logger.debug('Begin stage {}'.format(self.__stage))
                    if self.__stage == 'train':
                        self._trained_once = True
                        if (self.freeze_net_when_necessary
                                and not self._frozen_always):
                            self.melt_net()
                        self.net.train()
                        for self.__batch, it in enumerate(
                                self.__get_loader(self.__stage)):
                            # `it` is `(inputs, targets)`
                            self.__before_batch(self.__stage)
                            stats = self.__once(self.__stage, *it)
                            if stats:
                                stats_to_log = self._organize_stats(stats)
                                stats_to_log_repr = list(stats_to_log.items())
                                logger.info('epoch{}/{} batch{}: {}'
                                            .format(self.__epoch, self.__stage,
                                                    self.__batch,
                                                    stats_to_log_repr))
                                statsaver = self.__statsaver(self.__stage)
                                statsaver((self.__epoch, self.__batch),
                                          **stats_to_log)
                            else:
                                logger.info('epoch{}/{} batch{}'
                                            .format(self.__epoch,
                                                    self.__stage,
                                                    self.__batch))
                            checkpointsaver = self.__checkpointsaver()
                            checkpointsaver((self.__epoch, self.__batch))
                            self.__after_batch(self.__stage)
                    else:
                        if (self.freeze_net_when_necessary
                                and not self._frozen_always):
                            self.freeze_net()
                        self.net.eval()
                        for self.__batch, it in enumerate(
                                self.__get_loader(self.__stage)):
                            # `it` is `(inputs, targets)`
                            self.__before_batch(self.__stage)
                            stats = self.__once(self.__stage, *it)
                            if stats:
                                stats_to_log = self._organize_stats(stats)
                                stats_to_log_repr = list(stats_to_log.items())
                                logger.info('epoch{}/{} batch{}: {}'
                                            .format(self.__epoch, self.__stage,
                                                    self.__batch,
                                                    stats_to_log_repr))
                                statsaver = self.__statsaver(self.__stage)
                                statsaver((self.__epoch, self.__batch),
                                          **stats_to_log)
                            else:
                                logger.info('epoch{}/{} batch{}'
                                            .format(self.__epoch,
                                                    self.__stage,
                                                    self.__batch))
                            self.__after_batch(self.__stage)
                self.after_epoch()
            logger.info('Returns successfully')
            self.teardown()
        except BaseException as err:
            logger.error('Exception raised (of type {}): {}'
                         .format(type(err).__name__, err))
            self.teardown(error=err)
            raise

    # noinspection PyUnresolvedReferences
    def _organize_stats(self, stats: typing.Tuple) \
            -> typing.Dict[str, typing.Any]:
        logger = _l('_organize_stats')
        try:
            stat_names = self.stat_names
        except AttributeError:
            # to conform to the naming policy of ``numpy.savez``
            stat_names = ['arr_{}'.format(i) for i in range(len(stats))]

        if len(stat_names) != len(stats):
            logger.warning('len(stat_names) ({}) is different from '
                           'len(stats) ({})'
                           .format(len(stat_names), len(stats)))
        return collections.OrderedDict(zip(stat_names, stats))


class BasicEvaluator(BasicTrainer):
    """
    Abstract base evaluator adapting the framework by ``BasicTrainer`` such
    that it's dedicated to evaluating an existing network checkpoint.
    Please note the difference of the following convention with that of
    ``BasicTrainer``.

    +-----+--------------------+----------------------------------------------+
    | Man |      Function      |                 Description                  |
    +=====+====================+==============================================+
    | T   | STAGE_once         | Minibatch in loop STAGE                      |
    | T   | get_STAGEloader    | Dataloader of (inputs,targets) in loop STAGE |
    | F   | before_batch_STAGE | Launched before STAGE_once                   |
    | F   | after_batch_STAGE  | Launched after STAGE_once                    |
    +-----+--------------------+----------------------------------------------+

    where "Man" denotes "Mandatory". This table shows the signature. For
    example, if a function has signature ``(x, y) -> z``, then it accepts two
    positional arguments ``x`` and ``y``, and returns ``z``.

    +--------------------+-----------------------------------+
    |      Function      |             Signature             |
    +====================+===================================+
    | STAGE_once         | (inputs, targets) -> stats        |
    | get_STAGEloader    | () -> Iterator[(inputs, targets)] |
    | before_batch_STAGE | () -> None                        |
    | after_batch_STAGE  | () -> None                        |
    +--------------------+-----------------------------------+

    Instance variables need to be specified before ``init_monitors`` is
    called (by default ``init_monitors`` is called within ``setup``, which in
    turn is called at the beginning of ``run``):

    +----------------+----------------------+------------------------------+
    |    Variable    |       Default        |         Description          |
    +================+======================+==============================+
    | eval_basedir   | $basedir/eval_epEP   | Directory to hold evaluation |
    |                |                      | statistics, where EP denotes |
    |                |                      | the first element of the     |
    |                |                      | ``progress`` tuple specified |
    |                |                      | at ``__init__``. If this     |
    |                |                      | attribute is to set manually |
    |                |                      | it must be prefixed          |
    |                |                      | "$basedir"                   |
    | savedir        | $basedir/save        | Directory to hold            |
    |                |                      | checkpoints produced in      |
    |                |                      | previous loop train          |
    | statdir_STAGE  | $basedir             | Directory to hold statistics |
    |                | /stat_$STAGE         | produced in loop STAGE other |
    |                |                      | than train                   |
    | fired_STAGE    | fired_always         | The firing policy of         |
    |                |                      | StatSaver in loop STAGE      |
    |                |                      | other than train             |
    +----------------+----------------------+------------------------------+

    where ``basedir``, the ``basedir`` directory when previously training,
    must be provided at ``__init__``.

    Mandatory instance variables need to be specified before ``run``:

    +------------+------------------------------------------------+
    |  Variable  |                  Description                   |
    +============+================================================+
    | run_stages | The loops to run in each epoch; 'train' must   |
    |            | not be specified as one of them                |
    | stat_names | The names of statistics returned by STAGE_once |
    +------------+------------------------------------------------+
    """
    def __init__(self, net: nn.Module, progress: _EBProgress,
                 basedir: str, freeze_net_when_necessary: bool = False,
                 device: str = 'cpu'):
        """
        :param net: the network to evaluate
        :param progress: which to evaluate, must be of form (EPOCH, BATCH)
        :param freeze_net_when_necessary: if True, freeze the network
               parameters whenever ``self.__stage`` is not 'train', and
               always freeze the network if there's no 'train' in
               ``self.run_stages``. Do not specify as ``True`` if in non-train
               stages the network weights are used for backpropagation. This
               option can be useful if the inputs requires gradient
               backpropagation, in which case ``torch.no_grad`` cannot be
               used
        :param device: where to evaluate
        """
        super().__init__(net, progress[0] + 2, device=device,
                         progress=progress)
        if not os.path.isdir(basedir):
            raise FileNotFoundError('basedir "{}" not found'.format(basedir))
        self.basedir = basedir

    @property
    def default_basedir(self):
        raise AttributeError('No default basedir available')

    @property
    def default_eval_basedir(self):
        return 'eval_ep{}'.format(self.progress[0])

    def init_monitors(self):
        # eval_basedir
        if not hasattr(self, 'eval_basedir'):
            setattr(self, 'eval_basedir', os.path.join(
                self.basedir, self.default_eval_basedir))
            if os.path.isdir(getattr(self, 'eval_basedir')):
                raise FileExistsError('Default eval_basedir "{}" already '
                                      'exists; try setting a different name'
                                      .format(getattr(self, 'eval_basedir')))
        elif not os.path.samefile(os.path.commonprefix((
            getattr(self, 'eval_basedir'), self.basedir)), self.basedir):
            raise ValueError('Expecting eval_basedir to be a child of '
                             'self.basedir "{}", but got "{}"'
                             .format(getattr(self, 'eval_basedir'),
                                     self.basedir))
        elif os.path.exists(getattr(self, 'eval_basedir')):
            raise FileExistsError('eval_basedir "{}" already exists; '
                                  'try setting a different name'
                                  .format(getattr(self, 'eval_basedir')))

        # savedir
        if not hasattr(self, 'savedir'):
            setattr(self, 'savedir', os.path.join(
                getattr(self, 'basedir'), 'save'))

        # run_stages and others
        for stage in self.run_stages:
            if stage == 'train':
                raise ValueError('Expecting no \'train\' in `self.run_stages`'
                                 ', but got {}'.format(self.run_stages))
            if not hasattr(self, 'statdir_{}'.format(stage)):
                setattr(self, 'statdir_{}'.format(stage),
                        os.path.join(getattr(self, 'eval_basedir'),
                                     'stat_{}'.format(stage)))
            if not hasattr(self, 'fired_{}'.format(stage)):
                setattr(self, 'fired_{}'.format(stage), fired_always)

    def prepare_net(self, ext_savedir: str = None) -> None:
        savedir = ext_savedir if ext_savedir else getattr(self, 'savedir')
        load_checkpoint(self.net, savedir, type(self).checkpoint_tmpl,
                        self.progress, map_location=self.device)
        self.net.to(self.device)


class IniFunctionCaller:
    def __init__(self, cfg: configparser.ConfigParser,
                 varparam_policy='raise'):
        self.cfg = cfg
        self.varparam_policy = varparam_policy

    def call(self, f: typing.Callable[[typing.Any], T], **kwargs) -> T:
        """
        :param f: the callable to invoke
        :param scopes: INI sections to search for; if None, search in all
               sections in sequential order until the underlying key is found
        :type scopes: typing.Sequence[str]
        :param argname2inikey: translate certain argument name to INI key name
        :type argname2inikey: typing.Dict[str, str]
        :param argname2ty: translate certain argument name to unary function
               that converts string INI value to appropriate type; if not
               specified, the unary function will be taken from the type
               annotation, and if not found in annotation, default to ``str``
        :type argname2ty: typing.Dict[str, typing.Callable[ [str], typing.Any]]
        :return: whatever is returned by ``f``
        """
        logger = _l(self, 'call')
        argname2inikey = kwargs.get('argname2inikey', {})
        argname2ty = kwargs.get('argname2ty', {})
        scopes = kwargs.get('scopes', list(self.cfg.sections()))

        args = collections.OrderedDict()
        kwargs = collections.OrderedDict()
        params = inspect.signature(f).parameters
        for name, par in params.items():
            if par.kind in (inspect.Parameter.VAR_KEYWORD,
                            inspect.Parameter.VAR_POSITIONAL):
                if self.varparam_policy == 'ignore':
                    logger.info('Ignored var parameter in callable {}'
                                .format(f))
                    continue
                else:
                    raise RuntimeError('Unsupported parameter *args and/or '
                                       '**kwargs found in callable {}'
                                       .format(f))
            if par.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
                param_queue = args
            else:
                param_queue = kwargs

            inikey = argname2inikey.get(name, name)
            for sec in scopes:
                if inikey in self.cfg[sec]:
                    inivalue = self.cfg[sec][inikey]
                    if name in argname2ty:
                        ty = argname2ty[name]
                    elif par.annotation != inspect.Parameter.empty:
                        ty = par.annotation
                    else:
                        ty = str
                    inivalue = ty(inivalue)
                    break
            else:
                if par.default != inspect.Parameter.empty:
                    inivalue = par.default
                else:
                    raise KeyError('Argument `{}` (inikey={}) of '
                                   'callable {} not specified in self.cfg'
                                   .format(name, inikey, f))
            param_queue[name] = inivalue
        args = tuple(args.values())
        logger.info('Parsed args: {}'.format(args))
        logger.info('Parsed kwargs: {}'.format(kwargs))
        return f(*args, **kwargs)
