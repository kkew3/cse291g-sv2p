if __name__ == '__main__':

    import argparse

    def make_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('ini')
        parser.add_argument('action', choices=('train', 'infer'))
        return parser

    args = make_parser().parse_args()

    import logging

    logging.basicConfig(filename=args.ini + '.log', level=logging.INFO,
                        format='%(levelname)s|%(asctime)s'
                               '|%(name)s|%(message)s')

    import configparser
    import os

    from train.trainlib import IniFunctionCaller
    from train.train_cdna import CDNATrainer

    cfg = configparser.ConfigParser()
    cfg.read(args.ini)
    ifc = IniFunctionCaller(cfg)
    trainer = ifc.call(CDNATrainer,
                       scopes=('dataset', 'train', 'train_device'),
                       argname2ty={'indices': eval})
    basedir = 'runs-{}'.format(os.path.splitext(os.path.basename(args.ini))[0])
    if 'LUSTRE_SCRATCH' in os.environ:
        basedir = os.path.join(os.path.normpath(os.environ['LUSTRE_SCRATCH']),
                               'cse291g-wi19', 'cdna', basedir)
    trainer.basedir = basedir
    logging.info('basedir={}'.format(trainer.basedir))
    trainer.run()
