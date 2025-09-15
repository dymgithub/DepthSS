# train.py

import argparse
import os
import os.path as osp
import time
import warnings

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.logging import MMLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor using MMEngine')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically. '
             'If "--load-from" is specified, this option will be ignored.')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also supports nested key/value updating like a.b.c=d '
             'for find-grained adjustment.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 使用配置文件名自动创建工作目录
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume:
        cfg.resume = True

    # 确保工作目录存在
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = MMLogger.get_instance('MMSeg', log_file=log_file, log_level=cfg.log_level)

    runner = Runner.from_cfg(cfg)

    runner.train()


if __name__ == '__main__':
    main()