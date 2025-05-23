import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1 + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{args.loss_names}')
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    if not os.path.isdir(args.output_dir + '/img'):
        os.makedirs(args.output_dir + '/img')

    # 构建视频-文本数据集加载器
    train_loader, val_vid_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    # 注意：Evaluator 构造时传入视频加载器和文本加载器
    evaluator = Evaluator(val_vid_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)

    # 推理阶段
    logger.info(f"===================>start test")
    args.training = False
    # 构建测试数据加载器（视频和文本）
    test_vid_loader, test_txt_loader, num_classes = build_dataloader(args)

    # 加载并评估不同的 checkpoint
    for ckpt_name in ['best.pth', 'last.pth']:
        ckpt_path = op.join(args.output_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            model = build_model(args, num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=ckpt_path)
            model = model.cuda()
            do_inference(model, test_vid_loader, test_txt_loader)
