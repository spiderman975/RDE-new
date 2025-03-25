import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
from utils.comm import get_world_size

# 导入视频文本相关的数据加载类（注意：这里 VideoTextDataset 用于训练，VideoDataset 用于验证/测试，
# TextDataset 保持不变）
#from .bases_video import VideoTextDataset, TextDataset, VideoDataset
from .bases_video import VideoTextDataset, TextDataset, VideoDataset
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .msvd import MSVD

__factory = {
    'CUHK-PEDES': CUHKPEDES,
    'ICFG-PEDES': ICFGPEDES,
    'RSTPReid': RSTPReid,
    'MSVD': MSVD  # 当选择 MSVD 时，返回的是视频数据集类
}



def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform

'''
def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if k == "videos":
            # v 是一个列表，每个元素形状为 [T, 3, H, W]
            max_T = max(tensor.shape[0] for tensor in v)
            padded_v = []
            for tensor in v:
                T, C, H, W = tensor.shape
                if T < max_T:
                    pad = torch.zeros(max_T - T, C, H, W, dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, pad], dim=0)
                padded_v.append(tensor)
            batch_tensor_dict[k] = torch.stack(padded_v)
        elif isinstance(v[0], int):
            batch_tensor_dict[k] = torch.tensor(v)
        elif torch.is_tensor(v[0]):
            batch_tensor_dict[k] = torch.stack(v)
        else:
            raise TypeError(f"Unexpected data type: {type(v[0])} in a batch for key {k}.")
    return batch_tensor_dict
'''


def collate(batch):
    # 收集所有样本中的键
    keys = set([key for b in batch for key in b.keys()])
    # 构造字典，每个键对应一个列表，每个元素来自一个样本
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    batch_tensor_dict = {}

    # 对 videos 键单独处理：补齐每个视频的帧数，并生成有效性掩码
    if "videos" in dict_batch:
        videos_list = dict_batch["videos"]
        # 计算本批次中最大的帧数
        max_T = max(tensor.shape[0] for tensor in videos_list)
        padded_v = []
        valid_masks = []  # 存放每个视频的有效性掩码
        for tensor in videos_list:
            T, C, H, W = tensor.shape
            # 为原始帧创建一个全 1 掩码
            valid_mask = torch.ones(T, dtype=torch.long, device=tensor.device)
            if T < max_T:
                # 如果帧数不足，构造补齐张量（全 0）
                pad = torch.zeros(max_T - T, C, H, W, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=0)
                # 同时对掩码进行补齐：补齐部分标记为 0
                valid_mask = torch.cat([valid_mask, torch.zeros(max_T - T, dtype=torch.long, device=tensor.device)])
            padded_v.append(tensor)
            valid_masks.append(valid_mask)
        batch_tensor_dict["videos"] = torch.stack(padded_v)
        batch_tensor_dict["video_mask"] = torch.stack(valid_masks)

    # 对其他键进行处理
    for k, v in dict_batch.items():
        if k == "videos":
            continue  # 已经处理
        elif isinstance(v[0], int):
            batch_tensor_dict[k] = torch.tensor(v)
        elif torch.is_tensor(v[0]):
            batch_tensor_dict[k] = torch.stack(v)
        else:
            raise TypeError(f"Unexpected data type: {type(v[0])} in a batch for key {k}.")

    return batch_tensor_dict


def build_dataloader(args, transforms=None):
    logger = logging.getLogger("IRRA.dataset")
    num_workers = args.num_workers
    # __factory 返回的是视频数据集类（例如 MSVD），其中 dataset.train, dataset.test, dataset.val 的格式均为：
    # 列表或字典，其中训练部分的每个样本格式为 (pid, video_id, video_path, caption)
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    if args.training:
        # 构建训练和验证时的 transform
        train_transforms = build_transforms(img_size=args.img_size, aug=args.img_aug, is_train=True)
        val_transforms = build_transforms(img_size=args.img_size, is_train=False)

        # 训练集使用 VideoTextDataset（返回的视频张量、文本 token 序列等键与原 ImageTextDataset 保持一致）
        train_set = VideoTextDataset(dataset.train, args,
                                     transform=train_transforms,
                                     text_length=args.text_length)
        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader = DataLoader(train_set,
                                          batch_sampler=batch_sampler,
                                          num_workers=num_workers,
                                          collate_fn=collate)
            else:
                logger.info(f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}')
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # 验证集部分：对于视频数据集，使用 VideoDataset 加载视频数据，
        # 同时 TextDataset 用于加载文本描述。注意这里键名需要与数据集内部返回保持一致，
        # 如 "video_pids" 和 "video_paths"（与原图像版本 "image_pids"、"img_paths" 保持相同结构）。
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_vid_set = VideoDataset(ds['video_pids'], ds['video_paths'], transform=val_transforms, frame_rate=args.video_frame_rate)
        #val_vid_set = VideoDataset(ds['video_pids'], ds['video_paths'], transform=val_transforms, max_frames=args.max_frames)

        val_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=args.text_length)
        val_vid_loader = DataLoader(val_vid_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_vid_loader, val_txt_loader, num_classes

    else:
        if transforms:
            test_transforms = transforms
        else:
            test_transforms = build_transforms(img_size=args.img_size, is_train=False)
        ds = dataset.test
        test_vid_set = VideoDataset(ds['video_pids'], ds['video_paths'], transform=test_transforms, frame_rate=args.video_frame_rate)
        #test_vid_set = VideoDataset(ds['video_pids'], ds['video_paths'], transform=test_transforms, max_frames=args.max_frames)

        test_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=args.text_length)
        test_vid_loader = DataLoader(test_vid_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
        return test_vid_loader, test_txt_loader, num_classes

