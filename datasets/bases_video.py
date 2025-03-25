from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset


def inject_noisy_correspondence(dataset, noisy_rate, noisy_file=None):
    logger = logging.getLogger("RDE.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions = [i[3] for i in dataset_copy]
    images = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i] == i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i], image_ids[i], images[i], captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(
        real_correspondeces), nums - np.sum(real_correspondeces), nums))

    return dataset, np.array(real_correspondeces)


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RDE.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
            self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
            self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result



def extract_frames(video_path, target_fps=3, resize=(224, 224)):
    """
    从视频文件中按指定帧率提取帧，并调整每帧尺寸。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(orig_fps // target_fps) if target_fps is not None else 1
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if target_fps is None or frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame_rgb = cv2.resize(frame_rgb, resize)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
        frame_count += 1
    cap.release()
    return frames


class VideoTextDataset(Dataset):
    def __init__(self, dataset, args, transform=None, text_length: int = 77, truncate: bool = True):
        """
        dataset: 列表，每个元素格式为 (pid, video_id, video_path, caption)
        args: 参数对象，其中需要包含：
             - txt_aug: 是否对文本进行增强
             - video_frame_rate: 提取视频帧的目标帧率
             - noisy_rate, noisy_file 等（噪声注入相关）
        transform: 用于预处理视频帧（例如调整尺寸、归一化）的 transform
        """
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.video_frame_rate = args.video_frame_rate
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset, args.noisy_rate, args.noisy_file)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, video_id, video_path, caption = self.dataset[index]
        try:
            frames = extract_frames(video_path, target_fps=self.video_frame_rate)
            if len(frames) == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")
        except Exception as e:
            # 可以记录日志，这里简单打印错误
            print(f"Error extracting frames from {video_path}: {e}")
            # 如果出错，返回一个全零的dummy tensor，形状按照你的预期（例如1帧）
            #dummy = torch.zeros(1, 3, 384, 128)
            #frames = [dummy]
            # 注意：如果你返回 dummy 需要确保下游处理能够容忍这种情况

        if self.transform is not None:
            # 如果frames是PIL图像，则对每一帧应用transform；
            # 如果是dummy tensor，则跳过transform（或自行转换）
            try:
                frames = [self.transform(img) for img in frames]
            except Exception as e:
                print(f"Transform error: {e}")
                frames = frames  # 保持原样
        # 如果 frames 仍为 list of tensors，则堆叠它们
        try:
            video_tensor = torch.stack(frames)
        except Exception as e:
            print(f"Stack error for {video_path}: {e}")
            video_tensor = frames[0].unsqueeze(0)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                  truncate=self.truncate)
        if self.txt_aug:
            caption_tokens = self.txt_data_aug(caption_tokens.cpu().numpy())
        ret = {
            'pids': pid,
            'video_ids': video_id,
            'videos': video_tensor,
            'caption_ids': caption_tokens,
            'index': index,
        }
        return ret

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                if prob < 0.20:
                    prob /= 0.20
                    if prob < 0.6:
                        aug_tokens.append(mask)
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range))
                    else:
                        None
                else:
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)


class VideoTextDataset_0(Dataset):
    def __init__(self, dataset, args, transform=None, text_length: int = 77, truncate: bool = True):
        """
        dataset: 列表，每个元素格式为 (pid, video_id, video_path, caption)
        args: 参数对象，其中需要包含：
             - txt_aug: 是否对文本进行增强
             - video_frame_rate: 提取视频帧的目标帧率
             - noisy_rate, noisy_file 等（噪声注入相关）
        transform: 用于预处理视频帧（例如调整尺寸、归一化）的 transform
        """
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.video_frame_rate = args.video_frame_rate  # 新增视频帧率参数
        # 注入噪声（保持与原逻辑一致）
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset, args.noisy_rate, args.noisy_file)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 数据格式： (pid, video_id, video_path, caption)
        pid, video_id, video_path, caption = self.dataset[index]
        # 使用 extract_frames 函数提取视频帧（返回的是 PIL 图像列表）
        frames = extract_frames(video_path, target_fps=self.video_frame_rate)
        # 对每一帧进行 transform（保持和原图像 transform 一致）
        if self.transform is not None:
            frames = [self.transform(img) for img in frames]
        # 将帧列表堆叠成张量，形状为 [T, C, H, W]
        video_tensor = torch.stack(frames)
        # 对文本进行 tokenize
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.txt_aug:
            caption_tokens = self.txt_data_aug(caption_tokens.cpu().numpy())
        ret = {
            'pids': pid,
            'video_ids': video_id,
            'videos': video_tensor,
            'caption_ids': caption_tokens,
            'index': index,
        }
        return ret

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # 20% 概率进行增强
                if prob < 0.20:
                    prob /= 0.20
                    if prob < 0.6:
                        aug_tokens.append(mask)
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range))
                    else:
                        None  # 30% 概率直接删除（即不加入新的 token）
                else:
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)


# 用于验证/测试时的视频数据加载，只返回 pid 和视频张量
class VideoDataset(Dataset):
    def __init__(self, video_pids, video_paths, transform=None, frame_rate=3):
        self.video_pids = video_pids
        self.video_paths = video_paths
        self.transform = transform
        self.frame_rate = frame_rate

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        pid, video_path = self.video_pids[index], self.video_paths[index]
        frames = extract_frames(video_path, target_fps=self.frame_rate)
        if self.transform:
            frames = [self.transform(img) for img in frames]
        video_tensor = torch.stack(frames)
        return pid, video_tensor


# 文本数据集与原来一致
class TextDataset(Dataset):
    def __init__(self, caption_pids, captions, text_length: int = 77, truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        caption = tokenize(caption, tokenizer=self.tokenizer,
                           text_length=self.text_length, truncate=self.truncate)
        return pid, caption
