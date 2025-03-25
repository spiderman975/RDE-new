import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import random
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
from utils.iotools import read_json

# ------------------------------
# 修改的地方：修改 extract_frames 函数，使其支持基于最大帧数采样
# ------------------------------
def extract_frames(video_path, max_frames: int, resize=(224, 224)):
    """
    从视频文件中提取所有帧，
    若帧数小于 max_frames，则补充空帧使得总数达到 max_frames；
    若帧数大于 max_frames，则均匀采样 max_frames 帧。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame_rgb = cv2.resize(frame_rgb, resize)
        img = Image.fromarray(frame_rgb)
        frames.append(img)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return frames

    if total_frames < max_frames:
        # 修改1：补充空帧（用全黑图像补齐）
        dummy = Image.new("RGB", resize, (0, 0, 0))
        while len(frames) < max_frames:
            frames.append(dummy)
        return frames
    elif total_frames > max_frames:
        # 修改2：均匀采样 max_frames 帧
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
        return frames
    else:
        return frames

# ------------------------------
# 其他辅助函数保持不变
# ------------------------------
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
        tmp = (pids[i], image_ids[i], images[i], captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(
        noisy_rate, np.sum(real_correspondeces), nums - np.sum(real_correspondeces), nums))
    return dataset, np.array(real_correspondeces)

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

# ------------------------------
# 修改的地方：VideoTextDataset 中将 target_fps 改为使用 max_frames 参数
# ------------------------------
class VideoTextDataset(Dataset):
    def __init__(self, dataset, args, transform=None, text_length: int = 77, truncate: bool = True):
        """
        dataset: 列表，每个元素格式为 (pid, video_id, video_path, caption)
        args: 参数对象，其中需要包含：
             - txt_aug: 是否对文本进行增强
             - max_frames: 指定的最大帧数
             - noisy_rate, noisy_file 等（噪声注入相关）
        transform: 用于预处理视频帧（例如调整尺寸、归一化）的 transform
        """
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        # 修改：使用 args.max_frames 替代原来的 video_frame_rate
        self.max_frames = args.max_frames
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset, args.noisy_rate, args.noisy_file)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, video_id, video_path, caption = self.dataset[index]
        try:
            # 修改：调用 extract_frames 时使用 max_frames 参数
            frames = extract_frames(video_path, max_frames=self.max_frames)
            if len(frames) == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            dummy = torch.zeros(1, 3, 384, 128)
            frames = [dummy]
        if self.transform is not None:
            try:
                frames = [self.transform(img) for img in frames]
            except Exception as e:
                print(f"Transform error: {e}")
                frames = frames
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

class VideoDataset(Dataset):
    def __init__(self, video_pids, video_paths, transform=None, max_frames=3):
        self.video_pids = video_pids
        self.video_paths = video_paths
        self.transform = transform
        # 修改：使用 max_frames 替代 frame_rate 参数
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        pid, video_path = self.video_pids[index], self.video_paths[index]
        frames = extract_frames(video_path, max_frames=self.max_frames)
        if self.transform:
            frames = [self.transform(img) for img in frames]
        video_tensor = torch.stack(frames)
        return pid, video_tensor

# 文本数据集保持不变
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