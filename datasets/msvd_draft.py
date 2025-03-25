import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class MSVD(BaseDataset):
    """
    MSVD 视频数据集

    假设标注格式与 CUHK-PEDES 类似，但字段改为：
    - "video_path": 视频文件相对路径
    - "captions": 多个文本描述
    - "split": 数据集划分（train/test/val）
    - "id": 视频编号

    注意：此处仅修改数据加载部分，视频预处理（如帧提取）在后续流水线中处理。
    """
    dataset_dir = 'MSVD'  # 修改为你的视频数据集目录

    def __init__(self, root='', verbose=True):
        super(MSVD, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        # 假设视频文件存放在 dataset_dir 下的 "videos" 文件夹中
        self.video_dir = op.join(self.dataset_dir, 'videos/')
        self.anno_path = op.join(self.dataset_dir, 'transformed_video_data.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> MSVD Videos and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            video_id = 0
            for anno in annos:
                # 假设视频 id 从1开始，转换后从0开始
                pid = int(anno['id']) - 1
                pid_container.add(pid)
                # 拼接视频完整路径（如果 video_path 为相对路径，则需与 video_dir 拼接）
                video_path = op.join(self.video_dir, anno['video_path'])
                captions = anno['captions']  # caption list
                for caption in captions:
                    # 样本格式：(pid, video_id, video_path, caption)
                    dataset.append((pid, video_id, video_path, caption))
                video_id += 1
            for idx, pid in enumerate(pid_container):
                # 检查 pid 是否从 0 连续
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            video_paths = []
            captions = []
            video_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                video_path = op.join(self.video_dir, anno['video_path'])
                video_paths.append(video_path)
                video_pids.append(pid)
                caption_list = anno['captions']  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "video_pids": video_pids,
                "video_paths": video_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        """检查数据集目录及文件是否存在"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.video_dir):
            raise RuntimeError("'{}' is not available".format(self.video_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
