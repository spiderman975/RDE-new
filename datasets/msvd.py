import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class MSVD(BaseDataset):
    """
    MSVD 视频数据集——按外部 txt 列表来切分 train/val/test
    """
    dataset_dir = 'MSVD'

    def __init__(self, root='', verbose=True):
        super(MSVD, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.video_dir   = op.join(self.dataset_dir, 'vids')
        self.anno_path   = op.join(self.dataset_dir, 'transformed_video_data.json')
        self._check_before_run()

        # 现在 _split_anno 返回 (train_annos, test_annos, val_annos)
        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test,  self.test_id_container  = self._process_anno(self.test_annos)
        self.val,   self.val_id_container   = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> MSVD Videos and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        """
        不再根据 anno['split']，而是读取三个 txt：
        train_list.txt, test_list.txt, val_list.txt
        并以 video_path 文件名（去掉扩展名）为 key 来划分。
        """
        annos = read_json(anno_path)

        # 读三份列表
        def _read_list(fname):
            path = op.join(self.dataset_dir, fname)
            with open(path, 'r', encoding='utf-8') as fp:
                return { line.strip() for line in fp if line.strip() }

        train_ids = _read_list('train_list.txt')
        test_ids  = _read_list('test_list.txt')
        val_ids   = _read_list('val_list.txt')

        train_annos, test_annos, val_annos = [], [], []

        for anno in annos:
            # 取 video_path 的文件名（无后缀）来对比
            vid_name = op.splitext(op.basename(anno['video_path']))[0]
            if vid_name in train_ids:
                train_annos.append(anno)
            elif vid_name in test_ids:
                test_annos.append(anno)
            elif vid_name in val_ids:
                val_annos.append(anno)
            else:
                # 如果某个 id 不在任何列表中，可以根据需要警告或忽略
                self.logger.warning(f"Video {vid_name} not found in any split txt, ignored.")

        self.logger.info(f"Split counts: train={len(train_annos)}, test={len(test_annos)}, val={len(val_annos)}")
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        # —— 此处保持你原来的实现不变 —— #
        pid_container = set()
        if training:
            dataset = []
            video_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                video_path = op.join(self.video_dir, anno['video_path'])
                for caption in anno['captions']:
                    dataset.append((pid, video_id, video_path, caption))
                video_id += 1
            for idx, pid in enumerate(pid_container):
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            video_paths, video_pids = [], []
            captions, caption_pids  = [], []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                video_paths.append(op.join(self.video_dir, anno['video_path']))
                video_pids.append(pid)
                for caption in anno['captions']:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "video_pids":   video_pids,
                "video_paths":  video_paths,
                "caption_pids": caption_pids,
                "captions":     captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not op.exists(self.video_dir):
            raise RuntimeError(f"'{self.video_dir}' is not available")
        if not op.exists(self.anno_path):
            raise RuntimeError(f"'{self.anno_path}' is not available")
