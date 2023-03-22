# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import random

import mmcv

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class CompressedVideoDataset(BaseDataset):
    """Compressed video dataset for action recognition

    The dataset loads clips of raw videos and apply specified transforms to
    return a dict containing the frame tensors and other information. Not that
    for this dataset, `multi_class` should be False.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        sampling_strategy (str): The strategy to sample clips from raw videos.
            Choices are 'random' or 'positive'. Default: 'positive'.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 gop_size=12,
                 sampling_strategy='positive',
                 **kwargs):
        self.gop_size = gop_size
        super().__init__(ann_file, pipeline, start_index=0, data_prefix=data_prefix,**kwargs)
        self.sampling_strategy = sampling_strategy
        assert self.multi_class is False
        
        # If positive, we should only keep those raw videos with positive
        # clips
        

    # do not support multi_class
    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_path = line_split[0]
                label = int(line_split[1])
                num_frames = int(line_split[2])
                num_gop =  int(line_split[3])

                if self.data_prefix is not None:
                    video_path = osp.join(self.data_prefix, video_path)
                video_infos.append(
                    dict(
                        video_path=video_path,
                        label=label,
                        total_frames=num_frames,
                        num_gop=num_gop,
                        gop_size = self.gop_size))
        return video_infos
    
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        return self.pipeline(results)
