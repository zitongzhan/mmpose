import mmcv
import numpy as np
import os
import os.path as osp
import cv2
import json
import argparse
import torch
from mmengine.config import Config, DictAction
import mmpose
from mmpose.datasets import build_dataset
from mmengine.runner import Runner

if __name__ == '__main__':
    # config = 'configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_hrnet-w32_8xb64-20e_posetrack18-256x192.py'
    config = 'bu-hm_playground.py'
    cfg = Config.fromfile(config)
    cfg.work_dir = '/tmp/1'
    runner = Runner.from_cfg(cfg)
    train_dataset = runner.train_dataloader.dataset
    for data in train_dataset:
        print(data)
    exit()

