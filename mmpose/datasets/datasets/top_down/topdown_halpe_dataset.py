import os
import warnings

import numpy as np
from xtcocotools.coco import COCO

from ...registry import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownHalpeDataset(TopDownCocoDataset):
    """HalpeDataset for top-down pose estimation.

    'https://github.com/Fang-Haoshu/Halpe-FullBody'

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    In total, we have 136 keypoints for wholebody pose estimation.

    Halpe keypoint indexes::
        0-19: 20 body keypoints
        20-25: 6 foot keypoints
        26-93: 68 face keypoints
        94-135: 42 hand keypoints

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super(TopDownCocoDataset, self).__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        if 'image_thr' in data_cfg:
            warnings.warn(
                'image_thr is deprecated, '
                'please use det_bbox_thr instead', DeprecationWarning)
            self.det_bbox_thr = data_cfg['image_thr']
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['flip_pairs'] = self._make_flip_pairs()

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = \
            np.ones((self.ann_info['num_joints'], 1), dtype=np.float32)

        self.body_num = 20
        self.foot_num = 6
        self.face_num = 68
        self.left_hand_num = 21
        self.right_hand_num = 21

        # 'https://github.com/Fang-Haoshu/Halpe-FullBody/blob/'
        # 'master/cocoapi/PythonAPI/pycocotools/cocoeval.py#240'
        self.sigmas_body = [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.08, 0.08,
            0.08
        ]
        self.sigmas_foot = [0.089, 0.089, 0.089, 0.089, 0.089, 0.089]
        self.sigmas_face = [0.015] * 68
        self.sigmas_lefthand = [0.015] * 21
        self.sigmas_righthand = [0.015] * 21

        self.sigmas_wholebody = (
            self.sigmas_body + self.sigmas_foot + self.sigmas_face +
            self.sigmas_lefthand + self.sigmas_righthand)

        self.sigmas = np.array(self.sigmas_wholebody)

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'halpe'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    @staticmethod
    def _make_flip_pairs():
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[20, 21], [22, 23], [24, 25]]

        face = [[26, 42], [27, 41], [28, 40], [29, 39], [30, 38], [31, 37],
                [32, 36], [33, 35], [43, 52], [44, 51], [45, 50], [46, 49],
                [47, 48], [57, 61], [58, 60], [62, 71], [63, 70], [64, 69],
                [65, 68], [66, 73], [67, 72], [74, 80], [75, 79], [76, 78],
                [81, 85], [82, 84], [86, 90], [87, 89], [91, 93]]

        hand = [[94, 115], [95, 116], [96, 117], [97, 118], [98, 119],
                [99, 120], [100, 121], [101, 122], [102, 123], [103, 124],
                [104, 125], [105, 126], [106, 127], [107, 128], [108, 129],
                [109, 130], [110, 131], [111, 132], [112, 133], [113, 134],
                [114, 135]]

        return body + foot + face + hand

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3] > 0)

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec
