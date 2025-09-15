# core/datasets/transform.py

import os
import cv2
import numpy as np
import torch

from mmseg.registry import TRANSFORMS
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.datasets.transforms import PackSegInputs



@TRANSFORMS.register_module()
class LoadNormalAnnotations:
    def __init__(self):
        pass

    def _get_normal_path(self, img_path: str) -> str:
        try:
            norm_path = os.path.normpath(img_path)
            parts = norm_path.split(os.sep)
            img_dir_idx = parts.index('img_dir')
            split_folder = parts[img_dir_idx + 1]
            img_filename = parts[-1]
            norm_filename = f"norm_{img_filename}"

            base_dir_parts = parts[:img_dir_idx]
            base_dir = os.sep.join(base_dir_parts)

            final_base_dir = os.path.dirname(os.path.dirname(base_dir))
            normal_path = os.path.join(
                final_base_dir, 'output_results', split_folder, 'normal_maps', norm_filename
            )
            return normal_path
        except (ValueError, IndexError):
            return ""

    def __call__(self, results: dict) -> dict:
        img_path = results.get('img_path')
        if not img_path: return results
        normal_path = self._get_normal_path(img_path)
        if normal_path and os.path.exists(normal_path):
            normal_img = cv2.imread(normal_path, cv2.IMREAD_COLOR)
            if normal_img is not None:
                normal_img_rgb = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
                normal_map_float = (normal_img_rgb.astype(np.float32) / 255.0 - 0.5) * 2.0
                results['gt_edge_map'] = normal_map_float
                if 'seg_fields' not in results: results['seg_fields'] = []
                if 'gt_edge_map' not in results['seg_fields']:
                    results['seg_fields'].append('gt_edge_map')
        else:
            pass
        return results



@TRANSFORMS.register_module()
class CustomCrop:
    def __init__(self, crop_size=(512, 512)):
        self.crop_size = crop_size

    def _crop(self, img, crop_size):
        target_h, target_w = crop_size
        h, w = img.shape[0], img.shape[1]
        if h < target_h or w < target_w:
            pad_h = max(target_h - h, 0);
            pad_w = max(target_w - w, 0)
            if img.ndim == 3:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
            else:
                img = np.pad(img, ((0, pad_h), (0, pad_w)), 'constant')
        h, w = img.shape[0], img.shape[1]

        offset_h = np.random.randint(0, h - target_h + 1)
        offset_w = np.random.randint(0, w - target_w + 1)
        return img[offset_h:offset_h + target_h, offset_w:offset_w + target_w]

    def __call__(self, results: dict) -> dict:
        keys_to_crop = ['img'] + results.get('seg_fields', [])

        if 'gt_seg_map' in results:
            results['gt_seg_map'] = self._crop(results['gt_seg_map'], self.crop_size)
        if 'img' in results:
            results['img'] = self._crop(results['img'], self.crop_size)
        if 'gt_edge_map' in results:
            results['gt_edge_map'] = self._crop(results['gt_edge_map'], self.crop_size)

        results['img_shape'] = self.crop_size
        results['pad_shape'] = self.crop_size
        return results


@TRANSFORMS.register_module()
class CustomPackSegInputs(PackSegInputs):
    def __init__(self, meta_keys=(
    'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction'), **kwargs):
        super().__init__(meta_keys=meta_keys, **kwargs)

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if 'img' in results:
            img = results['img']
            if img.ndim < 3:
                img = np.expand_dims(img, -1)
            # HWC to CHW
            packed_results['inputs'] = torch.from_numpy(img.transpose(2, 0, 1).copy()).contiguous()

        data_sample = SegDataSample()


        if self.meta_keys is not None:
            meta_info = {}
            for key in self.meta_keys:
                if key in results:
                    meta_info[key] = results[key]
            data_sample.set_metainfo(meta_info)


        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            if gt_seg_map.ndim == 3: gt_seg_map = np.squeeze(gt_seg_map, axis=-1)
            gt_sem_seg_tensor = torch.from_numpy(gt_seg_map.copy()).long()
            data_sample.set_data(dict(gt_sem_seg=PixelData(data=gt_sem_seg_tensor)))


        if 'gt_edge_map' in results:
            gt_edge_map = results['gt_edge_map']
            # HWC to CHW
            gt_normal_tensor = torch.from_numpy(gt_edge_map.transpose(2, 0, 1).copy()).contiguous()
            data_sample.set_data(dict(gt_edge_map=PixelData(data=gt_normal_tensor)))

        packed_results['data_samples'] = data_sample
        return packed_results