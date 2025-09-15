# core/engine/normal_visualization_hook.py

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, List

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmseg.utils import get_palette, SampleList
from PIL import Image


@HOOKS.register_module()
class NormalSegVisHook(Hook):
    """
    A custom hook used to visually compare and save the following elements side by side as a single image during validation:
    1. Original input image
    2. Ground truth segmentation map
    3. Model-predicted segmentation map
    4. Ground truth normal map used for fusion
    """

    def __init__(self,
                 save_dir: str,
                 num_images_to_save: int = 5):
        super().__init__()
        self.save_dir = save_dir
        self.num_images_to_save = num_images_to_save
        self._saved_images_count_this_epoch = 0
        self._palette = None
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_palette(self, runner: Runner) -> List:
        if self._palette is None:

            dataset_meta = runner.val_dataloader.dataset.metainfo
            if 'palette' in dataset_meta:
                self._palette = dataset_meta['palette']
            else:
                num_classes = len(runner.val_dataloader.dataset.CLASSES)
                self._palette = get_palette('cityscapes', num_classes)
        return self._palette

    def _seg_to_pil(self, tensor: torch.Tensor, palette: list) -> Optional[Image.Image]:
        if tensor.ndim == 3:
            tensor = tensor.squeeze(0)
        seg_np = tensor.cpu().to(torch.uint8).numpy()
        color_seg = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
        for label_idx, color in enumerate(palette):
            color_seg[seg_np == label_idx, :] = color
        return Image.fromarray(color_seg)

    def _normal_to_pil(self, tensor: torch.Tensor) -> Optional[Image.Image]:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 3:
            return None

        tensor = (tensor.cpu().float() * 0.5 + 0.5).clamp(0, 1)

        normal_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(normal_np)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: SampleList) -> None:
        if self._saved_images_count_this_epoch >= self.num_images_to_save:
            return

        palette = self._get_palette(runner)

        for i in range(len(outputs)):
            if self._saved_images_count_this_epoch >= self.num_images_to_save:
                break

            data_sample = outputs[i]
            img_path = data_sample.get('img_path')

            try:
                original_img = Image.open(img_path).convert('RGB').resize(data_sample.pred_sem_seg.shape[1:][::-1])
            except Exception:

                original_img = Image.new('RGB', (512, 512), (255, 0, 0))

            gt_seg_img = self._seg_to_pil(data_sample.gt_sem_seg.data, palette)
            pred_seg_img = self._seg_to_pil(data_sample.pred_sem_seg.data, palette)


            if hasattr(data_sample, 'gt_normal_map') and data_sample.gt_normal_map is not None:
                gt_normal_img = self._normal_to_pil(data_sample.gt_normal_map.data)
            else:

                gt_normal_img = Image.new('RGB', original_img.size, (128, 128, 128))


            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            plot_items = [
                {'title': "Original Image", 'img': original_img},
                {'title': "Ground Truth Seg", 'img': gt_seg_img},
                {'title': "Predicted Seg", 'img': pred_seg_img},
                {'title': "Ground Truth Normals (Input)", 'img': gt_normal_img}
            ]

            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for ax, item in zip(axes.ravel(), plot_items):
                ax.imshow(item['img'])
                ax.set_title(item['title'])
                ax.axis('off')

            img_filename_stem = os.path.splitext(os.path.basename(img_path))[0]
            epoch_str = f"epoch_{runner.epoch + 1}"
            save_filename = f"{epoch_str}_val_{img_filename_stem}.png"
            save_path = os.path.join(self.save_dir, save_filename)

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
            self._saved_images_count_this_epoch += 1

    def before_val_epoch(self, runner: Runner) -> None:
        self._saved_images_count_this_epoch = 0