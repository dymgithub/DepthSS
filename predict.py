# predict.py

import argparse
import os
import torch
import mmcv
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import get_palette
from mmcv.transforms import Compose
from PIL import Image
import numpy as np

import core.datasets.transform
import core.models.DSEN
import core.models.hrnet_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with a custom segmentor')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image')
    parser.add_argument('output_path', help='path to save segmentation result')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope('mmseg')
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.to(args.device)
    model.eval()

    transforms = []
    for transform_cfg in cfg.test_pipeline:
        transform_type = transform_cfg.pop('type')
        if transform_type.startswith('core.datasets.'):
            module_name, class_name = transform_type.rsplit('.', 1)
            if class_name == 'LoadNormalAnnotations':
                from core.datasets.transform import LoadNormalAnnotations
                transforms.append(LoadNormalAnnotations(**transform_cfg))
            elif class_name == 'CustomPackSegInputs':
                from core.datasets.transform import CustomPackSegInputs
                transforms.append(CustomPackSegInputs(**transform_cfg))
        else:
            from mmcv.transforms import LoadImageFromFile, Resize
            if transform_type == 'LoadImageFromFile':
                transforms.append(LoadImageFromFile(**transform_cfg))
            elif transform_type == 'Resize':
                transforms.append(Resize(**transform_cfg))
            elif transform_type == 'LoadAnnotations':
                pass

    pipeline = Compose(transforms)

    data = dict(img_path=args.img_path)
    data = pipeline(data)

    inputs = data['inputs'].unsqueeze(0).to(args.device)
    data_samples = [data['data_samples'].to(args.device)]

    with torch.no_grad():
        results = model.predict(inputs, data_samples)

    pred_sem_seg = results[0].pred_sem_seg.cpu().data.numpy()

    palette = cfg.palette
    if palette is None:
        palette = get_palette('cityscapes', len(cfg.classes))

    color_seg = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[pred_sem_seg == label, :] = color

    result_img = Image.fromarray(color_seg)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result_img.save(args.output_path)
    print(f"Segmentation result saved to {args.output_path}")


if __name__ == '__main__':
    main()