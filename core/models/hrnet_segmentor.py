# core/models/hrnet_segmentor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from mmengine.structures import PixelData
from mmseg.utils import SampleList, add_prefix
from mmseg.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor


@MODELS.register_module()
class HRNetLiftFeatSegmentor(BaseSegmentor):

    def __init__(self,
                 backbone,
                 decode_head,
                 fusion_module,
                 data_preprocessor=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.fusion_module = MODELS.build(fusion_module)
        self.decode_head = MODELS.build(decode_head)
        if auxiliary_head is not None:
            self.auxiliary_head = MODELS.build(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:

        original_backbone_outputs = self.backbone(inputs)
        return original_backbone_outputs

    def _get_gt_normals(self, data_samples: SampleList) -> torch.Tensor:

        gt_normals_list = [
            ds.gt_edge_map.data for ds in data_samples
            if hasattr(ds, 'gt_edge_map') and ds.gt_edge_map is not None
        ]


        gt_normals = torch.stack(gt_normals_list, dim=0)
        return gt_normals

    def loss(self, inputs: torch.Tensor, data_samples: SampleList) -> dict:

        losses = dict()

        original_hrnet_outputs = self.extract_feat(inputs)

        features_to_fuse = original_hrnet_outputs[0]

        gt_normals = self._get_gt_normals(data_samples).to(features_to_fuse.device)

        f_enhanced = self.fusion_module(features_to_fuse, gt_normals)

        modified_hrnet_outputs = list(original_hrnet_outputs)
        modified_hrnet_outputs[0] = f_enhanced

        loss_decode = self._decode_head_forward_train(tuple(modified_hrnet_outputs), data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(original_hrnet_outputs, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: torch.Tensor, data_samples: SampleList) -> SampleList:

        original_hrnet_outputs = self.extract_feat(inputs)
        features_to_fuse = original_hrnet_outputs[0]

        gt_normals = self._get_gt_normals(data_samples).to(features_to_fuse.device)

        f_enhanced = self.fusion_module(features_to_fuse, gt_normals)

        modified_hrnet_outputs = list(original_hrnet_outputs)
        modified_hrnet_outputs[0] = f_enhanced

        seg_logits = self.decode_head.predict(tuple(modified_hrnet_outputs), data_samples, self.test_cfg)


        for i, data_sample in enumerate(data_samples):
            if hasattr(data_sample, 'gt_edge_map') and data_sample.gt_edge_map is not None:
                data_sample.gt_normal_map = data_sample.gt_edge_map

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: Optional[SampleList] = None) -> torch.Tensor:

        original_hrnet_outputs = self.extract_feat(inputs)
        features_to_fuse = original_hrnet_outputs[0]

        gt_normals = self._get_gt_normals(data_samples).to(features_to_fuse.device)

        f_enhanced = self.fusion_module(features_to_fuse, gt_normals)

        modified_hrnet_outputs = list(original_hrnet_outputs)
        modified_hrnet_outputs[0] = f_enhanced

        return self.decode_head.forward(tuple(modified_hrnet_outputs))

    def _decode_head_forward_train(self, x: Tuple[torch.Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(x, data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, x: Tuple[torch.Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(x, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(x, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses