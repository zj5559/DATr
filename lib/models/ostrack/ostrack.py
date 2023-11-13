"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cal_occ(bboxes1,bboxes2):
    #[x1,y1,x2,y2], occ_rate of box1
    int_xmin = torch.max(bboxes1[:,0], bboxes2[:,0])
    int_ymin = torch.max(bboxes1[:,1], bboxes2[:,1])
    int_xmax = torch.min(bboxes1[:,2], bboxes2[:,2])
    int_ymax = torch.min(bboxes1[:,3], bboxes2[:,3])

    int_h = torch.clamp(int_ymax - int_ymin, min=0.0)
    int_w = torch.clamp(int_xmax - int_xmin, min=0.0)

    int_vol = int_h*int_w
    vol1 = (bboxes1[:,2] - bboxes1[:,0])*(bboxes1[:,3] - bboxes1[:,1])
    occ = (int_vol + 1e-8) / (vol1 + 1e-8)
    return occ
class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward1(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z1=template, x1=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                is_train=False,
                template_extra=None,
                search_extra=None,
                search_anno=None,
                search_anno_extra=None,
                imix_occrate=0.5,
                imix_reverse_prob=0.0,
                norm_type='channel',
                ce_template_mask=None,
                ce_template_mask_extra=None,
                ce_keep_rate=None,
                mix_type=0,
                return_last_attn=False,
                ):
        if not is_train:
            out=self.forward1(template=template,search=search,ce_template_mask=ce_template_mask,ce_keep_rate=ce_keep_rate,return_last_attn=return_last_attn)
            return out
        x21=None
        if template_extra is not None:
            gt_bbox1 = search_anno[-1].clone()  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_bbox2 = search_anno_extra[-1].clone()
            # gt_bbox1[:, 2:] += gt_bbox1[:, :2]  # [x1,y1,x2,y2]
            # gt_bbox2[:, 2:] += gt_bbox2[:, :2]

            if mix_type==1:
                # return_last_attn=True
                x11, aux_dict11= self.backbone(z1=template, x1=search,z2=template_extra,x2=search_extra,
                                             gt_patch1=gt_bbox1, gt_patch2=gt_bbox2, \
                                             occ_thres=imix_occrate,norm_type=norm_type, \
                                        ce_template_mask=ce_template_mask,
                                             ce_template_mask2=ce_template_mask_extra,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )

            if mix_type==2:
                x11,x21, aux_dict11, aux_dict21 = self.backbone(z1=template, x1=search, z2=template_extra, x2=search_extra,
                                              gt_patch1=gt_bbox1, gt_patch2=gt_bbox2, \
                                            occ_thres=imix_occrate, norm_type=norm_type,reverse=True, \
                                              ce_template_mask=ce_template_mask,
                                              ce_template_mask2=ce_template_mask_extra,
                                              ce_keep_rate=ce_keep_rate,
                                              return_last_attn=return_last_attn, )
        else:
            x11, aux_dict11 = self.backbone(z1=template, x1=search,
                                          ce_template_mask=ce_template_mask,
                                          ce_keep_rate=ce_keep_rate,
                                          return_last_attn=return_last_attn, )
        feat_last11 = x11[-1] if isinstance(x11, list) else x11
        out = self.forward_head(feat_last11, None)

        if x21 is not None:
            feat_last21 = x21[-1] if isinstance(x21, list) else x21
            out21 = self.forward_head(feat_last21, None)
            return [out,out21],True
        else:
            return out, False



    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
    def process_feature(self,cat_feature):
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        return opt_feat
    def forward_head_imix(self, opt_feat, gt_score_map=None):
        bs, C, _,_ = opt_feat.size()
        Nq=1
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
