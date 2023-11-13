from . import BaseActor
from lib.utils.misc import NestedTensor
import random
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.models.ostrack.ostrack import OSTrack
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

class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        if not self.cfg.DA.imix:
            out_dict = self.forward_pass_test(data)
            # compute losses
            loss, status = self.compute_losses(out_dict, data)
        else:
            loss, status = self.forward_imix(data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict
    def forward_pass_test(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        # out_dict = self.net(template=template_list,
        #                     search=search_img,
        #                     ce_template_mask=box_mask_z,
        #                     ce_keep_rate=ce_keep_rate,
        #                     return_last_attn=False)
        if isinstance(self.net,OSTrack):
            out_dict = self.net.forward1(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        else:
            out_dict = self.net.module.forward1(template=template_list,
                                search=search_img,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        return out_dict
    def forward_imix(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        template_list_extra=None
        search_img_extra=None
        box_mask_z_extra = None
        if data['is_cutmix'][0]:
            template_list_extra = []
            for i in range(self.settings.num_template):
                template_img_i_extra = data['template_images_extra'][i].view(-1,
                                                                 *data['template_images_extra'].shape[
                                                                  2:])  # (batch, 3, 128, 128)
                # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
                template_list_extra.append(template_img_i_extra)

            search_img_extra = data['search_images_extra'][0].view(-1, *data['search_images_extra'].shape[2:])  # (batch, 3, 320, 320)
            # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z_extra = generate_mask_cond(self.cfg, template_list_extra[0].shape[0], template_list_extra[0].device,
                                                data['template_anno_extra'][0])
            if len(template_list_extra) == 1:
                template_list_extra = template_list_extra[0]
        search_anno=data['search_anno'] if data['is_cutmix'][0] else None
        search_anno_extra = data['search_anno_extra'] if data['is_cutmix'][0] else None
        out,is_imix = self.net(template=template_list,
                              search=search_img,
                              is_train=True,
                              template_extra=template_list_extra,
                              search_extra=search_img_extra,
                              search_anno=search_anno,
                              search_anno_extra=search_anno_extra,
                              imix_occrate=self.cfg.DA.imix_occrate,
                              norm_type=self.cfg.DA.norm_type,
                              imix_reverse_prob=self.cfg.DA.imix_reverse_prob,
                              ce_template_mask=box_mask_z,
                              ce_template_mask_extra=box_mask_z_extra,
                              mix_type=self.cfg.DA.mix_type,
                              ce_keep_rate=ce_keep_rate,
                              return_last_attn=False)
        if not is_imix:
            if 'new_bbox' in out and self.cfg.DA.update_bbox:
                data['search_anno']=out['new_bbox'].unsqueeze(0)
            loss, status=self.compute_losses(out,data)
            return loss,status
        else:
            # loss11,status11=self.compute_losses_imix(out[0],data['search_anno'])
            # loss22, status22 = self.compute_losses_imix(out[1], data['search_anno_extra'])
            if 'new_bbox' in out[0] and self.cfg.DA.update_bbox:
                data['search_anno']=out[0]['new_bbox'].unsqueeze(0)
                data['search_anno_extra'] = out[1]['new_bbox'].unsqueeze(0)
            loss_mix1, status_mix1 = self.compute_losses_imix(out[0], data['search_anno'])
            loss_mix2, status_mix2 = self.compute_losses_imix(out[1], data['search_anno_extra'])
            loss=(loss_mix1+loss_mix2)
            status = {"Loss/total": loss.item(),
                      "Loss/giou": (status_mix1["Loss/giou"]+status_mix2["Loss/giou"]),
                      "Loss/l1": (status_mix1["Loss/l1"]+status_mix2["Loss/l1"]),
                      "Loss/location": (status_mix1["Loss/location"]+status_mix2["Loss/location"]),
                      "IoU": (status_mix1["IoU"]+status_mix2["IoU"])/2}
            return loss,status
    def compute_losses_imix(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict[-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
