import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import cv2
import os
import numpy as np
import random
import math
from matplotlib import pyplot as plt

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x
def cal_occ(bboxes1, bboxes2):
    #[x1,y1,x2,y2]
    #occ rate of bboxes1
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    int_vol = np.multiply(int_h, int_w)
    vol1 = np.multiply(bboxes1[2] - bboxes1[0], bboxes1[3] - bboxes1[1])
    occ = (int_vol + 1e-8) / (vol1 + 1e-8)
    return occ

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, \
                 border_prob, sfactor, \
                 mode='pair', settings=None,occrate=0.5,scale=[0.7,2.0],center_range=2.0, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

        self.border_prob = border_prob
        self.sfactor = sfactor
        self.occrate = occrate
        self.scale = scale
        self.center_range = center_range

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    def _get_jittered_box_border(self, img,box, search_area_factor):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        h,w,_=img.shape
        xx,yy,ww,hh=box.numpy()
        tmp=random.random()
        crop_sz = math.ceil(math.sqrt(ww * hh) * search_area_factor)
        x1 = round(xx + 0.5 * ww - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(yy + 0.5 * hh - crop_sz * 0.5)
        y2 = y1 + crop_sz
        content_box = [x1,y1,x2,y2]
        crop_sz=content_box[2]-content_box[0]

        if tmp <= 0.25:
            #top x1,y1
            dy = int(random.uniform(crop_sz / 2 - hh / 4, crop_sz / 2+hh/4))
            dx=int(random.uniform(ww/2-crop_sz/2,crop_sz/2-ww/2))
        elif tmp <= 0.5:
            #left x1,y1
            dx = int(random.uniform(crop_sz / 2 - ww / 4, crop_sz / 2+ww/4))
            dy=int(random.uniform(hh/2-crop_sz/2,crop_sz/2-hh/2))
        elif tmp <= 0.75:
            #right x2,y2
            dx = -1*int(random.uniform(crop_sz / 2 - ww / 4, crop_sz / 2 + ww / 4))
            dy = int(random.uniform(hh / 2 - crop_sz / 2, crop_sz / 2 - hh / 2))
        else:
            # bottom x2,y2
            dy = -int(random.uniform(crop_sz / 2 - hh / 4, crop_sz / 2+hh/4))
            dx = int(random.uniform(ww / 2 - crop_sz / 2, crop_sz / 2 - ww / 2))
        content_box[0] += dx
        content_box[1] += dy
        content_box[2] += dx
        content_box[3] += dy
        jittered_sz = crop_sz / search_area_factor
        jittered_x1 = (content_box[0]+content_box[2])/2-jittered_sz/2
        jittered_y1 = (content_box[1]+content_box[3])/2-jittered_sz/2
        return torch.Tensor([jittered_x1,jittered_y1,jittered_sz,jittered_sz])
    def select_pos_center(self,box_ori,w,h,content_box,output_sz):
        tmp=max(content_box[0], box_ori[0] - self.center_range * box_ori[2])
        x1 = round(random.uniform(tmp,
                max(tmp,min(box_ori[0] + (self.center_range + 1) * box_ori[2], content_box[2] - w))))
        tmp=max(content_box[1], box_ori[1] - self.center_range * box_ori[3])
        y1 = round(random.uniform(tmp,
                max(tmp,min(box_ori[1] + (self.center_range + 1) * box_ori[3], content_box[3] - h))))
        x2 = round(min(content_box[2], x1 + w))
        y2 = round(min(content_box[3], y1 + h))
        bbox_obj = [x1, y1, x2, y2]
        return bbox_obj
    def crop_obj_by_box(self,data,crops,boxes,content_box,output_sz):
        crops=list(crops)
        for obj_idx in range(len(data['search_images_extra'])):
            for idx in range(len(data['search_images_extra'][obj_idx])):
                im=data['search_images_extra'][obj_idx][idx]
                target_bb=data['search_anno_extra'][obj_idx][idx]
                x, y, w, h=target_bb
                h_img, w_img, _ = im.shape
                w = min(x + w, w_img)
                h = min(y + h, h_img)
                x = max(0, x)
                y = max(0, y)
                w -= x
                h -= y
                if w < 1 or h < 1:
                    # print('cutmix: extra out of range')
                    continue
                x, y, w, h = int(x), int(y), math.ceil(w), math.ceil(h)
                scale = random.uniform(self.scale[0],self.scale[1])
                area_sz = boxes[idx][2] * boxes[idx][3] * scale * output_sz * output_sz
                h_new = np.sqrt(area_sz * h / w)
                w_new = w * h_new / h
                w_new, h_new = math.ceil(w_new), math.ceil(h_new)
                xx, yy, ww, hh = boxes[idx]
                xx, yy, ww, hh = output_sz * xx, output_sz * yy, output_sz * ww, output_sz * hh
                bbox_search = [int(max(content_box[idx][0], xx)), int(max(content_box[idx][1], yy)), \
                               math.ceil(min(content_box[idx][2], xx + ww)),
                               math.ceil(min(content_box[idx][3], yy + hh))]

                bbox_obj = self.select_pos_center([xx.item(), yy.item(), ww.item(), hh.item()], w_new, h_new, content_box[idx], output_sz)
                occ_rate=cal_occ(bbox_search,bbox_obj)
                ct=0
                cancel=False
                while occ_rate>self.occrate:
                    ct+=1
                    if ct>10:
                        # print('ct', ct)
                        w_new =w_new*0.9
                        h_new = h_new*0.9
                        w_new, h_new = math.ceil(w_new), math.ceil(h_new)
                    if ct>15:
                        cancel=True
                        # print('cancel cutmix')
                        break
                    bbox_obj = self.select_pos_center([xx.item(), yy.item(), ww.item(), hh.item()], w_new, h_new, content_box[idx], output_sz)
                    occ_rate = cal_occ(bbox_search, bbox_obj)
                if cancel:
                    continue
                obj = im[max(0,y):y + h, max(0,x):x + w, :]
                obj = cv2.resize(obj, (w_new, h_new))
                crops[idx][bbox_obj[1]:bbox_obj[3], bbox_obj[0]:bbox_obj[2], :] = obj[:bbox_obj[3] - bbox_obj[1],
                                                                                :bbox_obj[2] - bbox_obj[0], :]

        crops = tuple(crops)
        return crops,boxes
    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
            if data['is_cutmix']:
                data['template_images_extra'], data['template_anno_extra'], data['template_masks_extra'] = self.transform['joint'](
                    image=data['template_images_extra'], bbox=data['template_anno_extra'], mask=data['template_masks_extra'])
                data['search_images_extra'], data['search_anno_extra'], data['search_masks_extra'] = self.transform['joint'](
                    image=data['search_images_extra'], bbox=data['search_anno_extra'], mask=data['search_masks_extra'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            if s=='template':
                search_area_factor=self.search_area_factor[s]
            elif s=='search':
                search_area_factor =random.uniform(self.sfactor[0],self.sfactor[1])

            if s == 'search':
                if random.random() < self.border_prob:
                    jittered_anno = [self._get_jittered_box_border(x, a, search_area_factor) \
                                     for x, a in zip(data[s + '_images'], data[s + '_anno'])]
                else:
                    ok = False
                    ct = 0
                    while not ok:
                        if ct > 20:
                            break
                        jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
                        cxy_jitter = torch.stack(jittered_anno, dim=0)[:, :2] + torch.stack(jittered_anno, dim=0)[:,
                                                                                2:] / 2
                        cxy_box = torch.stack(data[s + '_anno'], dim=0)[:, :2] + torch.stack(data[s + '_anno'], dim=0)[
                                                                                 :, 2:] / 2
                        w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
                        min_sf = torch.max(torch.abs(cxy_jitter - cxy_box)) * 2 / torch.sqrt(w * h)
                        min_sf = min_sf.clip(self.sfactor[0])
                        if min_sf[0] <= self.sfactor[1]:
                            ok = True
                            search_area_factor = random.uniform(min_sf[0], self.sfactor[1])
                        ct += 1
            elif s == 'template':
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * search_area_factor)
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], search_area_factor,
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            w, h = torch.stack(boxes, dim=0)[:, 2]*self.output_sz[s], torch.stack(boxes, dim=0)[:, 3]*self.output_sz[s]
            if (w<1).any() or (h<1).any():
                data['valid'] = False
                # print(s,'is border:',is_border,search_area_factor,"Too small box is found. Replace it with new data.")
                return data
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            #_att: True: pad; False: img
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data
        if data['is_cutmix']:
            for s in ['template', 'search']:
                assert self.mode == 'sequence' or len(data[s + '_images_extra']) == 1, \
                    "In pair mode, num train/test frames must be 1"
                if not data['is_cutmix']:
                    break
                if s == 'template':
                    search_area_factor = self.search_area_factor[s]
                elif s == 'search':
                    search_area_factor = random.uniform(self.sfactor[0], self.sfactor[1])

                if s == 'search':
                    if random.random() < self.border_prob:
                        jittered_anno = [self._get_jittered_box_border(x, a, search_area_factor) \
                                         for x, a in zip(data[s + '_images_extra'], data[s + '_anno_extra'])]
                    else:
                        ok = False
                        ct = 0
                        while not ok:
                            if ct > 20:
                                break
                            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno_extra']]
                            cxy_jitter = torch.stack(jittered_anno, dim=0)[:, :2] + torch.stack(jittered_anno, dim=0)[:,
                                                                                    2:] / 2
                            cxy_box = torch.stack(data[s + '_anno_extra'], dim=0)[:, :2] + torch.stack(data[s + '_anno_extra'],
                                                                                                 dim=0)[
                                                                                     :, 2:] / 2
                            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
                            min_sf = torch.max(torch.abs(cxy_jitter - cxy_box)) * 2 / torch.sqrt(w * h)
                            min_sf = min_sf.clip(self.sfactor[0])
                            if min_sf[0] <= self.sfactor[1]:
                                ok = True
                                search_area_factor = random.uniform(min_sf[0], self.sfactor[1])
                            ct += 1
                elif s == 'template':
                    jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno_extra']]

                # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
                w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                crop_sz = torch.ceil(torch.sqrt(w * h) * search_area_factor)
                if (crop_sz < 1).any():
                    # data['is_cutmix'] = False
                    # break
                    data['valid']=False
                    return data
                    # print("Too small box is found. Replace it with new data.")


                # Crop image region centered at jittered_anno box and get the attention mask
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images_extra'], jittered_anno,
                                                                                  data[s + '_anno_extra'], search_area_factor,
                                                                                  self.output_sz[s],
                                                                                  masks=data[s + '_masks_extra'])
                w, h = torch.stack(boxes, dim=0)[:, 2] * self.output_sz[s], torch.stack(boxes, dim=0)[:, 3] * \
                       self.output_sz[s]
                if (w < 1).any() or (h < 1).any():
                    # data['is_cutmix'] = False
                    # # print(s,'is border:',is_border,search_area_factor,"Too small box is found. Replace it with new data.")
                    # break
                    data['valid'] = False
                    return data
                # Apply transforms
                data[s + '_images_extra'], data[s + '_anno_extra'], data[s + '_att_extra'], data[s + '_masks_extra'] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

                # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
                # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
                for ele in data[s + '_att_extra']:
                    if (ele == 1).all():
                        data['valid'] = False
                        return data
                # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
                for ele in data[s + '_att_extra']:
                    feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                    # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                    mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                    if (mask_down == 1).all():
                        data['valid'] = False
                        return data
        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        if data['is_cutmix']:
            if data["template_masks_extra"] is None or data["search_masks_extra"] is None:
                data["template_masks_extra"] = torch.zeros((1, self.output_sz["template_extra"], self.output_sz["template_extra"]))
                data["search_masks_extra"] = torch.zeros((1, self.output_sz["search_extra"], self.output_sz["search_extra"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
