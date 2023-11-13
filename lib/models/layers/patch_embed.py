import torch.nn as nn
import torch
from timm.models.layers import to_2tuple

def update_bbox(box1,box2):
    box2_int=box2.round()
    box1_new=box1.clone()
    if box1[0]>=box2_int[0] and box1[2]<=box2_int[2]:
        if box2_int[1]<=box1[1] and box2_int[3]>box1[1]:
            box1_new[1]=box2_int[3]
        elif box2_int[3]>=box1[3] and box2_int[1]<box1[3]:
            box1_new[3]=box2_int[1]
    elif box1[1]>=box2_int[1] and box1[3]<=box2_int[3]:
        if box2_int[0]<=box1[0] and box2_int[2]>box1[0]:
            box1_new[0]=box2_int[2]
        elif box2_int[2]>=box1[2] and box2_int[0]<box1[2]:
            box1_new[2]=box2_int[0]
    return box1_new
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
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x,x2=None,gt_patch1=None,gt_patch2=None,occ_thres=0.5,norm_type='channel'):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        # norm_type: 'channel': channel-wise; 'wo':without norm; 'global': calculate a global mean/variance for the whole object area
        if x2==None:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)
            return x
        else:
            w_ori=x.shape[-1]
            x = self.proj(x)
            x2=self.proj(x2)
            patch_sz=w_ori//x.shape[-1]
            gt_patch1=gt_patch1*w_ori
            gt_patch2 = gt_patch2 * w_ori
            gt_patch1[:,2:]=gt_patch1[:,:2]+gt_patch1[:,2:]-1
            gt_patch2[:, 2:] = gt_patch2[:, :2] + gt_patch2[:, 2:] - 1
            gt_patch1 = (gt_patch1 / patch_sz).round()
            gt_patch2 = (gt_patch2 / patch_sz).round()

            occ_rate1 = cal_occ(gt_patch1.round(), gt_patch2.round())
            occ_rate2 = cal_occ(gt_patch2.round(), gt_patch1.round())
            opt_feat1 = []
            opt_feat2 = []
            feat_sz=x.shape[2]
            gt_patch1_new=[]
            gt_patch2_new=[]
            for idx in range(gt_patch1.shape[0]):
                if occ_rate1[idx]>occ_thres or gt_patch2[idx, 3]<=gt_patch2[idx, 1] or gt_patch2[idx, 2]<=gt_patch2[idx, 0] or \
                    gt_patch2[idx, 0]<0 or gt_patch2[idx, 1]<0 or gt_patch2[idx, 0]>=feat_sz or gt_patch2[idx, 1]>=feat_sz:
                    # print('patch2 no target area / occ too much',gt_patch2[idx],gt_patch1[idx],occ_rate1[idx])
                    opt_feat1.append(x[idx])
                    gt_patch1_new.append(gt_patch1[idx])
                else:
                    tmp=x[idx].clone()
                    tmp2 = x2[idx].clone()

                    if norm_type=='channel':
                        _,h, w = tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3]),
                               int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2])].shape
                        mean1 = tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3]),
                                          int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2])].detach().mean(dim=(1,2)).unsqueeze(1).unsqueeze(1).repeat(1,h,w)
                        std1 = tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3]),
                                          int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2])].detach().std(dim=(1,2)).unsqueeze(1).unsqueeze(1).repeat(1,h,w)
                        mean2 = tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3]),
                                int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2])].detach().mean(dim=(1,2)).unsqueeze(1).unsqueeze(1).repeat(1,h,w)
                        std2 = tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3]),
                               int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2])].detach().std(dim=(1,2)).unsqueeze(1).unsqueeze(1).repeat(1,h,w)

                        tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                            int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )] = \
                            (tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                             int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )]-mean2)/(std2+0.0001)*(std1+0.0001)+mean1
                    elif norm_type=='wo':
                        tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                        int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )] = \
                            (tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                             int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )])
                    elif norm_type == 'global':
                        mean1 = tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                                int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].detach().mean()
                        std1 = tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                               int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].detach().std()
                        mean2 = tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                                int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )].detach().mean()
                        std2 = tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                               int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )].detach().std()

                        tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                        int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )] = \
                            (tmp2[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                             int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )] - mean2) / (std2 + 0.0001) * (
                                        std1 + 0.0001) + mean1
                    opt_feat1.append(tmp)
                    gt_patch1_new.append(update_bbox(gt_patch1[idx],gt_patch2[idx]))


                if occ_rate2[idx]>occ_thres or gt_patch1[idx, 3]<=gt_patch1[idx, 1] or gt_patch1[idx, 2]<=gt_patch1[idx, 0] or \
                    gt_patch1[idx, 0]<0 or gt_patch1[idx, 1]<0 or gt_patch1[idx, 0]>=feat_sz or gt_patch1[idx, 1]>=feat_sz:
                    # print('patch1 no target area / occ too much')
                    opt_feat2.append(x2[idx])
                    gt_patch2_new.append(gt_patch2[idx])
                else:
                    tmp=x2[idx].clone()
                    tmp2=x[idx].clone()
                    if norm_type == 'channel':
                        _, h, w = tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                                  int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].shape
                        mean1 = tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                                int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )].detach().mean(dim=(1, 2)).unsqueeze(
                            1).unsqueeze(1).repeat(1, h, w)
                        std1 = tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                               int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )].detach().std(dim=(1, 2)).unsqueeze(
                            1).unsqueeze(1).repeat(1, h, w)
                        mean2 = tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                                int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].detach().mean(dim=(1, 2)).unsqueeze(
                            1).unsqueeze(1).repeat(1, h, w)
                        std2 = tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                               int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].detach().std(dim=(1, 2)).unsqueeze(
                            1).unsqueeze(1).repeat(1, h, w)

                        tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                        int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )] = \
                            (tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                             int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )] - mean2) / (std2 + 0.0001) * (
                                        std1 + 0.0001) + mean1
                    elif norm_type == 'wo':
                        tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                        int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )] = \
                            (tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                             int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )])
                    elif norm_type == 'global':
                        mean1 = tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                                int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )].detach().mean()
                        std1 = tmp[:, int(gt_patch2[idx, 1]):int(gt_patch2[idx, 3] ),
                               int(gt_patch2[idx, 0]):int(gt_patch2[idx, 2] )].detach().std()
                        mean2 = tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                                int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].detach().mean()
                        std2 = tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                               int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )].detach().std()

                        tmp[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                        int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )] = \
                            (tmp2[:, int(gt_patch1[idx, 1]):int(gt_patch1[idx, 3] ),
                             int(gt_patch1[idx, 0]):int(gt_patch1[idx, 2] )] - mean2) / (std2 + 0.0001) * (
                                    std1 + 0.0001) + mean1
                    opt_feat2.append(tmp)
                    gt_patch2_new.append(update_bbox(gt_patch2[idx], gt_patch1[idx]))
            opt_feat1=torch.stack(opt_feat1,0)
            opt_feat2 = torch.stack(opt_feat2, 0)

            gt_patch2_new=torch.stack(gt_patch2_new,0)
            gt_patch1_new = torch.stack(gt_patch1_new, 0)
            gt_patch2_new[:,2:]-=gt_patch2_new[:,:2]
            gt_patch1_new[:, 2:] -= gt_patch1_new[:, :2]

            if self.flatten:
                opt_feat1 = opt_feat1.flatten(2).transpose(1, 2)  # BCHW -> BNC
                opt_feat2 = opt_feat2.flatten(2).transpose(1, 2)  # BCHW -> BNC
            opt_feat1 = self.norm(opt_feat1)
            opt_feat2 = self.norm(opt_feat2)
            return opt_feat1,opt_feat2,gt_patch1_new,gt_patch2_new


