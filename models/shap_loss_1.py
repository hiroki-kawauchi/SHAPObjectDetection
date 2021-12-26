import torch
from torch.autograd import Variable
from models.yolov3_shap import *
from utils.utils import *
import sys
sys.path.append('/home/linuxserver01/packages')
from captum.attr import GradientShap
from captum.attr import visualization as viz

import numpy as np

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def bb_mask(bboxes, attr, img_H=608, img_W=608, std=True):
    pixel_attr = np.sum(attr, axis=2)
    if std:
        pixel_attr = zscore(pixel_attr)
            
    pixel_mask = np.zeros((img_H,img_W))
    in_area = 0
    for [x1,y1,x2,y2] in bboxes:
        for i in range(int(y1), int(y2)):
            for j in range(int(x1),int(x2)):
                pixel_mask[i,j] = 1
                in_area += 1
    out_area = img_H*img_W - in_area
    
    in_pos = 0
    in_neg = 0
    out_pos = 0
    out_neg = 0
    for i in range(img_H):
        for j in range(img_W):
            if pixel_mask[i,j]>0:
                if pixel_attr[i,j]>=0:
                    in_pos += pixel_attr[i,j]
                else:
                    in_neg += pixel_attr[i,j]
            else:
                if pixel_attr[i,j]>=0:
                    out_pos += pixel_attr[i,j]
                else:
                    out_neg += pixel_attr[i,j]
    if in_area >0:
        in_pos = in_pos/in_area
        in_neg = in_neg/in_area
    if out_area>0:
        out_pos = out_pos/out_area
        out_neg = out_neg/out_area
    
    return in_pos, in_neg, out_pos, out_neg


def shaploss(imgs, labels, model, num_classes, confthre, nmsthre, stdevs=0.1, target_y='cls', 
              multiply_by_inputs=True, n_samples=5, alpha=1.0, beta=1.0):
    """
        Calculating SHAP-loss
        Args:
            imgs (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            labels (torch.Tensor) : label array whose shape is :math:`(N, 100, 5)`
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_imgs :
                info_img : tuple of h, w, nh, nw, dx, dy.
                    h, w (int): original shape of the image
                    nh, nw (int): shape of the resized image without padding
                    dx, dy (int): pad size
        Returns:
            loss (torch.Tensor)
    """
    # Find TP output
    model.eval()
    model(imgs)
    with torch.no_grad():
        outputss = model(imgs)
        outputss = postprocess(outputss, num_classes, confthre, nmsthre)
    
    labels = labels.cpu().data
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)# numbers of gt-objects in each image

    gt_bboxes_tp = [None for _ in range(len(nlabel))]
    output_id_tp = [None for _ in range(len(nlabel))]
    for outputs, label, i in zip(outputss, labels, range(len(nlabel))):
        gt_bboxes = []
        yolo_bboxes = []
        # gt bbox
        H = imgs.size(2)
        W = imgs.size(3)
        for j in range(nlabel[i]):
            _, xc, yc, w, h = label[j] # [class, xc, yc, w, h]
            xyxy_label = [(xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H] # [x1,y1,x2,y2]
            gt_bboxes.append([i,xyxy_label])

        if outputss[0] is not None:
            outputs = outputss[0].cpu().data
            for output in outputs:
                x1 = output[0].data.item()
                y1 = float(output[1].data.item())
                x2 = float(output[2].data.item())
                y2 = float(output[3].data.item())
                score = float(output[4].data.item() * output[5].data.item())
                label = int(output[6].data.item())
                yolo_bbox = [i,[x1,y1,x2,y2], score, label, False, 
                             int(output[7].data.item()),int(output[8].data.item()), int(output[9].data.item()), int(output[10].data.item())]#layer_num, anchor_num, x,y
                yolo_bboxes.append(yolo_bbox)

        # judge TP or FP
            # score sort
            yolo_bboxes = sorted(yolo_bboxes, key=lambda x: x[2])

            for k in range(len(yolo_bboxes)):
                a = None
                t = 0
                for gt_bbox in gt_bboxes:
                    iou = np_bboxes_iou(np.array(yolo_bboxes[k][1]), np.array(gt_bbox[1]).reshape(1,4))
                    if iou > max(0.5, t):
                        a = gt_bbox
                        t = iou
                if a != None:
                    gt_bboxes_tp[i] = a[1]
                    gt_bboxes.remove(a)
                    yolo_bboxes[k][4] = True
                    output_id_tp[i] = [yolo_bboxes[k][5], yolo_bboxes[k][6], yolo_bboxes[k][7], yolo_bboxes[k][8]]
                    break
      
    def yolo_wrapper(inp, output_id):
        layer_num, anchor_num, x, y = output_id
        output = model(inp, shap=True)
        return output[layer_num][:,anchor_num,y,x]
    if target_y == 'obj':
        target_y = 4
    elif target_y == 'cls':
        target_y = 5
    
    num_no_tp = 0
    inside_bb_sum = 0
    outside_bb_sum = 0
    
    for gt_bbox, output_id, img in zip(gt_bboxes_tp, output_id_tp, imgs):
        if gt_bbox == None:
            num_no_tp += 1
            continue
        
        img = img.reshape(1,3,img.size(1),img.size(2))

        baselines = img * 0
        with torch.no_grad():
            gs = GradientShap(yolo_wrapper, multiply_by_inputs=multiply_by_inputs)
            attr = gs.attribute(img, additional_forward_args=output_id, 
                                n_samples=n_samples, stdevs=stdevs,
                                baselines=baselines, target=target_y,
                                return_convergence_delta=False)
        attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1,2,0))
        in_pos, in_neg, out_pos, out_neg = bb_mask([gt_bbox], attr, img_H=imgs.size(2), img_W=imgs.size(3))
        inside_bb_sum -= in_neg
        outside_bb_sum += out_pos
    loss = alpha * inside_bb_sum + beta * outside_bb_sum
    if len(imgs)>num_no_tp:
        loss = loss*len(imgs)/(len(imgs) - num_no_tp) 
    return torch.tensor(loss, device="cuda:0",dtype=torch.float)