def vd_eval(l_annotations, l_outputs, fp_out=False):
    """
        calculating the number of TP,FP,FN, precision and recall
        Args:
            l_annotations (list):
            l_outputs (list):
            fp_out (bool):
                default:False
        Returns:
            l_evals (list):
                [TP,FP,FN,precision, recall]
            l_fpbb (list):
                If fp_out is True
    """
    if fp_out:
        return l_evals, l_fpbb
    else:
        return l_evals


import glob
import json
import os

import cv2
import numpy as np
#from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

#from dataset.cocodataset import *
from utils.utils import *


class VDEvaluator():
    """
    Vehicle Detection AP Evaluation class.
    All the data in the validation dataset are processed \
    and evaluated.
    """
    def __init__(self, data_dir, json_file, img_size, confthre, nmsthre,  min_size=1):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.img_size = img_size
        self.confthre = confthre # 0.005 from darknet
        self.nmsthre = nmsthre # 0.45 (darknet)
        self.min_size = min_size

    def evaluate(self, model):
        """
        VD average precision (AP) Evaluation. Iterate inference on the val dataset
        and the results are evaluated.
        Args:
            model : model object
        Returns:
            ap50 (float) : calculated AP for IoU=50
        """
        
        model.eval()
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
      
        img_list = glob.glob(os.path.join(self.data_dir, '*.jpg'))
        img_list.extend(glob.glob(os.path.join(self.data_dir,'*.png')))

        json_open = open(os.path.join(self.data_dir, self.json_file), 'r')
        json_load = json.load(json_open)

        all_gt_bboxes = []
        all_yolo_bboxes = []

        for image in img_list:
            gt_bboxes = []
            yolo_bboxes = []
            
            #predict bbox
            img = cv2.imread(image)
            if img is None:
                print('read image error')
            img, info_img = preprocess(img, self.img_size, jitter=0)  # info = (h, w, nh, nw, dx, dy)
            img = np.transpose(img / 255., (2, 0, 1))
            img = torch.from_numpy(img).float().unsqueeze(0)
            
            # gt bbox
            annotations = json_load[os.path.basename(image)]['regions']
            if len(annotations) > 0:
                for anno in annotations:
                    if anno['bb'][2] > self.min_size and anno['bb'][3] > self.min_size:
                        box = [anno['bb'][0], anno['bb'][1],
                            anno['bb'][0] + anno['bb'][2], anno['bb'][1] + anno['bb'][3]]
                        box = resized_bbox(box, info_img)
                        gt_bboxes.append([os.path.basename(image),box])
                        #gt_bboxes.append({"image":os.path.basename(image), "bbox":box})
            all_gt_bboxes.extend(gt_bboxes)
            
            

            with torch.no_grad():
                img = Variable(img.type(Tensor))
                outputs = model(img)
                # delete outputs with inf
                
                #outputs= postfilter(outputs)
                outputs = postprocess(outputs, 1, self.confthre, self.nmsthre)
            if outputs[0] is not None:
                outputs = outputs[0].cpu().data
                for output in outputs:
                    x1 = float(output[0])
                    y1 = float(output[1])
                    x2 = float(output[2])
                    y2 = float(output[3])
                    score = float(output[4].data.item() * output[5].data.item())
                    label = int(output[6])
                    yolo_bboxes.append([os.path.basename(image),[x1,y1,x2,y2], score, label, False])
                    #yolo_bboxes.append({"image":os.path.basename(image), "category_id": label,
                    # "bbox": [x1,y1,x2,y2],"score": score})
                
            # judge TP or FP
                # score sort
                yolo_bboxes = sorted(yolo_bboxes, key=lambda x: x[2])
                
                for j in range(len(yolo_bboxes)):
                    a = None
                    t = 0
                    for gt_bbox in gt_bboxes:
                        iou = np_bboxes_iou(np.array(yolo_bboxes[j][1]), np.array(gt_bbox[1]).reshape(1,4))
                        if iou > max(0.5, t):
                            a = gt_bbox
                            t = iou
                    if a != None:
                        gt_bboxes.remove(a)
                        yolo_bboxes[j][4] = True
            
            all_yolo_bboxes.extend(yolo_bboxes)
        # calculating AP
        ap50 = 0
        precision50 = 0
        recall50 = 0
        F_measure = 0
        if len(all_yolo_bboxes)==0:
            print('no pred')
        if len(all_gt_bboxes)==0:
            print('no gt')
        if len(all_yolo_bboxes) > 0 and len(all_gt_bboxes) > 0:
            tp_or_fp = [yolo_bbox[4] for yolo_bbox in all_yolo_bboxes]
            acc_tp = np.cumsum(tp_or_fp)
            acc_fp = np.cumsum(np.logical_not(tp_or_fp))

            precision = acc_tp /(acc_tp + acc_fp)
            recall = acc_tp / len(all_gt_bboxes)

            modified_recall = np.concatenate([[0], recall, [1]])
            modified_precision = np.concatenate([[0], precision, [0]])

            # 末尾から累積最大値を計算する。
            modified_precision = np.maximum.accumulate(modified_precision[::-1])[::-1]

            # AP50 を計算する。
            ap50 = (np.diff(modified_recall) * modified_precision[1:]).sum()
            
            precision50 = precision[-1]
            recall50 = recall[-1]
        if (precision50+recall50)>0:
            F_measure = 2*(precision50*recall50)/(precision50+recall50)
        
        outputs = 0
        
        return ap50, precision50, recall50, F_measure
