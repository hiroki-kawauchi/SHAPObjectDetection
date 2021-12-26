from __future__ import division
import torch
import numpy as np
import cv2


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1) + 1
    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        
        sum_area = bbox_area[i] + bbox_area[selec] - area + 1e-16
        iou = area / sum_area
        '''
        iou = np.zeros_like(area)
        if sum_area>0:
            iou = area / sum_area
        '''
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)

def postfilter(a):
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(len(a[i,j])):
                if a[i,j,k]>1 or a[i,j,k]<0:
                    a[i,j,k] = torch.tensor([1.])
                else:
                    a[i,j,k] = a[i,j,k]
    return a


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 8)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        class_pred = class_pred[0]
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get detections with higher confidence scores than the threshold
        ind = (image_pred[:, 5:5 + num_classes] * image_pred[:, 4][:, None] >= conf_thre).nonzero(as_tuple=False)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((
                image_pred[ind[:, 0], :5],
                image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
                ind[:, 1].float().unsqueeze(1),
                image_pred[ind[:, 0], 5 + num_classes:]
                ), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -5].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -5] == c]
            nms_in = detections_class.cpu().numpy()
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4]*nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def np_bboxes_iou(a_bbox, b_bboxes):
    """Calculate intersection over union (IOU).
    
    Args:
        a (array-like): 1-D Array with shape (4,) representing bounding box.
        b (array-like): 2-D Array with shape (NumBoxes, 4) representing bounding boxes.
    
    Returns:
        [type]: [description]

    from
    https://github.com/nekobean/pascalvoc_metrics/blob/master/pascalvoc_metrics.py
    """
    # 短形 a_bbox と短形 b_bboxes の共通部分を計算する。
    xmin = np.maximum(a_bbox[0], b_bboxes[:, 0])
    ymin = np.maximum(a_bbox[1], b_bboxes[:, 1])
    xmax = np.minimum(a_bbox[2], b_bboxes[:, 2])
    ymax = np.minimum(a_bbox[3], b_bboxes[:, 3])
    i_bboxes = np.column_stack([xmin, ymin, xmax, ymax])

    # 矩形の面積を計算する。
    a_area = calc_area(a_bbox)
    b_area = np.apply_along_axis(calc_area, 1, b_bboxes)
    i_area = np.apply_along_axis(calc_area, 1, i_bboxes)

    # IOU を計算する。
    iou = i_area / (a_area + b_area - i_area)

    return iou


def calc_area(bbox):
    """Calculate area of boudning box.
    
    Args:
        bboxes (array-like): 1-D Array with shape (4,) representing bounding box.
    
    Returns:
        float: Areea
    """
    # 矩形の面積を計算する。
    # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
    width = max(0, bbox[2] - bbox[0] + 1)
    height = max(0, bbox[3] - bbox[1] + 1)

    return width * height

    
def label2yolobox(labels, info_img, maxsize, lrflip):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy = info_img
    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h
    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize
    if lrflip:
        labels[:, 1] = 1 - labels[:, 1]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    h, w, nh, nw, dx, dy = info_img
    y1, x1, y2, x2 = box
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label


def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position

    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape
    img = img[:, :, ::-1]
    assert img is not None

    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw))\
                 / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h

    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy+nh, dx:dx+nw, :] = img

    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img

def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale

def random_distort(img, hue, saturation, exposure):
    """
    perform random distortion in the HSV color space.
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        hue (float): random distortion parameter.
        saturation (float): random distortion parameter.
        exposure (float): random distortion parameter.
    Returns:
        img (numpy.ndarray)
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors

def get_vd_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    label_names = ('vehicle')
    class_ids = [0]

    cls_colors = np.random.randint(128, 255, size=(1, 3))

    return label_names, class_ids, cls_colors

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bboxes_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics



def resized_bbox(box, info_img):
    """
    Transform gt-size bbox to resized bbox
    Args:
        box (list): [x1,y1,x2,y2]
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing

    Returns:
        r_box (list): [x1,y1,x2,y2](resized)
    """
    h, w, nh, nw, dx, dy = info_img

    x_1 = (box[0] / w) * nw + dx 
    y_1 = (box[1] / h) * nh + dy
    x_2 = (box[2] / w) * nw + dx
    y_2 = (box[3] / h) * nh + dy
    
    return [x_1,y_1,x_2,y_2]



from torch.autograd import Variable
import os
import glob
import json
import torch
import cv2

def make_data_dict(model, imgsize, input_dir, json_file='anno_data.json',
                   min_size=1,confthre=0.005, nmsthre=0.45, gpu=0):
    """
        returning lists of bounding box [x_1,y_1,x_2,y_2] of ground truth and yolo outputs
        no-skipping no object image

    Returns:
        pred_dict:
            list of dict below
            {"image_id": id_, "image_name":jpg/png, "category_id": label, "pred_bbox": bbox,
            "obj_score":, "class_score":, "score": score}
        gt_dict:
            list of dict below
            {"image_id": id_, "image_name":jpg/png, "gt_bbox":, "category_id": label}
    """
    img_list = glob.glob(os.path.join(input_dir,'*.jpg'))
    img_list.extend(glob.glob(os.path.join(input_dir,'*.png')))

    json_open = open(os.path.join(input_dir, json_file), 'r')
    json_load = json.load(json_open)

    pred_dict = []
    gt_dict = []

    for id_ in range(len(img_list)):
        image = img_list[id_]
        img = cv2.imread(image)
        img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        if gpu >= 0:
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))

        # gt bbox
        annotations = json_load[os.path.basename(image)]['regions']

        if len(annotations) > 0: 
            for anno in annotations:
                if anno['bb'][2] > min_size and anno['bb'][3] > min_size:
                    box = [anno['bb'][1], anno['bb'][0],
                        anno['bb'][1] + anno['bb'][3], anno['bb'][0] + anno['bb'][2]]
                    box = resized_bbox(box, info_img)
                    gt_dict.append({"image_id": id_, "image_name":os.path.basename(image),
                                    "gt_bbox":box})
                    classes.append(anno['class_id'])
        
        # yolo bbox
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, 1, confthre, nmsthre)
        
        bboxes = list()

        if outputs[0] is None:
            print("No Objects Deteted!!")
        
        else:   

            
            #classes = list()
            #colors = list()

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

                #cls_id = class_ids[int(cls_pred)]
                #print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
                #print('\t+ Label: %s, Conf: %.5f' %
                #      (class_names[cls_id], cls_conf.item()))
                # box = yolobox2label([y1, x1, y2, x2], info_img)
                bboxes.append([x1,y1,x2,y2])
                #classes.append(cls_id)
                #colors.append(class_colors[int(cls_pred)])


            #vis_bbox(
            #    img_raw, bboxes, #label=classes, label_names=class_names,
            #    instance_colors=colors, linewidth=2)
            #plt.show()

            #plt.savefig(os.path.join(folder_path, 'yolo_' + os.path.basename(image)))

        yolo_bboxes.append([os.path.basename(image), bboxes])



def bb_list(input_dir, cfg,
               min_size = 1, img_size=416, gpu=0,
               weights_path=None, ckpt=None):
    """
        returning lists of bounding box [x_1,y_1,x_2,y_2] of ground truth and yolo outputs
        no-skipping no object image

    Returns:
        gt_bboxes:
            [...[image_name,[list of bounding box]]...]
        yolo_bboxes:
    """
    with open(cfg, 'r') as f:
        cfg = yaml.load(f) 
    
    img_list = glob.glob(os.path.join(input_dir,'*.jpg'))
    img_list.extend(glob.glob(os.path.join(input_dir,'*.png')))
    
    imgsize = cfg['TEST']['IMGSIZE']
    model = YOLOv3(cfg['MODEL'])
    num_classes = cfg['MODEL']['N_CLASSES']

    confthre = cfg['TEST']['CONFTHRE']
    nmsthre = cfg['TEST']['NMSTHRE']

    if gpu >= 0:
        model.cuda(gpu)    

    assert weights_path or ckpt, 'One of --weights_path and --ckpt must be specified'

    if weights_path:
        print("loading yolo weights %s" % (weights_path))
        parse_yolo_weights(model, weights_path)
    elif ckpt:
        print("loading checkpoint %s" % (ckpt))
        state = torch.load(ckpt)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    model.eval()
    
    json_open = open(os.path.join(input_dir, 'anno_data.json'), 'r')
    json_load = json.load(json_open)

    gt_bboxes = []
    yolo_bboxes = []

    for image in img_list:
        
        #folder_name = os.path.splitext(os.path.basename(image))[0]
        #folder_path = os.path.join(output_dir, folder_name)
        #os.makedirs(folder_path, exist_ok=True)
        
        img = cv2.imread(image)
        # output raw image
        #cv2.imwrite(os.path.join(folder_path, os.path.basename(image)), img)
        
        
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        
        img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        if gpu >= 0:
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))


        # gt bbox
        annotations = json_load[os.path.basename(image)]['regions']
        
        bboxes = list()
        
        if len(annotations) == 0:
            print("No Objects Exist!!")
        
        else:
            
            #classes = list()
            #colors = list()
            
            for anno in annotations:
                if anno['bb'][2] > min_size and anno['bb'][3] > min_size:
                    box = [anno['bb'][1], anno['bb'][0],
                        anno['bb'][1] + anno['bb'][3], anno['bb'][0] + anno['bb'][2]]
                    box = resized_bbox(box, info_img)
                    bboxes.append(box)
                    #classes.append(anno['class_id'])
                    #colors.append(class_colors[anno['class_id']])

            #vis_bbox(
            #    img_raw, bboxes, #label=classes, label_names=class_names,
            #   instance_colors=colors, linewidth=2)
            #plt.show()

            
            #plt.savefig(os.path.join(folder_path, 'gt_' + os.path.basename(image)))        
            

        gt_bboxes.append([os.path.basename(image), bboxes])
        

        # yolo bbox

        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, num_classes, confthre, nmsthre)
        
        bboxes = list()

        if outputs[0] is None:
            print("No Objects Deteted!!")
        
        else:   

            
            #classes = list()
            #colors = list()

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

                #cls_id = class_ids[int(cls_pred)]
                #print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
                #print('\t+ Label: %s, Conf: %.5f' %
                #      (class_names[cls_id], cls_conf.item()))
                # box = yolobox2label([y1, x1, y2, x2], info_img)
                bboxes.append([x1,y1,x2,y2])
                #classes.append(cls_id)
                #colors.append(class_colors[int(cls_pred)])


            #vis_bbox(
            #    img_raw, bboxes, #label=classes, label_names=class_names,
            #    instance_colors=colors, linewidth=2)
            #plt.show()

            #plt.savefig(os.path.join(folder_path, 'yolo_' + os.path.basename(image)))

        yolo_bboxes.append([os.path.basename(image), bboxes])
        
    return gt_bboxes, yolo_bboxes
