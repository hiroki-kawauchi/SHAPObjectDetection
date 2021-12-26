import cv2
import glob
import json
import os
import shutil

def area(a, b):  # returns None if rectangles don't intersect
    """
    calculating intersection area
    Args:
        a, b(list) : [y1,x1,y2,x2]
    Returns:
        is_box: 重複したbox
    """
    is_box = [max(a[0], b[0]),
              max(a[1], b[1]),
              min(a[2], b[2]),
              min(a[3], b[3])] # [y1,x1,y2,x2]
    dy =  is_box[2] - is_box[0] 
    dx =  is_box[3] - is_box[1]

    if dx>0 and dy>0:
        return dx*dy, is_box
    else:
        return 0, []

def resize_gt(input_dir, imgsize):
    """
    Resizing input image and anno_data.json.
    Resized images do not have padding.
    Args:
        input_dir(str): path of GT-image dir
        imgsize (int): target image size after resizing(length of longer side)
    Returns:
        These returns are put to new dir named('input_dir name'_resized_'resized size')
        imgs(jpg):
        anno_data.json(json):
    """
    img_list = glob.glob(os.path.join(input_dir, '*.jpg'))
    img_list.extend(glob.glob(os.path.join(input_dir,'*.png')))

    json_open = open(os.path.join(input_dir, 'anno_data.json'), 'r')
    json_load = json.load(json_open)
    class_id = 0
    output_list = []

    output_dir = input_dir+'_'+'resized'+'_'+str(imgsize)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    for image in img_list:
        img = cv2.imread(image)
        h, w, _ = img.shape
        annotations = json_load[os.path.basename(image)]['regions']
        bboxes = []
        if len(annotations) > 0:
            for anno in annotations:
                box = [anno['bb'][1], anno['bb'][0],
                    anno['bb'][1] + anno['bb'][3], anno['bb'][0] + anno['bb'][2]]
                # [y1,x1,y2,x2]
                bboxes.append(box)

        div_h = h//imgsize + min(h%imgsize,1) # 高さ方向の分割回数
        div_w = w//imgsize + min(w%imgsize,1) 

        for i in range(div_h):
            for j in range(div_w):
                div_y1 = imgsize*i
                div_y2 = min(imgsize*(i+1)-1,h-1)
                div_x1 = imgsize*j
                div_x2 = min(imgsize*(j+1)-1,w-1)
                
                if (div_y2-div_y1)>=100 and (div_x2-div_x1)>=100:
                    
                    div_yxyx = [div_y1,div_x1,div_y2,div_x2]
        
                    div_img = img[div_y1:div_y2+1, div_x1:div_x2+1]
                    div_imagename = os.path.splitext(os.path.basename(image))[0] + '_'+str(i)+ '_'+str(j)+'.jpg'
                    cv2.imwrite(div_imagename,div_img)

                    l_regions = []
                    regions_dict = {}
                    for box in bboxes:
                        iou, is_box = area(div_yxyx,box)
                        
                        if iou/((box[3]-box[1])*(box[2]-box[0]))>=0.5:
                            l_regions.append(dict((['class_id', class_id],
                            ['bb', [is_box[1]-div_x1, is_box[0]-div_y1, is_box[3] - is_box[1], is_box[2] - is_box[0]]])))

                    
                    regions_dict['regions']=l_regions
                    output_list.append([div_imagename, regions_dict])

    with open('anno_data.json', 'w') as f:
        json.dump(dict(output_list), f, ensure_ascii=False)

    return

def remove_nogt(input_dir):
    """
    Removing images without gt object.
    Outputs are in the directory of 'out_dir'
    Args:
        input_dir(str): path of GT-image dir (inculding no-object images)
    """
    img_list = glob.glob(os.path.join(input_dir, '*.jpg'))
    img_list.extend(glob.glob(os.path.join(input_dir,'*.png')))

    json_open = open(os.path.join(input_dir, 'anno_data.json'), 'r')
    json_load = json.load(json_open)

    output_dir = input_dir+'_'+'rm'
    os.makedirs(output_dir, exist_ok=True)

    for image in img_list:
        annotations = json_load[os.path.basename(image)]['regions']
        if len(annotations) > 0:
            shutil.copyfile(image, os.path.join(output_dir, os.path.basename(image)))
    
    shutil.copyfile(os.path.join(input_dir, 'anno_data.json'),os.path.join(output_dir, 'anno_data.json'))
    
    return