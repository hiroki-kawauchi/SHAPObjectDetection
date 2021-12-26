from __future__ import division

from utils.utils import *
from utils.vd_evaluator import VDEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from models.shap_loss_1 import *
from dataset.dataset_vd import *

import os
import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import pandas as pd

import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_vd.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                            default=4000, help='interval between evaluations')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard', help='tensorboard path for logging', type=str, default=None)
    parser.add_argument('--anno_file', type=str,
                        default='anno_data.json', help='annotation data json file name')
    parser.add_argument('--shap_interval', type=int,
                        default=None, help='interval between updating shaploss')
    return parser.parse_args()


def main():
    """
    SHAP-regularized YOLOv3 trainer.
    
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    print("successfully loaded config file: ", cfg)

    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision
    at_alpha = cfg['TRAIN']['ATTENTION_ALPHA']
    at_beta = cfg['TRAIN']['ATTENTION_BETA']

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            factor = pow(i / burn_in, 4)# pow(x, y):x^y
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        print("using cuda") 
        model = model.cuda()

    if args.tfboard:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard)

    model.train()

    imgsize = cfg['TRAIN']['IMGSIZE']

    dataset = ListDataset(model_type=cfg['MODEL']['TYPE'],
                data_dir=cfg['TRAIN']['TRAIN_DIR'],
                json_file=args.anno_file,
                img_size=imgsize,
                augmentation=cfg['AUGMENTATION'])

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.n_cpu)

    dataiterator = iter(dataloader)

    evaluator = VDEvaluator(data_dir=cfg['TRAIN']['VAL_DIR'],
                    json_file=args.anno_file,
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'])

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # optimizer setup
    # set weight decay only on conv.weight
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)

    iter_state = 0

    if args.checkpoint:
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # start training loop
    log_col = ['time(min)', 'iter','lr','xy', 'wh',
               'conf', 'cls', 'shap', 'l2', 'imgsize',
               'ap50', 'precision50', 'recall50', 'F_measure']
    log = []
    ap50 = np.nan
    precision50 = np.nan
    recall50 = np.nan
    F_measure = np.nan
    shap_loss = torch.tensor(float('nan'),dtype=torch.float32)
    t_0 = time.time()
    for iter_i in range(iter_state, iter_size + 1):
        
        # VD evaluation

        if iter_i % args.eval_interval == 0 and iter_i > 0:
            ap50, precision50, recall50, F_measure = evaluator.evaluate(model)
            model.train()
            if args.tfboard:
                tblogger.add_scalar('val/COCOAP50', ap50, iter_i)

            print('[Iter {}/{}]:AP50:{}'.format(iter_i, iter_size,ap50))

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            loss_dict = model.loss_dict
            # adding SHAP-based loss
            if args.shap_interval is not None:
                if inner_iter_i % args.shap_interval == 0:
                    shap_loss_ = shaploss(imgs, targets, model,
                                          num_classes=cfg['MODEL']['N_CLASSES'],
                                          confthre=cfg['TEST']['CONFTHRE'],
                                          nmsthre=cfg['TEST']['NMSTHRE'],
                                          n_samples=cfg['TRAIN']['N_SAMPLES'], 
                                          alpha=at_alpha, beta=at_beta)
                    if shap_loss_ != 0 and shap_loss != torch.tensor(float('nan'),dtype=torch.float32):
                        shap_loss = shap_loss_
                        
                    model.train()
                    loss += shap_loss
            loss.backward()
            
        optimizer.step()
        scheduler.step()

        if iter_i % 10 == 0:
            # logging
            current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            t = (time.time() - t_0)//60
            print('[Time %d] [Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, att %f, total %f, imgsize %d, ap %f, precision %f, recall %f, F %f]'
                  % (t, iter_i, iter_size, current_lr,
                     loss_dict['xy'], loss_dict['wh'],
                     loss_dict['conf'], loss_dict['cls'], shap_loss,
                     loss_dict['l2'], imgsize, ap50, precision50, recall50, F_measure),
                  flush=True)
            log.append([t, iter_i, current_lr,
                        np.atleast_1d(loss_dict['xy'].to('cpu').detach().numpy().copy())[0],
                        np.atleast_1d(loss_dict['wh'].to('cpu').detach().numpy().copy())[0],
                        np.atleast_1d(loss_dict['conf'].to('cpu').detach().numpy().copy())[0],
                        np.atleast_1d(loss_dict['cls'].to('cpu').detach().numpy().copy())[0],
                        np.atleast_1d(shap_loss.to('cpu').detach().numpy().copy())[0],
                        np.atleast_1d(loss_dict['l2'].to('cpu').detach().numpy().copy())[0],
                        imgsize, ap50, precision50, recall50, F_measure])
            ap50 = np.nan
            precision50 = np.nan
            recall50 = np.nan
            F_measure = np.nan
            
            if args.tfboard:
                tblogger.add_scalar('train/total_loss', model.loss_dict['l2'], iter_i)

            # random resizing
            if random_resize:
                imgsize = (random.randint(0, 9) % 10 + 10) * 32
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataloader)

        # save checkpoint
        #if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
        if (0<iter_i<=1000 and (iter_i % 100 == 0))or(1000<iter_i and (iter_i % 500 == 0)):
            torch.save({'iter': iter_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                        os.path.join(args.checkpoint_dir, "snapshot"+str(iter_i)+".ckpt"))
            
            df_log = pd.DataFrame(log, columns=log_col)
            df_log.to_csv(os.path.join(args.checkpoint_dir, "log_"+str(iter_i)+".csv"))
            #log = []

    if args.tfboard:
        tblogger.close()


if __name__ == '__main__':
    main()
