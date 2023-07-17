###############################################################
##### @Title:  ICDAR 2023 DTT in image1: Text Manipulation Classification
##### @Time:  2023/03/2
##### @Author: ChenHao
###############################################################
import os
import pdb
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import time
import glob
import random
from PIL import Image
from model1 import task1_1
from scheduler import CosineScheduler

from cv2 import transform
# import cupy as cp # https://cupy.dev/ => pip install cupy-cuda102
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from config1 import get_hrnet_cfg, get_seg_model, DetectionHead

import torch  # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp  # https://pytorch.org/docs/stable/notes/amp_examples.html

from sklearn.model_selection import StratifiedGroupKFold, KFold  # Sklearn
import albumentations as A  # Augmentations
import timm
import segmentation_models_pytorch as smp  # smp


def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################
##### build_transforms & build_dataset & build_dataloader
###############################################################
def build_transforms(CFG):
    data_transforms = {
        "train": transforms.Compose([
                transforms.Resize(CFG.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]),

        "valid_test": transforms.Compose([
                transforms.Resize(CFG.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
    }
    return data_transforms


class build_dataset(Dataset):
    def __init__(self, df, train_val_flag=True, transforms=None):

        self.df = df
        self.train_val_flag = train_val_flag  #
        self.img_paths = df['img_path'].tolist()
        self.ids = df['img_name'].tolist()
        self.transforms = transforms

        if train_val_flag:
            self.label = df['img_label'].tolist()
            self.edge = df['edge_path'].tolist()

    def __len__(self):
        return len(self.df)
        # return 8

    def __getitem__(self, index):
        #### id
        id = self.ids[index]
        #### image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # [h, w, c]

        if self.train_val_flag:  # train
            ### augmentations
            img = self.transforms(Image.fromarray(img))

            gt = self.label[index]
            # print(type(torch.tensor(int(gt))))

            edge_p = self.edge[index]
            if edge_p != '0':


                edge = cv2.imread(edge_p, cv2.IMREAD_GRAYSCALE).astype('float32')#读入灰度图
                edge /= 255.0  # scale mask to [0, 1]
            else:
                edge = np.zeros((128,128))
            edge = np.expand_dims(edge, 0)#######?

            return img, torch.tensor(int(gt)), torch.tensor(edge)

        else:  # test
            ### augmentations
            img = self.transforms(Image.fromarray(img))

            return img, id, id


def build_dataloader(df, fold, data_transforms):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    train_dataset = build_dataset(train_df, train_val_flag=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_df, train_val_flag=True, transforms=data_transforms['valid_test'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://timm.fast.ai/create_model
# def build_model(CFG, pretrain_flag=False):
#     if pretrain_flag:
#         pretrain_weights = "imagenet"
#     else:
#         pretrain_weights = False
#     model = timm.create_model(CFG.backbone,
#                               pretrained=pretrain_flag,
#                               num_classes=CFG.num_classes)
#     model.to(CFG.device)
#     return model


###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    return {"CELoss": CELoss}


###############################################################
##### >>>>>>> build_metric <<<<<<
###############################################################
def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou
def dice_loss(out, gt, smooth = 1.0):
    gt = gt.view(-1)
    out = out.view(-1)

    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth)
    dice = (1.0 - dice).type(torch.float32)
    return dice

###############################################################
##### >>>>>>> train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler()
    losses_all, ce_all = 0, 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, gt, edge) in pbar:
        optimizer.zero_grad()

        images = images.cuda() # [b, c, w, h]
        gt = gt.cuda()
        edge = torch.sigmoid(edge).cuda()

        with amp.autocast(enabled=True):
            out_edges, x1, y_preds = model(images)######################################################
            # print(y_preds)
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            losses = ce_loss + dice_loss(out_edges, edge)

        scaler.scale(losses).backward()
        #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

        losses_all += losses.item() / images.shape[0]
        ce_all += ce_loss.item() / images.shape[0]

    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.8f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all, ce_all), flush=True)


@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    tamp = np.array([])
    untamp = np.array([])

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt, edge) in pbar:
        images = images.cuda()  # [b, c, w, h]
        gt = gt.cuda()

        _, _, y_preds = model(images)
        prob = torch.nn.functional.softmax(y_preds, dim=-1)[:, 1].detach().cpu().numpy()
        gt_s = gt.squeeze().cpu().numpy()

        tamper = prob[gt_s[:] == 1]
        untamper = prob[gt_s[:] == 0]

        tamp = np.r_[tamp, tamper]
        untamp = np.r_[untamp, untamper]

    thres = np.percentile(untamp, np.arange(90, 100, 1))
    recall = 100 * np.mean(np.greater(tamp[:][:, np.newaxis], thres).mean(axis=0))
    print("ave_recall: {:.6f}".format(recall), flush=True)

    return recall

@torch.no_grad()
def test_one_epoch(test_df, ckpt_paths1,ckpt_paths2,ckpt_paths3, test_loader, CFG):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids) in pbar:

        images = images.cuda()  # [b, c, w, h]

        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################

        #model = build_model(CFG, pretrain_flag=False)  # just dummy code
        # device_ids = [Id for Id in range(torch.cuda.device_count())]
        # net1 = get_seg_model(get_hrnet_cfg()).cuda()
        # net1 = nn.DataParallel(net1, device_ids=device_ids)
        # net1.load_state_dict(torch.load("/home/ch/code/TTI/PSCC-Net-main/checkpoint/HRNet_checkpoint/HRNet.pth"))
        #
        # net2 = DetectionHead().cuda()
        # net2 = nn.DataParallel(net2, device_ids=device_ids)
        # net2.load_state_dict(
        #     torch.load("/home/ch/code/TTI/PSCC-Net-main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth"))

        model1 = task1_1().cuda()
        #model = build_model(CFG, pretrain_flag=False)
        model1.load_state_dict(torch.load(ckpt_paths1))
        model1.eval()
        _, _, y_preds1 = model1(images)  # [b, c, w, h]
        prob1 = torch.nn.functional.softmax(y_preds1, dim=-1)[:, 1].detach().cpu().numpy()

        model2 = task1_1().cuda()
        # model = build_model(CFG, pretrain_flag=False)
        model2.load_state_dict(torch.load(ckpt_paths2))
        model2.eval()
        _, _, y_preds2 = model2(images)  # [b, c, w, h]
        prob2 = torch.nn.functional.softmax(y_preds2, dim=-1)[:, 1].detach().cpu().numpy()

        model3 = task1_1().cuda()
        # model = build_model(CFG, pretrain_flag=False)
        model3.load_state_dict(torch.load(ckpt_paths3))
        model3.eval()
        _, _, y_preds3 = model3(images)  # [b, c, w, h]
        prob3 = torch.nn.functional.softmax(y_preds3, dim=-1)[:, 1].detach().cpu().numpy()

        prob = (prob1 + prob2 + prob3)/3

        test_df.loc[test_df['img_name'].isin(ids), 'pred_prob'] = prob

    return test_df





if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 66
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "ckpt_ddt1"
        ckpt_name = "512_mvss_eff"  # for submit.
        tampered_img_paths = "/home/ch/code/TTI/data/train/tampered/imgs"
        tampered_edge_paths = "/home/ch/code/TTI/data/train/tampered/edge"
        untampered_img_paths = "/home/ch/code/TTI/data/train/untampered"

        # step2: data
        n_fold = 4
        img_size = [512, 512]
        train_bs = 16
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'efficientnet_b0'
        num_classes = 2
        # step4: optimizer
        epoch = 20
        lr = 1e-5
        wd = 1e-5
        lr_drop = 8
        # step5: infer
        thr = 0.5


    set_seed(CFG.seed)
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ###############################################################
    ##### train
    ###############################################################
    train_val_flag = True
    if train_val_flag:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'###################################################################################
        col_name = ['img_name', 'img_path', 'img_label', 'edge_path']
        imgs_info = []  # img_name, img_path, img_label
        # 篡改标签为1， 未篡改标签为0
        for img_name in os.listdir(CFG.tampered_img_paths):
            if img_name.endswith('.jpg'):  # pass other files
                edge_img_name = img_name.replace('.jpg', '.png')
                edge_path = os.path.join(CFG.tampered_edge_paths, edge_img_name)
                imgs_info.append(["p_" + img_name, os.path.join(CFG.tampered_img_paths, img_name), 1, edge_path])

        for img_name in os.listdir(CFG.untampered_img_paths):
            if img_name.endswith('.jpg'):  # pass other files
                imgs_info.append(["n_" + img_name, os.path.join(CFG.untampered_img_paths, img_name), 0, 0])

        imgs_info_array = np.array(imgs_info)
        df = pd.DataFrame(imgs_info_array, columns=col_name)


        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold

        for fold in range(CFG.n_fold):
            print(f'#' * 40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#' * 40, flush=True)


            data_transforms = build_transforms(CFG)
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms)  # dataset & dtaloader

            #model = build_model(CFG, pretrain_flag=False)  # model
            model = task1_1().cuda()


            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            # lr_scheduler1= torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop)
            lr_scheduler2 = CosineScheduler(optimizer, param_name='lr', t_max=CFG.epoch,
                                            value_min=CFG.lr * 1e-4,
                                            warmup_t=0, const_t=0)
            losses_dict = build_loss()  # loss

            best_recall = 0
            best_epoch = 0

            for epoch in range(1, CFG.epoch + 1):
                start_time = time.time()

                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)

                lr_scheduler2.step(epoch + 1)
                ave_recall = valid_one_epoch(model, valid_loader, CFG)

                is_best = (ave_recall >= best_recall)
                best_recall = max(best_recall, ave_recall)
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path)
                    torch.save(model.state_dict(), save_path)

                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.6f}\n".format(epoch, epoch_time, best_recall), flush=True)
    ###############################################################
    ##### test
    ###############################################################
    test_flag = False
    if test_flag:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        col_name = ['img_name', 'img_path', 'pred_prob']
        imgs_info = []  # img_name, img_path, pred_prob
        test_imgs = os.listdir(CFG.test_img_path)
        test_imgs.sort(key=lambda x: x[:-4])
        for img_name in test_imgs:
            if img_name.endswith('.jpg'):  # pass other files
                imgs_info.append([img_name, os.path.join(CFG.test_img_path, img_name), 0])

        imgs_info_array = np.array(imgs_info)
        test_df = pd.DataFrame(imgs_info_array, columns=col_name)

        data_transforms = build_transforms(CFG)
        test_loader = build_dataloader(test_df, False, None, data_transforms)  # dataset & dtaloader
        ckpt_paths1 = "/home/ch/code/TTI/ckpt_ddt1/512_mvss_eff/best_fold3_epoch1.pth"  # please use your ckpt path
        ckpt_paths2 = "/home/ch/code/TTI/ckpt_ddt1/512_mvss_eff/best_fold2_epoch1.pth"
        ckpt_paths3 = "/home/ch/code/TTI/ckpt_ddt1/512_mvss_eff/best_fold1_epoch2.pth"
        test_df = test_one_epoch(test_df, ckpt_paths1, ckpt_paths2, ckpt_paths2, test_loader, CFG)
        submit_df = test_df.loc[:, ['img_name', 'pred_prob']]
        submit_df.to_csv("submit_dummy.csv", header=False, index=False, sep=' ')
