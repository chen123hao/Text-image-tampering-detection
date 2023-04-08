import os
import timm
import torch
import torch.nn as nn
from config1 import get_hrnet_cfg, get_seg_model, DetectionHead
from MVSS_ import get_mvss

class task1(nn.Module):
    def __init__(self, net1 = get_seg_model(get_hrnet_cfg()) , net2 = DetectionHead() ):
        super(task1, self).__init__()
        self.exacter = net1
        self.detect_head = net2
    def forward(self,x):
        x = self.exacter(x)
        x = self.detect_head(x)
        return x

########################################################################################################
##################################              task1_new                ###############################
########################################################################################################
eff_net = timm.create_model('efficientnet_b0',pretrained=False,num_classes=2)
eff_net.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
mvss = get_mvss(sobel=True, n_input=3, constrain=True)
# model_dict = torch.load('/home/ch/code/TTI/MVSS-Net-master/ckpt/mvssnet_casia.pt',map_location=torch.device('cpu'))
# mvss.load_state_dict(model_dict)

class task1_1(nn.Module):
    def __init__(self, net1 = mvss , net2 = eff_net):
        super(task1_1, self).__init__()
        self.n1 = net1
        self.n2 = net2
    def forward(self,x):
        #x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        edge, x1 = self.n1(x)#edge, mask
        #x1 = torch.where(torch.isnan(x1), torch.full_like(x1, 0), x1)
        x = self.n2(x1)
        return torch.sigmoid(edge), x1, x


# for name, p in model.named_parameters():
#     if (name.startswith('n1')):
#         p.requires_grad = False

def dice_loss(out, gt, smooth = 1.0):
    gt = gt.view(-1)
    out = out.view(-1)

    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth) # TODO: need to confirm this matches what the paper says, and also the calculation/result is correct

    return 1.0 - dice

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device_ids = [Id for Id in range(torch.cuda.device_count())]
    #x = torch.rand(4,3,256,256)
    # net1 = get_seg_model(get_hrnet_cfg()).cuda()
    # net1 = nn.DataParallel(net1, device_ids=device_ids)
    # net1.load_state_dict(torch.load("/home/ch/code/TTI/PSCC-Net-main/checkpoint/HRNet_checkpoint/HRNet.pth"))
    #
    # net2 = DetectionHead().cuda()
    # net2 = nn.DataParallel(net2, device_ids=device_ids)
    # net2.load_state_dict(torch.load("/home/ch/code/TTI/PSCC-Net-main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth"))

    # net = task1(net1, net2).cuda()

    # for name, p in net.n1.named_parameters():
    #     if (not name.startswith('en.outc') and not name.startswith('route')):
    #         p.requires_grad = False

    net = eff_net
    getModelSize(net)
    x = torch.rand(4,1,512,512)
    a = net(x)
