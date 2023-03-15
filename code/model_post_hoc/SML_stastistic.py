import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
import time
import torch.nn as nn
def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--iter_num', type=int, default=8, help='latent dimension')
opt = parser.parse_args()

# checkpoint_path = '/data/local_userdata/tianxinyu/CVPR_2023/models/McA_2_channels/'
generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/base/Model_30_gen.pth'))

generator.cuda()
generator.eval()
generator.apply(apply_dropout)

def compute_entropy(pred):
    entropy_loss_fore = -pred * torch.log(pred + 1e-8)
    entropy_loss_back = -(1-pred) * torch.log(1-pred + 1e-8)
    entropy_loss_all = entropy_loss_fore+entropy_loss_back
    return entropy_loss_all


image_root = '/data/local_userdata/tianxinyu/DUTS/DUTS-TR/DUTS-TR-Image/'
# gt_root = '/data/local_userdata/tianxinyu/DUTS/DUTS-TR/DUTS-TR-Mask/'


test_loader = test_dataset(image_root, opt.testsize)
# pred_list = None

fg_logit_mean = 0
bg_logit_mean = 0
fg_pixel_num = 0
bg_pixel_num = 0
for i in tqdm(range(test_loader.size), desc='DUTS_Train'):

    image, HH, WW, name = test_loader.load_data()
    image = image.cuda()
    start = time.time()

    with torch.no_grad():
        logit = generator.forward(image)
        # logit = logit[:, 1, :, :].unsqueeze(1)
        max_logit, pred = torch.max(logit, dim=1, keepdim=True)
        fg_logit_mean += torch.sum(max_logit[pred == 1])
        bg_logit_mean += torch.sum(max_logit[pred == 0])
        fg_pixel_num += torch.sum(pred == 1)
        bg_pixel_num += torch.sum(pred == 0)

fg_logit_mean = fg_logit_mean/fg_pixel_num
bg_logit_mean = bg_logit_mean/bg_pixel_num

print(fg_logit_mean, bg_logit_mean)
print(fg_pixel_num, bg_pixel_num)


test_loader = test_dataset(image_root, opt.testsize)

fg_logit_var = 0
bg_logit_var = 0
fg_pixel_num = 0
bg_pixel_num = 0
for i in tqdm(range(test_loader.size), desc='DUTS_Train'):
    image, HH, WW, name = test_loader.load_data()
    image = image.cuda()
    start = time.time()

    with torch.no_grad():
        logit = generator.forward(image)
        # logit = logit[:, 1, :, :].unsqueeze(1)
        max_logit, pred = torch.max(logit, dim=1, keepdim=True)
        fg_logit_var += torch.sum(pow((max_logit[pred == 1] - fg_logit_mean), 2))
        bg_logit_var += torch.sum(pow((max_logit[pred == 0] - bg_logit_mean), 2))
        fg_pixel_num += torch.sum(pred == 1)
        bg_pixel_num += torch.sum(pred == 0)

fg_logit_var = fg_logit_var / fg_pixel_num
bg_logit_var = bg_logit_var / bg_pixel_num


print(fg_logit_mean, bg_logit_mean)
print(fg_logit_var, bg_logit_var)

print(fg_pixel_num, bg_pixel_num)

    # if i % 50 == 49 or i == test_loader.size - 1:
    #
    #     pred_list = pred_list.transpose(1, 3)
    #     pred_list, prediction = pred_list.max(3)
    #
    #
    #     class_max_logits = []
    #     mean_dict, var_dict = {}, {}
    #     for c in range(2):
    #         max_mask = pred_list[prediction == c]
    #         class_max_logits.append(max_mask)
    #
    #         mean = max_mask.mean(dim=0)
    #         var = max_mask.var(dim=0)
    #
    #
    #         mean_dict[c] = mean.item()
    #         var_dict[c] = var.item()
    #
    #     print(f"class mean: {mean_dict}")
    #     print(f"class var: {var_dict}")
    #     np.save(f'stats/DUTS_Train_MT_mean.npy', mean_dict)
    #     np.save(f'stats/DUTS_Train_MT_var.npy', var_dict)
    #
    #     break






