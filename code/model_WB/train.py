import torch
import torch.nn.functional as F
from torch.autograd import Variable
import inspect
import copy
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Pred_endecoder
from data import get_loader, EvalDataset
from utils import adjust_lr, AvgMeter
import logging
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from tools import *
from torch.utils.data import DataLoader
from metrics import Metrics
from tqdm import tqdm
from MaxNorm import MaxNorm_via_PGD


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=8, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.4, help='weight of the smooth loss')
parser.add_argument('--tro_train', type=float, default=1.0, help='weight of the smooth loss')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight of the smooth loss')

opt = parser.parse_args()

save_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_WB/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + "training_log.log", filemode='w', level=logging.INFO)
logging.info(opt)

current_file_name = inspect.getfile(inspect.currentframe())
logging.info(current_file_name)

print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/base/Model_30_gen.pth'))
generator.cuda()

image_root = '/data/local_userdata/tianxinyu/DUTS/DUTS-TR/DUTS-TR-Image/'
gt_root = '/data/local_userdata/tianxinyu/DUTS/DUTS-TR/DUTS-TR-Mask/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
multiclass_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training

save_path_temp = save_path + 'temp/'
if not os.path.exists(save_path_temp):
    os.makedirs(save_path_temp)


# model = copy.deepcopy(generator)

thresh = 0.1 #threshold value
pgdFunc = MaxNorm_via_PGD(thresh=thresh)
pgdFunc.setPerLayerThresh(generator) # set per-layer thresholds

# ## Finetune with MaxNorm and Weight Decay
for param in generator.parameters():  # freez all model paramters except the classifier layer
    # print(n)
    param.requires_grad = False

generator.decoder.output_conv[4].weight.requires_grad = True
generator.decoder.output_conv[4].bias.requires_grad = True

generator_optimizer = torch.optim.Adam([generator.decoder.output_conv[4].weight, generator.decoder.output_conv[4].bias], opt.lr_gen, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)


class binary_ce_loss(torch.nn.Module):
    def __init__(self):
        super(binary_ce_loss, self).__init__()

    def forward(self, input, target, weight=None):
        # input = input.view(input.shape[0], -1)
        # target = target.view(target.shape[0], -1)
        # weight = weight.view(weight.shape[0], -1)
        # loss = 0.0
        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         loss += -weight[i][j] * (target[i][j] * torch.log(input[i][j]) + (1 - target[i][j]) * torch.log(1 - input[i][j]))

        loss = -weight * (target * torch.log(input) + (1 - target) * torch.log(1 - input))

        # return loss/(input.shape[0]*input.shape[1]) # 默认取均值
        return loss.mean()


class Class_Balance_loss():

    def __init__(self, samples_per_cls=[9.05606316, 3.38788084], no_of_classes=2, beta=0.9999, gamma=2.0):
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float().cuda()

        self.weights = weights
        # self.multiclass_ce_loss_weight = torch.nn.CrossEntropyLoss(weight = weights, reduction='mean')
        self.bceloss = binary_ce_loss()

    def loss_compute(self, labels, logits):

        weights = self.weights[0] * labels[:, 0, :, :].detach() + self.weights[1] * labels[:, 1, :, :].detach()
        weights = weights.unsqueeze(1)
        # weights = torch.cat((weights, weights), dim=1)
        weights = weights.repeat(1, 2, 1, 1)

        # pred = logits.softmax(dim=1)
        pred = F.softmax(logits,dim=1)
        # cb_loss = F.binary_cross_entropy(input=pred, target=labels, weight=weights)
        cb_loss = self.bceloss(input=pred, target=labels, weight=weights)
        return cb_loss



# def CB_loss(labels, logits, samples_per_cls=np.array([905606316, 338788084]), no_of_classes=2, loss_type='softmax', beta=0.9999, gamma=2.0):
#     """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
#     Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
#     where Loss is one of the standard losses used for Neural Networks.
#     Args:
#       labels: A int tensor of size [batch].
#       logits: A float tensor of size [batch, no_of_classes].
#       samples_per_cls: A python list of size [no_of_classes].
#       no_of_classes: total number of classes. int
#       loss_type: string. One of "sigmoid", "focal", "softmax".
#       beta: float. Hyperparameter for Class balanced loss.
#       gamma: float. Hyperparameter for Focal loss.
#     Returns:
#       cb_loss: A float tensor representing class balanced loss
#     """
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes
#
#     # labels_one_hot = F.one_hot(labels, no_of_classes).float()
#     # labels_one_hot = labels_one_hot.cuda()
#     weights = torch.tensor(weights).float().cuda()
#     # weights = weights.unsqueeze(0)
#     # weights = weights.repeat(labels_one_hot.shape[0],1).cuda()
#     # weights = weights* labels_one_hot
#     # weights = weights.sum(1)
#     # weights = weights.unsqueeze(1)
#     # weights = weights.repeat(1,no_of_classes)
#     weights = weights[0] * labels[:, 0, :, :] + weights[1] * labels[:, 1, :, :]
#     weights = weights.unsqueeze(1)
#     weights = weights.repeat(1, no_of_classes, 1, 1)
#
#     pred = logits.softmax(dim = 1)
#     cb_loss = F.binary_cross_entropy(input = pred, target = labels, weight = weights)
#
#     return cb_loss


def compute_adjustment(train_loader, tro):
    """compute the base probabilities"""

    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.cuda()
        target = torch.cat((1-target,target),1)
        for m in range(target.shape[2]):
            for n in range(target.shape[2]):
                target_cur = target[:,:,m,n]
                for j in target_cur:
                    key = int(j.item())
                    label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments


def structure_loss(pred, mask, weight=None):
    # def generate_smoothed_gt(gts):
    #     epsilon = 0.001
    #     new_gts = (1-epsilon)*gts+epsilon/2
    #     return new_gts
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # new_gts = generate_smoothed_gt(mask)
    weit = weit.sum(dim=1)
    wbce = multiclass_ce_loss(pred, mask)
    wbce = (weit * wbce).sum(dim=(1, 2)) / weit.sum(dim=(1, 2))

    pred = F.softmax(pred,dim=1)
    inter_item = pred * mask
    union_item = pred + mask
    inter = ((inter_item.sum(dim=1)) * weit).sum(dim=(1, 2))
    union = ((union_item.sum(dim=1)) * weit).sum(dim=(1, 2))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def structure_loss_focal_loss(pred, mask, weight):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (((1-weight)**opt.focal_lamda)*weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def visualize_gt(var_map, save_path):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_pred(pred, save_path):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_pred_pred.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_original_img(rec_img, save_path):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

def no_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()

def yes_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def visualize_prediction_gt(pred,gts, save_path):
    for kk in range(gts.shape[0]):
        pred_back_kk, pred_fore_kk, gts_back_kk, gts_fore_kk = pred[kk, 0, :, :], pred[kk, 1, :, :], gts[kk, 0, :, :], gts[kk, 1, :, :]
        pred_back_kk = (pred_back_kk.detach().cpu().numpy()*255.0).astype(np.uint8)
        pred_fore_kk = (pred_fore_kk.detach().cpu().numpy() * 255.0).astype(np.uint8)
        gts_back_kk = (gts_back_kk.detach().cpu().numpy() * 255.0).astype(np.uint8)
        gts_fore_kk = (gts_fore_kk.detach().cpu().numpy() * 255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred_back_kk, pred_fore_kk, gts_back_kk, gts_fore_kk])
        name = '{:02d}_pback_pfore_gtback_gtfore.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)


def visualize_uncertainty(pred, save_path):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        pred_edge_kk = cv2.applyColorMap(pred_edge_kk, cv2.COLORMAP_JET)
        name = '{:02d}_aleatoric_rgb.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


CB_loss = Class_Balance_loss()
print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    log_str = '--------------------------Epoch:' + str(epoch) + '--------------------------'
    logging.info(log_str)
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            gts = torch.cat((1-gts, gts),1)
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # logit_adjustments = compute_adjustment(train_loader, opt.tro_train)

            pred = generator(images)
            # loss_all = structure_loss(pred, gts)
            loss_all = CB_loss.loss_compute(gts, pred)


            loss_all.backward()
            generator_optimizer.step()

            visualize_prediction_gt(F.softmax(pred, dim=1), gts, save_path_temp)

            if rate == 1:
                loss_record.update(loss_all.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

            loss = {}
            loss['loss_sum'] = loss_record.show().item()
            logging.info(loss)

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')


generator.eval()
dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
all_results = ''

for dataset in test_datasets:
    save_path_pred = save_path + 'pred/' + dataset + '/'
    if not os.path.exists(save_path_pred):
        os.makedirs(save_path_pred)
    save_path_uncer = save_path + 'uncertainty/' + dataset + '/'
    if not os.path.exists(save_path_uncer):
        os.makedirs(save_path_uncer)

    test_dataset = EvalDataset(dataset_path, gt_path, opt.trainsize, f'test_lists/{dataset}.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info(len(test_loader))
    metrics = Metrics()

    for i, (image, gt, HH, WW, name) in enumerate(tqdm(test_loader)):
        name = name[0].split('/')[-1].split('.')[0] + '.png'
        image = image.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            generator_pred = generator.forward(image)
        res = generator_pred
        res = F.upsample(F.softmax(res, dim=1), size=[HH, WW], mode='bilinear', align_corners=False)
        res = res[:, 1, :, :]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res_save = (255 * res).data.cpu().numpy().squeeze()
        cv2.imwrite(save_path_pred + name, res_save)

        uncert = -1 * res * torch.log(res + 1e-8)
        uncert = (uncert - uncert.min()) / (uncert.max() - uncert.min() + 1e-3)

        uncert_save = (255 * uncert).data.cpu().numpy().squeeze()
        uncert_save = uncert_save.astype(np.uint8)
        uncert_save = cv2.applyColorMap(uncert_save, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_uncer + name, uncert_save)

        metrics.update(res.squeeze(), uncert.squeeze(), gt.squeeze())

    logging.info(dataset)
    logging.info(metrics.get_results())
    logging.info('-' * 20)

    results_dict = metrics.get_results()
    all_results += '& ' + '%.3f' % results_dict['F_max'] + ' & ' + '%.3f' % results_dict['IoU'] + ' & ' + '%.3f' % results_dict['Accuracy']\
                   + ' & ' + '%.3f' % results_dict['FPR95'] + ' & ' + '%.3f' % results_dict['AUROC'] + ' '

logging.info(all_results)

