import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from data import EvalDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import logging
from torch.utils.data import DataLoader
from metrics import Metrics
from tqdm import tqdm
import pickle


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')

opt = parser.parse_args()

weight_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_mc_dropout/'
save_path = '/data/local_userdata/tianxinyu/CVPR_2023/metric_distribution/model_mc_dropout_eval_apply_drop_img/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load(weight_path + 'Model_30_gen.pth'))

generator.cuda()
generator.eval()
generator.apply(apply_dropout)

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + "sod_metrics_new_entropy.log", filemode='w', level=logging.INFO)

dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
all_results = ''

for dataset in test_datasets:

    save_path_pred = save_path + 'pred_new_entropy/' + dataset + '/'
    if not os.path.exists(save_path_pred):
        os.makedirs(save_path_pred)
    save_path_uncer = save_path + 'uncertainty_new_entropy/' + dataset + '/'
    if not os.path.exists(save_path_uncer):
        os.makedirs(save_path_uncer)

    test_dataset = EvalDataset(dataset_path, gt_path, opt.testsize, f'test_lists/{dataset}.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    metrics = Metrics()
    logging.info(len(test_loader))

    for i, (image, gt, HH, WW, name) in enumerate(tqdm(test_loader)):
        name = name[0].split('/')[-1].split('.')[0] + '.png'
        image = image.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            generator_pred = generator.forward(image)

        generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
        res = F.softmax(generator_pred, dim=1)
        # pred_list = res[:, 1, :, :].unsqueeze(1)
        pred_list = res

        for j in range(4):
            with torch.no_grad():
                pred = generator.forward(image)
                pred = F.upsample(pred, size=[HH, WW], mode='bilinear', align_corners=False)
                # pred = F.softmax(pred, dim=1)[:, 1, :, :].unsqueeze(1)
                pred = F.softmax(pred, dim=1)
                pred_list = torch.cat((pred_list, pred), dim=0)

        pred_cat_mean = torch.mean(pred_list, 0, keepdim=True)
        mean_pred = pred_cat_mean[:, 1, :, :]
        res = (mean_pred - mean_pred.min()) / (mean_pred.max() - mean_pred.min() + 1e-8)
        res = (255 * res).data.cpu().numpy().squeeze()
        cv2.imwrite(save_path_pred + name, res)

        # uncertainty = torch.var(pred_list, 1, keepdim=True)
        uncertainty = -1 * pred_cat_mean[:, 0, :, :] * torch.log(pred_cat_mean[:, 0, :, :] + 1e-8) - pred_cat_mean[:, 1, :, :] * torch.log(
            pred_cat_mean[:, 1, :, :] + 1e-8)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        res = (255 * uncertainty).data.cpu().numpy().squeeze()
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_uncer + name, res)

        metrics.update(mean_pred.squeeze(), uncertainty.squeeze(), gt.squeeze())

    logging.info(dataset)
    logging.info(metrics.get_results())
    logging.info('-' * 20)

    results_dict = metrics.get_results()
    all_results += '& ' + '%.3f' % results_dict['F_max'] + ' & ' + '%.3f' % results_dict['IoU'] + ' & ' + '%.3f' % \
                   results_dict['Accuracy'] \
                   + ' & ' + '%.3f' % results_dict['FPR95'] + ' & ' + '%.3f' % results_dict['AUROC'] + ' '

    pickle.dump(metrics.__dict__, open(save_path + dataset + '_metrics.pkl', 'wb'))

logging.info(all_results)

