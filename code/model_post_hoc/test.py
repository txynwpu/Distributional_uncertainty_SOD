import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from data import EvalDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from SML_utils import BoundarySuppressionWithSmoothing
import logging
from torch.utils.data import DataLoader
from metrics import Metrics
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')
parser.add_argument('--post_hoc_method', type=str, default='MCP', help='method of post-hoc, "MCP", "Energy", "ExGrad", "NorCal", "ReAct", "SML"')
# SML
# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4,
                    help='the number of boundary iterations')
# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                    help='kernel dilation rate of dilated smoothing')

opt = parser.parse_args()

# save_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_NorCal/'   # base model weight path
save_path = '/data/local_userdata/tianxinyu/CVPR_2023/metric_distribution/model_' + opt.post_hoc_method + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/base/Model_30_gen.pth'))

generator.cuda()
generator.eval()

logging.basicConfig(filename=save_path + "sod_metrics_new2.log", filemode='w', level=logging.INFO)


dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
# test_datasets = ['ECSSD']
all_results = ''

for dataset in test_datasets:
    # save_path_pred = save_path + 'pred_new/' + dataset + '/'
    # if not os.path.exists(save_path_pred):
    #     os.makedirs(save_path_pred)
    # save_path_uncer = save_path + 'uncertainty_new/' + dataset + '/'
    # if not os.path.exists(save_path_uncer):
    #     os.makedirs(save_path_uncer)

    test_dataset = EvalDataset(dataset_path, gt_path, opt.testsize, f'test_lists/{dataset}.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    metrics = Metrics()

    for i, (image, gt, HH, WW, name) in enumerate(tqdm(test_loader)):
        name = name[0].split('/')[-1].split('.')[0] + '.png'
        image = image.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            generator_pred = generator.forward(image)

        if opt.post_hoc_method == 'MCP':
            generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
            probability = F.softmax(generator_pred, dim=1)
            res = probability[:, 1, :, :]
            # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res_save = (255 * res_save).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res_save)

            conf, _ = torch.max(probability, dim=1, keepdim=True)
            uncertainty = 1 - conf

        elif opt.post_hoc_method == 'Energy':
            generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
            probability = F.softmax(generator_pred, dim=1)
            res = probability[:, 1, :, :]
            # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res_save = (255 * res_save).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res_save)

            temper = 1.0
            conf = temper * torch.logsumexp(generator_pred / temper, dim=1)
            uncertainty = 1 - conf

        elif opt.post_hoc_method == 'ExGrad':
            generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
            probability = F.softmax(generator_pred, dim=1)
            res = probability[:, 1, :, :]
            # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res_save = (255 * res_save).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res_save)

            uncertainty = probability[:, 1, :, :] * probability[:, 0, :, :] * 2    # fg * (1 - fg) + bg * (1 - bg) = 2 * fg * bg

        elif opt.post_hoc_method == 'SML':
            generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
            probability = F.softmax(generator_pred, dim=1)
            res = probability[:, 1, :, :]
            # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res_save = (255 * res_save).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res_save)

            # class_mean = [6.0126, 4.0575]            # compute by base model   [bg_max_logit_mean, fg_max_logit_mean]
            # class_var = [1.4823, 1.2701]             #  [bg_max_logit_var, fg_max_logit_var]
            class_mean = [6.0242, 4.1668]
            class_var = [1.4369, 1.1640]
            Smoothing = BoundarySuppressionWithSmoothing(
                boundary_suppression=opt.enable_boundary_suppression,
                boundary_width=opt.boundary_width,
                boundary_iteration=opt.boundary_iteration,
                dilated_smoothing=opt.enable_dilated_smoothing,
                kernel_size=opt.smoothing_kernel_size,
                dilation=opt.smoothing_kernel_dilation)

            anomaly_score, prediction = torch.max(generator_pred, dim=1, keepdim=True)    # max logit
            num_classes = generator_pred.shape[1]

            for c in range(num_classes):
                anomaly_score = torch.where(prediction == c,
                                            (anomaly_score - class_mean[c]) / np.sqrt(class_var[c]),
                                            anomaly_score)

            # anomaly_score = Smoothing(anomaly_score, prediction)
            anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-3)
            # uncertainty = 1 - anomaly_score.unsqueeze(1).detach()
            uncertainty = 1 - anomaly_score.detach()

        elif opt.post_hoc_method == 'NorCal':
            frequencies = torch.tensor([905606316, 338788084]) / 338788084    # in paper here is Nc
            gamma = 1 / frequencies ** 0.6

            generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
            probability = torch.exp(generator_pred)
            probability[:, 0, :, :] = probability[:, 0, :, :] * gamma[0]
            probability[:, 1, :, :] = probability[:, 1, :, :] * gamma[1]
            probability /= probability.sum(dim=1, keepdim=True)

            res = probability
            res = res[:, 1, :, :]
            # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res_save = (255 * res_save).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res_save)

            uncertainty = -1 * probability[:, 0, :, :] * torch.log(probability[:, 0, :, :] + 1e-8) - probability[:, 1, :, :] * torch.log(probability[:, 1, :, :] + 1e-8)

        elif opt.post_hoc_method == 'ReAct':
            generator_pred = generator_pred.clip(max=1)  # react

            generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
            probability = F.softmax(generator_pred, dim=1)
            res = probability[:, 1, :, :]
            # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res_save = (255 * res_save).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res_save)

            conf, _ = torch.max(probability, dim=1, keepdim=True)
            uncertainty = 1 - conf

        else:
            print('not support this post hoc method.')

        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-3)
        # uncert_save = (255 * uncertainty).data.cpu().numpy().squeeze()
        # uncert_save = uncert_save.astype(np.uint8)
        # uncert_save = cv2.applyColorMap(uncert_save, cv2.COLORMAP_JET)
        # cv2.imwrite(save_path_uncer + name, uncert_save)

        metrics.update(res.squeeze(), uncertainty.squeeze(), gt.squeeze())

    logging.info(dataset)
    logging.info(metrics.get_results())
    logging.info('-' * 20)

    results_dict = metrics.get_results()
    all_results += '& ' + '%.3f' % results_dict['F_max'] + ' & ' + '%.3f' % results_dict['IoU'] + ' & ' + '%.3f' % \
                   results_dict['Accuracy'] \
                   + ' & ' + '%.3f' % results_dict['FPR95'] + ' & ' + '%.3f' % results_dict['AUROC'] + ' '


    pickle.dump(metrics.__dict__, open(save_path + dataset + '_metrics.pkl', 'wb'))

logging.info(all_results)
