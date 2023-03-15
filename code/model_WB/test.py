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
import logging
from torch.utils.data import DataLoader
from metrics import Metrics
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')

opt = parser.parse_args()

weight_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_WB/'
save_path = '/data/local_userdata/tianxinyu/CVPR_2023/metric_distribution/model_WB/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load(weight_path + 'Model_30_gen.pth'))

generator.cuda()
generator.eval()

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + "sod_metrics_new.log", filemode='w', level=logging.INFO)

dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
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
        generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
        probability = F.softmax(generator_pred, dim=1)

        res = probability[:, 1, :, :]
        # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res_save = (255 * res_save).data.cpu().numpy().squeeze()
        # cv2.imwrite(save_path_pred + name, res_save)

        # uncert = -1 * res * torch.log(res + 1e-8)
        uncert = -1 * probability[:, 0, :, :] * torch.log(probability[:, 0, :, :] + 1e-8) - probability[:, 1, :, :] * torch.log(
            probability[:, 1, :, :] + 1e-8)
        uncert = (uncert - uncert.min()) / (uncert.max() - uncert.min() + 1e-3)

        # uncert_save = (255 * uncert).data.cpu().numpy().squeeze()
        # uncert_save = uncert_save.astype(np.uint8)
        # uncert_save = cv2.applyColorMap(uncert_save, cv2.COLORMAP_JET)
        # cv2.imwrite(save_path_uncer + name, uncert_save)

        metrics.update(res.squeeze(), uncert.squeeze(), gt.squeeze())

    logging.info(dataset)
    logging.info(metrics.get_results())
    logging.info('-' * 20)

    results_dict = metrics.get_results()
    all_results += '& ' + '%.3f' % results_dict['F_max'] + ' & ' + '%.3f' % results_dict['IoU'] + ' & ' + '%.3f' % results_dict['Accuracy']\
                   + ' & ' + '%.3f' % results_dict['FPR95'] + ' & ' + '%.3f' % results_dict['AUROC'] + ' '


    pickle.dump(metrics.__dict__, open(save_path + dataset + '_metrics.pkl', 'wb'))

logging.info(all_results)
