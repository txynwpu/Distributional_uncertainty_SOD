import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
from scipy import misc
from model.ResNet_models import Pred_endecoder
import torchvision.transforms as transforms
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import logging
from metrics import Metrics
from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')

opt = parser.parse_args()

weight_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_DC/'
save_path = '/data/local_userdata/tianxinyu/CVPR_2023/metric_distribution/model_DC/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load(weight_path + 'Model_30_gen.pth'))

generator.cuda()
generator.eval()

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + "sod_metrics_new_entropy.log", filemode='w', level=logging.INFO)

dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
all_results = ''

gaussian_blur_op = transforms.GaussianBlur(kernel_size=(7,13), sigma=(0.1,0.2))
with torch.no_grad():
    for dataset in test_datasets:
        # save_path_pred = save_path + 'pred_new_entropy/' + dataset + '/'
        # if not os.path.exists(save_path_pred):
        #     os.makedirs(save_path_pred)
        # save_path_uncer = save_path + 'uncertainty_new_entropy/' + dataset + '/'
        # if not os.path.exists(save_path_uncer):
        #     os.makedirs(save_path_uncer)

        metrics = Metrics()
        image_root = dataset_path + dataset + '/'
        gt_root = gt_path + dataset + '/'
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        for i in tqdm(range(test_loader.size), desc=dataset):
            image, gt, HH, WW, name = test_loader.load_data(gen_cor_flag=False)
            cimage = test_loader.load_data(gen_cor_flag=True)
            image = image.cuda()
            cimage = cimage.cuda()
            gt = gt.cuda()
            gc_img = gaussian_blur_op(cimage)
            generator_pred = generator.forward(gc_img, image - cimage)

            generator_pred = F.upsample(generator_pred, size=[WW, HH], mode='bilinear', align_corners=False)
            probability = F.softmax(generator_pred, dim=1)
            # res = probability[:, 1, :, :]
            pred_list = probability

            for iter in range(10):
                cimage = test_loader.load_data(gen_cor_flag = True)
                cimage = cimage.cuda()
                gc_img = gaussian_blur_op(cimage)
                generator_pred = generator.forward(gc_img, image-cimage)
                generator_pred = F.upsample(generator_pred, size=[WW, HH], mode='bilinear', align_corners=False)
                probability = F.softmax(generator_pred, dim=1)
                # res = probability[:, 1, :, :]
                pred_list = torch.cat((pred_list, probability), 0)

            # mean_pred = torch.mean(pred_list, dim=0, keepdim=True)
            # var_pred = torch.var(pred_list, dim=0, keepdim=True)
            pred_cat_mean = torch.mean(pred_list, 0, keepdim=True)
            mean_pred = pred_cat_mean[:, 1, :, :]
            # res = (mean_pred - mean_pred.min()) / (mean_pred.max() - mean_pred.min() + 1e-8)
            # res = (255 * res).data.cpu().numpy().squeeze()
            # cv2.imwrite(save_path_pred + name, res)

            var_pred = -1 * pred_cat_mean[:, 0, :, :] * torch.log(pred_cat_mean[:, 0, :, :] + 1e-8) - pred_cat_mean[:, 1, :, :] * torch.log(pred_cat_mean[:, 1, :, :] + 1e-8)
            var_pred = (var_pred - var_pred.min()) / (var_pred.max() - var_pred.min() + 1e-8)
            # res = (255 * var_pred).data.cpu().numpy().squeeze()
            # res = res.astype(np.uint8)
            # res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
            # cv2.imwrite(save_path_uncer + name, res)

            metrics.update(mean_pred.squeeze(), var_pred.squeeze(), gt.squeeze())

        logging.info(dataset)
        logging.info(metrics.get_results())
        logging.info('-' * 20)

        results_dict = metrics.get_results()
        all_results += '& ' + '%.3f' % results_dict['F_max'] + ' & ' + '%.3f' % results_dict['IoU'] + ' & ' + '%.3f' % \
                       results_dict['Accuracy'] \
                       + ' & ' + '%.3f' % results_dict['FPR95'] + ' & ' + '%.3f' % results_dict['AUROC'] + ' '

        pickle.dump(metrics.__dict__, open(save_path + dataset + '_metrics.pkl', 'wb'))

logging.info(all_results)
