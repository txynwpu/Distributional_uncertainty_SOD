import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
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
from grad_extractor import Gradient_Analysis
from einops import rearrange
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')
parser.add_argument('--post_hoc_method', type=str, default='GradNorm', help='method of post-hoc, "GradNorm"')

opt = parser.parse_args()

# save_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_GradNorm/'   # base model weight path
save_path = '/data/local_userdata/tianxinyu/CVPR_2023/metric_distribution/model_GradNorm_my/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/base/Model_30_gen.pth'))

generator.cuda()
generator.eval()

logging.basicConfig(filename=save_path + "sod_metrics_my.log", filemode='w', level=logging.INFO)


dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
all_results = ''
LOSS_CHOISE = 'ones'


for dataset in test_datasets:
    # save_path_pred = save_path + 'pred_my/' + dataset + '/'
    # if not os.path.exists(save_path_pred):
    #     os.makedirs(save_path_pred)
    # save_path_uncer = save_path + 'uncertainty_my/' + dataset + '/'
    # if not os.path.exists(save_path_uncer):
    #     os.makedirs(save_path_uncer)

    test_dataset = EvalDataset(dataset_path, gt_path, opt.testsize, f'test_lists/{dataset}.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    gradient_extractor = Gradient_Analysis(generator, ['decoder.path1.resConfUnit1.conv1'], opt.testsize, opt.testsize, 'max')
    gradient_extractor.cuda()

    metrics = Metrics()
    name_list = []
    res_list = []
    gt_list = []
    h_list = []
    w_list = []

    for i, (image, gt, HH, WW, name) in enumerate(tqdm(test_loader)):
        name = name[0].split('/')[-1].split('.')[0] + '.png'
        image = image.cuda()
        gt = gt.cuda()

        generator_pred = generator.forward(image)

        res = generator_pred
        res = F.upsample(F.softmax(res, dim=1), size=[HH, WW], mode='bilinear', align_corners=False)
        res = res[:, 1, :, :]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res_save = (255 * res).data.cpu().numpy().squeeze()
        # cv2.imwrite(save_path_pred + name, res_save)

        if LOSS_CHOISE == 'flip':
            image_flip = torch.flip(image, [3])
            generator_pred_flip = torch.flip(generator(image_flip), [3])
            loss = (F.softmax(generator_pred, dim=1) - F.softmax(generator_pred_flip, dim=1).detach()).pow(2).mean()
        elif LOSS_CHOISE == 'ones':
            generator_pred_conf = torch.max(F.softmax(generator_pred, dim=1), dim=1, keepdim=True)[0]
            loss = (generator_pred_conf - 1).pow(2).mean()

        loss.backward()
        h_list.append(HH)
        w_list.append(WW)
        name_list.append(name)

        gt = gt.squeeze()
        res_list.append(res.squeeze().detach())
        gt_list.append(gt)


    uncert_list = gradient_extractor.get_gradients()
    for res, gt, h, w, uncert, name in tqdm(zip(res_list, gt_list, h_list, w_list, uncert_list, name_list), desc=f'{dataset} process'):
        uncert = rearrange(torch.from_numpy(uncert), 'h w -> 1 1 h w')
        uncert = F.upsample(uncert, size=(h, w), mode='bilinear', align_corners=False)
        uncert = rearrange(uncert, '1 1 h w -> h w').cuda()

        uncert = (uncert - uncert.min()) / (uncert.max() - uncert.min() + 1e-3)
        # uncert_save = (255 * uncert).data.cpu().numpy().squeeze()
        # uncert_save = uncert_save.astype(np.uint8)
        # uncert_save = cv2.applyColorMap(uncert_save, cv2.COLORMAP_JET)
        # cv2.imwrite(save_path_uncer + name, uncert_save)

        metrics.update(res, uncert, gt)


    logging.info(dataset)
    logging.info(metrics.get_results())
    logging.info('-' * 20)

    results_dict = metrics.get_results()
    all_results += '& ' + '%.3f' % results_dict['F_max'] + ' & ' + '%.3f' % results_dict['IoU'] + ' & ' + '%.3f' % \
                   results_dict['Accuracy'] \
                   + ' & ' + '%.3f' % results_dict['FPR95'] + ' & ' + '%.3f' % results_dict['AUROC'] + ' '


    pickle.dump(metrics.__dict__, open(save_path + dataset + '_metrics.pkl', 'wb'))

logging.info(all_results)
