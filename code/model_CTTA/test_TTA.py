import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from model.loss_predictor import Pred_loss
from data import EvalDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import logging
from torch.utils.data import DataLoader
from metrics import Metrics
from tqdm import tqdm
import ttach as tta
import random
import torchvision.transforms as transforms
import pickle


class randomGaussian():
    def __init__(self, mean=0, std=0.15):
        self.mean = mean
        self.std = std

    def forward(self, img):

        noise = np.random.normal(loc=self.mean, scale=self.std, size=img.shape)
        img = img + torch.from_numpy(noise).float().cuda()

        return img

class trasform0():
    def __init__(self):
        self.num = 0
    def forward(self, image):
        return image


def visualize_pred(pred, save_path):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_pred_pred.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_original_img(rec_img, save_path, name):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        name = name + '_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')

opt = parser.parse_args()

weight_path = '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_CTTA/'
save_path = '/data/local_userdata/tianxinyu/CVPR_2023/metric_distribution/model_CTTA/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/base/Model_30_gen.pth'))
generator.cuda()
generator.eval()

loss_model = Pred_loss()
loss_model.load_state_dict(torch.load(weight_path + 'Model_30_gen.pth'))
loss_model.cuda()
loss_model.eval()

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + "sod_metrics_new.log", filemode='w', level=logging.INFO)

dataset_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/img/'
gt_path = '/data/local_userdata/tianxinyu/CVPR2022_multi-label-dataset/test/gt/'
test_datasets = ['DUTS_Test', 'ECSSD', 'DUT']
all_results = ''

transform0 = trasform0()
transform1 = randomGaussian(mean=0, std=0.1)
transform2 = randomGaussian(mean=0, std=0.05)
transform3 = randomGaussian(mean=0, std=0.08)
transform4 = randomGaussian(mean=0, std=0.12)
transform5 = randomGaussian(mean=0, std=0.02)
transform6 = tta.HorizontalFlip()
transform7 = tta.Rotate90(angles=[180])
transform8 = tta.Multiply(factors=[0.99])
transform9 = tta.Multiply(factors=[0.95])
transform10 = tta.Multiply(factors=[1.01])
transform11 = tta.Multiply(factors=[1.02])
transform12 = tta.Multiply(factors=[1.05])
transform13 = tta.Scale(scales=[0.90909])
transform14 = tta.Scale(scales=[1.09091])

transform_list = [transform0, transform1, transform2, transform3, transform4, transform5, transform6, transform7, transform8, transform9,
                  transform10, transform11, transform12, transform13, transform14]
param_list = [0, 0, 0, 0, 0, 0, True, 180, 0.99, 0.95, 1.01, 1.02, 1.05, 0.90909, 1.09091]

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
        image = image.cuda().detach()
        gt = gt.cuda()
        id = torch.ones(20)

        # visualize_original_img(image, 'temp/', '00')
        # image0 = transform0.forward(image)
        # # visualize_original_img(image, 'temp/', '1')
        # image1 = transform1.forward(image)
        # image2 = transform2.forward(image)
        # image3 = transform3.forward(image)
        # image4 = transform4.forward(image)
        # image5 = transform5.forward(image)
        # image6 = transform6.apply_aug_image(image, param_list[6])
        # image7 = transform7.apply_aug_image(image, param_list[7])
        # image8 = transform8.apply_aug_image(image, param_list[8])
        # image9 = transform9.apply_aug_image(image, param_list[9])
        # image10 = transform10.apply_aug_image(image, param_list[10])
        # image11 = transform11.apply_aug_image(image, param_list[11])
        # image12 = transform12.apply_aug_image(image, param_list[12])
        # image13 = transform13.apply_aug_image(image, param_list[13])
        # image14 = transform14.apply_aug_image(image, param_list[14])
        # #
        # visualize_original_img(image, 'temp/', '01')
        # visualize_original_img(image0, 'temp/', '0')
        # visualize_original_img(image1, 'temp/', '1')
        # visualize_original_img(image2, 'temp/', '2')
        # visualize_original_img(image3, 'temp/', '3')
        # visualize_original_img(image4, 'temp/', '4')
        # visualize_original_img(image5, 'temp/', '5')
        # visualize_original_img(image6, 'temp/', '6')
        # visualize_original_img(image7, 'temp/', '7')
        # visualize_original_img(image8, 'temp/', '8')
        # visualize_original_img(image9, 'temp/', '9')
        # visualize_original_img(image10, 'temp/', '10')
        # visualize_original_img(image11, 'temp/', '11')
        # visualize_original_img(image12, 'temp/', '12')
        # visualize_original_img(image13, 'temp/', '13')
        # visualize_original_img(image14, 'temp/', '14')
        #
        # if i==1:
        #     break

        org_image = image
        with torch.no_grad():
            generator_pred = generator.forward(image)

        generator_pred = F.upsample(generator_pred, size=[HH, WW], mode='bilinear', align_corners=False)
        pred_list = generator_pred[0].unsqueeze(0)

        for image_iter in range(3):       #对原始图片做三轮增强
            print(image_iter)
            image = org_image
            id = torch.ones(15)
            aug_iter = 0    # 每次选最合适的5个aug
            with torch.no_grad():
                for aug_iter in range(3):
                    loss = []
                    print(aug_iter)
                    for j in range(6):
                        transform = transform_list[j]
                        if id[j] == 1:
                            loss.append(loss_model(transform.forward(image)).item())
                        else:
                            loss.append(float("inf"))

                    for j in range(6, 13):
                        transform = transform_list[j]
                        if id[j] == 1:
                            loss.append(loss_model(transform.apply_aug_image(image, param_list[j])).item())
                        else:
                            loss.append(float("inf"))

                    if id[13] == 1:
                        image_320 = F.upsample(image, size=(320, 320), mode='bilinear', align_corners=True)
                        loss.append(loss_model(image_320))
                    else:
                        loss.append(float("inf"))

                    if id[14] == 1:
                        image_384 = F.upsample(image, size=(384, 384), mode='bilinear', align_corners=True)
                        loss.append(loss_model(image_384))
                    else:
                        loss.append(float("inf"))

                    index_tras = np.array(loss).argmin()
                    id[index_tras] = 0
                    transform_best = transform_list[index_tras]
                    if index_tras < 6:
                        image = transform_best.forward(image)
                    elif index_tras >= 6 and index_tras < 13:
                        image = transform_best.apply_aug_image(image, param_list[index_tras])
                    elif index_tras == 13:
                        image = image_320
                    else:
                        assert index_tras == 14
                        image = image_384

            # image_list = torch.cat((image_list, image), dim=0)
            # aug_id_list = torch.cat((aug_id_list, id.unsqueeze(0)), dim=0)
            assert torch.sum(id) == 12
            with torch.no_grad():
                pred = generator.forward(image)

            if id[6] == 0:
                pred = transform6.apply_deaug_mask(pred, apply=True)
            if id[7] == 0:
                pred = transform7.apply_deaug_mask(pred, angle=180)
            if id[13] == 0:
                pred = F.upsample(pred, size=(352, 352), mode='bilinear', align_corners=True)
            if id[14] == 0:
                pred = F.upsample(pred, size=(352, 352), mode='bilinear', align_corners=True)

            pred = F.upsample(pred, size=[HH, WW], mode='bilinear', align_corners=False)
            pred_list = torch.cat((pred_list, pred), dim=0)

        pred_list_prob = F.softmax(pred_list, dim=1)
        visualize_pred(pred_list_prob[:, 1, :, :].unsqueeze(1), '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_CTTA/temp/')

        entropy = -1 * pred_list_prob[:, 0, :, :] * torch.log(pred_list_prob[:, 0, :, :] + 1e-8) - pred_list_prob[:, 1, :, :] * torch.log(pred_list_prob[:, 1, :, :] + 1e-8)
        entropy = entropy.mean(dim=[1, 2])
        print(entropy.shape)
        entropy = F.softmax(entropy)

        pred_final = entropy[0] * pred_list[0] + entropy[1] * pred_list[1] + entropy[2] * pred_list[2] + entropy[3] * pred_list[3]

        probability = F.softmax(pred_final.unsqueeze(0), dim=1)

        res = probability[:, 1, :, :]
        # res_save = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res_save = (255 * res_save).data.cpu().numpy().squeeze()
        # cv2.imwrite(save_path_pred + name, res_save)

        # uncert = -1 * res * torch.log(res + 1e-8)
        uncert = -1 * probability[:, 0, :, :] * torch.log(probability[:, 0, :, :] + 1e-8) - probability[:, 1, :, :] * torch.log(
            probability[:, 1, :, :] + 1e-8)
        # uncert = torch.var(pred_list_prob[:, 1, :, :], dim=0, keepdim=True)
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
