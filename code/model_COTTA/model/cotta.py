from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import PIL
import torchvision.transforms as transforms
import model.my_transforms as my_transforms
from time import time
import logging
import numpy as np
import cv2


def get_tta_transforms(gaussian_std: float=0.005, soft=True, clip_inputs=False):
    img_shape = (352, 352, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        # my_transforms.Clip(0.0, 0.1),
        # my_transforms.ColorJitterPro(
        #     brightness=0.0001,
        #     contrast=0,
        #     saturation=0,
        #     hue=0,
        #     gamma=0
        # ),
        # transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        # transforms.RandomAffine(
        #     degrees=[-8, 8] if soft else [-15, 15],
        #     translate=(1/16, 1/16),
        #     scale=(0.95, 1.05) if soft else (0.9, 1.1),
        #     shear=None,
        #     resample=PIL.Image.BILINEAR,
        #     fillcolor=None
        # ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        # transforms.CenterCrop(size=n_pixels),
        # transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        # my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def visualize_original_img1(rec_img, save_path):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        name = '{:02d}_img_org.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

def visualize_original_img2(rec_img, save_path):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        name = '{:02d}_img_tra.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)



class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, loss = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs, loss

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, to_aug=True):
        outputs = self.model(x)
        self.model_ema.train()
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        anchor_prob, _ = torch.max(torch.nn.functional.softmax(self.model_anchor(x), dim=1), dim=1, keepdim=True)
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        # print(anchor_prob.mean(0).shape)
        # to_aug = anchor_prob.mean(0)<0.1
        visualize_original_img2(self.transform(x), '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_COTTA/temp/')
        visualize_original_img1(x,
                               '/data/local_userdata/tianxinyu/CVPR_2023/Models_distribution/model_COTTA/temp/')
        if to_aug:
            for i in range(N):

                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update

        # loss = (softmax_entropy(outputs, outputs_ema.detach())).mean()
        loss = structure_loss(F.softmax(outputs, dim=1), F.softmax(outputs_ema.detach(), dim=1))
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        # # Teacher update
        # self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)
        # # Stochastic restore
        # if True:
        #     for nm, m  in self.model.named_modules():
        #         for npp, p in m.named_parameters():
        #             if npp in ['weight', 'bias'] and p.requires_grad:
        #                 mask = (torch.rand(p.shape)<0.001).float().cuda()
        #                 with torch.no_grad():
        #                     p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema, loss


multiclass_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

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


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"