# Modeling the Distributional Uncertainty for Salient Object Detection Models
## CVPR 2023

### [Project Page(coming soon)](https://npucvr.github.io/Distributional_uncertainty_SOD) | [Paper(coming soon)](https://github.com/txynwpu/Distributional_uncertainty_SOD) | [Video(coming soon)](https://github.com/txynwpu/Distributional_uncertainty_SOD) | [LAB Page](http://npu-cvr.cn/)

## Abstract
Most of the existing salient object detection (SOD) models focus on improving the overall model performance, without explicitly explaining the discrepancy between the training and testing distributions. In this paper, we investigate a particular type of epistemic uncertainty, namely distributional uncertainty, for salient object detection. Specifically, for the first time, we explore the existing class-aware distribution gap exploration techniques, _i.e._ long-tail learning, single-model uncertainty modeling and test-time strategies, and adapt them to model the distributional uncertainty for our class-agnostic task. We define test sample that is dissimilar to the training dataset as being “out-of-distribution” (OOD) samples. Different from the conventional OOD definition, where OOD samples are those not belonging to the closed-world training categories, OOD samples for SOD are those break the basic priors of saliency, _i.e._ center prior, color contrast prior, compactness prior and _etc._, indicating OOD as being “continuous” instead of being discrete for our task. We’ve carried out extensive experimental results to verify effectiveness of existing distribution gap modeling techniques for SOD, and conclude that both train-time single-model uncertainty estimation techniques and weight-regularization solutions that preventing model activation from drifting too much are promising directions for modeling distributional uncertainty for SOD.

## Distributional Uncertainty

![image](https://github.com/txynwpu/Distributional_uncertainty_SOD/blob/main/image/uncertainty_explain.png#pic_center 500x1200)

<img src="https://github.com/txynwpu/Distributional_uncertainty_SOD/blob/main/image/uncertainty_explain.png" width="500" alt="" align=center />

Visualization of different types of uncertainty, where aleatoric uncertainty $p(y|x^\star,\theta)$ is caused by the inherent randomness of the data, model uncertainty $p(\theta|D)$ happens when there exists low-density region, leading to multiple solutions within this region, and distributional uncertainty $p(x^\star|D)$ occurs when the test sample $x^\star$ fails to fit in the model based on the training dataset $D$. 

## Motivation

<img src="https://github.com/txynwpu/Distributional_uncertainty_SOD/blob/main/image/sod_distributional_uncertainty.png" width="800" alt="" align=center />

“OOD” samples for salient object detection. Different from the class-aware tasks, OOD for saliency detection is continuous, which can be defined as attributes that break the basic saliency priors, _i.e._ center prior, contrast prior, compactness prior, _etc_. We aim to explore distributional uncertainty estimation for saliency detection.


## Environment
Pytorch 1.10.0  
Torchvision 0.11.1  
Cuda 11.4

## Dataset
we use DUTS training dataset to train our models, and use DUTS-test, ECSSD and DUT datasets for evaluation.


## Acknowledgement
We summarize many methods ([MCDropout](http://proceedings.mlr.press/v48/gal16.pdf), [DeepEnsemble](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf), [TALT](https://arxiv.org/pdf/2107.09249.pdf), [NorCal](https://proceedings.neurips.cc/paper/2021/file/14ad095ecc1c3e1b87f3c522836e9158-Paper.pdf), [WB](https://openaccess.thecvf.com/content/CVPR2022/papers/Alshammari_Long-Tailed_Recognition_via_Weight_Balancing_CVPR_2022_paper.pdf), [MCP](https://arxiv.org/pdf/1610.02136.pdf), [Energy](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf), [SML](http://openaccess.thecvf.com/content/ICCV2021/papers/Jung_Standardized_Max_Logits_A_Simple_yet_Effective_Approach_for_Identifying_ICCV_2021_paper.pdf), [GradNorm](https://proceedings.neurips.cc/paper/2021/file/063e26c670d07bb7c4d30e6fc69fe056-Paper.pdf), [ExGrad](https://arxiv.org/pdf/2205.10439), [TCP](https://proceedings.neurips.cc/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf), [DC](https://arxiv.org/pdf/2207.07235), [ReAct](https://proceedings.neurips.cc/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf), [CoTTA](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf), [CTTA](https://proceedings.mlr.press/v180/chun22a/chun22a.pdf),) and apply them to the SOD task to model distributional uncertainty. Thanks for these awesome and meaningful works.
