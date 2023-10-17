import torch.nn.functional as F
from fastNLP import LossBase
import numpy as np
import torch
import random
import os

class focal_loss(LossBase):
    def __init__(self,inputs,targets,seq_len=None, alpha=0.8, gamma=2, smooth_factor=2, size_average=True):

        super(focal_loss, self).__init__()
        self._init_param_map(inputs=inputs, targets=targets, seq_len=seq_len)    #确定参数映射
        self.size_average = size_average
        self.alpha = alpha      #负例权重
        self.gamma = gamma
        self.smooth_factor = smooth_factor

    def get_loss(self, inputs, targets, seq_len=None):
        P = F.softmax(inputs,dim=1) 
        # ---------one hot start--------------#
        class_mask = torch.zeros_like(inputs)    #生成和input一样shape的tensor
        class_mask = class_mask.requires_grad_()  # 需要更新， 所以加入梯度计算
        ids = targets.view(-1, 1)  # 取得目标的索引
        class_mask.data.scatter_(1, ids, 1.)  # 利用scatter将索引丢给mask
        # print('targets的one_hot形式\n', class_mask)  # one-hot target生成
        # ---------one hot end-------------------#
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = torch.pow((1 - probs), self.gamma) * log_p
        
        #注意，focal_loss中的α和class_weight_loss中的α不一样
        alpha = torch.ones_like(inputs).cuda()
        alpha[:,0] = 1 - self.alpha
        alpha[:,1] = self.alpha         #正例加权
        alpha = (alpha * class_mask).sum(1).view(-1,1)
        batch_loss = -alpha * loss  

        # 最终将每一个batch的loss加总后平均
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def label_smoothing(pred, target):
    ''' 
    Args:
        pred: prediction of model output    [N, M]
        target: ground truth of sampler [N]
    '''
    label_smooth = 0.05
    class_num = pred.size(1)

    # cross entropy loss with label smoothing
    logprobs = F.log_softmax(pred, dim=1)	# softmax + log
    target = F.one_hot(target, class_num)	# 转换成one-hot
    target = torch.clamp(target.float(), min=label_smooth/(class_num-1), max=1.0-label_smooth)
    loss = -1*torch.sum(target*logprobs, 1)


    return loss.mean()

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
