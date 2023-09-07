import math

from torch import nn
import torch
from torch.nn import functional as F

def MultiLableBCELoss(pred, target, weight=None, pos_weight=None):
    pred = torch.sigmoid(pred) # 对所有预测的结果进行sigmoid限制在0-1之间 不用softmax 因为类别之间不是互斥的
    gamma = 2
    a = 0.25

    # 抑制背景类
    alpha = torch.zeros(pred.size()[1])
    print(alpha)
    alpha[0] += a
    alpha[1:] += (1 - a)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
    print(alpha)

    # 处理正负样本不均衡问题
    if pos_weight is None:
        label_size = pred.size()[1]
        pos_weight = torch.ones(label_size) # 如果没有设置权重 大家都是一样的权重
    # 处理多标签不平衡问题
    if weight is None:
        label_size = pred.size()[1]
        weight = torch.ones(label_size)


    # 统计每一个类别做二分类交叉熵损失的loss结果
    val = 0
    val_fl = 0

    for li_x, li_y in zip(pred, target):
        # 取每一个样本的 预测结果 和 其真实标签  如 tensor([0.6900, 0.7109, 0.5744]) tensor([1., 1., 0.])
        print(li_x, li_y)
        for i, xy in enumerate(zip(li_x,li_y)):
            # 取每个样本中的第i个预测结果和第i个位置上的label  （所以需要是 one——hot类型的label， 并且是 float类型， 才可后续计算 binarycrossentorpy)
            x,y = xy
            print(x, y) # tensor(0.6900) tensor(1.)
            # 对每一个类别进行二分类交叉熵损失计算 然后根据正负样本的权重计算一下 pos_weight是正样本的权重
            loss_val = pos_weight[i] * y * math.log(x, math.e) + (1-y) * math.log(1-x, math.e)
            # focal loss
            loss_fl_val = y * (1-x)**gamma * math.log(x,math.e) + (1-y) * x**gamma * math.log(1-x, math.e)
            # 求和每一个类别的损失 并且给权重
            val += weight[i] * loss_val
            val_fl+= alpha[i] * loss_fl_val

    return -val / (pred.size()[0] * pred.size()[1]), -val_fl / (pred.size()[0] * pred.size()[1])

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds_softmax, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        print(preds_softmax.size()) # [B C]
        preds_logsoft = torch.log(preds_softmax) # log(pt)
        # index = labels.view(-1,1)
        # dim = 1
        print(labels.view(-1,1))
        # preds_softmax = preds_softmax.gather(dim=1,index=labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(dim=1,index=labels)   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))

        # pred_softmax是对应类别为1的预测结果 由于我的target是one-hot类型的

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # 类别权重 [ α, 1-α, 1-α, 1-α, 1-α, ...]
        loss = torch.mul(alpha, loss.t())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

if __name__ == '__main__':

    fc_loss = focal_loss()

    input = torch.tensor([[0.8, 0.9, 0.3], [0.8, 0.9, 0.3], [0.8, 0.9, 0.3], [0.8, 0.9, 0.3]])

    target_onehot = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],dtype=torch.int64)

    target = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]],dtype=torch.int64)

    print(target_onehot.dtype)

    input_softmax = torch.softmax(input,dim=1)

    # print(input_softmax)

    # exit()

    # loss = fc_loss(input_softmax, target)

    # input_softmax_gather = input_softmax.gather(dim=1,index=target)
    # print(input_softmax)
    # print(target)
    # print(input_softmax_gather)

    weight =  torch.Tensor([0.8, 0.5, 0.8]) # 每个标签的权重

    # # 内置函数 求多标签多分类的损失结果
    # loss = nn.MultiLabelSoftMarginLoss(weight=weight)
    # loss_val = loss(input, target_onehot)
    # print("nn.MultiLabelSoftMarginLoss:",loss_val.item())

    # 自己写的函数
    loss_my, loss_my_fl = MultiLableBCELoss(input, target_onehot, weight)
    print("myself MultiLabelSoftMarginLoss:",loss_my.item())
    print("myself MultiLabelSoftMarginLoss add focal loss :",loss_my_fl.item())

    # BCEWithLogitsLoss
    # 合并sigmoid 和 BCE 也就是不需要对input进行sigmoid操作
    loss_BCE = nn.BCEWithLogitsLoss(weight=weight)
    loss_val = loss_BCE(input, target_onehot)
    print("nn.BCEWithLogitsLoss:",loss_val.item())


