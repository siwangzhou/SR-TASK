# from model import CS,Deep_resconsturction,CS_stage1
from ops.OmniSR import OmniSR,LR_SR_x8,LR_SR_x4,LR_SR_x3,LR_SR_x2,CS_LR_SR_x2,unshuffle_LR_SR_x2,TAD_LR_SR_x2,CNNCR_EDSR_LR_SR_x2,\
    CNNCR_EDSR_LR_SR_x4,OmniSR_CSUP,CNNCR_EDSR_CSUP_LR_SR_x2
from ops.edsr.edsr import LR_SR_x8,LR_SR_x4,LR_SR_x2,LR_SR_x3,EDSR
from ops.myloss import BlockL1Loss
import torch
import numpy as np
import torchvision.transforms as transforms
import math
import copy
# from data import MyDataset
from torch.utils.checkpoint import checkpoint
from torch import nn, optim
from data import MyDataset3 as MyDataset,data_prefetcher,Test
import time

def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def random_block_zero(tensor,tensor_lr, scale, block_size):
    n,_,w,h=tensor.shape
    col=int(w/block_size)
    row=int(h/block_size)
    list_arr=[]
    for j in range(n):
        randarr = np.arange(col*row)
        np.random.shuffle(randarr)
        list_arr.append(randarr)
        for i in randarr[:int(len(randarr)/2)]:
            tensor[j,:,int(i/col)*block_size:int(i/col)*block_size+block_size,(i%row)*block_size:(i%row)*block_size+block_size]=0
        block_size=int(block_size/scale)
        for i in randarr[:int(len(randarr)/2)]:
            tensor_lr[j,:,int(i/col)*block_size:int(i/col)*block_size+block_size,(i%row)*block_size:(i%row)*block_size+block_size]=0
    return tensor,tensor_lr,list_arr

if __name__ == '__main__':
    kwards = {'upsampling': 4,
              'res_num': 5,
              'block_num': 1,
              'bias': True,
              'block_script_name': 'OSA',
              'block_class_name': 'OSA_Block',
              'window_size': 8,
              'pe': True,
              'ffn_bias': True}
    kwards_lr = {'channel_in' : 3,
                 'channel_out': 3,
                 'block_num': [8, 8],
                 'down_num': 2,
                 'down_scale': 4
    }

    # 判别器模型参数
    kernel_size_d = 3  # 所有卷积模块的核大小
    n_channels_d = 64  # 第1层卷积模块的通道数, 后续每隔1个模块通道数翻倍
    n_blocks_d = 8  # 卷积模块数量
    fc_size_d = 1024  # 全连接层连接数

    # net_G=OmniSR_CSUP(kwards=kwards).cuda()
    # net_G = EDSR(scale=2).cuda()
    net_G = OmniSR(kwards=kwards).cuda()

    data_train = MyDataset(lrsize=64,scales=4)
    data_test = Test(scales=4)
    train_data = torch.utils.data.DataLoader(dataset=data_train, batch_size=32, shuffle=True, num_workers=8,pin_memory=True)
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0,
                                             pin_memory=True)
    # lossMSE = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss()
    blockloss=BlockL1Loss()

    # lossBCE = torch.nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=0.0005, weight_decay=1e-4, betas=[0.9, 0.999])
    #学习率调度器
    step_schedule = optim.lr_scheduler.StepLR(step_size=250, gamma=0.5, optimizer=optimizer_G)
    epochs = 1000
    epoch = 0
    chpoint = torch.load("./Net/Omni_x4/Omni_x4_149.pt")
    epoch = chpoint['epoch']
    net_G.load_state_dict(chpoint['net'])
    optimizer_G.load_state_dict(chpoint['optimizer_net'])
    step_schedule.load_state_dict(chpoint['scheduler_net'])
    # step_schedule.step_size=250

    for i in range(epoch, epochs):
        # net_G.requires_grad_(True)
        psnr_list = []
        net_list = []

        sum_loss = 0
        losses = []
        t1 = time.time()
        t2 = time.time()
        save_num = int(len(train_data) / 4)
        for batch, (cropimg,sourceimg) in enumerate(train_data, start=0):
            cropimg=cropimg.cuda()
            sourceimg=sourceimg.cuda()
            # sourceimg, cropimg,list_arr = random_block_zero(sourceimg, cropimg, block_size=32, scale=8)
            # print(cropimg)

            ##--------生成器训练--------##
            #清空梯度流
            optimizer_G.zero_grad()

            hr_img=net_G(cropimg)
            # lr_img,hr_img = net_G(sourceimg,list_arr)
            # a = transforms.ToPILImage()(lr_img[0] / 2 + 0.5)
            # a.show()
            # a = transforms.ToPILImage()(hr_img[0] / 2 + 0.5)
            # a.show()
            # print(sourceimg.shape,lr_img.shape,hr_img.shape)

            loss1 = lossL1(hr_img, sourceimg)
            # loss2 = lossL1(lr_img, cropimg)
            # loss3=blockloss(hr_img,sourceimg,list_arr)
            # loss_G=loss1+loss2/16+loss3*8
            loss_G=loss1

            sum_loss += loss_G

            loss_G.backward()
            # print("loss_G:", loss1.item(), loss3.item(), loss4.item())
            optimizer_G.step()

            #判断最高psnr 并保存
            if batch%save_num==0 or cropimg is None:
                psnr1_sum=0
                # net_list.append(copy.deepcopy(net_G).cpu())
                # a=LR_SR_INV(kward_lr=kwards_lr, kwards_sr=kwards)
                # a.load_state_dict(net_G.state_dict())
                # a.eval()
                a=copy.deepcopy(net_G.state_dict())
                net_list.append(a)
                net_G.eval()
                for test_batch, (lrimg,hrimg) in enumerate(test_data, start=0):
                    lrimg=lrimg.cuda()
                    hrimg=hrimg.cuda()
                    # hrimg,lrimg,list_arr=random_block_zero(hrimg,lrimg,block_size=32,scale=8)

                    # print(hrimg)
                    # lr_img,hr_img = checkpoint(net_G,hrimg,list_arr, use_reentrant=True)
                    hr_img = checkpoint(net_G, lrimg, use_reentrant=True)
                    # lr_img = lr_img / 2 + 0.5
                    # lr_img = lr_img.clamp(0, 1)
                    # hr_img = net_G.layer3(lr_img)
                    # hr_img = checkpoint(net_G.layer3, (lr_img-0.5)*2, use_reentrant=True)

                    # hr_img = checkpoint(net_G, lrimg, use_reentrant=True)
                    i1 = hrimg.cpu().detach().numpy()[0]
                    i2 = hr_img.cpu().detach().numpy()[0]
                    # i1 = (i1 + 1.0) / 2.0
                    i1 = np.clip(i1, 0.0, 1.0)
                    # i2 = (i2 + 1.0) / 2.0
                    i2 = np.clip(i2, 0.0, 1.0)

                    i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
                    i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
                    # print(i1.shape,i2.shape)
                    psnr1 = psnr_get(i2, i1)
                    psnr1_sum += psnr1
                psnr1_sum=psnr1_sum/(test_batch+1)
                psnr_list.append(psnr1_sum)
                net_G.train()

        psnr_max=max(psnr_list)
        index=psnr_list.index(psnr_max)
        if (i + 1)  >= 0:
            torch.save({
                'epoch': i + 1,
                'net': net_list[index],
                # 'net4': net_list[index].SRnet4.state_dict(),
                # 'net8': net_list[index].SRnet8.state_dict(),
                # 'deblock_net': net_list[index].deblock_model.state_dict(),
                'optimizer_net': optimizer_G.state_dict(),
                'scheduler_net': step_schedule.state_dict(),
            }, './Net/Omni_x4/Omni_x4_{0}.pt'.format(i + 1))
            del net_list
        print('{2}|{3}    avg_loss={0}    time={1}min   psnr:{4}'.format(sum_loss / batch, (time.time() - t1) / 60, i + 1,epochs, psnr_max),index,psnr_list)
        str_list = [str(item) for item in psnr_list]
        # 使用join方法将新列表中的元素连接成一个字符串，元素之间由空格分隔
        str_list = ' '.join(str_list)
        str_write='{0}|{1}    avg_loss={2}    time={3}min   psnr_max:{4}   index={5}'.format(i + 1,epochs, sum_loss / batch, (time.time() - t1) / 60,  psnr_max,index)+'  '+str_list+'\n'
        fp = open('./Net/Omni_x4/Omni_x4.txt', 'a+')
        fp.write(str_write)
        fp.close()
        step_schedule.step()