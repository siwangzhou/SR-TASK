import os
import shutil
import time
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config as config
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset, TrainValidContextDataset,TrainValidContextDataset_CV2
from image_quality_assessment import PSNR, SSIM
from model import Generator, ContentLoss,Generator_X8,Generator_X12,Generator_X6
from scripts.SSLoss import RegionStyleLoss,PCPFeat

from MultiD import Discriminator_with_semseg,Discriminator_cat_mask_with_semseg


import imgproc
import numpy as np

# put all samples to GPU
def dict_to_device(sample):
    """Send dictionary of tensors to device (GPU/CPU)

    Args:
        sample (dict): Dictionary of tensors (image and targets)
        gpu_info(dict): Dictionary with required GPU information
    Returns:
        tensor (tensor): The tensor sent to the device
    """
    sample['image'] = sample['image'].to(config.device)
    sample['image_lr'] = sample['image_lr'].to(config.device)


    for key, target in sample['labels'].items():
        sample['labels'][key] = sample['labels'][key].to(config.device)

    return sample


def main():
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0
    best_mIoU = 0.0

    train_prefetcher, valid_prefetcher, test_prefetcher, valid_dataloader = load_dataset()
    print("Load all datasets successfully.")

    discriminator, generator = build_model()
    print("Build ESRGAN model successfully.")

    psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion, crossentorpy_criterion, multilabel_criterion = define_loss()
    ss_criterion, vgg_model = define_SSLoss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    if config.resume:
        print("Loading RRDBNet model weights")
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        generator.load_state_dict(checkpoint["state_dict"])
        print("Loaded RRDBNet model weights.")

    print("Check whether the pretrained discriminator model is restored...")
    if config.resume_d:
        # Load checkpoint model
        checkpoint = torch.load(config.resume_d, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = discriminator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        # new_state_dict.pop("features.0.weight")

        model_state_dict.update(new_state_dict)
        discriminator.load_state_dict(model_state_dict)
        # Load the optimizer model
        # d_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        # d_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained discriminator model weights.")

    print("Check whether the pretrained generator model is restored...")
    if config.resume_g:
        # Load checkpoint model
        checkpoint = torch.load(config.resume_g, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = generator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        generator.load_state_dict(model_state_dict)
        # Load the optimizer model
        g_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained generator model weights.")

    print(f"start epoch is {config.start_epoch}")

    results_dir = os.path.join("./results", config.exp_name, config.upscale_factor)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("E:/jingou/ESRGAN/samples", "logs", config.exp_name))

    best_psnr = 0.0
    best_ssim = 0.0



    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    for epoch in range(config.start_epoch, config.epochs):

        # train(discriminator,
        #       generator,
        #       train_prefetcher,
        #       pixel_criterion,
        #       content_criterion,
        #       adversarial_criterion,
        #       crossentorpy_criterion,
        #       multilabel_criterion,
        #       d_optimizer,
        #       g_optimizer,
        #       epoch,
        #       writer)

        train_concat_mask(discriminator,
              generator,
              vgg_model,
              train_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              crossentorpy_criterion,
              ss_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              writer)


        psnr, ssim = test(generator, test_prefetcher, epoch, writer, psnr_model, ssim_model, "Test")
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        # is_best = ssim > best_ssim
        # is_best = mIoU > best_mIoU

        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        # best_mIoU = max(mIoU, best_mIoU)
        # 只保存最好的
        if is_best:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_mIoU": best_mIoU,
                        "state_dict": discriminator.state_dict(),
                        "optimizer": d_optimizer.state_dict(),
                        "scheduler": d_scheduler.state_dict()},
                       os.path.join(results_dir, f"d_best_0701.pth.tar"))
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_mIoU": best_mIoU,
                        "state_dict": generator.state_dict(),
                        "optimizer": g_optimizer.state_dict(),
                        "scheduler": g_scheduler.state_dict()},
                       os.path.join(results_dir, f"g_best_0701.pth.tar"))

        if (epoch + 1) == config.epochs:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_mIoU": best_mIoU,
                        "state_dict": discriminator.state_dict(),
                        "optimizer": d_optimizer.state_dict(),
                        "scheduler": d_scheduler.state_dict()},
                       os.path.join(results_dir, f"d_last_0701.pth.tar"))
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_mIoU": best_mIoU,
                        "state_dict": generator.state_dict(),
                        "optimizer": g_optimizer.state_dict(),
                        "scheduler": g_scheduler.state_dict()},
                       os.path.join(results_dir, f"g_last_0701.pth.tar"))

def load_dataset():
    # Load train, test and valid datasets
    train_datasets = TrainValidContextDataset_CV2(config.train_image_dir, config.image_size, config.upscale_factor, image_set='train', txt_name='train.txt')
    valid_datasets = TrainValidContextDataset_CV2(config.valid_image_dir, 512, config.upscale_factor, image_set='val', txt_name='val.txt')
    test_datasets  = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_dataloader, valid_dataloader, test_prefetcher, valid_dataloader


def build_model() -> [nn.Module, nn.Module]:
    # Not cat mask
    # discriminator = Discriminator_with_semseg()
    # 换成cat mask label的D
    discriminator = Discriminator_cat_mask_with_semseg()
    if config.upscale_factor == 4:
        generator = Generator() # X4
    elif config.upscale_factor == 6:
        generator = Generator_X6()
    elif config.upscale_factor == 8:
        generator = Generator_X8() # X8
    else:
        generator = Generator_X12()

    # Transfer to CUDA
    discriminator = discriminator.to(device=config.device)
    generator     = generator.to(device=config.device)

    return discriminator, generator


def define_loss() -> [nn.MSELoss, nn.L1Loss, ContentLoss, nn.MSELoss, nn.BCEWithLogitsLoss, nn.CrossEntropyLoss, nn.MultiLabelSoftMarginLoss()]:
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.L1Loss().to(config.device)
    content_criterion = ContentLoss(config.feature_model_extractor_node,
                                    config.feature_model_normalize_mean,
                                    config.feature_model_normalize_std).to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)
    crossentorpy_criterion = nn.CrossEntropyLoss(ignore_index=255).to(config.device)
    multilabel_criterion = nn.MultiLabelSoftMarginLoss().to(config.device)

    return psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion, crossentorpy_criterion, multilabel_criterion

def define_SSLoss():
    ss_criterion = RegionStyleLoss(reg_num=21).to(config.device)
    vgg_model_feat = PCPFeat(weight_path='./results/pre-trained/vgg19-dcbb9e9d.pth').to(config.device)
    return ss_criterion, vgg_model_feat

def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(discriminator.parameters(), config.model_lr, config.model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.model_lr, config.model_betas)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.MultiStepLR,
                                                                           lr_scheduler.MultiStepLR]:
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma)

    return d_scheduler, g_scheduler



def train(discriminator: nn.Module,
          generator: nn.Module,
          train_dataloader,
          pixel_criterion: nn.L1Loss,
          content_criterion: ContentLoss,
          adversarial_criterion: nn.BCEWithLogitsLoss,
          ce_criterion: nn.CrossEntropyLoss,
          multice_criterion: nn.MultiLabelSoftMarginLoss,
          d_optimizer: optim.Adam,
          g_optimizer: optim.Adam,
          epoch: int,
          writer: SummaryWriter) -> None:

    # Calculate how many batches of data are in each Epoch
    batches = len(train_dataloader)

    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    ce_losses = AverageMeter('Semseg CrossEntorpy loss', ":6.6f")
    multice_losses = AverageMeter('Multi Label Softmax loss', ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")

    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses, ce_losses, multice_losses,
                              d_hr_probabilities, d_sr_probabilities,
                              psnres],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the adversarial network model in training mode
    discriminator.train()
    generator.train()


    # Get the initialization training time
    end = time.time()

    for idx,samples in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # all to tensor
        dict_to_device(samples)

        # get data
        lr = samples["image_lr"]
        hr = samples["image"]
        semseg_mask = samples["labels"]["semseg"]
        labels      = samples["labels"]["label"]

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        discriminator.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        # Use the generator model to generate fake samples
        sr = generator(lr) # x_fake

        # d_hr_real_or_fake_logits,  d_hr_pred_semseg_logits = discriminator(hr) # x_real
        # d_sr_real_or_fake_logits,  d_sr_pred_semseg_logits = discriminator(sr.detach().clone()) # E[x_fake] = torch.mean(sr_output) sr.detach()此时sr不需要计算梯度

        # concat
        hr_concat = torch.concat((hr, semseg_mask),dim=1)
        sr_concat = torch.concat((sr, semseg_mask),dim=1)
        d_hr_real_or_fake_logits, d_hr_pred_semseg_logits = discriminator(hr_concat)  # x_real
        d_sr_real_or_fake_logits, d_sr_pred_semseg_logits = discriminator(sr_concat.detach().clone())  # E[x_fake] = torch.mean(sr_output) sr.detach()此时sr不需要计算梯度

        # d hr loss
        d_adv_loss_hr = adversarial_criterion(d_hr_real_or_fake_logits - torch.mean(d_sr_real_or_fake_logits), real_label) * 0.5 # (x_real - E[x_fake] , 1)
        d_semseg_loss_hr = ce_criterion(d_hr_pred_semseg_logits, semseg_mask)
        d_loss_hr = d_adv_loss_hr + d_semseg_loss_hr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        d_loss_hr.backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples

        # d_sr_real_or_fake_logits,  d_sr_pred_semseg_logits = discriminator(sr.detach().clone()) # E[x_fake] = torch.mean(sr_output) sr.detach()此时sr不需要计算梯度

        d_sr_real_or_fake_logits,  d_sr_pred_semseg_logits = discriminator(sr_concat.detach().clone()) # E[x_fake] = torch.mean(sr_output) sr.detach()此时sr不需要计算梯度

        # d sr loss
        d_adv_loss_sr = adversarial_criterion(d_sr_real_or_fake_logits - torch.mean(d_hr_real_or_fake_logits), fake_label) * 0.5 # (x_fake - E[x_real] , 0)
        d_semseg_loss_sr = ce_criterion(d_sr_pred_semseg_logits, semseg_mask)
        d_loss_sr = d_adv_loss_sr + d_semseg_loss_sr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        d_loss_sr.backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_hr + d_loss_sr

        # Improve the discriminator model's ability to classify real and fake samples
        d_optimizer.step()
        # Finish training the discriminator model



        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        # Output discriminator to discriminate object probability
        # g_hr_real_or_fake_logits,  g_hr_pred_semseg_logits = discriminator(hr.detach().clone())
        # g_sr_real_or_fake_logits,  g_sr_pred_semseg_logits = discriminator(sr) # sr此时是需要梯度的 训练G了

        # concat
        g_hr_real_or_fake_logits,  g_hr_pred_semseg_logits = discriminator(hr_concat.detach().clone())
        g_sr_real_or_fake_logits,  g_sr_pred_semseg_logits = discriminator(sr_concat) # sr此时是需要梯度的 训练G了

        pixel_loss = config.pixel_weight * pixel_criterion(sr, hr)
        content_loss = config.content_weight * content_criterion(sr, hr)
        g_adv_loss_hr = adversarial_criterion(g_hr_real_or_fake_logits - torch.mean(g_sr_real_or_fake_logits), fake_label) * 0.5
        g_adv_loss_sr = adversarial_criterion(g_sr_real_or_fake_logits - torch.mean(g_hr_real_or_fake_logits), real_label) * 0.5
        adversarial_loss = config.adversarial_weight * (g_adv_loss_hr + g_adv_loss_sr)
        # Computational multi tasks network loss
        g_semseg_loss_sr = config.semseg_weight * ce_criterion(g_sr_pred_semseg_logits, semseg_mask)
        # Calculate the generator total loss value
        g_loss = pixel_loss + content_loss + adversarial_loss + g_semseg_loss_sr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        g_loss.backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        g_optimizer.step()
        # Finish training the generator model

        # Calculate the score of the discriminator on real samples and fake samples, the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(d_hr_real_or_fake_logits.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(d_sr_real_or_fake_logits.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        ce_losses.update(g_semseg_loss_sr.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if idx % config.print_frequency == 0:
            iters = idx + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/Semseg_Loss", g_semseg_loss_sr.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(idx)



def train_concat_mask(discriminator: nn.Module,
          generator: nn.Module,
          vgg_model,
          train_dataloader,
          pixel_criterion: nn.L1Loss,
          content_criterion: ContentLoss,
          adversarial_criterion: nn.BCEWithLogitsLoss,
          ce_criterion: nn.CrossEntropyLoss,
          ss_criterion: RegionStyleLoss,
          d_optimizer: optim.Adam,
          g_optimizer: optim.Adam,
          epoch: int,
          writer: SummaryWriter) -> None:

    # Calculate how many batches of data are in each Epoch
    batches = len(train_dataloader)

    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    ce_losses = AverageMeter('Semseg CrossEntorpy loss', ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")

    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses, ce_losses,
                              d_hr_probabilities, d_sr_probabilities,
                              psnres],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the adversarial network model in training mode
    discriminator.train()
    generator.train()


    # Get the initialization training time
    end = time.time()

    for idx,samples in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # all to tensor
        dict_to_device(samples)

        # get data
        lr = samples["image_lr"]
        hr = samples["image"]
        # use to cal semseg loss
        semseg_mask = samples["labels"]["semseg"]
        # use to concat with hr/sr,then input to Discriminator
        semseg_mask_onehot = samples["labels"]["semseg_onehot"]

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)


        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        discriminator.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        # Use the generator model to generate fake samples
        sr = generator(lr) # x_fake


        # !!!!D中 送入D的 hr 和 SR 都要和 mask 进行 concat 然后再卷积
        hr_cat_mask = torch.cat((hr, semseg_mask_onehot), dim=1) # 按channel cat
        sr_cat_mask = torch.cat((sr.detach(), semseg_mask_onehot), dim=1) # 按channel cat

        # !!!!D中 送入D的就是hr和sr 不需要cat --- 消融实验 只要ssloss
        # hr_cat_mask = hr
        # sr_cat_mask = sr.detach()

        # D result
        d_hr_real_or_fake_logits,  d_hr_pred_semseg_logits = discriminator(hr_cat_mask) # x_real
        d_sr_real_or_fake_logits,  d_sr_pred_semseg_logits = discriminator(sr_cat_mask.clone()) # E[x_fake] = torch.mean(sr_output) sr.detach()此时sr不需要计算梯度
        # d hr loss
        d_adv_loss_hr = adversarial_criterion(d_hr_real_or_fake_logits - torch.mean(d_sr_real_or_fake_logits), real_label) * 0.5 # (x_real - E[x_fake] , 1)
        d_semseg_loss_hr = ce_criterion(d_hr_pred_semseg_logits, semseg_mask)
        d_loss_hr = d_adv_loss_hr + d_semseg_loss_hr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        d_loss_hr.backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples
        d_sr_real_or_fake_logits,  d_sr_pred_semseg_logits = discriminator(sr_cat_mask.clone()) # E[x_fake] = torch.mean(sr_output) sr.detach()此时sr不需要计算梯度
        # d sr loss
        d_adv_loss_sr = adversarial_criterion(d_sr_real_or_fake_logits - torch.mean(d_hr_real_or_fake_logits), fake_label) * 0.5 # (x_fake - E[x_real] , 0)
        d_semseg_loss_sr = ce_criterion(d_sr_pred_semseg_logits, semseg_mask)
        d_loss_sr = d_adv_loss_sr + d_semseg_loss_sr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        d_loss_sr.backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_hr + d_loss_sr

        # Improve the discriminator model's ability to classify real and fake samples
        d_optimizer.step()
        # Finish training the discriminator model


        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        # Output discriminator to discriminate object probability

        g_hr_real_or_fake_logits,  g_hr_pred_semseg_logits = discriminator(hr_cat_mask.clone())
        g_sr_real_or_fake_logits,  g_sr_pred_semseg_logits = discriminator(sr_cat_mask) # sr此时是需要梯度的 训练G了
        pixel_loss = config.pixel_weight * pixel_criterion(sr, hr)
        content_loss = config.content_weight * content_criterion(sr, hr)

        g_adv_loss_hr = adversarial_criterion(g_hr_real_or_fake_logits - torch.mean(g_sr_real_or_fake_logits), fake_label) * 0.5
        g_adv_loss_sr = adversarial_criterion(g_sr_real_or_fake_logits - torch.mean(g_hr_real_or_fake_logits), real_label) * 0.5
        adversarial_loss = config.adversarial_weight * (g_adv_loss_hr + g_adv_loss_sr)
        # Computational multi tasks network loss
        g_semseg_loss_sr = config.semseg_weight * ce_criterion(g_sr_pred_semseg_logits, semseg_mask)

        # Cal SSLoss
        # 1、 提取特征
        sr_feat = vgg_model(sr)
        hr_feat = vgg_model(hr.detach())
        # 2、计算ss loss
        g_semseg_ss_loss_sr = ss_criterion(sr_feat,hr_feat,semseg_mask_onehot)

        # # Calculate the generator total loss value with out g_semseg_ss_loss_sr
        # g_loss = pixel_loss + content_loss + adversarial_loss + g_semseg_loss_sr

        # Add ss Loss to G loss
        g_loss = pixel_loss + content_loss + adversarial_loss + g_semseg_loss_sr + g_semseg_ss_loss_sr

        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        g_loss.backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        g_optimizer.step()
        # Finish training the generator model

        # Calculate the score of the discriminator on real samples and fake samples, the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(d_hr_real_or_fake_logits.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(d_sr_real_or_fake_logits.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        ce_losses.update(g_semseg_loss_sr.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if idx % config.print_frequency == 0:
            iters = idx + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/Semseg_Loss", g_semseg_loss_sr.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(idx)


def validate(model,
             valid_dataloader,
             epoch,
             writer,
             psnr_model: nn.Module,
             ssim_model: nn.Module,
             mode) -> [float, float]:

    batches = len(valid_dataloader)
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix=f"{mode}: ")

    # Put the model in verification mode
    model.eval()

    # Calculate the time it takes to test a batch of data
    end = time.time()

    with torch.no_grad():

        for batch_index,samples in enumerate(valid_dataloader): # 512*512 padding size

            dict_to_device(samples)

            # measure data loading time
            lr = samples["image_lr"]
            hr = samples["image"]
            size = samples["meta"]["im_size"]

            # Mixed precision
            sr = model(lr)

            # 去掉填充的地方 只对 原始尺寸的图像 计算PSNR
            sr = imgproc.process_image_depadding(sr,size[0],size[1])
            hr = imgproc.process_image_depadding(hr,size[0],size[1])

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % 500 == 0:
                progress.display(batch_index)

            # print metrics
        progress.display_summary()

        if mode == "Valid" or mode == "Test":
            writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
            writer.add_scalar(f"{mode}/SSIM", psnres.avg, epoch + 1)
        else:
            raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

        return psnres.avg, ssimes.avg


def validate_IoU(model,valid_dataloader,epoch):


    # Put the model in verification mode
    # model is Single-D

    confmat = evaluate(model, valid_dataloader, num_classes=21)
    val_info = str(confmat)
    print(f"[{epoch}/{config.epochs}]\n")
    print(val_info)
    return confmat.return_mIoU()


def test(model: nn.Module,
             data_prefetcher: CUDAPrefetcher,
             epoch: int,
             writer: SummaryWriter,
             psnr_model: nn.Module,
             ssim_model: nn.Module,
             mode: str) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        psnr_model (nn.Module): The model used to calculate the PSNR function
        ssim_model (nn.Module): The model used to compute the SSIM function
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():

        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", psnres.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg



import distributed_utils as utils
def evaluate(model, data_loader, num_classes=21):
    model.eval() # 任务模型
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    with torch.no_grad():

        for samples in metric_logger.log_every(data_loader, 100, header):

            dict_to_device(samples)

            image  = samples["image"]
            target = samples["labels"]["semseg"]

            # 真实图像输入single_D 执行任务 并且计算准确率
            _,output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

# Copy form "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\n".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
