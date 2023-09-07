import os
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
from dataset import CUDAPrefetcher, TestImageDataset,TrainValidContextDataset_CV2
from image_quality_assessment import PSNR, SSIM
from model import Generator, ContentLoss,Generator_X8,Generator_X12,Generator_X6
from scripts.SSLoss import RegionStyleLoss,PCPFeat

import imgproc

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
    best_mIoU = 0.0
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, valid_prefetcher, test_prefetcher, valid_dataloader = load_dataset()
    print("Load all datasets successfully.")

    generator = build_model()
    print("Build ESRGAN model successfully.")

    psnr_criterion, pixel_criterion, content_criterion, crossentorpy_criterion = define_loss()
    ss_criterion, vgg_model = define_SSLoss()
    print("Define all loss functions successfully.")

    g_optimizer = define_optimizer(generator)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    if config.resume:
        print("Loading RRDBNet model weights")
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        generator.load_state_dict(checkpoint["state_dict"])
        print("Loaded RRDBNet model weights.")


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
    writer = SummaryWriter(os.path.join("./logs", config.exp_name))


    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    for epoch in range(config.start_epoch, config.epochs):

        train(generator, train_prefetcher, pixel_criterion, content_criterion, g_optimizer, epoch, writer)
        psnr, ssim = test(generator, test_prefetcher, epoch, writer, psnr_model, ssim_model, "Test")
        print("\n")

        # Update LR
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr & ssim > best_ssim

        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)

        # 只保存最好的
        if is_best:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_mIoU": best_mIoU,
                        "state_dict": generator.state_dict(),
                        "optimizer": g_optimizer.state_dict(),
                        "scheduler": g_scheduler.state_dict()},
                       os.path.join(results_dir, f"g_best.pth.tar"))

        if (epoch + 1) == config.epochs:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_mIoU": best_mIoU,
                        "state_dict": generator.state_dict(),
                        "optimizer": g_optimizer.state_dict(),
                        "scheduler": g_scheduler.state_dict()},
                       os.path.join(results_dir, f"g_last.pth.tar"))

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


def build_model():
    if config.upscale_factor == 4:
        generator = Generator() # X4
    elif config.upscale_factor == 6:
        generator = Generator_X6()
    elif config.upscale_factor == 8:
        generator = Generator_X8() # X8
    else:
        generator = Generator_X12()

    # Transfer to CUDA
    generator     = generator.to(device=config.device)

    return generator


def define_loss() -> [nn.MSELoss, nn.L1Loss, ContentLoss, nn.CrossEntropyLoss]:
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.L1Loss().to(config.device)
    content_criterion = ContentLoss(config.feature_model_extractor_node,
                                    config.feature_model_normalize_mean,
                                    config.feature_model_normalize_std).to(config.device)
    crossentorpy_criterion = nn.CrossEntropyLoss(ignore_index=255).to(config.device)

    return psnr_criterion, pixel_criterion, content_criterion, crossentorpy_criterion

def define_SSLoss():
    ss_criterion = RegionStyleLoss(reg_num=21).to(config.device)
    vgg_model_feat = PCPFeat(weight_path='./results/pre-trained/vgg19-dcbb9e9d.pth').to(config.device)
    return ss_criterion, vgg_model_feat

def define_optimizer(generator: nn.Module):
    g_optimizer = optim.Adam(generator.parameters(), config.model_lr, config.model_betas)

    return g_optimizer


def define_scheduler(g_optimizer: optim.Adam) :
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma)
    return g_scheduler



def train(generator: nn.Module,
          train_dataloader,
          vgg_model,
          pixel_criterion: nn.L1Loss,
          content_criterion: ContentLoss,
          ce_criterion: nn.CrossEntropyLoss,
          ss_criterion: RegionStyleLoss,
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
    ce_losses = AverageMeter('Semseg CrossEntorpy loss', ":6.6f")
    ss_losses = AverageMeter('Segmentation Aware Style loss', ":6.6f")

    psnres = AverageMeter("PSNR", ":4.2f")

    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, ce_losses,ss_losses, psnres],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator network model in training mode
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
        semseg_mask_onehot = samples["labels"]["semseg_onehot"]


        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        sr = generator(lr) # x_fake

        pixel_loss = config.pixel_weight * pixel_criterion(sr, hr)
        content_loss = config.content_weight * content_criterion(sr, hr)
        # Computational semseg task network loss
        sr_semseg_pred_logits = semsegTaskNetwork(config.semseg_network_weight, sr)
        semseg_loss_sr = config.semseg_weight * ce_criterion(sr_semseg_pred_logits, semseg_mask)

        sr_feat = vgg_model(sr)
        hr_feat = vgg_model(hr.detach())
        ss_loss_sr = ss_criterion(sr_feat, hr_feat, semseg_mask_onehot)

        # Calculate the generator total loss value
        g_loss = pixel_loss + content_loss + semseg_loss_sr + ss_loss_sr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        g_loss.backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        g_optimizer.step()
        # Finish training the generator model


        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        ce_losses.update(semseg_loss_sr.item(), lr.size(0))
        ss_losses.update(ss_loss_sr.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if idx % config.print_frequency == 0:
            iters = idx + epoch * batches + 1
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Semseg_Loss",semseg_loss_sr.item(), iters)
            writer.add_scalar("Train/SS_Loss",ss_loss_sr.item(), iters)
            progress.display(idx)





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
