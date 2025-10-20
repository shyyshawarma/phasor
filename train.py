from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, ssim

from utils import save_img, VGGPerceptualLoss, torchPSNR
from network1 import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, rgb_to_y
from model_with_eca import Restormer
import kornia


# ===============================
# Training settings
# ===============================
parser = argparse.ArgumentParser(description='Phaseformer')
parser.add_argument('--dataset', required=False, default='./uw_data/', help='dataset folder')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--finetune', default=False, help='to finetune')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=400, help='# of epochs at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=400, help='# of epochs to decay learning rate')
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy')
parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed')
opt = parser.parse_args()
print(opt)


# ===============================
# Custom Dataset
# ===============================
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dirs, reference_dirs, transform=None, transformH=None):
        self.input_dirs = input_dirs
        self.reference_dirs = reference_dirs
        self.transform = transform
        self.transformH = transformH
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        self.input_images = []
        self.reference_images = []
        for input_dir, reference_dir in zip(input_dirs, reference_dirs):
            input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
            reference_files = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(valid_extensions)])
           
            for file in input_files:
                if file in reference_files:
                    self.input_images.append(os.path.join(input_dir, file))
                    self.reference_images.append(os.path.join(reference_dir, file))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx]).convert("RGB")
        reference_image = Image.open(self.reference_images[idx]).convert("RGB")

        input_image = self.transform(input_image)
        reference_imageL = self.transform(reference_image)
        reference_imageH = self.transformH(reference_image)  

        return input_image, reference_imageL, reference_imageH, idx


# ===============================
# Data loading
# ===============================
def get_dataset(root_dirs):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformH = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_dirs = root_dirs['input']
    reference_dirs = root_dirs['reference']
    return CustomImageDataset(input_dirs, reference_dirs, transform, transformH)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# Initialization
# ===============================
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")

if opt.finetune:
    G_path = 'Provide the path to the checkpoint you wish to continue from'
    net_g = torch.load(G_path).to(device)
else:
    net_g = Restormer().to(device)

print(f'Trainable parameters: {count_parameters(net_g)}')    

print('===> Loading datasets')

root_dirs_train = {
    'input': ['./uw_data/train/a'],      
    'reference': ['./uw_data/train/b']
}
dataset_train = get_dataset(root_dirs_train)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
print(len(data_loader_train))

root_dirs_test = {
    'input': ['./uw_data/test/a'],
    'reference': ['./uw_data/test/b']
}
dataset_test = get_dataset(root_dirs_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)

print('===> Building models')


# ===============================
# Loss Functions
# ===============================
class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()
        kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x, xx):
        gradient_x = F.conv2d(x, self.weight_g, groups=3)
        gradient_xx = F.conv2d(xx, self.weight_g, groups=3)
        l = nn.L1Loss()
        return l(gradient_x, gradient_xx)


class WeightedLoss(nn.Module):
    def __init__(self, num_weights):
        super(WeightedLoss, self).__init__()
        self.num_weights = num_weights
        self.weights = nn.Parameter(torch.rand(1, num_weights))
        self.softmax_l = nn.Softmax(dim=1)

    def forward(self, *argv):
        loss = 0
        weights = self.softmax_l(self.weights)
        for idx, arg in enumerate(argv):
            loss += arg * weights[0, idx]
        return loss


# Initialize loss functions
Weighted_Loss4 = WeightedLoss(4).to(device)
Weighted_Loss2 = WeightedLoss(2).to(device)
Gradient_Loss = Gradient_Loss().to(device)
L_per = VGGPerceptualLoss().to(device)
MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
Charbonnier_loss = nn.SmoothL1Loss().to(device)


# ===============================
# Optimizer & Scheduler
# ===============================
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)


# ===============================
# Training Loop
# ===============================
output_dir_train = './images_train'
os.makedirs(output_dir_train, exist_ok=True)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    net_g.train()
    for iteration, batch in enumerate(data_loader_train, 1):
        rgb, tarL, tarH, indx = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
        fake_b, fake_b1 = net_g(rgb)
        optimizer_g.zero_grad()

        loss_g_L = Weighted_Loss4(Charbonnier_loss(tarL, fake_b), L_per(fake_b, tarL),
                                  Gradient_Loss(fake_b, tarL), MS_SSIM_loss(fake_b, tarL))
        loss_g_H = Weighted_Loss4(Charbonnier_loss(tarH, fake_b1), L_per(fake_b1, tarH),
                                  Gradient_Loss(fake_b1, tarH), MS_SSIM_loss(fake_b1, tarH))
        loss_g = Weighted_Loss2(loss_g_L, loss_g_H)
        
        loss_g.backward()
        optimizer_g.step()

        if iteration % 100 == 0:
            out_image = torch.cat((rgb, fake_b, tarL), 3)
            save_img(out_image[0].detach().cpu(), f'{output_dir_train}/{iteration}.png')
            print(f"===> Epoch[{epoch}]({iteration}/{len(data_loader_train)}): Loss_G: {loss_g.item()}")

    update_learning_rate(net_g_scheduler, optimizer_g)

    # ===============================
    # Evaluation Loop
    # ===============================
    output_dir = './images_test'
    os.makedirs(output_dir, exist_ok=True)

    net_g.eval()
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for test_iter, batch in enumerate(data_loader_test, 1):
            rgb_input, target, targetH, idx = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            prediction = net_g(rgb_input)[0]

            # Compute PSNR
            psnr_val = torchPSNR(prediction.clamp(0, 1), target.clamp(0, 1))
            total_psnr += psnr_val
            count += 1

            out = torch.cat((rgb_input, prediction, target), 3)
            filename = f'{output_dir}/{idx[0]}.png'
            save_img(out[0].detach().cpu(), filename)

    avg_psnr = total_psnr / count
    print(f"===> Epoch {epoch} Average PSNR: {avg_psnr:.4f} dB")

    # ===============================
    # Save Model Checkpoint
    # ===============================
    checkpoint_dir = os.path.join("checkpoint", opt.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    net_g_model_out_path = os.path.join(checkpoint_dir, f"netG_model_epoch_{epoch}_psnr_{avg_psnr:.4f}.pth")
    torch.save(net_g, net_g_model_out_path)
    print(f"Checkpoint saved at {net_g_model_out_path}")
