from pathlib import Path
import random

import cv2
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose

image_types = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

def get_image_names(seq_dir, pattern=None):
    seq_path = Path(seq_dir)
    files = []

    for image_type in image_types:
        files.extend(seq_path.glob(image_type))

    if pattern is not None:
        files = [file for file in files if pattern in file.name]

    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.name))))
    return [str(file) for file in files]

class Normalize_for_RVRT(nn.Module):
    def forward(self, x):
        #print("cnm", x.shape)
        x = x.float() / 255.0
        # 保持 NFCHW 格式，不再合并 F 和 C 维度
        return x


class Augment_for_RVRT(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_names = [
            'do_nothing', 'flipud', 'rot90', 'rot90_flipud',
            'rot180', 'rot180_flipud', 'rot270', 'rot270_flipud', 'add_noise'
        ]
        self.weights = [32, 12, 12, 12, 12, 12, 12, 12, 12]

    def augment(self, op_name, img):
        # 输入 img 格式: [C, H, W]
        match op_name:
            case 'do_nothing':
                return img
            case 'flipud':
                return torch.flip(img, dims=[1])  # 沿 H 维度翻转
            case 'rot90':
                return torch.rot90(img, k=1, dims=[1, 2])  # 在 H, W 平面旋转
            case 'rot90_flipud':
                return torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[1])
            case 'rot180':
                return torch.rot90(img, k=2, dims=[1, 2])
            case 'rot180_flipud':
                return torch.flip(torch.rot90(img, k=2, dims=[1, 2]), dims=[1])
            case 'rot270':
                return torch.rot90(img, k=3, dims=[1, 2])
            case 'rot270_flipud':
                return torch.flip(torch.rot90(img, k=3, dims=[1, 2]), dims=[1])
            case 'add_noise':
                noise = torch.randn(1, 1, 1, device=img.device) * (5.0 / 255.0)
                return torch.clamp(img + noise.expand_as(img), 0.0, 1.0)
            case _:
                raise ValueError(f"Unsupported op name: {op_name}")

    def forward(self, x):
        # 输入格式: N, F, C, H, W
        N, F, C, H, W = x.shape
        out = torch.empty_like(x)
        op_name = random.choices(self.op_names, weights=self.weights, k=1)[0]
        #print(op_name, "new")

        for n in range(N):
            for f in range(F):
                img = x[n, f]  # [C, H, W]
                out[n, f] = self.augment(op_name, img)
        return out


def normalize_augment_for_RVRT(data_input, ctrl_fr_idx):
    video_transform = Compose([
        Normalize_for_RVRT(),
        Augment_for_RVRT(),
    ])
    img_train = video_transform(data_input)
    gt_train = img_train
    return img_train, gt_train

class Normalize(nn.Module):
    def forward(self, x):
        #print("cnm",x.shape)
        if x.max() > 1:
            x = x.float() / 255.0
        else:
            x = x.float()
        n, f, c, h, w = x.shape
        return x.view(n, f * c, h, w)


class Augment(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_names = [
            'do_nothing', 'flipud', 'rot90', 'rot90_flipud',
            'rot180', 'rot180_flipud', 'rot270', 'rot270_flipud', 'add_noise'
        ]
        self.weights = [32, 12, 12, 12, 12, 12, 12, 12, 12]

    def augment(self, op_name, img):

        match op_name:
            case 'do_nothing':
                return img
            case 'flipud':
                return torch.flip(img, dims=[1])
            case 'rot90':
                return torch.rot90(img, k=1, dims=[1, 2])
            case 'rot90_flipud':
                return torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[1])
            case 'rot180':
                return torch.rot90(img, k=2, dims=[1, 2])
            case 'rot180_flipud':
                return torch.flip(torch.rot90(img, k=2, dims=[1, 2]), dims=[1])
            case 'rot270':
                return torch.rot90(img, k=3, dims=[1, 2])
            case 'rot270_flipud':
                return torch.flip(torch.rot90(img, k=3, dims=[1, 2]), dims=[1])
            case 'add_noise':
                noise = torch.randn(1, 1, 1, device=img.device) * (5.0 / 255.0)
                return torch.clamp(img + noise.expand_as(img), 0.0, 1.0)
            case _:
                raise ValueError(f"Unsupported op name: {op_name}")

    def forward(self, x):
        N, FC, H, W = x.shape
        F = FC // 3
        out = torch.empty_like(x,requires_grad=x.requires_grad,device=x.device)
        op_name = random.choices(self.op_names, weights=self.weights, k=1)[0]
        #print(op_name, "new")
        for n in range(N):
            for f in range(F):
                img = x[n, f * 3:f * 3 + 3]  # [3, H, W]
                out[n, f * 3:f * 3 + 3] = self.augment(op_name, img)
        return out
def normalize_augment(data_input, ctrl_fr_idx):
    video_transform = Compose([
        Normalize(),
        Augment(),
    ])
    img_train = video_transform(data_input)
    gt_train = img_train[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
    return img_train, gt_train

def orthogonal_conv_weights(layer):
    if not isinstance(layer, nn.Conv2d):
        return
    weight_tmp = layer.weight.data.clone()
    c_out, c_in, kh, kw = weight_tmp.shape
    dtype = weight_tmp.dtype
    weight_flat = weight_tmp.permute(2, 3, 1, 0).contiguous().view(-1, c_out)
    try:
        U, _, V = torch.linalg.svd(weight_flat, full_matrices=False)
        weight_ortho = torch.matmul(U, V)

        weight_new = weight_ortho.view(kh, kw, c_in, c_out).permute(3, 2, 0, 1).contiguous()
        layer.weight.data.copy_(weight_new.to(dtype))
    except RuntimeError as e:
        print(f"SVD failed for {layer.__class__.__name__}: {e}")



def batch_psnr(images, images_clean, data_range):
    images_cpu = images.data.cpu().numpy().astype(np.float32)
    images_clean = images_clean.data.cpu().numpy().astype(np.float32)
    psnr = 0.0
    for index in range(images_cpu.shape[0]):
        psnr += peak_signal_noise_ratio(images_clean[index, :, :, :], images_cpu[index, :, :, :],
                                        data_range=data_range)
    return psnr / images_cpu.shape[0]


def get_image_names(seq_dir, pattern=None):
    seq_path = Path(seq_dir)
    files = []

    for image_type in image_types:
        files.extend(seq_path.glob(image_type))

    if pattern is not None:
        files = [file for file in files if pattern in file.name]

    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.name))))
    return [str(file) for file in files]


def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    if gray_mode:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)

    if expand_axis0:
        img = np.expand_dims(img, axis=0)

    expanded_h, expanded_w = False, False
    h, w = img.shape[-2], img.shape[-1]

    if expand_if_needed:
        if h % 2 == 1:
            expanded_h = True
            last_row = img[..., -1:, :]
            img = np.concatenate((img, last_row), axis=-2)

        if w % 2 == 1:
            expanded_w = True
            last_col = img[..., -1:]  # slice last col
            img = np.concatenate((img, last_col), axis=-1)

    if normalize_data:
        img = img.astype(np.float32) / 255.0

    return img, expanded_h, expanded_w

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
    file_paths = get_image_names(seq_dir)
    file_paths = file_paths[:max_num_fr]

    print(f"Open sequence in folder: {seq_dir} ({len(file_paths)} frames)")

    seq_list = []
    expanded_h, expanded_w = False, False

    for fpath in file_paths:
        img, h_exp, w_exp = open_image(
            fpath,
            gray_mode=gray_mode,
            expand_if_needed=expand_if_needed,
            expand_axis0=False
        )
        seq_list.append(img)
        expanded_h |= h_exp
        expanded_w |= w_exp

    # Stack to [T, C, H, W]
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w


def save_pretrain_model_checkpoint(args, model, model_name):
    log_dir = Path(args.pretrain_model)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), log_dir / '{}.pth'.format(model_name))

def save_model_checkpoint(model, config, optimizer, train_pars, epoch, role="global", client_id=None):
    """
    Save model and optimizer state dicts.

    Args:
        model: torch.nn.Module
        config: dict, contains training configuration including 'log_dir' and 'save_every_epochs'
        optimizer: torch.optim.Optimizer
        train_pars: dict, training info like epoch, loss, etc.
        epoch: int
        role: 'client' or 'global'
        client_id: int, required if role is 'client'
    """

    # === Set subdirectory path ===
    if role == "client":
        assert client_id is not None, "client_id must be provided when role is 'client'"
        log_dir = Path(config['log_dir']) / f"client_{client_id}"
    else:
        log_dir = Path(config['log_dir']) / f"{role}"

    log_dir.mkdir(parents=True, exist_ok=True)

    # === Save model only for inference ===
    torch.save(model.state_dict(), log_dir / 'net.pth')

    # === Save full checkpoint for resume training ===
    save_dict = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'training_params': train_pars,
        'args': config
    }

    torch.save(save_dict, log_dir / 'ckpt.pth')

    if epoch % config['save_every_epochs'] == 0:
        epoch_ckpt_path = log_dir / f'ckpt_e{epoch + 1}.pth'
        torch.save(save_dict, epoch_ckpt_path)

    del save_dict
