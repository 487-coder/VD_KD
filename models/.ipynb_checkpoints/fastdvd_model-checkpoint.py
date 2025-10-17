"""
Definition of the FastDVDnet model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import batch_psnr

class CvBlock(nn.Module):
    """(Conv2d => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)"""
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(
                num_in_frames * (3 + 1),
                num_in_frames * self.interm_ch,
                kernel_size=3,
                padding=1,
                groups=num_in_frames,
                bias=False
            ),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_in_frames * self.interm_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Downscale + (Conv2d => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """(Conv2d => BN => ReLU) x 2 + Upscale"""
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    """Conv2d => BN => ReLU => Conv2d"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """Definition of the denoising block of FastDVDnet.

    Args:
        num_input_frames (int): number of input frames

    Inputs of forward():
        inX: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """
    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        # Input convolution block
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        # Estimation
        x = self.outc(x0 + x1)
        # Residual
        x = in1 - x
        return x


class FastDVDnet(nn.Module):
    """Definition of the FastDVDnet model.

    Inputs of forward():
        x: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """
    def __init__(self, num_input_frames=5):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        # Define models of each denoising stage
        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        # Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(
            x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames)
        )

        # First stage
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)

        # Second stage
        x = self.temp2(x20, x21, x22, noise_map)
        return x




def frame_denoise(model, noise_frame, sigma_map, context):
    #print("test")
    #print( noise_frame.shape, sigma_map.shape)
    _, _, h, w = noise_frame.shape
    pad_h = (4 - h % 4)
    pad_w = (4 - w % 4)
    if pad_h or pad_w:
        noise_frame = F.pad(noise_frame, (0, pad_w, 0, pad_h), mode="reflect")
        sigma_map = F.pad(sigma_map, (0, pad_w, 0, pad_h), mode="reflect")
    with context:
        denoise_frame = model(noise_frame, sigma_map)
        denoise_frame = torch.clamp(denoise_frame, 0.0, 1.0)
    if pad_h:
        denoise_frame = denoise_frame[:, :, :-pad_h, :]
    if pad_w:
        denoise_frame = denoise_frame[:, :, :, :-pad_w]

    return denoise_frame

'''
def denoise_seq_fastdvdnet(seq, noise_std, model, temporal_window=5, is_training=False):
    frame_num, c, h, w = seq.shape
    center = (temporal_window - 1) // 2
    denoise_frames = torch.empty_like(seq).to(seq.device)
    noise_map = noise_std.view(1, 1, 1, 1).expand(1, 1, h, w).to(seq.device)
    model.to(seq.device)
    context = torch.enable_grad() if is_training else torch.no_grad()
    frames = []
    with context:
        for denoise_index in range(frame_num):
            # load input frames
            if not frames:

                for index in range(temporal_window):
                    rel_index = abs(index - center)  # handle border conditions, reflect
                    frames.append(seq[rel_index])
            else:
                del frames[0]
                rel_index = min(denoise_index + center,
                                -denoise_index + 2 * (frame_num - 1) - center)  # handle border conditions
                frames.append(seq[rel_index])

            input_tensor = torch.stack(frames, dim=0).view(1, temporal_window * c, h, w).to(seq.device)
            x = frame_denoise(model, input_tensor, noise_map, context)

            denoise_frames[denoise_index] = x.squeeze(0)
        del frames
        del input_tensor
        torch.cuda.empty_cache()
        return denoise_frames
'''
def denoise_seq_fastdvdnet(seq, noise_std, model, temporal_window=5, is_training=False):
    """
    支持两种输入：
      - noise_std.shape == (1,)：整段序列使用统一噪声
      - noise_std.shape == (T, 1, 1, 1)：每帧使用独立噪声
    """
    frame_num, c, h, w = seq.shape
    center = (temporal_window - 1) // 2
    denoise_frames = torch.empty_like(seq).to(seq.device)
    model.to(seq.device)
    context = torch.enable_grad() if is_training else torch.no_grad()
    frames = []

    # 🔹 兼容判断逻辑
    if noise_std.dim() == 1 and noise_std.numel() == 1:
        # 统一噪声
        per_frame_noise = False
    elif noise_std.dim() == 4 and noise_std.shape[0] == frame_num:
        # 每帧噪声
        per_frame_noise = True
    else:
        raise ValueError(f"Unsupported noise_std shape: {noise_std.shape}")

    with context:
        for denoise_index in range(frame_num):
            # 构造时序窗口帧
            if not frames:
                for index in range(temporal_window):
                    rel_index = abs(index - center)
                    frames.append(seq[rel_index])
            else:
                del frames[0]
                rel_index = min(
                    denoise_index + center,
                    -denoise_index + 2 * (frame_num - 1) - center
                )
                frames.append(seq[rel_index])

            # 输入拼接
            input_tensor = torch.stack(frames, dim=0).view(1, temporal_window * c, h, w).to(seq.device)

            # 🔹 根据模式生成噪声图
            if per_frame_noise:
                sigma_t = noise_std[denoise_index]  # shape [1,1,1,1]
            else:
                sigma_t = noise_std.view(1, 1, 1, 1)

            noise_map = sigma_t.expand(1, 1, h, w).to(seq.device)

            # 前向推理
            x = frame_denoise(model, input_tensor, noise_map, context)
            denoise_frames[denoise_index] = x.squeeze(0)

        del frames
        torch.cuda.empty_cache()
        return denoise_frames



def validate_fastdvd_model(model, dataset_val, valnoisestd, temp_psz,idx, device,role = "client"):
    model.eval()
    total_psnr = 0.0
    cnt = 0

    # 将噪声标准差归一化到 [0,1]
    noise_std = valnoisestd
    #print(noise_std)

    with torch.no_grad():
        for batch_idx, seq in enumerate(dataset_val):
            if role == "global_test":
                print(f"{batch_idx} / {len(dataset_val)}")
                #print(id(model))
            # 如果 dataset_val 返回的是 [1, T, C, H, W]，去掉 batch 维
            if seq.dim() == 5 and seq.shape[0] == 1:
                seq = seq.squeeze(0)  # -> [T, C, H, W]

            seq = seq.to(device)  # GT 干净序列 [T, C, H, W]
            noise = torch.empty_like(seq).normal_(mean=0, std=noise_std)
            noisy_seq = torch.clamp(seq + noise, 0.0, 1.0)
            sigma_noise = torch.tensor([noise_std], device=device)
            out_val = denoise_seq_fastdvdnet(
                seq=noisy_seq,
                noise_std=sigma_noise,
                temporal_window=temp_psz,
                model=model
            )
            #if role == "global_test":
                #save_seq(out_val, "pretrain_test", "denoised")
            #if role == "client":
            #save_seq(seq, "gt_frames", "gt")
            #save_seq(noisy_seq,"noise_seq","seqn")

                #save_seq(out_val, f"client_denoised_frames_{idx}", "denoised")
            #if role == "distill":
                #save_seq(out_val, f"disitll_{idx}", "denoised")
            total_psnr += batch_psnr(out_val.cpu(), seq.squeeze().cpu(), data_range=1.0)
            cnt += 1
    avg_psnr = total_psnr / cnt
    return avg_psnr


import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import os


def save_seq(seq_tensor, save_dir, prefix="frame"):
    os.makedirs(save_dir, exist_ok=True)
    T = seq_tensor.shape[0]
    for i in range(T):
        frame = seq_tensor[i].cpu()
        frame = TF.to_pil_image(frame)
        frame.save(os.path.join(save_dir, f"{prefix}_{i:02d}.png"))

