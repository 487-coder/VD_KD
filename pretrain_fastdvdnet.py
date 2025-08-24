import torch
from torch.utils.data import DataLoader
from models import FastDVDnet,denoise_seq_fastdvdnet
from pathlib import Path
import torch.optim as optim
from dataset import ServerDataset
from dataset import partition_server_data
from options import args_parser
import torch.nn as nn
from utils import normalize_augment,orthogonal_conv_weights,batch_psnr,save_pretrain_model_checkpoint
from torch.utils.tensorboard import SummaryWriter
def pretrain_fastdvdnet(args,pretrain_data,val_dataset):
    pretrain_dataset = ServerDataset(pretrain_data,args.temp_patch_size,args.patch_size,args.max_number_patches,
                                    random_shuffle=True,temp_stride= 3)
    pretrain_loader = DataLoader(pretrain_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)
    ctrl_fr_idx = (args.temp_patch_size - 1) // 2
    writer = SummaryWriter("./logs/pretrain")

    device_ids = [0]
    torch.backends.cudnn.benchmark = True
    model = FastDVDnet()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_decay_epochs = [6, 8]  # 学习率衰减的epoch
    lr_decay_factor = 0.1  # 学习率衰减因子
    orthog_every_steps = 10  # 每10个step进行正交化
    stop_orthog_epoch = 8  # 第8轮后停止正交化
    for epoch in range(args.pretrain_epoch):
        current_lr = args.lr
        for decay_epoch in lr_decay_epochs:
            if epoch >= decay_epoch:
                current_lr *= lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        enable_orthog = epoch < stop_orthog_epoch
        step_count = 0
        model.train()
        for i,(seq,gt) in enumerate(pretrain_loader):

            optimizer.zero_grad()

            # 数据处理和前向传播（保持原样）
            img_train, gt_train = normalize_augment(seq, ctrl_fr_idx)
            device = next(model.parameters()).device
            img_train = img_train.to(device, non_blocking=True)
            gt_train = gt_train.to(device, non_blocking=True)

            N, _, H, W = img_train.size()
            stdn = torch.empty((N, 1, 1, 1), device=device).uniform_(
                args.noise_ival[0], args.noise_ival[1]
            )
            noise = torch.zeros_like(img_train)
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
            imgn_train = img_train + noise
            noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True)

            out_train = model(imgn_train, noise_map)
            loss = criterion(gt_train, out_train) / (N * 2)
            loss.backward()
            writer.add_scalar('loss', loss.item())
            optimizer.step()
            step_count += 1
            if step_count % orthog_every_steps == 0:
                if enable_orthog:
                    model.apply(orthogonal_conv_weights)
                    print(f"Applied orthogonalization at epoch {epoch}, step {step_count}")
        model.eval()
        validate_and_log(
            model=model,
            dataset_val=val_dataset,
            valnoisestd=args.val_noiseL,
            temp_psz=args.temp_patch_size,
            writer=writer,
            epoch=epoch,
            lr=current_lr,
        )
        save_pretrain_model_checkpoint(args,model,model_name = "fastdvdent")
    writer.close()


def validate_and_log(model, dataset_val, valnoisestd, temp_psz, writer,
                     epoch, lr):
    """Run validation and log PSNR + sample images."""

    psnr_val = 0
    for seq_val in dataset_val:
        # Add Gaussian noise
        noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
        seqn_val = (seq_val + noise).cuda()
        sigma_noise = torch.tensor([valnoisestd], device='cuda')
        out_val = denoise_seq_fastdvdnet(
            seq=seqn_val,
            noise_std=sigma_noise,
            temporal_window=temp_psz,
            model=model
        )
        psnr_val += batch_psnr(out_val.cpu(), seq_val.squeeze_(), data_range=1.0)

    psnr_val /= len(dataset_val)


    print(f"\n[epoch {epoch + 1}] PSNR_val: {psnr_val:.4f}")
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)




