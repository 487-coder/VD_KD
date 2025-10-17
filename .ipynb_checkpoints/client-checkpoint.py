import copy
from torch.utils.data import DataLoader
import torch
from dataset import LocalDataset,ServerDataset
from torch import nn
from models.SwinIR_model import CharbonnierLoss,validate_SwinIR_model
from models.fastdvd_model import validate_fastdvd_model
from models.RVRT_model import CharbonnierLoss,validate_RVRT_model
from utils import normalize_augment,orthogonal_conv_weights,save_model_checkpoint,batch_psnr,normalize_augment_for_RVRT

class Client(object):
    def __init__(self,args,model,train_dataset_path,testloader,logger,device,model_name,idx):
        '''client = Client(self.args, copy.deepcopy(self.model_dict[model_name]),
                                    self.client_dataloader[idx],self.local_test_dataloader[idx],
                                    logger=self.logger,device=self.device,model_name=model_name, idx=idx)'''
        self.args = args
        self.model = copy.deepcopy(model)
        self.train_dataset_path = train_dataset_path
        self.testloader = testloader
        self.device = device
        self.logger = logger
        self.model_name = model_name
        self.idx = idx
        if self.model_name == 'fastdvdnet':
            self.ctrl_fr_idx = (self.args.temp_psz - 1) // 2
            self.trainset = ServerDataset(self.train_dataset_path,sequence_length= 5,crop_size=96,
                                         epoch_size = self.args.max_number_patches,
                                         random_shuffle= True, temp_stride= 3)
            
            self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size,
                                           shuffle=False, num_workers=4)
            self.criterion = nn.MSELoss(reduction='sum').to(device)
        elif self.model_name == 'SwinIR':
            self.trainset = ServerDataset(self.train_dataset_path,sequence_length=1,
                                         crop_size= 128,epoch_size= -1,random_shuffle=True,
                                         temp_stride=-1)
            self.train_loader = DataLoader(self.trainset,batch_size= 2,
                                           shuffle=False, num_workers= 4)
            self.ctrl_fr_idx = 0
            self.criterion = CharbonnierLoss(1e-9).to(device)
        elif self.model_name == 'RVRT':
            self.trainset = ServerDataset(self.train_dataset_path,sequence_length= 16,
                                         crop_size=256,epoch_size=-1,random_shuffle=True,
                                         temp_stride=1)
            self.train_loader = DataLoader(self.trainset,batch_size=8,
                                           shuffle=False, num_workers=4)
            self.ctrl_fr_idx = 0
            self.criterion = CharbonnierLoss(1e-9).to(device)
    def update_weights(self,global_round,lr):
        if self.model_name == 'fastdvdnet':
            self.update_fastdvd_weights(global_round, lr)
        elif self.model_name == "SwinIR":
            self.update_SwinIR_weights(global_round,lr)
        elif self.model_name == "RVRT":
            self.update_RVRT_weights(global_round,lr)
    def update_RVRT_weights(self,global_round,lr):
        self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = torch.optim.Adam(optim_params, lr=4e-4,
                                     betas=(0.9, 0.99),
                                     weight_decay=0)
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            step_count = 0
            for i, (seq, gt) in enumerate(self.train_loader):
                self.model.train()
                optimizer.zero_grad()
                img_train, gt_train = normalize_augment(seq, 0)

                img_train = img_train.to(self.device, non_blocking=True)
                gt_train = gt_train.to(self.device, non_blocking=True)
                N, T, C, H, W = img_train.size()
                stdn = torch.empty((N, 1, 1, 1, 1), device=self.device).uniform_(
                    self.args.noise_level[0], self.args.noise_level[1]
                )
                noise = torch.zeros_like(img_train)
                noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
                imgn_train = img_train + noise
                imgn_train = torch.clamp(imgn_train, 0.0, 1.0)
                noise_channel = stdn.expand(N, T, 1, H, W)

                # 拼接噪声通道到图像，形成 (N, T, C+1, H, W)
                imgn_train_with_noise = torch.cat([imgn_train, noise_channel], dim=2)

                out_train = self.model(imgn_train_with_noise)
                loss = criterion(out_train, gt_train)
                loss.backward()
                optimizer.step()
                step_count += 1
                batch_loss.append(loss.item())
                if i % 10 == 0:
                    print(
                        f"| Global Round: {global_round} | Client: {self.idx} ,{self.model_name}| Local Epoch: {epoch} | "
                        f"[{i * len(img_train)}/{len(self.train_loader)} ({100. * i / len(self.train_loader):.0f}%)] "
                        f"\tLoss: {loss.item():.6f}")
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            save_model_checkpoint(
                model=self.model,
                config={
                    'log_dir': self.args.save_dir,
                    'save_every_epochs': 5  # 每轮都保存，或自定义
                },
                optimizer=optimizer,
                train_pars={
                    'epoch_losses': epoch_loss[-1],
                    'epoch': global_round
                },
                epoch=global_round,
                role="client",
                client_id=self.idx
            )

            return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def update_SwinIR_weights(self,global_round,lr):
        self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = torch.optim.Adam(optim_params, lr=2e-4,
                         betas=(0.9, 0.999),
                         weight_decay=0)
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            step_count = 0
            for i, (seq, gt) in enumerate(self.train_loader):
                self.model.train()
                optimizer.zero_grad()
                img_train, gt_train = normalize_augment_for_RVRT(seq, 0)
                if img_train.dim() == 5:
                    img_train = img_train.squeeze(1)  # Remove time dimension
                if gt_train.dim() == 5:
                    gt_train = gt_train.squeeze(1)
                img_train = img_train.to(self.device, non_blocking=True)
                gt_train = gt_train.to(self.device, non_blocking=True)
                N, _, H, W = img_train.size()
                stdn = torch.empty((N, 1, 1, 1), device=self.device).uniform_(
                    self.args.noise_level[0], self.args.noise_level[1]
                )
                noise = torch.zeros_like(img_train)
                noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
                imgn_train = img_train + noise
                imgn_train = torch.clamp(imgn_train, 0.0, 1.0)
                out_train = self.model(imgn_train)
                loss = 1.0 * criterion(out_train, gt_train)
                loss.backward()
                optimizer.step()
                step_count += 1
                batch_loss.append(loss.item())
                if i % 10 == 0:
                    print(
                        f"| Global Round: {global_round} | Client: {self.idx} ,{self.model_name}| Local Epoch: {epoch} | "
                        f"[{i * len(img_train)}/{len(self.train_loader)} ({100. * i / len(self.train_loader):.0f}%)] "
                        f"\tLoss: {loss.item():.6f}")
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        save_model_checkpoint(
            model=self.model,
            config={
                'log_dir': self.args.save_dir,
                'save_every_epochs': 5  # 每轮都保存，或自定义
            },
            optimizer=optimizer,
            train_pars={
                'epoch_losses': epoch_loss[-1],
                'epoch': global_round
            },
            epoch=global_round,
            role="client",
            client_id=self.idx
        )

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)






    def update_fastdvd_weights(self,global_round,lr):
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = self.criterion.to(self.device)

        epoch_loss = []
        orthog_every_steps = 10  # 每10个step进行正交化
        stop_orthog_epoch = 8  # 第8轮后停止正交化
        for epoch in range(self.args.local_ep):
            batch_loss = []
            enable_orthog = epoch < stop_orthog_epoch
            step_count = 0
            for batch_idx, (seq, gt) in enumerate(self.train_loader):
                self.model.train()
                optimizer.zero_grad()
                img_train, gt_train = normalize_augment(seq, self.ctrl_fr_idx)
                img_train = img_train.to(self.device, non_blocking=True)
                gt_train = gt_train.to(self.device, non_blocking=True)
                N, _, H, W = img_train.size()
                stdn = torch.empty((N, 1, 1, 1), device=self.device).uniform_(
                    self.args.noise_level[0], self.args.noise_level[1]
                )
                noise = torch.normal(mean=0.0, std=stdn.expand_as(img_train))
                # print("testValue")
                # print(img_train.max(), img_train.min())
                imgn_train = img_train + noise
                imgn_train = torch.clamp(imgn_train, 0.0, 1.0)
                noise_map = stdn.expand((N, 1, H, W)).to(self.device)
                out_train = self.model(imgn_train, noise_map)
                #save_seq(out_train, f"client_train_output{self.idx}", "step_count", batch_idx=step_count)
                loss = criterion(out_train, gt_train) / (N * 2)
                loss.backward()
                optimizer.step()
                step_count += 1
                if step_count % orthog_every_steps == 0:
                    if enable_orthog:
                        self.model.apply(orthogonal_conv_weights)
                        print(f"Applied orthogonalization at epoch {epoch}, step {step_count}")
                if batch_idx % 10 == 0:
                    print(f"| Global Round: {global_round} | Client: {self.idx} ,{self.model_name}| Local Epoch: {epoch} | "
                          f"[{batch_idx * len(img_train)}/{len(self.train_loader)} ({100. * batch_idx / len(self.train_loader):.0f}%)] "
                          f"\tLoss: {loss.item():.6f}")
                    self.logger.add_scalar(f'client {self.idx},{self.model_name}loss', loss.item())

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        save_model_checkpoint(
            model=self.model,
            config={
                'log_dir': self.args.save_dir,
                'save_every_epochs': 5  # 每轮都保存，或自定义
            },
            optimizer=optimizer,
            train_pars={
                'epoch_losses': epoch_loss[-1],
                'epoch': global_round
            },
            epoch=global_round,
            role="client",
            client_id=self.idx
        )

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    '''
    def test_psnr(self):
        self.model.eval()
        if self.model_name == "fastdvdnet":
            avg_psnr = validate_fastdvd_model(
                model=self.model,
                dataset_val=self.testloader,
                valnoisestd=self.args.test_noise,
                temp_psz=self.args.temp_psz,
                device=self.device
            )
        elif self.model_name == "SwinIR":
            avg_psnr = validate_SwinIR_model(
                args=self.args,
                model=self.model,
                dataset_val=self.testloader,
                valnoisestd=self.args.test_noise,
                device=self.device
            )
        print(f"[Client {self.idx},{self.model_name}] PSNR : {avg_psnr:.2f}")
        return avg_psnr

    '''
    


# 4. 在client.py中的使用示例
    def test_psnr(self,idx,role = "client"):
        """
        替换原来的test_psnr函数，用于调试
        """
       
        
        if self.model_name == "fastdvdnet":
            # 设置CUDNN
         
            #print(f"Model {self.idx} hash after test_psnr: {hash(str(list(self.model.parameters())))}")
            # 使用调试版本的验证函数
            avg_psnr = validate_fastdvd_model(
                model=self.model,
                dataset_val=self.testloader,
                valnoisestd=self.args.test_noise,
                temp_psz=self.args.temp_psz,
                device=self.device,
                role = role,
                idx = idx
        
            )
            
        elif self.model_name == "SwinIR":
            avg_psnr = validate_SwinIR_model(
                args=self.args,
                model=self.model,
                dataset_val=self.testloader,
                valnoisestd=self.args.test_noise,
                device=self.device,
                role = role
            )
        elif self.model_name == "RVRT":
            avg_psnr = validate_RVRT_model(
                args=self.args,
                model=self.model,
                dataset_val=self.testloader,
                valnoisestd=self.args.test_noise,
                device=self.device,
                role=role
            )
        
        print(f"[Client {self.idx},{self.model_name}] PSNR : {avg_psnr:.2f}")
        return avg_psnr



    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)

import os
import torchvision.transforms.functional as TF

def save_seq(seq_tensor, save_dir, prefix="frame", batch_idx=0):
    """
    保存序列或batch的图像:
    - [T, C, H, W]       → 保存 T 张
    - [B, C, H, W]       → 保存 B 张
    - [B, T, C, H, W]    → 保存 B*T 张
    """
    os.makedirs(save_dir, exist_ok=True)

    # 情况一: batch of sequences
    if seq_tensor.dim() == 5:   # [B, T, C, H, W]
        B, T, C, H, W = seq_tensor.shape
        for b in range(B):
            for t in range(T):
                frame = TF.to_pil_image(seq_tensor[b, t].cpu())
                frame.save(os.path.join(save_dir, f"{prefix}_b{batch_idx+b:03d}_t{t:02d}.png"))

    # 情况二: sequence only
    elif seq_tensor.dim() == 4: # [T, C, H, W] or [B, C, H, W]
        T = seq_tensor.shape[0]
        for i in range(T):
            frame = TF.to_pil_image(seq_tensor[i].cpu())
            frame.save(os.path.join(save_dir, f"{prefix}_b{batch_idx:03d}_{i:02d}.png"))

    # 情况三: single image
    elif seq_tensor.dim() == 3: # [C, H, W]
        frame = TF.to_pil_image(seq_tensor.cpu())
        frame.save(os.path.join(save_dir, f"{prefix}_b{batch_idx:03d}.png"))
    else:
        raise ValueError(f"Unsupported tensor shape {seq_tensor.shape}")



