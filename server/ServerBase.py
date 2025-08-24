import copy
from typing import List

from utils import batch_psnr,normalize_augment
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_model_checkpoint
from models import denoise_seq_fastdvdnet
import models
from pathlib import Path
from pretrain_fastdvdnet import pretrain_fastdvdnet
import torch
from client import Client
from dataset import ServerDataset
import numpy as np
class Server(object):
    def __init__(self,args,model_names,pretrain_data,val_dataset,distill_data,
                 client_dataloaders,local_test_dataloader,
                 logger,mode,device):
        super(Server, self).__init__()
        self.args = args
        # global model
        self.model_names = model_names
        self.model_dict = {model_name: models.get_model(model_name)() for model_name in self.model_names}
        # global distill datasets
        self.pretrain_data = pretrain_data
        self.distill_criterion = nn.MSELoss()
        self.distill_optimizers = {model_name: torch.optim.Adam(
            self.model_dict[model_name].parameters(), lr=1e-4)
            for model_name in self.model_names}
        self.distill_data = ServerDataset(distill_data, args.temp_patch_size, args.patch_size,
                                          random_shuffle=True, temp_stride=3)
        self.distill_loader = DataLoader(self.distill_data, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        # 其他
        self.device = device
        self.mode = mode
        self.logger = logger
        #验证
        self.val_dataset = val_dataset
        self.global_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.ctrl_fr_idx = (self.args.temp_psz - 1) // 2
        # clients
        self.clients: List[Client] = []  # 添加类型
        self.client_dataloader = client_dataloaders
        self.local_test_dataloader = local_test_dataloader


    def create_clients(self):
        model_config = dict(zip(self.args.model_names, self.args.model_counts))
        idx = 0
        for model_name, count in model_config.items():
            for _ in range(count):
                client = Client(self.args, copy.deepcopy(self.model_dict[model_name]),
                                self.client_dataloader[idx],self.local_test_dataloader[idx],
                                logger=self.logger,device=self.device,model_name=model_name, idx=idx)
                self.clients.append(client)
                idx += 1
    def get_pretrain_model(self):
        pretrain_func_map = {"fastdvdnet": pretrain_fastdvdnet}
        for model_name, model in self.model_dict.items():
            exist_pretrain_model = (Path(f"{self.args.pretrain_model}") / f"{model_name}.pth").exists()
            if not exist_pretrain_model:
                if model_name in pretrain_func_map:
                    pretrain_func_map[model_name](self.args, self.pretrain_data, self.val_dataset)
            else:
                pretrain_state_dict = torch.load(Path(f"{self.args.pretrain_model}") / f"{model_name}.pth")
                loaded_layers = [key for key in pretrain_state_dict if key in model.state_dict()]
                model.load_state_dict(pretrain_state_dict, strict=False)
                # 向client初始化模型

    '''def aggregate_same_models(self):
        def average_weights(w):
            w_avg = copy.deepcopy(w[0])
            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], len(w))
            return w_avg
        for model_name in self.model_names:
            client_weights = [client.model.state_dict() for client in self.clients if client.model_name == model_name]
            self.model_dict[model_name].load_state_dict(average_weights(client_weights))
            '''

    def aggregate_same_models(self):
        """Aggregate models with the same architecture across all clients in self.clients."""

        def average_weights(w):
            w_avg = copy.deepcopy(w[0])
            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], len(w))
            return w_avg

        for model_name in self.model_names:
            client_weights = []
            for client in self.clients:  # ✅ 不需要传参，直接用 self.clients
                if client.model_name == model_name:
                    # 确保所有参数搬到 self.device
                    state_dict = {k: v.to(self.device) for k, v in client.model.state_dict().items()}
                    client_weights.append(state_dict)
            if client_weights:
                self.model_dict[model_name].load_state_dict(average_weights(client_weights))
    def distill(self,epochs =1):
        for model_name in self.model_names:
            model = self.model_dict[model_name]
            optimizer = self.distill_optimizers[model_name]
            model.to(self.device)
            model.train()
            criterion = self.distill_criterion.to(self.device)
            epoch_loss = []
            for teacher in self.clients:
                teacher.model.eval()
            for epoch in range(epochs):
                batch_loss = []
                step_count = 0
                for batch_idx, (seq,gt) in enumerate(self.distill_loader):
                    model.train()
                    optimizer.zero_grad()
                    img_train, gt_train = normalize_augment(seq, self.ctrl_fr_idx)
                    img_train = img_train.to(self.device, non_blocking=True)
                    gt_train = gt_train.to(self.device, non_blocking=True)
                    N, _, H, W = img_train.size()
                    stdn = torch.empty((N, 1, 1, 1), device=self.device).uniform_(
                        self.args.noise_level[0], self.args.noise_level[1]
                    )

                    noise = torch.normal(mean=0.0, std=stdn.expand_as(img_train))
                    imgn_train = img_train + noise
                    imgn_train = torch.clamp(imgn_train, 0.0, 1.0)
                    noise_map = stdn.expand((N, 1, H, W)).to(self.device)
                    student_outputs = model(imgn_train, noise_map)
                    soft_target = []
                    if self.mode == "mean output":
                        teacher_outputs = []
                        with torch.no_grad():
                            for teacher in self.clients:
                                teacher_denoised = teacher.model(imgn_train, noise_map)
                                teacher_outputs.append(teacher_denoised)
                        if teacher_outputs:
                            soft_target = torch.stack(teacher_outputs).mean(dim=0)
                        else:
                            continue

                    loss = criterion(student_outputs, soft_target)
                    aux_loss_weight = 0.2
                    aux_loss = nn.MSELoss()(student_outputs, gt_train)
                    total_loss_batch = (1 - aux_loss_weight) * loss + aux_loss_weight * aux_loss
                    total_loss_batch = total_loss_batch / (N*2)
                    total_loss_batch.backward()
                    optimizer.step()
                    step_count += 1
                    if step_count % 5 ==0:
                        print(f'Epoch: [{epoch}/{epochs}], Step: [{step_count}],model:[{model_name}], loss:[{total_loss_batch.item():.6f}]')
                    batch_loss.append(total_loss_batch.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
    def train(self):
        train_loss = []
        for epoch in tqdm(range(self.args.num_epochs)):
            current_lr = self.args.lr * (0.1 ** (epoch // 3))
            global_psnr = 0
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch + 1} |\n')
            m = max(int(self.args.sampling_rate * self.args.client_numbers), 1)
            idxs_users = np.random.choice(range(self.args.client_numbers), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.clients[idx].load_model(self.model_dict[self.clients[idx].model_name].state_dict())
                w, loss = self.clients[idx].update_weights(global_round=epoch,lr=current_lr)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                local_psnr = self.clients[idx].test_psnr()
                global_psnr += local_psnr
                self.logger.add_scalar(f'Client_{idx}_{self.clients[idx].model_name}/Loss', loss, epoch)
                self.logger.add_scalar(f'Client_{idx}_{self.clients[idx].model_name}/PSNR', local_psnr, epoch)
            # update global weights
            self.aggregate_same_models()
            self.distill()
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average test psnr:', global_psnr / m)
            self.logger.add_scalar('Global/Average_Loss', loss_avg, epoch)
            self.logger.add_scalar('Global/Average_PSNR', global_psnr / m, epoch)
            if (epoch + 1) % 3 == 0 or epoch == 0:
                self.global_test_psnr(epoch)
            for model_name in self.model_names:
                save_model_checkpoint(
                    model=self.model_dict[model_name],
                    config={
                        'log_dir': self.args.save_dir,
                        'save_every_epochs': 5,
                        'model_name': model_name  # 添加模型名称以区分不同模型
                    },
                    optimizer=self.distill_optimizers.get(model_name, None),  # 使用对应的optimizer
                    train_pars={
                        'epoch_loss': loss_avg,
                        'epoch': epoch,
                        'model_name': model_name  # 在训练参数中也加上模型名称
                    },
                    epoch=epoch,
                    role=f'global_{model_name}'  # 角色名称中包含模型名称
                )

            print('Training is completed.')
    def global_test_psnr(self,epoch):
        for model_name in self.model_names:
            model = self.model_dict[model_name]
            model.eval()
            total_psnr = 0.0
            total_noisy_psnr = 0.0
            cnt = 0

            for idx, (seq,) in enumerate(self.global_test_loader):
                seq = seq.to(self.device)  # [1, T, C, H, W]
                if seq.dim() == 5 and seq.shape[0] == 1:
                    seq = seq.squeeze(0)  # [T, C, H, W]

                noise = torch.empty_like(seq).normal_(mean=0, std=self.args.test_noise).to(self.device)
                noisy_seq = seq + noise
                noisy_seq = torch.clamp(noisy_seq, 0.0, 1.0)

                noise_map = torch.tensor([self.args.test_noise], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    denoised_seq = denoise_seq_fastdvdnet(
                        seq=noisy_seq,
                        noise_std=noise_map,
                        temporal_window=self.args.temp_psz,
                        model=model
                    )

                # 中心帧比较
                gt = seq[self.ctrl_fr_idx].unsqueeze(0)
                pred = denoised_seq[self.ctrl_fr_idx].unsqueeze(0)
                noisy_center = noisy_seq[self.ctrl_fr_idx].unsqueeze(0)
                #print("GT min/max:", gt.min().item(), gt.max().item())
                #print("Pred min/max:", pred.min().item(), pred.max().item())

                psnr_clean = batch_psnr(pred, gt, data_range=1.0)
                psnr_noisy = batch_psnr(noisy_center, gt, data_range=1.0)

                total_psnr += psnr_clean
                total_noisy_psnr += psnr_noisy
                cnt += 1

                print(f"{model_name} [{idx}] PSNR_noisy: {psnr_noisy:.2f} dB | PSNR_denoised: {psnr_clean:.2f} dB")

            avg_psnr = total_psnr / cnt
            avg_psnr_noisy = total_noisy_psnr / cnt
            self.logger.add_scalar(f'Global_{model_name}/Test_PSNR', avg_psnr, epoch)
            self.logger.add_scalar(f'Global_{model_name }/Test_PSNR_Noisy', avg_psnr_noisy, epoch)

            print(f"\n[Global Test PSNR] {model_name} Clean: {avg_psnr:.4f} dB | Noisy: {avg_psnr_noisy:.4f} dB\n")




    def Save_CheckPoint(self, save_path,model_name):
        torch.save(self.model_dict[model_name].state_dict(), save_path)
















