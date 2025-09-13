import copy
from typing import List

from utils import normalize_augment
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_model_checkpoint
from models.SwinIR_model import SwinIR,validate_SwinIR_model
from models.fastdvd_model import FastDVDnet,validate_fastdvd_model
from pathlib import Path
from pretrain_fastdvdnet import pretrain_fastdvdnet
from pretrain_SwinIR import pretrain_SwinIR
import torch
from client import Client
from dataset import ServerDataset
import numpy as np
class Server(object):
    def __init__(self,args,model_names,pretrain_data,val_dataset,distill_data,
                 client_dataset_path,local_test_dataloader,
                 logger,mode,device):
        super(Server, self).__init__()
        self.args = args

        self.model_names = model_names
        self.model_dict = {"fastdvdnet":FastDVDnet,
                           "SwinIR": SwinIR}

        self.pretrain_data = pretrain_data
        #
        self.distill_criterion = nn.MSELoss()
        self.distill_optimizers = {model_name: torch.optim.Adam(
            self.model_dict[model_name].parameters(), lr=1e-4)
            for model_name in self.model_names}
        self.model_dataset_configs = {
            'fastdvdnet': {
                'temp_patch_size': args.temp_patch_size,
                'patch_size': args.patch_size,
                'temp_stride': 3
            },
            'SwinIR': {
                'temp_patch_size': 1,  # sequence_length=1 对应
                'patch_size': 128,  # crop_size=128
                'temp_stride': -1
            }
        }
        self.distill_data_dict = {}
        self.distill_loader_dict = {}

        for model_name in self.model_names:
            config = self.model_dataset_configs[model_name]
            # 每个模型都使用相同的distill_data，但用不同的参数处理
            self.distill_data_dict[model_name] = ServerDataset(
                distill_data,  # 相同的原始数据
                config['temp_patch_size'],
                config['patch_size'],
                random_shuffle=True,
                temp_stride=config['temp_stride']
            )
            self.distill_loader_dict[model_name] = DataLoader(
                self.distill_data_dict[model_name],
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=4
            )
        '''
        self.distill_data = ServerDataset(distill_data, args.temp_patch_size, args.patch_size,
                                          random_shuffle=True, temp_stride=3)
        self.distill_loader = DataLoader(self.distill_data, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        '''
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
        self.client_dataset_path = client_dataset_path
        self.local_test_dataloader = local_test_dataloader


    def create_clients(self):
        model_config = dict(zip(self.args.model_names, self.args.model_counts))
        idx = 0
        for model_name, count in model_config.items():
            for _ in range(count):
                client = Client(self.args, copy.deepcopy(self.model_dict[model_name]),
                                self.client_dataset_path[idx],self.local_test_dataloader[idx],
                                logger=self.logger,device=self.device,model_name=model_name, idx=idx)
                self.clients.append(client)
                idx += 1
    def get_pretrain_model(self):
        pretrain_func_map = {"fastdvdnet": pretrain_fastdvdnet,
                             "SwinIR": pretrain_SwinIR}
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
            distill_loader = self.distill_loader_dict[model_name]
            model.to(self.device)
            model.train()
            criterion = self.distill_criterion.to(self.device)
            epoch_loss = []
            for teacher in self.clients:
                teacher.model.eval()
            for epoch in range(epochs):
                batch_loss = []
                step_count = 0
                for batch_idx, (seq,gt) in enumerate(distill_loader):
                    model.train()
                    optimizer.zero_grad()
                    if model_name == 'SwinIR':
                        ctrl_fr_idx = 0  # 单帧情况下使用第一帧
                    else:
                        ctrl_fr_idx = self.ctrl_fr_idx  # FastDVDnet使用中心帧
                    img_train, gt_train = normalize_augment(seq, ctrl_fr_idx)
                    if img_train.dim() == 5:
                        img_train = img_train.squeeze(1)
                        print("test_size")# Remove time dimension
                    if gt_train.dim() == 5:
                        gt_train = gt_train.squeeze(1)
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
                    if model_name == "fastdvdnet":
                        student_outputs = model(imgn_train, noise_map)
                    elif model_name == "SwinIR":
                        student_outputs = model(imgn_train)
                    soft_target = []
                    if self.mode == "mean output":
                        teacher_outputs = []
                        with torch.no_grad():
                            for teacher in self.clients:
                                if teacher.model_name == "fastdvdnet":
                                    teacher_denoised = teacher.model(imgn_train, noise_map)
                                elif teacher.model_name == "SwinIR":
                                    teacher_denoised = teacher.model(imgn_train)
                                teacher_outputs.append(teacher_denoised)
                        if teacher_outputs:
                            soft_target = torch.stack(teacher_outputs).mean(dim=0)
                        else:
                            continue

                    loss = criterion(student_outputs, soft_target)
                    aux_loss_weight = 0.2
                    aux_loss = nn.MSELoss()(student_outputs, gt_train)
                    total_loss_batch = (1 - aux_loss_weight) * loss + aux_loss_weight * aux_loss
                    total_loss_batch.backward()
                    optimizer.step()
                    step_count += 1
                    if step_count % 5 ==0:
                        print(f'Epoch: [{epoch}/{epochs}], Step: [{step_count}],model:[{model_name}], loss:[{total_loss_batch}]')
                    batch_loss.append(total_loss_batch.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
    def train(self):
        train_loss = []
        for epoch in tqdm(range(self.args.num_epochs)):
            current_lr = self.args.lr * (0.1 ** (epoch // 3)) ## ?
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

            if model_name == "fastdvdnet":
                avg_psnr = validate_fastdvd_model(
                    model = model,
                    dataset_val=self.global_test_loader,
                    valnoisestd=self.args.test_noise,
                    temp_psz= self.args.temp_psz,
                    device=self.device
                    )
            elif model_name == "SwinIR":
                avg_psnr = validate_SwinIR_model(
                    args=self.args,
                    model=model,
                    dataset_val=self.global_test_loader,
                    valnoisestd=self.args.test_noise,
                    device=self.device
                )
            self.logger.add_scalar(f'Global_{model_name}/Test_PSNR', avg_psnr, epoch)
            print(f"\n[Global Test PSNR] {model_name} Clean: {avg_psnr:.4f} dB")




    def Save_CheckPoint(self, save_path,model_name):
        torch.save(self.model_dict[model_name].state_dict(), save_path)
















