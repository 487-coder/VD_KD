import copy

import torch
from torch import nn
from models import denoise_seq_fastdvdnet
from utils import normalize_augment,orthogonal_conv_weights,save_model_checkpoint,batch_psnr
class Client(object):
    def __init__(self,args,model,trainloader,testloader,logger,device,model_name,idx):
        '''client = Client(self.args, copy.deepcopy(self.model_dict[model_name]),
                                    self.client_dataloader[idx],self.local_test_dataloader[idx],
                                    logger=self.logger,device=self.device,model_name=model_name, idx=idx)'''
        self.args = args
        self.model = copy.deepcopy(model)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.logger = logger
        self.model_name = model_name
        self.idx = idx
        self.ctrl_fr_idx = (self.args.temp_psz - 1) // 2
        self.criterion = nn.MSELoss(reduction='sum').to(device)

    def update_weights(self,global_round,lr):
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = self.criterion.to(self.device)

        epoch_loss = []
        orthog_every_steps = 10  # 每10个step进行正交化
        stop_orthog_epoch = 8  # 第8轮后停止正交化
        for epoch in range(self.args.local_ep):
            batch_loss = []
            enable_orthog = epoch < stop_orthog_epoch
            step_count = 0
            for batch_idx, (seq, gt) in enumerate(self.trainloader):
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
                          f"[{batch_idx * len(img_train)}/{len(self.trainloader.dataset)} ({100. * batch_idx / len(self.trainloader):.0f}%)] "
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

    def test_psnr(self):
        self.model.eval()
        total_psnr = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, seq in enumerate(self.testloader):
                seq = seq.to(self.device)  # [1, T, C, H, W]
                if seq.dim() == 5 and seq.shape[0] == 1:
                    seq = seq.squeeze(0)  # [T, C, H, W]

                noise = torch.empty_like(seq).normal_(mean=0, std=self.args.test_noise).to(self.device)
                noisy_seq = seq + noise
                noisy_seq = torch.clamp(noisy_seq, 0.0, 1.0)
                noise_map = torch.tensor([self.args.test_noise], dtype=torch.float32).to(self.device)

                denoised_seq = denoise_seq_fastdvdnet(
                    seq=noisy_seq,
                    noise_std=noise_map,
                    temporal_window=self.args.temp_psz,
                    model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                )

                # 只评估中心帧
                gt = seq[self.ctrl_fr_idx].unsqueeze(0)
                pred = denoised_seq[self.ctrl_fr_idx].unsqueeze(0)
                #print("GT min/max:", gt.min().item(), gt.max().item())
                #print("Pred min/max:", pred.min().item(), pred.max().item())

                psnr = batch_psnr(pred, gt, data_range=1.0)
                total_psnr += psnr
                cnt += 1
                print(f"[Client {self.idx},{self.model_name}] PSNR on sample {cnt}: {psnr:.2f}")
        return total_psnr / cnt

    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)




