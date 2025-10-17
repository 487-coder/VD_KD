import copy
from typing import List

from utils import normalize_augment
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_model_checkpoint
from models.SwinIR_model import SwinIR, validate_SwinIR_model,frame_denoise_swinir
from models.fastdvd_model import FastDVDnet, validate_fastdvd_model,frame_denoise,denoise_seq_fastdvdnet

from pathlib import Path
from pretrain_fastdvdnet import pretrain_fastdvdnet
from pretrain_SwinIR import pretrain_SwinIR
from pretrain_RVRT import pretrain_RVRT
import torch
from client import Client
from dataset import ServerDataset,TestDataset
import numpy as np
from models.RVRT_model import RVRT,validate_RVRT_model,test_clip


class Server(object):
    def __init__(self, args, model_names, pretrain_data, val_dataset, distill_data,
                 client_dataset_path, local_test_dataloader,
                 logger, mode, device):
        super(Server, self).__init__()
        self.args = args

        self.model_names = model_names
        self.model_dict = {"fastdvdnet": FastDVDnet().to(device),
                           "SwinIR": SwinIR(upscale=1, in_chans=3,
                                            img_size=128,
                                            window_size=8,
                                            img_range=1.0,
                                            depths=[6, 6, 6, 6, 6, 6],
                                            embed_dim=180,
                                            num_heads=[6, 6, 6, 6, 6, 6],
                                            mlp_ratio=2,
                                            upsampler=None,
                                            resi_connection="1conv").to(device),
                           "RVRT": RVRT(upscale=1,clip_size=2,img_size=[2, 64, 64],
                                        window_size=[2, 8, 8],num_blocks=[1, 2, 1],
                                        depths= [2, 2, 2],embed_dims=[192, 192, 192],
                                        num_heads=[6, 6, 6],inputconv_groups=[1, 3, 4, 6, 8, 4],
                                        spynet_path="/root/autodl-tmp/pretrain_model/spynet_sintel_final-3d2a1287.pth",
                                        deformable_groups=12,attention_heads=12,attention_window=[3,3],
                                        nonblind_denoising=True,use_checkpoint_attn=False,
                                        use_checkpoint_ffn=False,no_checkpoint_attn_blocks=[],
                                        no_checkpoint_ffn_blocks=[],cpu_cache_length=100)}

        self.pretrain_data = pretrain_data
        #
        self.distill_criterion = nn.MSELoss()
        self.distill_optimizers = {model_name: torch.optim.Adam(
            self.model_dict[model_name].parameters(), lr=1e-4)
            for model_name in self.model_names}
        self.distill_data = ServerDataset(
            distill_data, self.args.temp_patch_size,
            96,
            epoch_size = 2560,
            random_shuffle=True,
            temp_stride=1
        )
        self.distill_loader = DataLoader(
            self.distill_data,
            batch_size=1,
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
        # 验证
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
                                self.client_dataset_path[idx], self.local_test_dataloader[idx],
                                logger=self.logger, device=self.device, model_name=model_name, idx=idx)
                self.clients.append(client)
                idx += 1

    def get_pretrain_model(self):
        def load_clean_state_dict(model, ckpt_path, model_name):
            state_dict = torch.load(ckpt_path, map_location=self.device)

            # 如果是 dict 包装的 checkpoint（SwinIR 常见）
            if isinstance(state_dict, dict) and "params" in state_dict:
                state_dict = state_dict["params"]

            # 如果 key 前缀带 module.，去掉
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            # 统计加载比例
            total_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(v.numel() for k, v in state_dict.items() if k in model.state_dict())
            ratio = 100.0 * loaded_params / total_params

            print(f"🔍 Loading pretrain for {model_name} from {ckpt_path}")
            print(f"    Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            print(f"    Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            print(f"    ✅ Loaded {loaded_params}/{total_params} params ({ratio:.2f}%)")

            if len(missing) > 0:
                print(f"⚠️ Warning: Some layers of {model_name} were not loaded from checkpoint!")

            return model

        pretrain_func_map = {
            "fastdvdnet": pretrain_fastdvdnet,
            "SwinIR": pretrain_SwinIR,
            "RVRT": pretrain_RVRT
        }

        for model_name, model in self.model_dict.items():
            ckpt_path = Path(self.args.pretrain_model) / f"{model_name}.pth"
            if not ckpt_path.exists():
                if model_name in pretrain_func_map:
                    print(f"⚠️ No pretrain found for {model_name}, running pretrain function...")
                    pretrain_func_map[model_name](self.args, self.pretrain_data, self.val_dataset)
            else:
                self.model_dict[model_name] = load_clean_state_dict(model, ckpt_path, model_name)
                self.model_dict[model_name] = self.model_dict[model_name].to(self.device)

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
                self.model_dict[model_name] = self.model_dict[model_name].to(self.device)
        for client in self.clients:
            client.model.to("cpu")
            torch.cuda.empty_cache()

    def distill_old(self, epochs: int = 1) -> None:
        for model_name in self.model_names:
            model = self.model_dict[model_name]
            optimizer = self.distill_optimizers[model_name]
            criterion = self.distill_criterion.to(self.device)
            distill_loader = self.distill_loader
    
            
        
            for teacher in self.clients:
                teacher.model.to(self.device)
                teacher.model.eval()
    
            epoch_loss = []
    
            for epoch in range(epochs):
                batch_loss, step = [], 0
                for seq in distill_loader:
                    model.to(self.device)
                    model.train()
                    
                    optimizer.zero_grad()
    
                    img_train, gt_train = normalize_augment(seq, self.ctrl_fr_idx)
                
                    print(f"image_train shape {img_train.shape}")
                    img_train = img_train.to(self.device, non_blocking=True)
                    gt_train = gt_train.to(self.device, non_blocking=True)
    
                    N, FC, H, W  = img_train.shape
                    F = FC // 3 
                    img_train = img_train.view(N, F, 3, H, W).squeeze(0)  # [F, 3, H, W]
                    print(f"new_image_train shape {img_train.shape}")
                    n, C, h, w = img_train.shape
                    stdn = torch.empty((n, 1, 1, 1), device=self.device).uniform_(
                        self.args.noise_level[0], self.args.noise_level[1]
                    )
                    noise = torch.normal(mean=0.0, std=stdn.expand_as(img_train))
                    

                    imgn_train = torch.clamp(img_train + noise, 0.0, 1.0)
                    gtn_train = torch.clamp(
                        imgn_train[:, 3 * self.ctrl_fr_idx:3 * self.ctrl_fr_idx + 3, :, :],
                        0.0,
                        1.0,
                    )

                    noise_map = stdn.expand((n, 1, h, w))
            
                    nonblind_img = torch.cat([imgn_train, noise_map], dim=1)
    
                    # student forward
                    if model_name == "fastdvdnet":
                        student_outputs = denoise_seq_fastdvdnet(seq=imgn_train,noise_std=stdn,
                        temporal_window=5,model=model,is_training= True)
                        #seq, noise_std, model, temporal_window=5, is_training=False
                        #save_seq(student_outputs, "student_fast", "step", batch_idx=step)
                        
                    elif model_name == "SwinIR":
                        student_outputs = torch.empty_like(imgn_train, device=self.device)

                        for t in range(n):
                            noisy_frame = imgn_train[t].unsqueeze(0).to(self.device)
                            noise_map_t = noise_map[t].unsqueeze(0).to(self.device)
                            denoised_frame = frame_denoise_swinir(model, noisy_frame, noise_map_t, device= self.device,
                                                                   context = torch.enable_grad())
                            student_outputs[t] = denoised_frame.squeeze(0)
                        #save_seq(student_outputs, "student_Swin", "step", batch_idx=step)
                    elif model_name == "RVRT":
                        nonblind_img = nonblind_img.unsqueeze(0).to(self.device)
                        student_outputs = test_clip(nonblind_img, model,context = torch.enable_grad())
                        student_outputs = student_outputs[:, :n, :, :, :]
                        student_outputs = student_outputs.squeeze(0)
                        #save_seq(student_outputs, "student_RVRT", "step", batch_idx=step)



                    #print("student")
                    #print(model_name)
                    #print(student_outputs.shape)
    
                    # --- collect teachers by architecture ---
                    teacher_outputs = {"fastdvdnet": [], "SwinIR": [],"RVRT":[]}
                    with torch.no_grad():
                        for teacher in self.clients:
                            if teacher.model_name == "fastdvdnet":
                                #print("before distill")
                                
                                idx = teacher.idx
                                #teacher.test_psnr(role = "distill", idx = idx)
                                #print(f"Model {idx} hash after test_psnr: {hash(str(list(teacher.model.parameters())))}")
                                #print(imgn_train.shape, noise_map.shape)
                                #x = teacher.model(imgn_train,noise_map)
                                teacher_output = denoise_seq_fastdvdnet(seq=imgn_train, noise_std=stdn,
                                                                         temporal_window=5, model=teacher.model,
                                                                         is_training=False)

                                teacher_outputs["fastdvdnet"].append(teacher_output)
                                
                                #save_seq(teacher_output, f"{idx}", "step", batch_idx=step)
                                
                            elif teacher.model_name == "SwinIR":
                                teacher_output = torch.empty_like(imgn_train, device=self.device)

                                for t in range(n):
                                    noisy_frame = imgn_train[t].unsqueeze(0).to(self.device)
                                    noise_map_t = noise_map[t].unsqueeze(0).to(self.device)
                                    denoised_frame = frame_denoise_swinir(teacher.model, noisy_frame, noise_map_t,
                                                                          device=self.device,
                                                                          context=torch.no_grad())
                                    teacher_output[t] = denoised_frame.squeeze(0)
                                #x = frame_denoise_swinir(teacher.model, gtn_train, noise_map, device= self.device)
                                teacher_outputs["SwinIR"].append(teacher_output)
                                idx = teacher.idx
                                save_seq(teacher_output, f"{idx}", "step", batch_idx=step)
                            elif teacher.model_name == "RVRT":
                                nonblind_img = nonblind_img.unsqueeze(0).to(self.device)
                                teacher_output = test_clip(nonblind_img, teacher.model)
                                teacher_output = teacher_output[:, :n, :, :, :]
                                teacher_output = teacher_output.squeeze(0)
                                teacher_outputs["RVRT"].append(teacher_output)
                                idx = teacher.idx
                                save_seq(teacher_output, f"{idx}", "step", batch_idx=step)

                                #save_seq(x, f"{idx}", "step", batch_idx=step)
                            #print("teacher")
                            #print(teacher.model_name)
                            #print(x.shape)
    
                    # average each architecture separately
                    soft_targets = {}

                    for arch, outs in teacher_outputs.items():
                        if outs:  # 检查outs是否非空
                            mean_output = torch.stack(outs).mean(0)
                            soft_targets[arch] = mean_output
                            save_seq(mean_output, f"target_{arch}", "step", batch_idx=step)

                            
                    if not soft_targets:
                        continue  # no teacher available
    
                    # compute distillation loss against each architecture
                    losses = [
                        criterion(student_outputs, target)
                        for target in soft_targets.values()
                    ]
                    distill_loss = torch.stack(losses).mean()
    
                    aux_loss_weight = 0.8
                    aux_loss = nn.MSELoss()(student_outputs, gt_train)
                    total_loss = (1 - aux_loss_weight) * distill_loss + aux_loss_weight * aux_loss
    
                    total_loss.backward()
                    optimizer.step()
    
                    step += 1
                    batch_loss.append(total_loss.item())
                    #print(f"🔎 当前 batch 来自文件: {video_name}")
                    #print(f"当前中心帧: {center}")
                
                    if step % 10 == 0:
                        print(f"Epoch [{epoch}/{epochs}] Step [{step}] "
                              f"Model [{model_name}] Loss [{total_loss:.4f}]")
                        #if model_name == "fastdvdnet":
                            #print(step)
                            #self.global_test_psnr(epoch)
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
    def distill(self, epochs: int = 1) -> None:
        """
        目标：用“聚合后的全局模型”互相蒸馏。
        关键：开头对全局模型做一次 snapshot（teacher_snapshots），后续所有学生都只向这些冻结老师学习；
             即使先蒸了 A→B，随后做 B→A，B 的 teacher 也仍是 snapshot（聚合后的原始权重），不受 A→B 影响。
        """
        device = self.device
        def freeze_bn_running_stats(model):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()                       # 固定running_mean/var
                    m.track_running_stats = True
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)

    
        # -----------------------------
        # 1) 冻结老师快照（关键！）
        # -----------------------------
        teacher_snapshots = {}
        for name in self.model_names:
            t = copy.deepcopy(self.model_dict[name])
            t.eval()
            for p in t.parameters():
                p.requires_grad = False
            teacher_snapshots[name] = t.to(self.device)
    
        # （如果你内存吃紧，可以只在用到时上 GPU，用完移回 CPU；
        #  为了清晰，这里一次性放到 device。）
    
        # -----------------------------
        # 2) 共同的损失
        # -----------------------------
        criterion = self.distill_criterion.to(device)
    
        # -----------------------------
        # 3) 各学生逐一蒸馏（学生是“可变的”，老师是 snapshot 固定）
        # -----------------------------
        for student_name in self.model_names:
            student = self.model_dict[student_name]
            optimizer = self.distill_optimizers[student_name]
    
            # 老师集合 = 除了自己之外的所有快照（互相蒸馏；如需把同构也算老师，可删除这个 if）
            teacher_set = {k: v for k, v in teacher_snapshots.items() if k != student_name}
    
            # 若没有老师（只剩一个模型），直接跳过
            if not teacher_set:
                continue
    
            epoch_loss = []
    
            for epoch in range(epochs):
                batch_loss, step = [], 0
    
                for seq in self.distill_loader:
                    # DataLoader 可能返回 [tensor]，这里统一成 tensor，避免 normalize_augment 报 list.max()
                    if isinstance(seq, (list, tuple)):
                        seq = seq[0]
    
                    student.to(device).train()
                    freeze_bn_running_stats(student)
                    optimizer.zero_grad()
    
                    # --------- 数据预处理 ---------
                    img_train, gt_train = normalize_augment(seq, self.ctrl_fr_idx)   # [1, F*3, H, W], [1, 3, H, W]
                    img_train = img_train.to(device, non_blocking=True)
                    gt_train = gt_train.to(device, non_blocking=True)
    
                    N, FC, H, W = img_train.shape
                    F = FC // 3
                    img_train = img_train.view(N, F, 3, H, W).squeeze(0)             # [F, 3, H, W]
                    n, C, h, w = img_train.shape
    
                    # 每帧不同噪声（兼容 Tensor(shape=(1,)) 的写法见你已有的 denoise_seq_fastdvdnet 修改）
                    #stdn_uniform = torch.empty(1, device=device).uniform_(
                    #self.args.noise_level[0], 
                    #self.args.noise_level[1]
                    #) 
                    stdn = torch.empty(1, device=device).uniform_(
                    self.args.noise_level[0],
                    self.args.noise_level[1]
                    )

                    # 生成噪声也用张量
                    noise = torch.normal(mean=0.0, std=stdn.item(), size=img_train.shape).to(device)
                    # 或者
                    #noise = torch.randn_like(img_train) * stdn
                    
                    # noise_map 也用张量
                    noise_map = stdn.view(1, 1, 1, 1).expand(n, 1, h, w)
                    
                    # 传给函数
                    
                
                    #noise = torch.normal(mean=0.0, std=stdn.expand_as(img_train))
                    #imgn_train = torch.clamp(img_train + noise, 0.0, 1.0)
                    #noise = torch.normal(
                    
                    imgn_train = torch.clamp(img_train + noise, 0.0, 1.0)

                    # 为 SwinIR 和 RVRT 准备统一的 noise_map
                    #noise_map = stdn_uniform.expand((n, 1, h, w))  # [F, 1, H, W]
                   
                    # RVRT 用的 nonblind_img
                    nonblind_img = torch.cat([imgn_train, noise_map], dim=1)  # [F, 4, H, W]

    
                    # 中心帧 gt（如果你有整序列干净 gt，可用序列监督，这里先保持中心帧）
                    # gtn_train = torch.clamp(
                    #     img_train[:, 3 * self.ctrl_fr_idx:3 * self.ctrl_fr_idx + 3, :, :],  # 如果 img_train 是干净序列
                    #     0.0, 1.0
                    # )
                    # 但根据你上面的实现，这里沿用 normalize_augment 的中心帧 gt_train:
                    # gt_train: [1, 3, H, W]
                    center = self.ctrl_fr_idx
    
                    #noise_map = stdn.expand((n, 1, h, w))
                    nonblind_img = torch.cat([imgn_train, noise_map], dim=1)  # RVRT 用
    
                    # --------- 学生前向 ---------
                    if student_name == "fastdvdnet":
                        save_seq(imgn_train, f"fastdvd_student_raw", "step", batch_idx=step)
                        student.to(device).train()
                        student_outputs = denoise_seq_fastdvdnet(
                            seq=imgn_train, noise_std=stdn, temporal_window=5,
                            model=student, is_training=True
                        )  # [F, 3, H, W]
                        save_seq(student_outputs, f"fastdvd", "step", batch_idx=step)
    
                    elif student_name == "SwinIR":
                        save_seq(imgn_train, f"swinir_student_input_full", "step", batch_idx=step)
                        
                        student.to(device).train()
                        student_outputs = torch.empty_like(imgn_train, device=device)
                    
                        # 🔑 关键：SwinIR 在训练模式下会计算梯度，导致计算图堆积
                        # 处理每一帧，及时释放计算图
                        for t in range(n):
                            noisy_frame = imgn_train[t:t+1]  # [1, 3, H, W]
                            noise_map_t = noise_map[t:t+1]   # [1, 1, H, W]
                            #save_seq(noisy_frame, f"swinir_student_frame_{t:02d}_input", "step", batch_idx=step)
        
                            
                            denoised_frame = frame_denoise_swinir(
                                student, noisy_frame, noise_map_t, 
                                device=device, context=torch.enable_grad()
                            )
                            student_outputs[t] = denoised_frame.squeeze(0)
                            
                            
                    
                        save_seq(student_outputs, f"SwinIR", "step", batch_idx=step)
    
                    elif student_name == "RVRT":
                        student.to(device).train()
                        nb = nonblind_img.unsqueeze(0).to(device)                   # [1, F, 4, H, W]
                        student_outputs = test_clip(nb, student, context=torch.enable_grad())  # [1, F, 3, H, W]
                        student_outputs = student_outputs[:, :n].squeeze(0)
    
                    # --------- 老师前向（固定 snapshot，不更新） ---------
                    with torch.no_grad():
                        teacher_sum = torch.zeros_like(student_outputs, device=device)
                        teacher_count = 0

                        #teacher_outs = []
                        for tname, tmodel in teacher_set.items():
                            tmodel = tmodel.to(self.device)
                            tmodel.eval()
                            if tname == "fastdvdnet":
                                save_seq(imgn_train, f"fastdvd_teacher_raw", "step", batch_idx=step)
                                tout = denoise_seq_fastdvdnet(
                                    seq=imgn_train, noise_std=stdn, temporal_window=5,
                                    model=tmodel, is_training=False
                                )
                                save_seq(tout, f"fastdvd_teacher", "step", batch_idx=step)
                            elif tname == "SwinIR":
                                save_seq(imgn_train, f"swinir_teacher_input_full", "step", batch_idx=step)
                                tout = torch.empty_like(imgn_train, device=device)
                                for tt in range(n):
                                    nf = imgn_train[tt].unsqueeze(0)
                                    nm = noise_map[tt].unsqueeze(0)
                                    #save_seq(nf, f"swinir_teacher_frame_{t:02d}_input", "step", batch_idx=step)
        
                                    df = frame_denoise_swinir(tmodel, nf, nm, device=device)
                                    tout[tt] = df.squeeze(0)
                                save_seq(tout, f"Swin_teacher", "step", batch_idx=step)
                            elif tname == "RVRT":
                                tmodel.eval()
                                nb_t = nonblind_img.unsqueeze(0).to(device)
                                tout = test_clip(nb_t, tmodel)[:, :n].squeeze(0)
                            #teacher_outs.append(tout)
                            teacher_sum += tout
                            teacher_count += 1
                            tmodel.to("cpu")
                            del tout
                            torch.cuda.empty_cache()

                        soft_target = teacher_sum / max(1, teacher_count)
                        #save_seq(soft_target, f"soft", "step", batch_idx=step)


    
                    # --------- 蒸馏损失（对多个老师取均值） ---------
                    #soft_target = torch.stack(teacher_outs).mean(0)                # [F, 3, H, W]
                    distill_loss = criterion(student_outputs, soft_target)
    
                    # 可选：只对中心帧做监督，避免广播问题
                    aux_loss_weight = 0.5
                    if seq.dim() == 5:  # [1, T, 3, H, W]
                        seq = seq.squeeze(0) 
                    seq = seq.to(self.device, non_blocking=True)
                    aux_loss = nn.MSELoss()(student_outputs, seq)
                    total_loss = (1 - aux_loss_weight) * distill_loss + aux_loss_weight * aux_loss
    
                    total_loss.backward()
                    optimizer.step()
                    del imgn_train, noise, student_outputs, soft_target
                    torch.cuda.empty_cache()
    
                    step += 1
                    batch_loss.append(total_loss.item())
                
    
                    if step % 10 == 0:
                        print(f"[Mutual KD] Student[{student_name}] Epoch[{epoch}/{epochs}] "
                              f"Step[{step}] Loss[{total_loss:.4f}]")
    
                epoch_loss.append(sum(batch_loss) / max(1, len(batch_loss)))
            #student.to("cpu")  # ✅ 修改⑤：每个学生训练完后移回CPU
            torch.cuda.empty_cache()



    def train(self):
        train_loss = []
        for epoch in tqdm(range(self.args.num_epochs)):
            current_lr = self.args.lr * (0.1 ** (epoch // 3))  ## ?
            global_psnr = 0
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch + 1} |\n')
            m = max(int(self.args.sampling_rate * self.args.client_numbers), 1)
            idxs_users = np.random.choice(range(self.args.client_numbers), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.clients[idx].load_model(self.model_dict[self.clients[idx].model_name].state_dict())
                if self.clients[idx].model_name == "fastdvdnet":
                    w, loss = self.clients[idx].update_fastdvd_weights(global_round=epoch, lr=current_lr)
                else:
                    w, loss = self.clients[idx].update_SwinIR_weights(global_round=epoch, lr=current_lr)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                local_psnr = self.clients[idx].test_psnr(idx = idx, role = "client")
                global_psnr += local_psnr
                self.logger.add_scalar(f'Client_{idx}_{self.clients[idx].model_name}/Loss', loss, epoch)
                self.logger.add_scalar(f'Client_{idx}_{self.clients[idx].model_name}/PSNR', local_psnr, epoch)
           
            self.aggregate_same_models()
            

            self.distill()
            
            
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average test psnr:', global_psnr / m)
            print("➡️ Writing to TensorBoard...")
            self.logger.add_scalar('Global/Average_Loss', loss_avg, epoch)
            self.logger.add_scalar('Global/Average_PSNR', global_psnr / m, epoch)
            if epoch % 1 == 0 or epoch == 0:
                print("➡️ Running global_test_psnr...")
                self.global_test_psnr(epoch)
            for model_name in self.model_names:
                print(f"➡️ Saving checkpoint for {model_name}...")
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

    def global_test_psnr(self, epoch):
        for model_name in self.model_names:
            model = self.model_dict[model_name]
            model.eval()

            if model_name == "fastdvdnet":
                avg_psnr = validate_fastdvd_model(
                    model=model,
                    dataset_val=self.global_test_loader,
                    valnoisestd=self.args.test_noise,
                    temp_psz=self.args.temp_psz,
                    device=self.device,
                    role="global_test",
                    idx = -1
                )
                self.logger.add_scalar(f'Global_{model_name}/Test_PSNR', avg_psnr, epoch)
                print(f"\n[Global Test PSNR] {model_name} Clean: {avg_psnr:.4f} dB")
            elif model_name == "SwinIR":
                avg_psnr = validate_SwinIR_model(
                    args=self.args,
                    model=model,
                    dataset_val=self.global_test_loader,
                    valnoisestd=self.args.test_noise,
                    device=self.device,
                    role="global_test"
                )
                print(f"\n[Global Test PSNR] {model_name} Clean: {avg_psnr:.4f} dB")
            elif model_name == "RVRT":
                avg_psnr = validate_RVRT_model(
                    args=self.args,
                    model=model,
                    dataset_val=self.global_test_loader,
                    valnoisestd=self.args.test_noise,
                    device=self.device,
                    role="global_test"
                )
                self.logger.add_scalar(f'Global_{model_name}/Test_PSNR', avg_psnr, epoch)
                print(f"\n[Global Test PSNR] {model_name} Clean: {avg_psnr:.4f} dB")

    def Save_CheckPoint(self, save_path, model_name):
        torch.save(self.model_dict[model_name].state_dict(), save_path)


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














