import os

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from dataset import ServerDataset,TestDataset,LocalDataset
from dataset import partition_server_data,partition,partition_test_dataset
from options import args_parser
from pretrain_fastdvdnet import pretrain_fastdvdnet
from server.ServerBase import Server
from client import Client
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = args_parser()
logger = SummaryWriter('./logs')
args.save_dir = './model_checkpoint/'
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


client_dataset_path= partition(args.train_dataset,args.client_numbers)
val_dataset = TestDataset(args.val_data)


client_dataset = {}
client_testset = partition_test_dataset(val_dataset,args.client_numbers)
client_dataloader = {}
client_test_dataloader = {}
for i in range(args.client_numbers):
    client_dataset[i] = client_dataset_path[i]
    '''
    if i < args.model_counts[0]:
        client_dataset[i] = LocalDataset(client_dataset_path[i],sequence_length= 1,crop_size= 128,
                                         epoch_size= -1,random_shuffle= True, temp_stride= -1)
        client_dataloader[i] = DataLoader(client_dataset[i], batch_size=2, shuffle=False, num_workers=4)
    else:
        client_dataset[i] = LocalDataset(client_dataset_path[i],sequence_length= 5, crop_size=args.patch_size,
                                        epoch_size=args.max_number_patches,random_shuffle=True,temp_stride= 3)
        client_dataloader[i] = DataLoader(client_dataset[i],batch_size=args.batch_size,shuffle=False,num_workers=4)
    '''
    client_test_dataloader[i] = DataLoader(client_testset[i],batch_size=1)
#分发数据集
pretrain_data, distill_data = partition_server_data(args.server_data,split_ratio =0.3,shuffle=True,seed= 10000)

#初始化server
server = Server(args,args.model_names,pretrain_data,val_dataset,distill_data,
                client_dataset_path,client_test_dataloader,logger,args.mode,device)
#初始化clients
server.get_pretrain_model()
server.create_clients()
#初始化模型，分发给对应客户端

server.train()
server.global_test_psnr(epoch = args.num_epochs)
logger.close()
if args.upload_model == True:
    for model_name in args.model_names:
        save_path = os.path.join(args.save_dir, f'{model_name}.pth')
        server.Save_CheckPoint(save_path, model_name)
        print(f'{model_name} model is saved on: {save_path}')










