from torch.utils.data import DataLoader
from pathlib import Path
from dataset import ServerDataset,TestDataset
from dataset import partition_server_data
from options import args_parser
from pretrain import pretrain

args = args_parser()
pretrain_data, distill_data = partition_server_data(args.server_data,split_ratio =0.3,shuffle=True,seed= 10000)
val_dataset = TestDataset(args.val_dataset)
exist_pretrain_model = (Path(f"{args.pretrain_model}") / "fastdvdent.pth").exists()
if not exist_pretrain_model:
    pretrain(args,pretrain_data,val_dataset)





