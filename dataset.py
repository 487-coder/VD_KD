import random
from pathlib import Path
from torch.utils.data import Subset, Dataset
import numpy as np
import imageio.v3 as iio
import torch
from torchvision import transforms as T
from pathlib import Path
import random
from utils import open_sequence
image_types = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

from utils import get_image_names
VALSEQPATT = '*'
NUMFRXSEQ_VAL = 15

def partition(file_root,client_number):
    np.random.seed(10000)
    video_paths = sorted(Path(file_root).glob("*.mp4"))
    dataset_length = len(video_paths)
    index = np.arange(dataset_length)
    data_per_client = dataset_length // client_number
    np.random.shuffle(index)
    client_dataset = {}
    for i in range(client_number):
        start = i * data_per_client
        end = start + data_per_client
        client_index = index[start:end]
        client_dataset[i] = []
        for j in client_index:
            client_dataset[i].append(video_paths[j])
    return client_dataset
def partition_server_data(parent_dir,split_ratio=0.3, shuffle=True, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    server_image = Path(parent_dir)
    all_dirs = [str(p) for p in server_image.iterdir() if p.is_dir()]

    random.shuffle(all_dirs)
    split_point = int(len(all_dirs) * split_ratio)

    return all_dirs[:split_point], all_dirs[split_point:]
class ServerDataset(Dataset):
    def __init__(self, data_path, sequence_length, crop_size, epoch_size=-1,
                 random_shuffle=True, temp_stride=-1):
        super(ServerDataset, self).__init__()
        self.data_path = data_path
        self.videos = []
        for p in self.data_path:
            img_files = get_image_names(p)  # p 是帧图片所在的目录
            frames = [iio.imread(str(img)) for img in img_files]  # 逐张读取
            self.videos.append(np.stack(frames, axis=0))  # [T, H, W, 3]
        self.seq_len = sequence_length
        self.half = (sequence_length - 1) // 2
        self.stride = temp_stride if temp_stride > 0 else sequence_length
        self.crop_size = crop_size
        self.random_shuffle = random_shuffle
        self.to_tensor = T.ToTensor()
        self.pairs = []
        for vid, frames in enumerate(self.videos):
            T_frames = frames.shape[0]
            valid_centers = list(range(
                self.half * self.stride,
                T_frames - self.half * self.stride
            ))
            for c in valid_centers:
                self.pairs.append((vid, c))

        self.epoch_size = epoch_size if epoch_size > 0 else len(self.pairs)
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.random_shuffle:
            vid, center = random.choice(self.pairs)
        else:
            vid, center = self.pairs[idx % len(self.pairs)]

        video = self.videos[vid]  # [T, H, W, 3]

        seq_imgs = []
        for i in range(self.seq_len):
            frame_idx = center + (i - self.half) * self.stride
            arr = video[frame_idx]  # H,W,3 uint8
            t = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0)
            seq_imgs.append(t)
        seq = torch.stack(seq_imgs, dim=0)

        i, j, h, w = T.RandomCrop.get_params(seq[0], output_size=(self.crop_size, self.crop_size))
        seq = seq[:, :, i:i + h, j:j + w]  # [F, C, Hc, Wc]
        # 中心帧 GT： [1, C, H, W]
        gt = seq[self.half]  # 取第 self.half 帧
        return seq, gt






class LocalDataset(Dataset):
    def __init__(self,file_paths,sequence_length,crop_size,epoch_size = -1,
                 random_shuffle = True,temp_stride = -1):
        super().__init__()
        video_paths = file_paths
        self.video_paths = video_paths
        # 预加载所有视频帧到内存
        self.videos = []
        for p in self.video_paths:
            frames = iio.imread(str(p), plugin="pyav")  # [T, H, W, 3], RGB
            self.videos.append(np.asarray(frames))

        self.seq_len = sequence_length
        self.half = (sequence_length - 1) // 2
        self.stride = temp_stride if temp_stride > 0 else sequence_length
        self.crop_size = crop_size
        self.random_shuffle = random_shuffle
        self.to_tensor = T.ToTensor()

        self.pairs = []
        for vid, frames in enumerate(self.videos):
            T_frames = frames.shape[0]
            valid_centers = list(range(
                self.half * self.stride,
                T_frames - self.half * self.stride
            ))
            for c in valid_centers:
                self.pairs.append((vid, c))

        self.epoch_size = epoch_size if epoch_size > 0 else len(self.pairs)


    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.random_shuffle:
            vid, center = random.choice(self.pairs)
        else:
            vid, center = self.pairs[idx % len(self.pairs)]

        video = self.videos[vid]  # [T, H, W, 3]

        seq_imgs = []
        for i in range(self.seq_len):
            frame_idx = center + (i - self.half) * self.stride
            arr = video[frame_idx]  # H,W,3 uint8
            t = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0)
            seq_imgs.append(t)
        seq = torch.stack(seq_imgs, dim=0)

        i, j, h, w = T.RandomCrop.get_params(seq[0], output_size=(self.crop_size, self.crop_size))
        seq = seq[:, :, i:i + h, j:j + w]  # [F, C, Hc, Wc]
        # 中心帧 GT： [1, C, H, W]
        gt = seq[self.half]  # 取第 self.half 帧
        return seq, gt

class TestDataset(Dataset):

    def __init__(self, data_dir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
        self.gray_mode = gray_mode
        self.data_dir = Path(data_dir)
        self.num_input_frames = num_input_frames

        seq_dirs = sorted(self.data_dir.glob(VALSEQPATT))
        seq_dirs = [p for p in seq_dirs if p.is_dir() and not p.name.startswith('.')]
        self.sequences = []
        for seq_path in seq_dirs:
            seq, _, _ = open_sequence(
                str(seq_path),  # open_sequence likely expects string path
                gray_mode=gray_mode,
                expand_if_needed=False,
                max_num_fr=num_input_frames
            )
            self.sequences.append(seq)

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index])

    def __len__(self):
        return len(self.sequences)

def partition_test_dataset(global_test_dataset, client_number):
    np.random.seed(10000)
    testset_length = len(global_test_dataset)
    index = np.arange(testset_length)
    data_per_client = testset_length // client_number
    np.random.shuffle(index)
    client_testset = {}
    for i in range(client_number):
        start = i * data_per_client
        end = start + data_per_client
        client_index = index[start:end]
        sub_dataset = Subset(global_test_dataset, client_index)
        client_testset[i] = sub_dataset
    return client_testset






