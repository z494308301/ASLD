import os
import time

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import torch.utils.data as data
import torchvision.transforms as transforms
from thop import profile_origin


class MSTARDataset(data.Dataset):
    def __init__(self, data_dir, split='train', subset='CT'):
        self.data_path = os.path.join(data_dir, split, f"{subset}_data.npy")
        self.label_path = os.path.join(data_dir, split, f"{subset}_labels.npy")
        self.angle_path = os.path.join(data_dir, split, f"{subset}_angle.npy")

        self.data = np.load(self.data_path)
        self.labels = np.load(self.label_path)  # 保留原始标签（字符串）
        self.angles = np.load(self.angle_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = np.expand_dims(img, axis=-1)  # (H,W) -> (H,W,1)
        img = self.transform(img)  # -> (1,H,W)
        label = self.labels[idx]
        # angle = self.angles[idx]
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)
        return img, label, angle


def load_datasets(data_dir):
    all_raw_labels = []
    # 优先收集训练集标签，确保编码器正确拟合

    train_subsets = [('train', 'ANT')]#('train', 'CT'),
    other_subsets = [('val', 'CT'), ('test', 'ANT'), ('test', 'UNT')]#, ('test', 'CT')

    # 先收集训练集标签
    for split, subset in train_subsets:
        label_path = os.path.join(data_dir, split, f"{subset}_labels.npy")
        raw_labels = np.load(label_path)
        all_raw_labels.extend(raw_labels.tolist())
        print(f"训练集{split}-{subset}原始标签: {np.unique(raw_labels)}")

    # 再收集其他子集标签
    for split, subset in other_subsets:
        label_path = os.path.join(data_dir, split, f"{subset}_labels.npy")
        raw_labels = np.load(label_path)
        all_raw_labels.extend(raw_labels.tolist())

    # 拟合编码器
    global_le = LabelEncoder()
    global_le.fit(all_raw_labels)
    print("全局编码器类别:", global_le.classes_, "共", len(global_le.classes_), "类")

    # 第三步：定义数据集加载函数，使用全局编码器
    def create_dataset(split, subset):
        dataset = MSTARDataset(data_dir, split, subset)
        dataset.le = global_le  # 替换为全局编码器
        dataset.labels = global_le.transform(dataset.labels)  # 用全局编码器编码
        assert np.min(dataset.labels) == 0, f"{split}-{subset}标签最小值不为0"
        assert np.max(dataset.labels) == len(global_le.classes_) - 1, f"{split}-{subset}标签最大值错误"
        return dataset

    # 加载所有数据集
    # ct_train = create_dataset('train', 'CT')
    ant_train = create_dataset('train', 'ANT')
    combined_train = torch.utils.data.ConcatDataset([ant_train])#torch.utils.data.ConcatDataset([ct_train, ant_train])

    val_ct = create_dataset('val', 'CT')
    ant_test = create_dataset('test', 'ANT')
    unt_test = create_dataset('test', 'UNT')
    # ct_test = create_dataset('test', 'CT')

    # 验证拼接后训练集标签分布
    all_labels = []
    for ds in combined_train.datasets:
        all_labels.extend(ds.labels.tolist())
    print("拼接后训练集标签分布:", np.unique(all_labels, return_counts=True))
    print("拼接后训练集标签范围:", min(all_labels), max(all_labels))

    return {
        'train': combined_train,
        'val': val_ct,
        'test_ant': ant_test,
        'test_unt': unt_test
        # 'test_ct': ct_test
    }

def datasets_class_num(concat_dataset):
    all_classes = set()

    for sub_dataset in concat_dataset.datasets:
        if hasattr(sub_dataset, 'le'):
            classes = sub_dataset.le.classes_
            all_classes.update(classes)

    num_classes = len(all_classes)

    return num_classes
