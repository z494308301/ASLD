import os.path

import random

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms



from PIL import Image
from model import UNet
from train_test import train_model, model_val, model_test

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

class DynamicKMeansSmoothingLoss:
    def __init__(self, num_classes, feature_extractor, train_dataset,
                 update_interval=2, max_epsilon=0.2, batch_size=64, device='cuda'):
        """
        动态K-Means自适应平滑损失（支持定期更新聚类中心）
        :param update_interval: 每隔多少个epoch更新一次聚类中心
        :param feature_extractor: 模型的特征提取器（用于动态提取特征）
        :param train_dataset: 训练集（用于重新聚类）
        """
        self.current_epoch = 0  # 记录当前epoch，用于判断是否更新
        self.kmeans = KMeans(n_clusters=num_classes, n_init=20, max_iter=500,
                             algorithm='elkan', random_state=42)
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor  # 特征提取器（随模型更新）
        self.train_dataset = train_dataset  # 训练集（用于重新聚类）
        self.update_interval = update_interval  # 更新间隔（epoch）
        self.max_epsilon = max_epsilon
        self.batch_size = batch_size
        self.device = device
        self.epsilon = None  # 初始化为 None
        self.cluster_centers = None
        self.dist_matrix = None  # 新增：保存类别间 pairwise 距离矩阵（后续转为CUDA张量）

        # ================================
        # self.update_if_needed()  # 首次更新聚类和 epsilon
        # ================================

        if isinstance(self.epsilon, np.ndarray):
            self.epsilon = torch.tensor(
                self.epsilon,
                device=self.device,
                dtype=torch.float32
            )

    def _extract_current_features(self):
        """用当前模型提取最新的训练集特征"""
        self.feature_extractor.eval()
        all_features = []

        loader = self.train_dataset

        with torch.no_grad():
            for images, _, _ in tqdm(loader, desc="提取当前特征用于更新聚类"):
                images = images.to(self.device)
                features = self.feature_extractor(images)[0]  # 提取cls_project特征
                all_features.append(features.cpu().numpy())

        raw_features = np.concatenate(all_features, axis=0)
        print(f"PCA 降维前特征维度: {raw_features.shape[1]}")
        pca = PCA(n_components=0.95, random_state=42)
        reduced_features = pca.fit_transform(raw_features)
        print(f"PCA 降维后特征维度: {reduced_features.shape[1]}")
        print(f"PCA 实际保留信息比例: {pca.explained_variance_ratio_.sum():.4f}")

        # 增强归一化：Z-score + Min-Max
        norm_features = (reduced_features - reduced_features.mean(axis=0)) / (reduced_features.std(axis=0) + 1e-8)
        norm_features = (norm_features - norm_features.min(axis=0)) / (
                    norm_features.max(axis=0) - norm_features.min(axis=0) + 1e-8)

        print(f"归一化后特征均值: {norm_features.mean():.6f}（目标≈0.5）")
        print(f"归一化后特征标准差: {norm_features.std():.6f}")
        return norm_features

    def update_if_needed(self):
        """每隔update_interval个epoch，用当前特征更新聚类中心和ε"""
        self.current_epoch += 1
        is_first_run = (self.cluster_centers is None) or (self.epsilon is None)

        if is_first_run or (self.current_epoch % self.update_interval == 0):
            print(f"\n===== Epoch {self.current_epoch}：更新聚类中心和平滑参数 =====")
            current_features = self._extract_current_features()
            self.kmeans.fit(current_features)
            self.cluster_centers = self.kmeans.cluster_centers_

            # 计算类别间 pairwise 距离矩阵（10×10）
            dist_matrix = np.full((self.num_classes, self.num_classes), np.inf)  # 对角线设为无穷大
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    dist = np.linalg.norm(self.cluster_centers[i] - self.cluster_centers[j])
                    dist_matrix[i][j] = dist
                    dist_matrix[j][i] = dist

            self.dist_matrix = torch.tensor(dist_matrix, device=self.device, dtype=torch.float32)

            # 计算每个类的最小混淆距离（用于生成ε）
            min_distances = torch.min(self.dist_matrix, dim=1).values.cpu().numpy()
            print(f"各类别最小混淆距离: {np.round(min_distances, 4)}")

            dist_range = min_distances.max() - min_distances.min()
            print(f"距离范围: {dist_range:.6f}")

            if dist_range < 1e-6:
                normalized_dist = min_distances / (min_distances.max() + 1e-8)
            else:
                normalized_dist = (min_distances - min_distances.min()) / dist_range
            self.epsilon = self.max_epsilon * (1 - normalized_dist)
            self.epsilon = np.clip(self.epsilon, 0.01, self.max_epsilon)  # 避免ε=0导致软标签失效

            # 打印更新后的参数
            print("更新后的平滑参数ε：")
            for cls in range(self.num_classes):
                print(f"类别 {cls}：ε = {self.epsilon[cls]:.4f}")

            assert self.epsilon.shape == (self.num_classes,), \
                f"epsilon维度错误: {self.epsilon.shape}，预期({self.num_classes},)"
            self.epsilon = torch.tensor(
                self.epsilon,
                device=self.device,
                dtype=torch.float32
            )
            assert self.epsilon.shape == (self.num_classes,), f"epsilon 维度错误"


    def __call__(self, logits, labels):

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("logits中包含NaN或Inf值，导致log_softmax失败")

        batch_size = logits.size(0)
        num_classes = self.num_classes
        soft_labels = torch.zeros((batch_size, num_classes), device=self.device, dtype=logits.dtype)

        # 遍历每个样本，生成距离感知的软标签
        for idx in range(batch_size):
            true_cls = labels[idx].item()  # 当前样本的真实类别
            epsilon = self.epsilon[true_cls].item()  # 当前类别的平滑系数

            # 1. 获取真实类别与所有类别的pairwise距离（排除自身，已设为inf）
            pairwise_dists = self.dist_matrix[true_cls]  # 形状(10,)

            # 2. 距离→相似度转换：距离越近，相似度越高（用倒数避免除以0）
            similarities = 1 / (pairwise_dists + 1e-8)  # 加1e-8防止距离为0时溢出
            similarities[true_cls] = 0  # 排除真实类别自身的相似度

            # 3. 归一化相似度（确保总和为1，方便分配ε权重）
            similarities = similarities / similarities.sum()  # 形状(10,)，总和=1

            # 4. 生成软标签：真实类别占1-ε，其他类别按相似度分配ε
            soft_labels[idx, true_cls] = 1 - epsilon  # 真实类别核心权重
            soft_labels[idx, :] += epsilon * similarities  # 其他类别按相似度分配平滑权重

        # 验证软标签合法性（每行总和应为1）
        soft_sum = soft_labels.sum(dim=1)
        assert torch.allclose(soft_sum, torch.ones(batch_size, device=self.device), atol=1e-3), \
            f"软标签每行总和应为1，实际为{soft_sum.cpu().numpy()}"

        # 计算损失
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(soft_labels * log_probs, dim=1).mean()

        # 可选：打印第一个样本的软标签分布（调试用）
        # print(f"样本{0}（真实标签{true_cls}）软标签分布: {torch.round(soft_labels[0], 4).cpu().numpy()}")

        return loss


class AdaptiveAzimuthDisentangleLoss(nn.Module):
    """
    自适应分布回归软标签损失（方位解缠专用）
    对标分类聚类自适应软标签 → 回归自适应分布权重
    输入：
        theta_pred: 预测角度 (B, 1) 0~360°
        theta_true: 真实角度 (B, 1) 0~360°
        conf: 角度置信度 (B, 1) 0~1（模型额外输出）
        disen_score: 解缠评分 (B, 1) 0~1（越小越纠缠）
    输出：自适应软标签分布损失
    """

    def __init__(self, kappa_base=10.0, lambda_disen=0.5):
        super().__init__()
        self.kappa_base = kappa_base
        self.lambda_disen = lambda_disen

    def forward(self, theta_pred, theta_true, conf, disen_score):
        # 角度差（周期性）
        true_vec = torch.stack([
            torch.sin(theta_true),
            torch.cos(theta_true)
        ], dim=1)
        kappa = self.kappa_base * conf * torch.exp(-self.lambda_disen * disen_score)
        cos_diff = F.cosine_similarity(theta_pred, true_vec)
        nll = -kappa * cos_diff + torch.log(2 * np.pi * torch.special.i0(kappa))

        return nll.mean()

class MSTARDataset(Dataset):
    def __init__(self, data_root, angle_root=None, transform=None, is_test=False):
        self.data_root = data_root
        self.angle_root = angle_root
        self.transform = transform
        self.is_test = is_test

        # 固定类别列表（训练/测试必须完全一致）
        self.classes = ['2S1', 'BMP2', 'BRDM_2', 'BTR_60', 'BTR70',
                        'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']
        # self.classes = ['BMP2', 'BRDM_2', 'BTR70', 'T72']
        # self.classes = ['2s1', 'bmp2', 'btr70', 'm1', 'm2',
        #                 'm35', 'm60', 'm548', 't72', 'zsu23']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []  # 保存：(图片路径, 标签, 角度文件路径)

        # 遍历所有类别
        for cls in self.classes:
            cls_dir = os.path.join(self.data_root, cls)
            if not os.path.isdir(cls_dir):
                continue

            # 递归遍历所有子文件夹（支持多层）
            for dirpath, _, filenames in os.walk(cls_dir):
                for fname in filenames:
                    if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        img_path = os.path.join(dirpath, fname)
                        label = self.class_to_idx[cls]

                        # 训练/验证集：匹配角度文件
                        angle_path = None
                        if not self.is_test and self.angle_root:
                            base_name = os.path.splitext(fname)[0]
                            angle_path = os.path.join(self.angle_root, base_name)

                        self.samples.append((img_path, label, angle_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, angle_path = self.samples[idx]

        # 读取灰度图
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)

        # 读取角度
        angle = 0.0
        if not self.is_test:
            with open(angle_path, 'r') as f:
                angle = float(f.read().strip())

        return img, label, angle


def load_image_from_path(img_path, transform):
    """从路径加载图像并转换为模型输入格式"""
    img = Image.open(img_path).convert('L')
    if transform:
        img = transform(img)
    return img.unsqueeze(0)  # 添加batch维度

if __name__ == '__main__':

    # 初始化参数
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 若用GPU
    torch.backends.cudnn.deterministic = True  # 禁用cudnn的随机优化

    TRAIN_FOLDER = r".\SOC\train-aspect"
    TEST_FOLDER = r".\SOC\test"
    ANGLE_FOLDER = r".\angle_new_1"




    transform = transforms.Compose([
        transforms.Resize((88, 88)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)
    ])

    full_train_set = MSTARDataset(
        data_root=TRAIN_FOLDER,
        angle_root=ANGLE_FOLDER,
        transform=transform,
        is_test=False
    )

    # 7:3 划分（无数据泄露）
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 2. 测试集（无角度）
    test_set = MSTARDataset(
        data_root=TEST_FOLDER,
        transform=None,
        is_test=True
    )

    BATCH_SIZE = 64
    NUM_WORKERS = 0
    num_classes = 10

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"训练集数量：{len(train_set)}")
    print(f"验证集数量：{len(val_set)}")
    print(f"测试集数量：{len(test_set)}")

    model = UNet(n_channels=1, n_classes=num_classes, m_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#
    if device.type == 'cuda':
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    model = model.to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=0.001, weight_decay=5e-4)


    best_acc = 0.0
    epochs = 50
    patience = 5 # 最多容忍5轮无提升
    no_improve_epochs = 0  # 当前连续无提升的轮次
    model_save_path = r'.\weights\best_model.pth'

    # 注意：使用适合验证指标的调度器（如ReduceLROnPlateau）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.3, patience=2  # 下降幅度更大，容忍更少轮次无提升
    )

    '''    '''
    loss_fn = DynamicKMeansSmoothingLoss(
        num_classes = num_classes,
        feature_extractor = model,  # 直接传入模型，而非 lambda 函数
        train_dataset = train_loader,
        update_interval = 10,
        max_epsilon = 0.2,
        device = device)

    loss_fa = AdaptiveAzimuthDisentangleLoss(kappa_base=10.0, lambda_disen=0.5)


    for epoch in range(1, epochs + 1):
        print(f"\n##### EPOCH {epoch}/{epochs} #####")


        # 策略：按 当前验证精度 选择损失函数
        if best_acc >= 50:
            # 后期：高阶软标签
            current_loss_fn = loss_fn
            print("=> 使用 动态软标签损失")
        else:
            # 初期：强制硬标签，让特征先稳定
            current_loss_fn = None
            print("=> 使用 硬标签交叉熵")

        # 只有使用软标签时，才允许内部按interval更新
        if current_loss_fn is not None:
            loss_fn.update_if_needed()  # 内部自己判断是否到轮次
            print('=> 动态软标签已激活')

        # 训练
        loss = train_model(model, train_loader, opt, current_loss_fn, loss_fa)
        print(f"Loss on train set: ", loss)

        # 验证
        print("Evaluating on val set:")
        current_val_acc, best_acc, best_acc_map = model_val(
            model=model,
            dataloader=val_loader,
            save_path=model_save_path,
            current_best=best_acc
        )

        print(f"Current val accuracy: {current_val_acc:.2f}% | Best val accuracy: {best_acc:.2f}%")

        # 早停
        if current_val_acc < best_acc:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            no_improve_epochs = 0

        scheduler.step(current_val_acc)


# 训练结束后，在测试集上评估最佳模型
    print("\nTesting best model on test set:")
    test_acc = model_test(
        model=model,
        dataloader=test_loader,  # 测试集数据加载器
        model_path=model_save_path
    )

    # # ===================== 解耦重构测试 =====================
    # # ===================== 解耦重构测试 =====================
    # # ===================== 解耦重构测试 =====================
    # # ===================== 解耦重构测试 =====================
    test_samples = []

    # 遍历训练集找合适的测试样本
    for sample in full_train_set.samples:
        img_path, label, angle_path = sample
        if angle_path:
            with open(angle_path, 'r') as f:
                angle = float(f.read().strip())
            test_samples.append((img_path, label, angle))

    if len(test_samples) >= 2:
        # 随机选择两个不同方位角的样本 316 1616

        # #  316（HB03941.001.jpeg、 角度: 3.64），1616（HB19867.025.jpeg、角度: 1.24）
        # #  316（HB03941.001.jpeg、 角度: 3.64），1626（HB19878.025.jpeg、角度: 2.29）
        # #  316（HB03941.001.jpeg、 角度: 3.64），1248（HB03935.015.jpeg、角度: 3.98）
        # #  316（HB03941.001.jpeg、 角度: 3.64），890（HB04033.004.jpeg、角度: 5.78）
        sample1, sample2 = test_samples[316], test_samples[890]
        # sample1, sample2 = random.sample(test_samples, 2)
        path1, label1, angle1 = sample1
        path2, label2, angle2 = sample2

        print(f"\n=== 解耦重构测试 ===")
        print(f"身份图像: {os.path.basename(path1)}, 标签: {label1}, 角度: {angle1:.2f}")
        print(f"姿态图像: {os.path.basename(path2)}, 标签: {label2}, 角度: {angle2:.2f}")

        # 加载图像
        img1 = load_image_from_path(path1, transform)
        img2 = load_image_from_path(path2, transform)

        # 加载最佳模型
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # 执行解耦重构
        visualize_disentangle_reconstruction(model, img1, img2, device=device)
        print("解耦重构测试完成！")
    else:
        print("测试样本不足，跳过解耦重构测试")

