import time
from calendar import day_abbr
import umap
import cv2
import os
import numpy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
def loss_fn(recon_x, x):
    # 将reduction='sum'改为'reduction='mean''，降低损失量级
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 88*88), x.view(-1, 88*88), reduction='mean')
    return BCE  # 无需除以batch_size（mean已做）
#
#
# class ShapePoseLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Sobel 边缘提取
#         self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
#         self.sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
#
#     def get_edge(self, x):
#         gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
#         gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
#         return torch.sqrt(gx**2 + gy**2 + 1e-8)
#
#     def get_gradient(self, x):
#         # 梯度 = 角度信息
#         return self.get_edge(x)  # 梯度和边缘在SAR里等价表示角度
#
#     def forward(self, recon, shape_img, pose_img):
#         # ========== 核心三损失 ==========
#         recon_loss = F.mse_loss(recon, shape_img)         # 整体图像
#         shape_loss = F.mse_loss(self.get_edge(recon), self.get_edge(shape_img))  # 形状
#         pose_loss = F.mse_loss(self.get_gradient(recon), self.get_gradient(pose_img)) # 角度
#
#         # ========== 最优权重 ==========
#         loss = recon_loss + 1.5 * shape_loss + 1.2 * pose_loss
#         return loss
#


def compute_disentanglement_score(feat_cls, feat_azi):
    """
    计算解缠质量评分 D_i
    输入：
        feat_cls: 类别特征 [B, C]
        feat_azi: 角度特征 [B, A]
    输出：
        D: 每个样本的解缠评分 [B]，越大=越纠缠
    """
    # 拉平
    f_c = feat_cls.flatten(1)
    f_a = feat_azi.flatten(1)

    # 余弦相似度
    norm_c = F.normalize(f_c, dim=1)
    norm_a = F.normalize(f_a, dim=1)
    cos_sim = (norm_c * norm_a).sum(dim=1)

    # 解缠评分：越大越纠缠
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    D = torch.abs(cos_sim)
    return D

def train_model(model, dataloader, opt, loss_fn_k, loss_fa_k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_train_loss = 0.0
    batch_count = 0

    if loss_fn_k is None:
        loss_fn_k = nn.CrossEntropyLoss()
    loss_fn_ik = nn.CrossEntropyLoss()
    # loss_shape = ShapePoseLoss()

    for inputs, labels, angles in dataloader:

        if batch_count == 0:
            print("训练时批次标签范围:", labels.min().item(), labels.max().item())
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        angles = angles.float().to(device)

        # 获得模型所有输出
        feature_class, feature_pose, predict_class, predict_pose, predict_map, conf = model(inputs) # torch.Size([32, 10])

        # 计算解缠得分
        D = compute_disentanglement_score(feature_class, feature_pose)

        # 计算分类损失
        loss1 = loss_fn_k(predict_class, labels) # class predict_class=torch.Size([32, 10]) classification ce-loss

        # 计算姿态损失
        loss2 = loss_fa_k(predict_pose, angles, conf, D)

        # 重构损失
        loss3 = loss_fn(predict_map, inputs) # map angle pseudo_angle=torch.Size([32, 1]) regression

        # ===================== 身份-姿态交换重构训练 =====================
        # loss4 = torch.tensor(0.0, device=device)  # 交换重构损失
        # loss5 = torch.tensor(0.0, device=device)  # 身份保持损失
        #
        #
        # if inputs.size(0) >= 2:
        #     # 创建批次内的身份-姿态交换
        #     batch_size = inputs.size(0)
        #     half_size = batch_size // 2
        #
        #     # 获取前半和后半的特征（使用4D特征进行拼接）
        #     cls_feat1 = feature_class[:half_size].reshape(half_size, 1024, 5, 5)  # torch.Size([64, 25600])
        #     cls_feat2 = feature_class[half_size:2 * half_size].reshape(half_size, 1024, 5, 5)
        #
        #     pose_feat1 = feature_pose[:half_size].reshape(half_size, 1024, 5, 5)
        #     pose_feat2 = feature_pose[half_size:2 * half_size].reshape(half_size, 1024, 5, 5)
        #
        #     # 获取对应的标签
        #     label1 = labels[:half_size]
        #     label2 = labels[half_size:2 * half_size]
        #
        #     # 交换1: 样本1的身份 + 样本2的姿态
        #     x_swap1 = torch.cat([cls_feat1, pose_feat2], dim=1)
        #     # 交换2: 样本2的身份 + 样本1的姿态
        #     x_swap2 = torch.cat([cls_feat2, pose_feat1], dim=1)
        #
        #     # 使用交换后的特征进行重构（使用身份图像的编码器特征）
        #     recon_swap1 = model.decoder(x_swap1)
        #     recon_swap2 = model.decoder(x_swap2)
        #
        #     input1 = inputs[:half_size]
        #     input2 = inputs[half_size:2 * half_size]
        #
        #     loss3_1 = loss_shape(recon_swap1, input1, input2)
        #     loss3_2 = loss_shape(recon_swap2, input2, input1)
        #     loss3 = (loss3_1 + loss3_2) * 0.5
        # else:
        #     loss3 = 0
            # ===================== 姿态保持损失 =====================
            # loss_swap1 = loss_fn(recon_swap1, inputs[:half_size]) * 0.5 + loss_fn(recon_swap1, inputs[half_size:2*half_size]) * 0.5
            # loss_swap2 = loss_fn(recon_swap2, inputs[half_size:2*half_size]) * 0.5 + loss_fn(recon_swap2, inputs[:half_size]) * 0.5
            # loss4 = (loss_swap1 + loss_swap2) * 0.5

            # ===================== 身份保持损失 =====================
            # with torch.enable_grad():
            #     # 对重构图像进行前向传播
            #     _, _, recon1_pred, _, _, _ = model(recon_swap1)
            #     _, _, recon2_pred, _, _, _ = model(recon_swap2)
            #
            #     # 重构后的图像应该被分类为原始身份
            #     loss_identity1 = loss_fn_ik(recon1_pred, label1)  # recon_swap1应该被分类为label1
            #     loss_identity2 = loss_fn_ik(recon2_pred, label2)  # recon_swap2应该被分类为label2
            #     loss5 = (loss_identity1 + loss_identity2) * 0.5

        total_loss = loss1 + loss2 + loss3 #+ loss4 + loss5
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        total_train_loss += total_loss.item()
        batch_count += 1

        # if batch_count % 10 == 0:
        #     pred_probs = F.softmax(predict_class, dim=1).mean(dim=0)  # 类别概率均值
        #     print(f"Batch {batch_count} 类别概率分布: {pred_probs.detach().cpu().numpy()}")

    return total_train_loss / batch_count



def plot_umap_pose_regression(model, dataloader, save_path="./"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_feat_pose = []
    all_angles = []

    with torch.no_grad():
        for inputs, labels, angles in dataloader:
            inputs = inputs.to(device)

            feat_id, feat_pose, pred_cls, pred_pose, pred_map, conf = model(inputs)

            fp = feat_pose.detach().cpu().flatten(start_dim=1).numpy()
            ang = angles.cpu().numpy()  # 关键：必须是原始角度（0~2π）
            all_feat_pose.append(fp)
            all_angles.append(ang)

    X_pose = np.concatenate(all_feat_pose, axis=0)
    A = np.concatenate(all_angles, axis=0)

    # 检查角度范围（关键！）
    print("角度范围：", A.min(), "~", A.max())
    # 如果角度是 -π~π，转成 0~2π
    if A.min() < 0:
        print("检测到负角度，正在转换为 0~2π 范围")
        A = (A + 2 * np.pi) % (2 * np.pi)
    print("转换后角度范围：", A.min(), "~", A.max())

    print("正在计算 UMAP 降维 (姿态特征)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_pose)

    # 绘制回归友好版 UMAP：按角度上色
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=A,  # 颜色 = 真实方位角（回归核心）
        cmap="hsv",  # 环形渐变更适合 0~2π 的角度
        s=12,
        alpha=0.8
    )
    plt.colorbar(scatter, label="True Aspect Angle (rad)")
    plt.title("UMAP Visualization of Pose Features (Colored by Aspect Angle)", fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "umap_pose_by_angle_fixed.png"), dpi=300)
    plt.close()

    print("✅ 修复后的UMAP姿态特征图已保存：umap_pose_by_angle_fixed.png")


def plot_umap_identity_class(model, dataloader, save_path="./"):
    """
    身份特征专用 UMAP 可视化（按目标类别着色，分类友好）
    输出：umap_identity_by_class.png
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_feat_id = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, angles in dataloader:
            inputs = inputs.to(device)

            # 和你模型的 forward 输出保持一致
            feat_id, feat_pose, pred_cls, pred_pose, pred_map, conf = model(inputs)

            # 展平特征向量
            fid = feat_id.detach().cpu().flatten(start_dim=1).numpy()
            lbs = labels.cpu().numpy()

            all_feat_id.append(fid)
            all_labels.append(lbs)

    # 拼接所有数据
    X_id = np.concatenate(all_feat_id, axis=0)
    Y = np.concatenate(all_labels, axis=0)

    print("正在计算 UMAP 降维 (身份特征)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap_id = reducer.fit_transform(X_id)

    # 绘图：按类别着色，显示聚类效果
    plt.figure(figsize=(8, 6))
    for c in np.unique(Y):
        mask = Y == c
        plt.scatter(
            X_umap_id[mask, 0],
            X_umap_id[mask, 1],
            label=f"Class {c}",
            s=12,
            alpha=0.8
        )
    plt.title("UMAP Visualization of Identity Features (Colored by Target Class)", fontsize=12)
    plt.legend(loc="best", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "umap_identity_by_class.png"), dpi=300)
    plt.close()

    print("✅ 身份特征 UMAP 图已保存：umap_identity_by_class.png")

def plot_tsne_features(model, dataloader, save_path="./"):
    """
    分类友好版：身份特征按类别着色
    回归友好版：姿态特征按【真实方位角】着色（连续颜色）
    输出：
    tsne_identity.png   身份特征（分类）
    tsne_pose.png       姿态特征（回归友好，角度着色）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_feat_id = []
    all_feat_pose = []
    all_labels = []
    all_angles = []  # 姿态回归专用：保存真实角度

    with torch.no_grad():
        for inputs, labels, angles in dataloader:
            inputs = inputs.to(device)

            # 完全沿用你的模型输出
            feat_id, feat_pose, pred_cls, pred_pose, pred_map, conf = model(inputs)

            # 展平特征
            fid = feat_id.detach().cpu().flatten(start_dim=1).numpy()
            fpose = feat_pose.detach().cpu().flatten(start_dim=1).numpy()
            lbs = labels.cpu().numpy()
            ang = angles.cpu().numpy()  # 真实角度

            all_feat_id.append(fid)
            all_feat_pose.append(fpose)
            all_labels.append(lbs)
            all_angles.append(ang)

    # 拼接全部数据
    X_id = np.concatenate(all_feat_id, axis=0)
    X_pose = np.concatenate(all_feat_pose, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    A = np.concatenate(all_angles, axis=0)

    print("Computing t-SNE for identity...")
    X_tsne_id = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_id)
    print("Computing t-SNE for pose...")
    X_tsne_pose = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pose)

    # ==========================
    # 图1：身份特征（分类着色）
    # ==========================
    plt.figure(figsize=(8, 6))
    for c in np.unique(Y):
        mask = Y == c
        plt.scatter(X_tsne_id[mask, 0], X_tsne_id[mask, 1], s=12, alpha=0.8, label=f"Class {c}")
    plt.title("t-SNE of Identity Features (Colored by Class)")
    plt.legend(fontsize=7, ncol=2)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tsne_identity.png"), dpi=300)
    plt.close()

    # ==========================
    # 图2：姿态特征（回归友好！按真实方位角着色）
    # ==========================
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne_pose[:, 0], X_tsne_pose[:, 1],
        c=A,  # 按真实方位角上色
        cmap="viridis",  # 连续渐变色（回归专用）
        s=12, alpha=0.8
    )
    plt.colorbar(scatter, label="True Aspect Angle (rad)")  # 颜色条=角度
    plt.title("t-SNE of Pose Features (Colored by True Aspect Angle)")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tsne_pose.png"), dpi=300)
    plt.close()

    print("✅ 回归友好版 t-SNE 已保存完成！")

def calculate_map_overall_accuracy(pred, label):
    """
    计算分割结果的总体准确率（Overall Accuracy）

    参数:
        pred: 模型预测的二值图，形状为 [batch_size, H, W]，值为0或1
        label: 真实分割标签，形状为 [batch_size, H, W] 或 [batch_size, 1, H, W]，值为0或1

    返回:
        accuracy: 总体准确率，范围 [0, 1]
    """
    # 确保标签与预测图形状一致（去除可能的通道维度）
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)  # 从 [B,1,H,W] 变为 [B,H,W]

    # 检查形状是否匹配
    if pred.shape != label.shape:
        raise ValueError(f"预测形状 {pred.shape} 与标签形状 {label.shape} 不匹配")

    # 计算所有像素中预测正确的数量
    correct_pixels = torch.sum(pred == label)

    # 计算总像素数量
    total_pixels = pred.numel()  # 计算张量中元素的总个数

    # 计算总体准确率
    accuracy = correct_pixels.float() / total_pixels

    return accuracy

def model_val(model, dataloader, save_path='best_model.pth', current_best=0.0, current_best_map=0.0):
    """
    模型验证函数，保留验证精度最高的模型参数

    参数:
        model: 待验证的模型
        dataloader: 验证集数据加载器
        save_path: 最佳模型保存路径
        current_best: 当前最佳精度，用于比较是否更新模型

    返回:
        float: 本次验证的精度
        float: 更新后的最佳精度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    # map_correct = 0

    with torch.no_grad():
        for data, target, _ in dataloader:  # 假设数据格式为 (data, target, other_info)
            data, target = data.to(device), target.to(device)
            _, _, predict_cls, _, _, _ = model(data)

            # 获取预测结果（假设输出为分类概率分布）
            pred = predict_cls.max(1, keepdim=True)[1]  # 取概率最大的类别索引
            correct += pred.eq(target.view_as(pred)).sum().item()


    # 计算验证精度
    val_acc = 100. * correct / len(dataloader.dataset)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    # 若当前精度高于历史最佳，保存模型并更新最佳精度
    if val_acc >= current_best:
        print(f"New best model found! Saving to {save_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': val_acc,
            'epoch': getattr(model, 'current_epoch', 'unknown')  # 假设模型有current_epoch属性
        }, save_path)
        current_best = val_acc

    return val_acc, current_best, current_best_map


def model_test(model, dataloader, model_path='best_model.pth'):
    """
    模型测试函数，加载最佳模型并在测试集上评估

    参数:
        model: 测试用的模型结构
        dataloader: 测试集数据加载器
        model_path: 最佳模型参数路径

    返回:
        float: 测试精度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    # 加载最佳模型参数
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target, _ in dataloader:
            data, target = data.to(device), target.to(device)
            _, _, predict_cls, _, _, _ = model(data)
            # 获取预测结果
            pred = predict_cls.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    # 计算测试精度
    test_acc = 100. * correct / total
    print(f"Test Accuracy (using best model): {test_acc:.2f}%")
    print(f"Best validation accuracy: {checkpoint['accuracy']:.2f}%")  # 打印最佳验证精度作为参考

    return test_acc


def visualize_disentangle_reconstruction(model, img_target, img_pose, device="cuda"):
    """
    固定身份 + 更换方位角 → 生成激活图/重构图
    输出和你图2一样的热力图效果！
    """
    model.eval()
    with torch.no_grad():
        # ===================== 1. 处理【身份图像】 =====================
        x = img_target.to(device)
        x1 = model.inc(x)
        x2 = model.down1(x1)
        x3 = model.down2(x2)
        x4 = model.down3(x3)
        x5 = model.down4(x4)

        feature1_target = torch.cat([
            model.avg(x1).repeat(1, 2, 1, 1),
            model.avg(x2),
            model.avg(x3),
            model.avg(x4)
        ], 1).mul(model.avg(x5).repeat(1, 2, 1, 1))

        # 提取 4D 身份特征
        cls_feature = model.cla(feature1_target)  # [1, 512, 5, 5]

        # ===================== 2. 处理【姿态图像】 =====================
        xp = img_pose.to(device)
        x1p = model.inc(xp)
        x2p = model.down1(x1p)
        x3p = model.down2(x2p)
        x4p = model.down3(x3p)
        x5p = model.down4(x4p)

        feature1_pose = torch.cat([
            model.avg(x1p).repeat(1, 2, 1, 1),
            model.avg(x2p),
            model.avg(x3p),
            model.avg(x4p)
        ], 1).mul(model.avg(x5p).repeat(1, 2, 1, 1))

        # 提取 4D 姿态特征
        pose_feature = model.pose(feature1_pose)  # [1, 512, 5, 5]

        x_combined = torch.cat([cls_feature, pose_feature], dim=1)  # [1, 1024, 5, 5]

        # ===================== 4. 解码器重构（完全匹配你的代码！） =====================
        recon = model.decoder(x_combined)  # [1, 1, 88, 88]

    # ===================== 🔥 关键：画成你要的激活图（热力图） =====================
    recon_np = recon.squeeze().cpu().numpy()  # (88,88)

    plt.figure(figsize=(12, 4))

    # 原始身份图
    plt.subplot(1, 3, 1)
    plt.imshow(img_target[0,0].cpu().numpy(), cmap="gray")
    plt.title("Identity Image")
    plt.axis("off")

    # 原始姿态图
    plt.subplot(1, 3, 2)
    plt.imshow(img_pose[0,0].cpu().numpy(), cmap="gray")
    plt.title("Pose Image")
    plt.axis("off")

    # ✅ 重构图：热力图激活图（和你图2完全一样！）
    plt.subplot(1, 3, 3)
    plt.imshow(recon_np, cmap="viridis", interpolation="bilinear")
    plt.title("Reconstructed Activation Map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_result.png", dpi=300)
    plt.show()