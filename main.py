import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.loader import get_loaders
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from Configure import config
from Model.Googlenet import GoogleNet
from utils.kaiming import init_weights
from torch.utils.tensorboard import SummaryWriter

# 与 dataset/dataset.py 中 transforms.Normalize 使用的参数一致
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# =========================================================
# Activation Probe
# =========================================================

activations = {}


def save_activation(name):
    def hook(module, inputs, output):
        activations[name] = output.detach()

    return hook


# =========================================================
# Seed
# =========================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Grad-CAM visualization helpers
# =========================================================


def denormalize_cifar10(images):
    """把送入模型的标准化张量恢复为 TensorBoard 可显示的 RGB 图像。"""

    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)

    return (images * std + mean).clamp(0.0, 1.0)


def colorize_gradcam(cam):
    """把 [B,1,H,W]、范围 0~1 的 CAM 转成蓝-青-黄-红热力图。"""

    value = cam.clamp(0.0, 1.0)

    red = (1.5 - torch.abs(4.0 * value - 3.0)).clamp(0.0, 1.0)
    green = (1.5 - torch.abs(4.0 * value - 2.0)).clamp(0.0, 1.0)
    blue = (1.5 - torch.abs(4.0 * value - 1.0)).clamp(0.0, 1.0)

    return torch.cat([red, green, blue], dim=1)


def make_gradcam_visualizations(images, cam):
    """生成原图、彩色热力图和映射到原图相同位置的叠加图。"""

    original_images = denormalize_cifar10(images)
    heatmap_images = colorize_gradcam(cam)

    # CAM 越强，彩色热力图越明显；CAM 接近 0 时尽量保留原图。
    overlay_alpha = 0.45 * cam.clamp(0.0, 1.0)
    overlay_images = (
        (1.0 - overlay_alpha) * original_images + overlay_alpha * heatmap_images
    ).clamp(0.0, 1.0)

    return (
        original_images.detach(),
        heatmap_images.detach(),
        overlay_images.detach(),
    )


# =========================================================
# Grad-CAM
# =========================================================


def generate_gradcam(model, images):

    gradcam_data = {}

    # -------------------------
    # 只临时监听 inception5b
    # -------------------------
    def forward_hook(module, inputs, output):
        gradcam_data["activation"] = output

        def save_gradient(grad):
            gradcam_data["gradient"] = grad

        output.register_hook(save_gradient)

    handle = model.inception5b.register_forward_hook(forward_hook)

    model.eval()

    # 注意：
    # Grad-CAM 绝对不能放在 torch.no_grad() 里面
    logits = model(images)

    # [B, 10]
    pred_classes = logits.argmax(dim=1)

    # 每张图片取自己的预测类别 logit
    scores = logits[torch.arange(logits.size(0), device=images.device), pred_classes]

    model.zero_grad()

    # 所有图片的目标 score 一次 backward
    scores.sum().backward()

    # -------------------------
    # 获取 5b activation
    # 和对应 gradient
    # -------------------------

    activation = gradcam_data["activation"]
    # [B, 1024, 8, 8]

    gradient = gradcam_data["gradient"]
    # [B, 1024, 8, 8]

    # -------------------------
    # 每张 feature map 的重要性
    # -------------------------

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    # [B, 1024, 1, 1]

    # -------------------------
    # 权重乘回 feature maps
    # 然后沿 channel 求和
    # -------------------------

    cam = (weights * activation).sum(dim=1)
    # [B, 8, 8]

    # 只保留正向支持该类别的区域
    cam = torch.relu(cam)

    # -------------------------
    # 每张图片单独归一化到 0~1
    # -------------------------

    cam_min = cam.amin(dim=(1, 2), keepdim=True)

    cam_max = cam.amax(dim=(1, 2), keepdim=True)

    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    # -------------------------
    # 8x8 -> 32x32
    # -------------------------

    cam = F.interpolate(
        cam.unsqueeze(1),
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    # [B, 1, 32, 32]

    # 用完立即移除 Grad-CAM hook
    handle.remove()

    return (
        cam.detach(),
        pred_classes.detach(),
    )


# =========================================================
# Main
# =========================================================


def main():

    writer = SummaryWriter(log_dir="runs/googlenet_cifar10")

    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # =====================================================
    # Dataset
    #
    # 当前阶段：
    # train -> 训练
    # val   -> 选择超参数
    # test  -> 暂时不要碰
    # =====================================================

    train_loader, val_loader, _ = get_loaders(batch_size=config.BATCH_SIZE)

    # =====================================================
    # Model
    # =====================================================

    model = GoogleNet().to(device)

    model.apply(init_weights)

    # =====================================================
    # Activation hooks
    #
    # 这些 hook 可以一直挂着，因为只 detach 后保存输出
    # =====================================================

    model.inception3b.register_forward_hook(save_activation("3b"))

    model.inception4a.register_forward_hook(save_activation("4a"))

    model.inception4d.register_forward_hook(save_activation("4d"))

    model.inception5b.register_forward_hook(save_activation("5b"))

    # =====================================================
    # 固定一批 validation 图片
    #
    # 整个训练过程都观察完全相同的输入
    # =====================================================

    probe_images, probe_targets = next(iter(val_loader))

    probe_images = probe_images.to(device)

    # =====================================================
    # Grad-CAM 固定图片
    #
    # 暂时直接取 probe batch 前 8 张
    # =====================================================

    cam_images = probe_images[:8]

    cam_targets = probe_targets[:8].to(device)

    # =====================================================
    # Loss + Optimizer
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_top1 = 0.0

    # 只在这些 epoch 做 Grad-CAM
    # epoch 从 0 开始
    gradcam_epochs = [0, 9, 15, 20, 29]

    # =====================================================
    # Train
    # =====================================================

    for epoch in range(config.EPOCHS):

        # =================================================
        # Train
        # =================================================

        (
            train_loss,
            train_top1,
            train_top5,
            epoch_main_loss,
            epoch_aux1_loss,
            epoch_aux2_loss,
        ) = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            topk=config.TOPK,
        )

        # =================================================
        # Validation
        # =================================================

        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            topk=config.TOPK,
        )

        # =================================================
        # TensorBoard：Loss
        # =================================================

        writer.add_scalar(
            "Loss/train_total",
            train_loss,
            epoch,
        )

        writer.add_scalar(
            "Loss/train_aux1",
            epoch_aux1_loss,
            epoch,
        )

        writer.add_scalar(
            "Loss/train_aux2",
            epoch_aux2_loss,
            epoch,
        )

        writer.add_scalars(
            "Loss/main",
            {
                "train": epoch_main_loss,
                "validation": val_loss,
            },
            epoch,
        )

        # =================================================
        # TensorBoard：Accuracy
        # =================================================

        writer.add_scalars(
            "Accuracy/Top1",
            {
                "train": train_top1,
                "validation": val_top1,
            },
            epoch,
        )

        writer.add_scalar(
            "Optimization/learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        # =================================================
        # Console
        # =================================================

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Top1: {train_top1 * 100:.2f}% "
            f"Train Top5: {train_top5 * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Top1: {val_top1 * 100:.2f}% "
            f"Val Top5: {val_top5 * 100:.2f}%"
        )

        # =================================================
        # Best checkpoint
        # =================================================

        if val_top1 > best_val_top1:

            best_val_top1 = val_top1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "weight_decay": config.WEIGHT_DECAY,
                    "lr": config.LR,
                    "epochs": epoch + 1,
                    "val_top1": val_top1,
                },
                "googlenet_cifar10_best.pth",
            )

        # =================================================
        # 固定 Probe Batch
        #
        # 观察：
        # ActivationStd
        # SampleVariation
        # =================================================

        model.eval()

        with torch.no_grad():
            _ = model(probe_images)

        for name in [
            "3b",
            "4a",
            "4d",
            "5b",
        ]:

            x = activations[name]
            # [B,C,H,W]

            # ---------------------------------------------
            # 1. Activation 数值尺度
            # ---------------------------------------------

            writer.add_scalar(
                f"ActivationStd/{name}",
                x.std().item(),
                epoch,
            )

            # ---------------------------------------------
            # 2. 每张图压成 C 维 representation
            # ---------------------------------------------

            feature = x.mean(dim=(2, 3))
            # [B,C]

            # ---------------------------------------------
            # 3. 不同样本在每个 channel 上的变化
            # ---------------------------------------------

            sample_std = feature.std(dim=0)
            # [C]

            sample_variation = sample_std.mean()

            writer.add_scalar(
                f"SampleVariation/{name}",
                sample_variation.item(),
                epoch,
            )

        # =================================================
        # Grad-CAM
        #
        # 固定 8 张图
        # 只在部分 epoch 做
        # =================================================

        if epoch in gradcam_epochs:

            cam, pred_classes = generate_gradcam(
                model,
                cam_images,
            )

            (
                original_images,
                heatmap_images,
                overlay_images,
            ) = make_gradcam_visualizations(
                cam_images,
                cam,
            )

            # 三组图的样本顺序完全一致：
            # 第 i 张 Original、Raw_Heatmap、Overlay 都对应同一输入。
            writer.add_images(
                "GradCAM/Original",
                original_images.cpu(),
                epoch,
            )

            # 这是 inception5b Grad-CAM 的原始 0~1 响应图。
            writer.add_images(
                "GradCAM/Raw_Heatmap",
                cam.cpu(),
                epoch,
            )

            # 彩色响应图按相同空间位置覆盖到反归一化后的原图。
            writer.add_images(
                "GradCAM/Overlay",
                overlay_images.cpu(),
                epoch,
            )

            writer.flush()

            # 打印 true / predicted
            print(f"\nGrad-CAM Epoch {epoch + 1}")

            for i in range(cam_images.size(0)):
                print(
                    f"Image {i}: "
                    f"true={cam_targets[i].item()} "
                    f"pred={pred_classes[i].item()}"
                )

    # =====================================================
    # Finish
    # =====================================================

    print(f"Best Validation Top1: " f"{best_val_top1 * 100:.2f}%")

    writer.close()


if __name__ == "__main__":
    main()
