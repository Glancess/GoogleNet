import random
import numpy as np
import torch
import torch.nn as nn
from dataset.loader import get_loaders
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from Configure import config
from Model.Googlenet import GoogleNet
from utils.kaiming import init_weights
from torch.utils.tensorboard import SummaryWriter

activations = {}


def save_activation(name):
    def hook(module, inputs, output):
        activations[name] = output.detach()

    return hook


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    writer = SummaryWriter(log_dir="runs/googlenet_cifar10")
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # 当前阶段：
    # train 用来训练
    # val 用来选择超参数
    # test 暂时不要碰
    train_loader, val_loader, _ = get_loaders(batch_size=config.BATCH_SIZE)

    model = GoogleNet().to(device)
    model.apply(init_weights)

    model.inception3b.register_forward_hook(save_activation("3b"))
    model.inception4a.register_forward_hook(save_activation("4a"))
    model.inception4d.register_forward_hook(save_activation("4d"))
    model.inception5b.register_forward_hook(save_activation("5b"))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_top1 = 0.0
    # 探针！
    probe_images, probe_targets = next(iter(val_loader))
    probe_images = probe_images.to(device)
    for epoch in range(config.EPOCHS):

        # =========================
        # Train
        # =========================
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

        # =========================
        # Validation
        # =========================
        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            topk=config.TOPK,
        )
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/train_aux1", epoch_aux1_loss, epoch)
        writer.add_scalar("Loss/train_aux2", epoch_aux2_loss, epoch)

        writer.add_scalars(
            "Loss/main",
            {
                "train": epoch_main_loss,
                "validation": val_loss,
            },
            epoch,
        )

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
        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Top1: {train_top1 * 100:.2f}% "
            f"Train Top5: {train_top5 * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Top1: {val_top1 * 100:.2f}% "
            f"Val Top5: {val_top5 * 100:.2f}%"
        )

        # 用 validation Top1 选择当前 run 的最佳模型
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
        # =========================
        # 固定 probe batch
        # =========================
        model.eval()

        with torch.no_grad():
            _ = model(probe_images)

        for name in ["3b", "4a", "4d", "5b"]:
            x = activations[name]

            writer.add_scalar(
                f"ActivationStd/{name}",
                x.std().item(),
                epoch,
            )

    print(f"Best Validation Top1: " f"{best_val_top1 * 100:.2f}%")


if __name__ == "__main__":
    main()
