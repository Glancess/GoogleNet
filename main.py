import random
import numpy as np
import torch
import torch.nn as nn
from dataset.loader import get_loaders
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from Configure import config
from Model.Googlenet import GoogleNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # 当前阶段：
    # train 用来训练
    # val 用来选择超参数
    # test 暂时不要碰
    train_loader, val_loader, _ = get_loaders(batch_size=config.BATCH_SIZE)

    model = GoogleNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_top1 = 0.0

    for epoch in range(config.EPOCHS):

        # =========================
        # Train
        # =========================
        train_loss, train_top1, train_top5 = train_one_epoch(
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

    print(f"Best Validation Top1: " f"{best_val_top1 * 100:.2f}%")


if __name__ == "__main__":
    main()
