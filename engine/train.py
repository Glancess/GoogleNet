from utils.metric import accuracy


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    topk,
):
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        main, aux1, aux2 = model(images)
        loss_main = criterion(main, targets)
        loss_aux1 = criterion(aux1, targets)
        loss_aux2 = criterion(aux2, targets)

        loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
        print(
            f"main={loss_main.item():.4f}, "
            f"aux1={loss_aux1.item():.4f}, "
            f"aux2={loss_aux2.item():.4f}, "
            f"total={loss.item():.4f}"
        )
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(
            main,
            targets,
            topk=topk,
        )

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_top1 = total_top1 / total_samples
    epoch_top5 = total_top5 / total_samples

    return epoch_loss, epoch_top1, epoch_top5
