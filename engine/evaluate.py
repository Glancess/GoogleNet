import torch

from utils.metric import accuracy


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    criterion,
    device,
    topk,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss = criterion(outputs, targets)

        top1, top5 = accuracy(
            outputs,
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
