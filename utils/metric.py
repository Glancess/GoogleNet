import torch


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    output: [B, C]，模型输出的 logits
    target: [B]，真实类别
    topk: 例如 (1,) 或 (1, 5)
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # 转置

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        acc = correct_k / batch_size
        result.append(acc)

    return result
