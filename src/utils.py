import torch
from typing import Tuple


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """バッチ単位の分類精度を返す（%）"""
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / targets.size(0)


def train_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += (outputs.argmax(dim=1) == targets).sum().item()
        n += bs

    epoch_loss = running_loss / n
    epoch_acc = 100.0 * running_acc / n
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, targets)

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += (outputs.argmax(dim=1) == targets).sum().item()
        n += bs

    epoch_loss = running_loss / n
    epoch_acc = 100.0 * running_acc / n
    return epoch_loss, epoch_acc


def save_checkpoint(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer = None, epoch: int = None):
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch
    torch.save(data, path)
