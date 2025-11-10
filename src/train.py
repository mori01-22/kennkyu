import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

from model import SimpleCNN
from utils import train_epoch, evaluate, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Simple CIFAR-10 training script")
    p.add_argument("--data-dir", type=str, default="./data", help="dataset root")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out, exist_ok=True)

    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"  train_loss: {train_loss:.4f}  train_acc: {train_acc:.2f}%")
        print(f"  val_loss:   {val_loss:.4f}  val_acc:   {val_acc:.2f}%")

        # 保存
        ckpt_path = os.path.join(args.out, f"model_epoch{epoch}.pt")
        save_checkpoint(model, ckpt_path, optimizer=optimizer, epoch=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.out, "best_model.pt")
            save_checkpoint(model, best_path, optimizer=optimizer, epoch=epoch)
            print(f"  New best model saved (acc={best_acc:.2f}%)")

    print("Training finished.")


if __name__ == "__main__":
    main()
