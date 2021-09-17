from vin import ViN
from loader import get_trainloader, get_testloader
import argparse
import torch
from utils.drawer import Drawer
import os

parser = argparse.ArgumentParser(description="train food 500")

parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()


def train():
    net = ViN(512, 4, 0.1, 100, 768, 12, 3072)
    net.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=args.lr)
    os.makedirs('./params', exist_ok=True)

    for epoch in range(args.num_epoch):
        total_loss, total_acc = train_epoch(net, epoch, criterion, updater)
        print(f'Epoch: {epoch}, loss: {total_loss},')
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), f'./params/vin{epoch:0>3d}.params')


def train_epoch(net, epoch, criterion, updater):
    dataloader = get_testloader(args.batch_size, num_workers=4, shuffle=True)
    total_loss, total_acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(args.device)
        y = y.to(args.device)
        y_hat = net(x)
        updater.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        updater.step()
        total_loss += loss
        total_acc += accuracy(y_hat, y)
        

        if (batch + 1) % 100 == 0:
            print(f'\t [epoch{epoch} batch:{batch+1}]: loss:{total_loss/args.batch_size/batch}, acc:{total_acc/args.batch_size/batch}')
    return total_loss, total_acc


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    y_hat = y_hat.argmax(1)
    acc = y_hat == y.T
    return acc.sum()


if __name__ == '__main__':
    train()