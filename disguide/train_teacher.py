# Code to train a new classifier to use as teacher/victim from scratch.

import argparse
import torch
import torch.optim as optim
import network

from dataloader import get_dataloader


def test(model, test_loader, device):
    """Test model on dataset. Returns mean top 1 accuracy"""
    model.eval()
    correct = 0
    count = 0
    for (data, target) in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += data.shape[0]
    return (1.0 * correct) / count


def train_epoch(model, dataloader, optimizer, device, loss_fn=torch.nn.CrossEntropyLoss()):
    """Train model for one epoch on training dataset"""
    model.train()
    for idx, (img, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = img.to(device), labels.to(device)
        preds = model(data)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()


def train(model, train_loader, optimizer, scheduler, test_loader, args):
    """Outer training loop. Train model for args.epoch epochs on training dataset and incrementally post information"""
    best_acc = 0
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, args.device)
        scheduler.step()
        accuracy = test(model, test_loader, args.device)

        if accuracy > best_acc:
            torch.save(model.state_dict(), f"checkpoint/teacher/{args.dataset}-resnet18_8x_{args.suffix}.pt")
            best_acc = accuracy
        print(f"Epoch {epoch} accuracy: {accuracy}")
        if accuracy >= args.max_accuracy:
            return


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Training Script for Teacher Models")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-accuracy', type=float, default=0.785)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--steps', nargs='+', default=[0.25, 0.5, 0.75], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['svhn', 'cifar10', 'mnist', 'cifar100'],
                        help='dataset name (default: cifar100)')

    args = parser.parse_args()
    args.input_space = "post-transform"
    print("Running training script for fresh teacher model")

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    print(f"Device is {args.device}")

    steps = sorted([int(step * args.epochs) for step in args.steps])
    print("Learning rate scheduling at steps: {}\n".format(steps))

    args.num_classes = 10 if args.dataset in ['cifar10', 'svhn', 'mnist'] else 100
    model = network.resnet_8x.ResNet18_8x(num_classes=args.num_classes)
    model = model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.2)

    train_loader, test_loader, identity = get_dataloader(args)
    assert isinstance(identity, torch.nn.Identity)

    train(model, train_loader, optimizer, scheduler, test_loader, args)


if __name__ == '__main__':
    print("torch version", torch.__version__)
    main()
