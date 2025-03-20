"""CNN-IMG: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ShardPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
)

from flwr.common.typing import UserConfig
from datetime import datetime
from pathlib import Path
import json

from tqdm import tqdm

"""
# Cifar-10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)ShardPartitioner

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 20)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)

        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout1(x)
Resize
        # Flatten and fully connected layers
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
"""


# Cifar-100
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Fifth convolutional block (thêm mới)
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Block 5
        x = F.relu(self.bn5(self.conv5(x)))

        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(-1, 512)

        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR100 data with augmentation."""
    global fds
    if fds is None:
        # partitioner = DirichletPartitioner(
        #     num_partitions=num_partitions,
        #     partition_by="coarse_label",
        #     alpha=0.3,
        #     seed=42,
        #     min_partition_size=50,
        # )

        # partitioner = ShardPartitioner(
        #     num_partitions=num_partitions, partition_by="label", num_shards_per_partition=6
        # )

        partitioner = IidPartitioner(partition_id, num_partitions)

        fds = FederatedDataset(
            dataset="sarath2003/BreakHis",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Tăng cường dữ liệu cho tập huấn luyện
    train_transforms = Compose(
        [   Resize((400, 700)),
            ToTensor(),
            # Thêm các phép biến đổi để tăng cường dữ liệu
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),  # Giá trị mean và std chính xác cho CIFAR-100
        ]
    )

    # Biến đổi cho tập kiểm tra
    test_transforms = Compose(
        [Resize((400, 700)),ToTensor(), Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
    )

    def apply_train_transforms(batch):
        batch["image"] = [train_transforms(img) for img in batch["image"]]
        return batch

    def apply_test_transforms(batch):
        batch["image"] = [test_transforms(img) for img in batch["image"]]
        return batch

    train_data = partition_train_test["train"].with_transform(apply_train_transforms)
    test_data = partition_train_test["test"].with_transform(apply_test_transforms)

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device, lr=0.01):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    net.train()
    running_loss = 0.0
    for _ in tqdm(range(epochs)):
        # print(f"Epoch{_}")
        epoch_loss = 0.0
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        running_loss += epoch_loss

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    print(f"Average training loss: {avg_trainloss}")
    return avg_trainloss


"""
def test(net, testloader, device):
    # Validate the model on the test set with detailed metrics.
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate overall metrics
    avg_loss = test_loss / len(testloader)
    accuracy = correct / total
    
    print(f'Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}')
    
    # Print per-class accuracy
    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return avg_loss, accuracy
"""


def test(net, testloader, device):
    """Validate the model on the test set with detailed metrics."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 8
    class_total = [0] * 8

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy - fixed to handle single-sample batches
            c = predicted == labels  # Don't squeeze
            for i in range(len(labels)):
                label = labels[i].item()  # Convert tensor to Python number
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate overall metrics
    avg_loss = test_loss / len(testloader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    # Print per-class accuracy
    for i in range(8):
        if class_total[i] > 0:
            print(
                f"Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%"
            )

    return avg_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


if __name__ == "__main__":
    trainloader, testloader = load_data(0, 1)
    net = Net()
    train(net, trainloader, 100, torch.device("cuda:0"))
    test(net, testloader, torch.device("cuda:0"))
    print("Success!")
