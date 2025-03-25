import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Set Random Seed
def set_seed(seed=77):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Parameters
BATCH_SIZE = 64
EPOCHS = 35
PATIENCE = 10
LEARNING_RATE = 1e-4
LEARNING_RATE2 = 1e-3
NAMING = "ResNeXt101_v4"

data_transforms = {
    'train': v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomRotation(degrees=20),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': v2.Compose([
        v2.Resize(360),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
cutmix = v2.CutMix(num_classes=100, alpha=1.0)


class myDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def load_dataset(path='data/train/'):
    images = []
    labels = []
    for i in range(100):
        folder_path = os.path.join(path, str(i))
        for file in os.listdir(folder_path):
            images.append(os.path.join(folder_path, file))
            labels.append(i)
    return images, labels


train_images, train_labels = load_dataset("data/train")
val_images, val_labels = load_dataset("data/val")
train_dataset = myDataset(train_images, train_labels,
                          transform=data_transforms['train'])
val_dataset = myDataset(val_images, val_labels,
                        transform=data_transforms['val'])


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.resnext = models.resnext101_64x4d(pretrained=True)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 100)

    def get_parameter_size(self):
        print('On GPU', os.environ["CUDA_VISIBLE_DEVICES"])
        print('NAMING:', NAMING)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x):
        return self.resnext(x)


def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (inputs, labels) in bar:
        inputs, labels = cutmix(inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        bar.set_description(f"Training Loss: {loss.item():.5f}")

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for i, (inputs, labels) in bar:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            bar.set_description(f"Validate Loss: {loss.item():.5f}")

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(seed=77):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE, shuffle=False)

    model = ResNeXt().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    layer_params = [param for name, param in model.named_parameters()
                    if param.requires_grad and "fc" not in name]
    fc_params = [param for name, param in model.named_parameters()
                 if param.requires_grad and "fc" in name]

    optimizer = optim.AdamW([
        {'params': layer_params, 'lr': LEARNING_RATE},
        {'params': fc_params, 'lr': LEARNING_RATE2},
    ], weight_decay=1e-5)

    model.get_parameter_size()

    no_improvement_epochs = 0
    train_losses = []
    val_losses = []
    max_acc = 0

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion,
                           optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Acc:  {val_acc:.4f}")

        no_improvement_epochs += 1
        if val_acc > max_acc:
            print(f"Saving model, Best Accuracy: {val_acc:.4f}")
            os.makedirs('soup', exist_ok=True)
            torch.save(model.state_dict(),
                       f'soup/model_{NAMING}_seed{seed}.pth')
            max_acc = val_acc
            no_improvement_epochs = 0

        if no_improvement_epochs >= PATIENCE:
            print("Early stopping")
            break

    print(f"Best Accuracy: {max_acc:.4f}")

    # plot
    Epochs = range(epoch + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(Epochs, train_losses, label='Training Loss')
    plt.plot(Epochs, val_losses, label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plot_{NAMING}.png')


if __name__ == "__main__":
    train_model(seed=77)
    # train_model(seed=156)
    # train_model(seed=514)
    # train_model(seed=7810)
    # train_model(seed=672)
    # train_model(seed=379)
    # train_model(seed=668)
    # train_model(seed=67)
    # train_model(seed=78)
    # train_model(seed=74)
    # train_model(seed=102)
    # train_model(seed=132)
    # train_model(seed=158)
    # train_model(seed=140)
    # train_model(seed=192)
    # train_model(seed=33363)
    # train_model(seed=960)
    # train_model(seed=373)
    # train_model(seed=694)
    # train_model(seed=9996)
    # train_model(seed=1025)
    # train_model(seed=5751)
