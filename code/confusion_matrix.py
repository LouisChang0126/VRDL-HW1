import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import models
import os
import PIL
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NAMING = "soup/model_ResNeXt101_v4_seed77.pth"

val_transform = v2.Compose([
    v2.Resize(360),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class myDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_dataset(path='data/val/'):
    images = []
    labels = []
    for i in range(100):
        folder_path = os.path.join(path, str(i))
        for file in os.listdir(folder_path):
            images.append(os.path.join(folder_path, file))
            labels.append(i)
    return images, labels


val_images, val_labels = load_dataset("data/val")
val_dataset = myDataset(val_images, val_labels, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.resnext = models.resnext101_64x4d(pretrained=True)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 100)

    def forward(self, x):
        return self.resnext(x)


model = ResNeXt().to(device)
model.load_state_dict(torch.load(f'{NAMING}', map_location=device))
model.eval()


def compute_confusion_matrix(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader,
                                   desc='Computing Confusion Matrix'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(25, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')


cm = compute_confusion_matrix(model, val_loader, device)
plot_confusion_matrix(cm, classes=[str(i) for i in range(100)])
