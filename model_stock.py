import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import timm
import os
import numpy as np
import PIL
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Set Random Seed
SEED = 77
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
BATCH_SIZE = 64

data_transforms = {
    'val': v2.Compose([
        v2.Resize(360),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


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


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.resnext = timm.create_model('resnext101_64x4d',
                                         pretrained=True, num_classes=100)

    def get_parameter_size(self):
        print('On GPU', os.environ["CUDA_VISIBLE_DEVICES"])
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x):
        return self.resnext(x)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for i, data in bar:
            inputs, labels = data
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


def compute_model_stock_weights(w0, w1, w2):
    w_h = {}
    for key in w0.keys():
        w1_vec = w1[key].flatten().to('cpu').float()
        w2_vec = w2[key].flatten().to('cpu').float()

        cos_theta = torch.dot(w1_vec, w2_vec) / (torch.norm(w1_vec)
                                                 * torch.norm(w2_vec))
        cos_theta = cos_theta.clamp(-1, 1)

        t = (2 * cos_theta) / (1 + cos_theta)

        w_12 = (w1[key] + w2[key]) / 2
        w_h[key] = t * w_12 + (1 - t) * w0[key]

    return w_h


# run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_images, val_labels = load_dataset("data/val")
val_dataset = myDataset(val_images, val_labels,
                        transform=data_transforms['val'])

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model_pretrain = ResNeXt().to(device)

criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler()

w0 = model_pretrain.state_dict()
w1 = torch.load('soup/model_ResNeXt101_v4_seed74.pth')
w2 = torch.load('soup/model_ResNeXt101_v4_seed960.pth')

w_h = compute_model_stock_weights(w0, w1, w2)

model_stock = ResNeXt().to(device)
model_stock.load_state_dict(w_h)

val_loss, val_acc = validate(model_stock, val_loader, criterion, device)
print(f"Model Stock - Validation Loss: {val_loss:.4f}, \
      Validation Accuracy: {val_acc:.4f}")

torch.save(model_stock.state_dict(), 'model_stock.pth')
print(f"Model Stock Accuracy: {val_acc:.4f}")
