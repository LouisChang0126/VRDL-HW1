import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
import os
from tqdm import tqdm
import PIL
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

BATCH_SIZE = 64
SEEDS = [1025, 102, 132, 140, 156, 158, 192, 33363, 373,
         379, 514, 5751, 668, 672, 67, 694, 74, 77, 7810, 78, 960, 9996]

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


def load_dataset(path='data/val/'):
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
        self.resnext = models.resnext101_64x4d(pretrained=False)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 100)

    def forward(self, x):
        return self.resnext(x)


def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def greedy_soup_with_ratios(model_paths, val_loader, output_path,
                            device='cuda', ratios=[0.25, 0.5, 0.75]):
    criterion = nn.CrossEntropyLoss()

    state_dicts = []
    for path in model_paths:
        temp_model = ResNeXt().to(device)
        temp_model = load_model_weights(temp_model, path, device)
        state_dicts.append(temp_model.state_dict())

    best_acc = -1
    best_model_idx = 0
    for idx, path in enumerate(model_paths):
        model = ResNeXt().to(device)
        model = load_model_weights(model, path, device)
        _, acc = validate(model, val_loader, criterion, device)
        print(f"Model {idx+1} ({path}): Validation Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_idx = idx

    soup_model = ResNeXt().to(device)
    soup_model.load_state_dict(state_dicts[best_model_idx])
    soup_state_dict = state_dicts[best_model_idx].copy()
    soup_size = 1
    current_acc = best_acc

    print(f"Starting with best model {best_model_idx+1} \
          (Accuracy: {best_acc:.4f})")

    remaining_indices = [i for i in range(len(model_paths))
                         if i != best_model_idx]
    for idx in remaining_indices:
        print(f"\nTrying to add model {idx+1} \
              ({model_paths[idx]}) with multiple ratios:")

        best_temp_state_dict = soup_state_dict
        best_new_acc = current_acc
        best_ratio = None

        for ratio in ratios:
            new_model_weight = ratio
            soup_weight = 1.0 - new_model_weight

            temp_state_dict = soup_state_dict.copy()
            for key in temp_state_dict.keys():
                temp_state_dict[key] = soup_weight * soup_state_dict[key]
                + new_model_weight * state_dicts[idx][key]

            temp_model = ResNeXt().to(device)
            temp_model.load_state_dict(temp_state_dict)
            _, new_acc = validate(temp_model, val_loader, criterion, device)

            print(f"Ratio {soup_weight:.2f} (Soup) + {new_model_weight:.2f} \
                  (New Model): Accuracy = {new_acc:.4f}")

            if new_acc > best_new_acc:
                best_new_acc = new_acc
                best_temp_state_dict = temp_state_dict.copy()
                best_ratio = new_model_weight

        if best_new_acc > current_acc:
            soup_state_dict = best_temp_state_dict
            soup_size += 1
            current_acc = best_new_acc
            print(f"Added model {idx+1} with ratio {best_ratio:.2f}. \
                  New best accuracy: {current_acc:.4f}")
        else:
            print(f"Model {idx+1} discarded (no improvement with any ratio).")

    soup_model.load_state_dict(soup_state_dict)
    torch.save(soup_model.state_dict(), output_path)
    print(f"\nGreedy Soup completed with {soup_size} models. \
          Best Accuracy: {current_acc:.4f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    val_images, val_labels = load_dataset("data/val")
    val_dataset = myDataset(val_images, val_labels,
                            transform=data_transforms['val'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_paths = [f'soup/model_ResNeXt101_v4_seed{seed}.pth'
                   for seed in SEEDS]
    output_path = 'greedy_soup_ratio.pth'

    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    greedy_soup_with_ratios(model_paths, val_loader,
                            output_path, device, ratios)
