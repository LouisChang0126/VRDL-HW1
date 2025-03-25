import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
import os
import numpy as np
import PIL
from tqdm import tqdm
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Parameters
RESIZE_SIZE = 360
MODEL_NAME = "greedy_soup_ratio.pth"

# Set Random Seed
SEED = 77
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_transforms = v2.Compose([
    v2.Resize(RESIZE_SIZE),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, image_name


def load_test_images(path='data/test/'):
    images = [os.path.join(path, file) for file in os.listdir(path)]
    return images


test_images = load_test_images("data/test")
test_dataset = TestDataset(test_images, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.resnext = models.resnext101_64x4d(pretrained=True)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 100)

    def forward(self, x):
        return self.resnext(x)


# run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNeXt().to(device)
model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
model.eval()

results = []

with torch.no_grad():
    for images, names in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for name, pred in zip(names, predicted.cpu().numpy()):
            results.append([os.path.basename(name), pred])

output_file = "prediction.csv"
df = pd.DataFrame(results, columns=["image_name", "pred_label"])
df.to_csv(output_file, index=False)

print(f"Testing complete. Results saved in {output_file}")
