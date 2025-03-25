import torch
import torch.nn as nn
from torchvision import models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# parameters
SEEDS = [1025, 102, 132, 140, 156, 158, 192, 33363, 373,
         379, 514, 5751, 668, 672, 67, 694, 74, 77, 7810, 78, 960, 9996]


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.resnext = models.resnext101_64x4d(pretrained=False)
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 100)

    def forward(self, x):
        return self.resnext(x)


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def model_soup(model_paths, output_path, device='cuda'):
    base_model = ResNeXt().to(device)

    base_model = load_model_weights(base_model, model_paths[0], device)
    avg_state_dict = base_model.state_dict()

    num_models = len(model_paths)
    for i, model_path in enumerate(model_paths[1:], 1):
        print(f"Averaging model {i+1}/{num_models}: {model_path}")
        temp_model = ResNeXt().to(device)
        temp_model = load_model_weights(temp_model, model_path, device)
        temp_state_dict = temp_model.state_dict()

        for key in avg_state_dict.keys():
            avg_state_dict[key] += temp_state_dict[key]

    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / num_models

    base_model.load_state_dict(avg_state_dict)

    torch.save(base_model.state_dict(), output_path)
    print(f"Model Soup saved to {output_path}")


if __name__ == "__main__":
    model_paths = [f'soup/model_ResNeXt101_v4_seed{seed}.pth'
                   for seed in SEEDS]
    output_path = 'model_soup.pth'

    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_soup(model_paths, output_path, device)

    print("Model Soup completed successfully!")
