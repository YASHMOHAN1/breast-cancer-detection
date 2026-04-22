import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform

        for label_name in ["benign", "malignant"]:
            folder = os.path.join(root_dir, label_name)

            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".png") or file.endswith(".jpg"):
                        path = os.path.join(root, file)

                        label = 0 if label_name == "benign" else 1

                        self.data.append((path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label