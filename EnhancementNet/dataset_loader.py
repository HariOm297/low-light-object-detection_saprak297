import os, cv2, torch
from torch.utils.data import Dataset

class EnhancementDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.images = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        inp = cv2.imread(os.path.join(self.input_dir, name))
        tgt = cv2.imread(os.path.join(self.target_dir, name))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB) / 255.0
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB) / 255.0

        inp = torch.tensor(inp).permute(2,0,1).float()
        tgt = torch.tensor(tgt).permute(2,0,1).float()

        return inp, tgt