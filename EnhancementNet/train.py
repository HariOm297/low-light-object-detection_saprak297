import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EnhancementNet
from dataset_loader import EnhancementDataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "/nlsasfs/home/gpucbh/vyakti22/saprak297/rclone-v1.73.0-linux-amd64/AI_SUMMIT_ENHANCEMENT/inputs"
TARGET_DIR = "/nlsasfs/home/gpucbh/vyakti22/saprak297/rclone-v1.73.0-linux-amd64/AI_SUMMIT_ENHANCEMENT/targets"

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50
SAVE_PATH = "/nlsasfs/home/gpucbh/vyakti22/saprak297/rclone-v1.73.0-linux-amd64/enhancement_net.pth"


dataset = EnhancementDataset(INPUT_DIR, TARGET_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = EnhancementNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.L1Loss()


for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for inp, tgt in progress:
        inp = inp.to(device)
        tgt = tgt.to(device)

        out = model(inp)
        loss = loss_fn(out, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {epoch_loss / len(loader):.4f}")


torch.save(model.state_dict(), SAVE_PATH)
print("✅ Training Finished & Model Saved")
