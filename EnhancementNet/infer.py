import torch
import cv2
import os
from model import EnhancementNet
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
INPUT_DIR = "/home/saprak297/AI_SUMMIT_ENHANCEMENT/inputs"
OUTPUT_DIR = "/home/saprak297/AI_SUMMIT_ENHANCEMENT/enhanced_images"
MODEL_PATH = "enhancement_model.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = EnhancementNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

for name in tqdm(os.listdir(INPUT_DIR)):
    img_path = os.path.join(INPUT_DIR, name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    inp = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(inp)

    out = out.squeeze().permute(1,2,0).cpu().numpy()
    out = (out * 255).astype("uint8")

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), out)

print("✅ Inference completed, images saved.")