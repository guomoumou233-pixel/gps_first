# file: train_distill_remoteclip_FINAL_WORKING.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
from tqdm import tqdm
from transformers import CLIPTokenizer
from sklearn.model_selection import train_test_split

# --------------------------- 你的学生模型 ---------------------------
from tiny_student_model import LightweightStudentCLIP

# --------------------------- 路径 ---------------------------
TEACHER_CHECKPOINT_PATH = "/root/mnist-clip/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = "/root/mnist-clip/data/RSICD_images"
JSON_PATH = "/root/mnist-clip/data/RSICD-en_cleaned.json"

# --------------------------- 数据集（返回 PIL） ---------------------------
class RSICDDataset(Dataset):
    def __init__(self, data_list, img_dir):
        self.data_list = data_list
        self.img_dir = img_dir
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = os.path.join(self.img_dir, item["imged_id"])
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return None
        return img, item["caption"]

# --------------------------- 损失函数 ---------------------------
def compute_distillation_loss(s_logits, t_logits, alpha=0.5, temp=4.0, device="cuda"):
    hard_loss = F.cross_entropy(s_logits, torch.arange(s_logits.size(0), device=device))
    soft_targets = F.softmax(t_logits / temp, dim=-1)
    kd_loss = F.kl_div(F.log_softmax(s_logits / temp, dim=-1), soft_targets, reduction='batchmean') * (temp**2)
    return alpha * hard_loss + (1-alpha) * kd_loss

# --------------------------- 主脚本 ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--vision_variant', type=str, default='L1')
    parser.add_argument('--save_dir', type=str, default='./remoteclip_student_FINAL')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 加载 Teacher (RemoteCLIP) --------------------
    import open_clip
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=TEACHER_CHECKPOINT_PATH, device=device
    )
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False
    teacher_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # 手动创建兼容的 preprocess（关键！）
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    teacher_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    # -------------------- 加载 Student --------------------
    student_model = LightweightStudentCLIP(vision_variant=args.vision_variant, projection_dim=512).to(device)
    student_model.train()
    for p in student_model.text_model.parameters(): p.requires_grad = False
    for p in student_model.text_projection.parameters(): p.requires_grad = True
    student_model.logit_scale.requires_grad = True

    student_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # -------------------- 数据 --------------------
    with open(JSON_PATH) as f:
        data = json.load(f)
    train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
    dataset = RSICDDataset(train_data, IMG_DIR)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True,
                        collate_fn=lambda x: [i for i in x if i is not None])

    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6
    )

    # -------------------- 训练 --------------------
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            if len(batch) == 0: continue
            pil_images = [x[0] for x in batch]
            captions = [x[1] for x in batch]

            # Teacher
            with torch.no_grad():
                img_tensor = torch.stack([teacher_preprocess(img) for img in pil_images]).to(device)
                text_tokens = teacher_tokenizer(captions).to(device)
                img_feat = teacher_model.encode_image(img_tensor)
                txt_feat = teacher_model.encode_text(text_tokens)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                logit_scale = teacher_model.logit_scale.exp()
                t_logits_i = logit_scale * img_feat @ txt_feat.t()
                t_logits_t = t_logits_i.t()

            # Student
            student_inputs = student_tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            img_tensor_student = img_tensor  # 已经归一化好了，直接复用
            s_logits_i, s_logits_t = student_model(
                image=img_tensor_student,
                input_ids=student_inputs.input_ids,
                attention_mask=student_inputs.attention_mask
            )

            # Loss
            loss = (compute_distillation_loss(s_logits_i, t_logits_i, device=device) +
                    compute_distillation_loss(s_logits_t, t_logits_t, device=device)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(student_model.state_dict(),
                       f"{args.save_dir}/student_epoch{epoch}.pt")

    print("训练完成！模型已保存")

if __name__ == "__main__":
    main()