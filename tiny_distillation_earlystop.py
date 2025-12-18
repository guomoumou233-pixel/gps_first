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
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import open_clip

# --------------------------- 你的学生模型 ---------------------------
# 确保 tiny_student_model.py 和其依赖（如 image_encoder.py）在路径中
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
        # 注意: 你的键名是 "imged_id"，保持不变
        img_path = os.path.join(self.img_dir, item["imged_id"]) 
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return None
        return img, item["caption"]

# --------------------------- 损失函数 (接受 alpha 和 temp 参数) ---------------------------
def compute_distillation_loss(s_logits, t_logits, alpha=0.1, temp=4.0, device="cuda"): # 默认 alpha 设为 0.1
    # Hard Loss (对比学习中的交叉熵)
    # 目标是主对角线
    hard_loss = F.cross_entropy(s_logits, torch.arange(s_logits.size(0), device=device))
    
    # Soft Loss (KL 散度)
    soft_targets = F.softmax(t_logits / temp, dim=-1)
    kd_loss = F.kl_div(F.log_softmax(s_logits / temp, dim=-1), soft_targets, reduction='batchmean') * (temp**2)
    
    # 混合损失
    return alpha * hard_loss + (1.0 - alpha) * kd_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--vision_variant', type=str, default='L1')
    parser.add_argument('--save_dir', type=str, default='./remoteclip_student_with_val2')
    parser.add_argument('--patience', type=int, default=7)
    
    # 🚀 关键修改 1: 添加蒸馏超参数
    parser.add_argument('--distill_T', type=float, default=4.0, help="蒸馏温度 T")
    parser.add_argument('--distill_alpha', type=float, default=0.1, help="硬标签损失权重 alpha (建议 0.1~0.3)")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 1. 加载 Teacher ====================
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=TEACHER_CHECKPOINT_PATH, device=device
    )
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False
    teacher_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # 🚀 关键修改 2: 定义训练集和验证集专用预处理
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # 训练集预处理 (必须包含随机增强，解决过拟合)
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5), # 随机翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 验证集预处理 (标准 CenterCrop，用于稳定评估)
    val_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # 原始的 teacher_preprocess 变量已弃用

    # ==================== 2. 加载 Student ====================
    student_model = LightweightStudentCLIP(vision_variant=args.vision_variant, projection_dim=512).to(device)
    student_model.train()
    for p in student_model.text_model.parameters(): p.requires_grad = False
    for p in student_model.text_projection.parameters(): p.requires_grad = True
    student_model.logit_scale.requires_grad = True
    student_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # ==================== 3. 数据划分：80% 训练 + 20% 验证 ====================
    with open(JSON_PATH) as f:
        data = json.load(f)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=None)
    print(f"训练集: {len(train_data)} 张, 验证集: {len(val_data)} 张")

    train_dataset = RSICDDataset(train_data, IMG_DIR)
    val_dataset   = RSICDDataset(val_data,   IMG_DIR)

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0: return None
        return [b[0] for b in batch], [b[1] for b in batch]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6
    )

    # ==================== 4. 训练 + 验证 + 早停 ====================
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ----- Train -----
        student_model.train()
        train_loss = 0.0
        for pil_images, captions in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            if pil_images is None: continue

            with torch.no_grad():
                # 🚀 关键修改 3: 训练时使用 train_preprocess (Teacher 和 Student 看到增强图)
                img_tensor = torch.stack([train_preprocess(img) for img in pil_images]).to(device)
                
                text_tokens = teacher_tokenizer(captions).to(device)
                img_f = teacher_model.encode_image(img_tensor)
                txt_f = teacher_model.encode_text(text_tokens)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                logit_scale = teacher_model.logit_scale.exp()
                t_logits_i = logit_scale * img_f @ txt_f.t()
                t_logits_t = t_logits_i.t()

            text_inputs = student_tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            s_logits_i, s_logits_t = student_model(img_tensor, text_inputs.input_ids, text_inputs.attention_mask)

            # 🚀 关键修改 4: 训练损失使用 argparse 传入的参数
            loss = (compute_distillation_loss(s_logits_i, t_logits_i, device=device, alpha=args.distill_alpha, temp=args.distill_T) +
                    compute_distillation_loss(s_logits_t, t_logits_t, device=device, alpha=args.distill_alpha, temp=args.distill_T)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ----- Valid -----
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pil_images, captions in val_loader:
                if pil_images is None: continue
                
                # 🚀 关键修改 5: 验证时使用 val_preprocess (标准 CenterCrop)
                img_tensor = torch.stack([val_preprocess(img) for img in pil_images]).to(device)
                
                text_tokens = teacher_tokenizer(captions).to(device)
                img_f = teacher_model.encode_image(img_tensor)
                txt_f = teacher_model.encode_text(text_tokens)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                logit_scale = teacher_model.logit_scale.exp()
                t_logits_i = logit_scale * img_f @ txt_f.t()
                t_logits_t = t_logits_i.t()

                text_inputs = student_tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                s_logits_i, s_logits_t = student_model(img_tensor, text_inputs.input_ids, text_inputs.attention_mask)

                # 验证损失使用 argparse 传入的参数
                loss = (compute_distillation_loss(s_logits_i, t_logits_i, device=device, alpha=args.distill_alpha, temp=args.distill_T) +
                        compute_distillation_loss(s_logits_t, t_logits_t, device=device, alpha=args.distill_alpha, temp=args.distill_T)) / 2
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end=" ")

        # ----- 早停 + 保存最佳模型 -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(student_model.state_dict(), f"{args.save_dir}/BEST_student_model.pt")
            print("← New Best!")
        else:
            patience_counter += 1
            print(f"(patience {patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\n早停触发！最佳模型在 Epoch {best_epoch}，Val Loss = {best_val_loss:.4f}")
            break

        # 每 5 轮也保存一下
        if epoch % 5 == 0:
            torch.save(student_model.state_dict(), f"{args.save_dir}/student_epoch{epoch}.pt")

    print(f"训练结束！最佳模型已保存至 {args.save_dir}/BEST_student_model.pt")

if __name__ == "__main__":
    main()