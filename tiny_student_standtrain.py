import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
import time
import sys
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import CLIPTokenizer 

# --- 修改点 1: 导入新的轻量化模型 ---
try:
    # 假设 tiny_student_model.py 和 image_encoder.py 在当前目录下
    from tiny_student_model import LightweightStudentCLIP
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 tiny_student_model.py 和 image_encoder.py 在当前目录下，并且您已安装所有依赖库。")
    sys.exit(1)


# --- 1. 配置参数 ---
# 数据集路径
DATA_DIR = "/root/mnist-clip/data/RSICD_images" # 图像文件所在的根目录
CAPTION_FILE_PATH = "/root/mnist-clip/data/RSICD-en_cleaned.json" 

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
TRAIN_SPLIT_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer 使用的标准 CLIP 模型 (TinyCLIP 的文本部分通常兼容标准 CLIP Tokenizer)
TOKENIZER_MODEL_NAME = 'openai/clip-vit-base-patch32'
# TinyCLIP 预训练权重名称 (用于初始化 Student 的文本部分)
TINY_CLIP_MODEL_NAME = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"

MAX_TEXT_LENGTH = 77 # CLIP标准长度
WEIGHT_DECAY = 1e-4 


class RemoteSensingDataset(Dataset):
    def __init__(self, data_list, image_dir, tokenizer, transform):
        """
        Args:
            data_list (list): 包含 {'imged_id': <filename>, 'caption': <caption>} 的字典列表。
            image_dir (str): 图像文件所在的根目录。
            tokenizer: 用于文本编码的 CLIPTokenizer。
            transform: 用于图像预处理的 torchvision.transforms。
        """
        self.data_list = data_list
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 提取文件名并替换所有反斜杠 (\) 为正斜杠 (/)
        image_filename_fixed = item['imged_id'].replace('\\', '/')
        
        # 2. 构造最终路径
        img_path = os.path.join(self.image_dir, image_filename_fixed)
        
        # 图像处理
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ 无法加载图像 {img_path}: {e}")
            return None 

        # 文本处理
        caption = item['caption']
        tokenized_text = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_TEXT_LENGTH, 
            return_tensors="pt"
        )
        
        return image, tokenized_text['input_ids'].squeeze(), tokenized_text['attention_mask'].squeeze()

# --- 3. 图像预处理和数据加载 ---
def load_and_split_data():
    if not os.path.exists(CAPTION_FILE_PATH):
        print(f"❌ 错误：未找到描述文件，请确保文件位于: {CAPTION_FILE_PATH}")
        sys.exit(1)
        
    with open(CAPTION_FILE_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"✅ 成功加载 {len(full_data)} 条数据。")

    # 随机划分 80% 训练集, 20% 验证集
    train_data, val_data = train_test_split(
        full_data, 
        test_size=(1 - TRAIN_SPLIT_RATIO), 
        random_state=42 
    )
    print(f"训练集大小: {len(train_data)} | 验证集大小: {len(val_data)}")
    
    # CLIP 标准图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                              std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # 初始化 Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

    # 实例化 Dataset
    train_dataset = RemoteSensingDataset(train_data, DATA_DIR, tokenizer, preprocess) 
    val_dataset = RemoteSensingDataset(val_data, DATA_DIR, tokenizer, preprocess)
    
    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None, None
        
        images, input_ids, attention_masks = zip(*batch)
        
        return (
            torch.stack(images),
            torch.stack(input_ids),
            torch.stack(attention_masks)
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader


# --- 4. 训练和验证函数 ---

def contrastive_loss(logits):
    """标准的 CLIP 对比学习损失 (交叉熵)"""
    targets = torch.arange(logits.shape[0]).long().to(DEVICE)
    return nn.CrossEntropyLoss()(logits, targets)

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch): 
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        images, input_ids, attention_mask = batch
        
        if images is None:
            continue
            
        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        optimizer.zero_grad()
        
        # --- 修改点 2: 前向传播解包 ---
        # LightweightStudentCLIP 只返回两个值 (logits_per_image, logits_per_text)
        logits_per_image, logits_per_text = model(
            images, 
            input_ids, 
            attention_mask
        )
        
        # 计算损失
        loss_i = contrastive_loss(logits_per_image)
        loss_t = contrastive_loss(logits_per_text)
        loss = (loss_i + loss_t) / 2
        
        loss.backward()
        optimizer.step()
        scheduler.step() 
        
        total_loss += loss.item()
        
        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{NUM_EPOCHS} | Step {step+1} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    
    for batch in dataloader:
        images, input_ids, attention_mask = batch
        
        if images is None:
            continue
            
        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # --- 修改点 3: 验证集前向传播解包 ---
        logits_per_image, logits_per_text = model(
            images, 
            input_ids, 
            attention_mask
        )
        
        loss_i = contrastive_loss(logits_per_image)
        loss_t = contrastive_loss(logits_per_text)
        loss = (loss_i + loss_t) / 2
        
        total_val_loss += loss.item()

    return total_val_loss / len(dataloader)


# --- 5. 主程序 ---
def main():
    print(f"🚀 开始训练 LightweightStudentCLIP 模型 (设备: {DEVICE})")
    print(f"📚 数据目录: {DATA_DIR} | 描述文件: {CAPTION_FILE_PATH}")

    # 步骤 1: 加载数据并划分
    train_loader, val_loader = load_and_split_data()

    # 步骤 2: 初始化模型
    # --- 修改点 4: 实例化 LightweightStudentCLIP ---
    print("Initializing Model...")
    model = LightweightStudentCLIP(
        vision_variant='L1', # 可选 L1, L2, L3, L4 (确保与 image_encoder.py 支持的一致)
        projection_dim=512,
        tinyclip_model_name=TINY_CLIP_MODEL_NAME
    ).to(DEVICE)
    
    # ************************************************
    # *** 冻结文本编码器参数 ***
    # ************************************************
    print("Freezing Text Encoder parameters...")
    # LightweightStudentCLIP 同样使用了 self.text_model 和 self.text_projection
    try:
        for param in model.text_model.parameters():
            param.requires_grad = False
        print("✅ 文本 Transformer 参数已冻结。")
    except AttributeError:
        print("⚠️ 警告: 无法找到 model.text_model。")
    
    try:
        for param in model.text_projection.parameters():
            param.requires_grad = False
        print("✅ 文本投影层参数已冻结。")
    except AttributeError:
        print("⚠️ 警告: 无法找到 model.text_projection。")
    
    # 确保 logit_scale (温度系数) 可训练
    try:
        model.logit_scale.requires_grad = True
    except AttributeError:
        pass

    # ************************************************
    # *** 初始化优化器和学习率调度器 ***
    # ************************************************
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE) 
    
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    # 步骤 3: 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch) 
        val_loss = validate(model, val_loader)
        
        print(f"\n======== Epoch {epoch} Summary ========")
        print(f"Average Training Loss: {train_loss:.4f}")
        print(f"Average Validation Loss: {val_loss:.4f}")
        print("=======================================\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f"/root/mnist-clip/tiny_student_finetuned.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ 验证损失降低，模型已保存至 {model_save_path}")

if __name__ == "__main__":
    main()