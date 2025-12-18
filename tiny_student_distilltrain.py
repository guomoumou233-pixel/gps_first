import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import time
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR 
from transformers import CLIPModel, CLIPTokenizer
from sklearn.model_selection import train_test_split

# 依赖您的本地文件
try:
    # ---!!! 修改点 1: 导入轻量化学生模型 !!!---
    from tiny_student_model import LightweightStudentCLIP as StudentCLIP 
    # 假设 image_encoder 存在于 StudentCLIP 的导入路径中
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 tiny_student_model.py (包含 LightweightStudentCLIP) 和 image_encoder.py 在当前目录下。")
    sys.exit(1)


# --- 1. 配置参数 ---
# ---!!! 修改点 2: 更新数据路径 !!!---
# 数据集图片路径 (根据您的描述)
DATA_DIR = "/root/mnist-clip/data/RSICD_images"
# 描述文件路径 (根据您的描述)
CAPTION_FILE_NAME = "RSICD-en_cleaned.json" 
# CAPTION_FILE_PATH 应该指向 JSON 文件所在的目录
CAPTION_FILE_PATH = os.path.join("/root/mnist-clip/data", CAPTION_FILE_NAME) 

# Teacher Model 路径 (保持不变)
REMOTECLIP_PATH = "checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"

# 训练参数 (可以根据您的硬件调整)
BATCH_SIZE = 64  
NUM_EPOCHS = 15      
LEARNING_RATE = 5e-5 
TRAIN_SPLIT_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用 Teacher 的基准模型作为 Tokenizer/结构参考
TEACHER_MODEL_NAME = 'openai/clip-vit-base-patch32' 
MAX_TEXT_LENGTH = 77 

# --- 知识蒸馏超参数 ---
TEMPERATURE = 4.0   
ALPHA = 0.5         
PATIENCE = 5        


# --- 2. 教师模型加载函数 (保持不变) ---
def load_remoteclip_teacher(model_path, device):
    """加载 RemoteCLIP 预训练权重到标准 CLIP 模型结构中"""
    print(f"🔄 正在加载 Teacher 模型: {model_path}...")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"❌ 错误：未找到教师模型权重文件: {model_path}")
        sys.exit(1)
    
    # 实例化标准 CLIP 模型 (Teacher 是 ViT-B/32)
    teacher_model = CLIPModel.from_pretrained(TEACHER_MODEL_NAME).to(device)

    try:
        teacher_model.load_state_dict(state_dict, strict=True) 
    except RuntimeError as e:
        print("⚠️ 无法直接加载 RemoteCLIP 权重。请确保权重文件结构与 CLIPModel 匹配。")
        print(f"原始加载错误信息: {e}")
        
    teacher_model.eval()
    # 冻结所有 Teacher 参数
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    return teacher_model


# --- 3. 自定义数据集类 (!!! 需要调整，因为您的 json 文件格式不同!!!) ---
class RemoteSensingDataset(Dataset):
    def __init__(self, data_list, image_dir, tokenizer, transform):
        self.data_list = data_list
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # ---!!! 修改点 3: 适应新的 JSON 文件键名 ('imged_id' 和 'caption') !!!---
        image_filename = item['imged_id']
        caption = item['caption']
        # ------------------------------------------------------------------------
        
        img_path = os.path.join(self.image_dir, image_filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            # print(f"⚠️ 无法加载图片 {img_path}: {e}") # 调试时启用
            return None 

        tokenized_text = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_TEXT_LENGTH, 
            return_tensors="pt"
        )
        
        return image, tokenized_text['input_ids'].squeeze(), tokenized_text['attention_mask'].squeeze()


# --- 4. 数据加载 (保持增强逻辑，更新路径) ---
def load_and_split_data():
    if not os.path.exists(CAPTION_FILE_PATH):
        print(f"❌ 错误：未找到描述文件，请确保文件位于: {CAPTION_FILE_PATH}")
        sys.exit(1)
        
    with open(CAPTION_FILE_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"✅ 成功加载 {len(full_data)} 条数据。")

    train_data, val_data = train_test_split(
        full_data, 
        test_size=(1 - TRAIN_SPLIT_RATIO), 
        random_state=42 
    )
    print(f"训练集大小: {len(train_data)} | 验证集大小: {len(val_data)}")
    
    # 图像预处理保持不变 (适配 CLIP/RemoteCLIP ViT-B/32 的输入)
    train_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), 
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    val_preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    tokenizer = CLIPTokenizer.from_pretrained(TEACHER_MODEL_NAME)

    train_dataset = RemoteSensingDataset(train_data, DATA_DIR, tokenizer, train_preprocess)
    val_dataset = RemoteSensingDataset(val_data, DATA_DIR, tokenizer, val_preprocess)
    
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=4) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    
    return train_loader, val_loader


# --- 5. 损失函数 (保持不变) ---

def contrastive_loss_hard(logits):
    """标准的 CLIP 对比学习损失 (交叉熵)"""
    targets = torch.arange(logits.shape[0]).long().to(DEVICE)
    return nn.CrossEntropyLoss()(logits, targets)

def compute_distillation_loss(student_logits, teacher_logits, alpha, temperature, device):
    """计算组合损失：ALPHA * 硬损失 + (1 - ALPHA) * 软蒸馏损失"""
    
    # 1. 软目标蒸馏损失 (Soft KD Loss)
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=-1),
        soft_targets
    ) * (temperature ** 2) 

    # 2. 硬目标对比损失 (Hard Contrastive Loss)
    hard_loss = contrastive_loss_hard(student_logits)

    # 3. 组合损失
    combined_loss = alpha * hard_loss + (1.0 - alpha) * kd_loss
    return combined_loss


# --- 6. 训练和验证函数 (保持不变) ---

def train_one_epoch(student_model, teacher_model, dataloader, optimizer, scheduler, epoch):
    student_model.train()
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
        
        # --- Teacher Model 前向传播 (无梯度) ---
        with torch.no_grad():
            teacher_outputs = teacher_model(
                pixel_values=images, 
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_loss=False 
            )
            
            t_img_embeds = teacher_outputs.image_embeds / teacher_outputs.image_embeds.norm(dim=-1, keepdim=True)
            t_text_embeds = teacher_outputs.text_embeds / teacher_outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            teacher_logit_scale = teacher_model.logit_scale.exp()
            teacher_logits_per_image = teacher_logit_scale * t_img_embeds @ t_text_embeds.T
            teacher_logits_per_text = teacher_logits_per_image.T
        
        # --- Student Model 前向传播 ---
        # LightweightStudentCLIP 的 forward 方法返回 (logits_per_image, logits_per_text)
        logits_per_image, logits_per_text = student_model(
            images, 
            input_ids, 
            attention_mask
        )
        
        # 计算组合蒸馏损失
        loss_i = compute_distillation_loss(logits_per_image, teacher_logits_per_image, ALPHA, TEMPERATURE, DEVICE)
        loss_t = compute_distillation_loss(logits_per_text, teacher_logits_per_text, ALPHA, TEMPERATURE, DEVICE)
        
        loss = (loss_i + loss_t) / 2
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{NUM_EPOCHS} | Step {step+1} | Distill Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.2f}s")
            
    scheduler.step()
    
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
        
        # LightweightStudentCLIP 的 forward 方法返回 (logits_per_image, logits_per_text)
        logits_per_image, logits_per_text = model(
            images, 
            input_ids, 
            attention_mask
        )
        
        # 只使用标准的硬对比损失进行验证
        loss_i = contrastive_loss_hard(logits_per_image)
        loss_t = contrastive_loss_hard(logits_per_text)
        loss = (loss_i + loss_t) / 2
        
        total_val_loss += loss.item()

    if len(dataloader) > 0:
        return total_val_loss / len(dataloader)
    return 0.0


# --- 7. 主程序 (更新学生模型初始化和冻结逻辑) ---
def main():
    print(f"🚀 开始 StudentCLIP 知识蒸馏训练 (设备: {DEVICE})")

    # 步骤 1: 加载数据并划分
    train_loader, val_loader = load_and_split_data()

    # 步骤 2: 初始化 Teacher 和 Student 模型
    teacher_model = load_remoteclip_teacher(REMOTECLIP_PATH, DEVICE)
    
    # ---!!! 修改点 4: 初始化 LightweightStudentCLIP !!!---
    # LightweightStudentCLIP 的默认参数与 student_large_distill_train.py 的逻辑匹配
    student_model = StudentCLIP().to(DEVICE) 
    # --------------------------------------------------------
    
    # 冻结 Student 文本编码器参数 (与 large 脚本的冻结逻辑保持一致，只训练视觉和投影层)
    print("🔒 冻结 Student Model 文本编码器参数...")
    for param in student_model.text_model.parameters():
        param.requires_grad = False
        
    # 确保 logit_scale 和投影层可训练
    student_model.logit_scale.requires_grad = True
    for param in student_model.text_projection.parameters():
        param.requires_grad = True
    
    # ---!!! 修改点 5: 优化器只关注可训练参数 !!!---
    # 只需要训练视觉编码器、logit_scale 和 文本投影层。
    trainable_params = filter(lambda p: p.requires_grad, student_model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # 引入学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0 
    
    # 步骤 3: 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(student_model, teacher_model, train_loader, optimizer, scheduler, epoch)
        
        val_loss = validate(student_model, val_loader)
        
        print(f"\n======== Epoch {epoch} Summary ========")
        print(f"Average Training Distillation Loss: {train_loss:.4f}")
        print(f"Average Validation Hard Loss: {val_loss:.4f}")
        print("=======================================\n")
        
        # --- 早停和保存最佳模型逻辑 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0 
            # 保存的是 StudentCLIP 的状态字典
            model_save_path = f"./student_clip_remote_distilled_best_model.pth" 
            torch.save(student_model.state_dict(), model_save_path)
            print(f"✨ 验证损失降低，模型已保存至 {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ 验证损失未降低. Patience: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve == PATIENCE:
            print(f"🛑 提前停止训练! 验证损失已连续 {PATIENCE} 个 Epoch 未降低。")
            break 

if __name__ == "__main__":
    main()