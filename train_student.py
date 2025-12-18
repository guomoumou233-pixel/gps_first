import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel, LoraConfig
from tqdm import tqdm
import json
import os
from PIL import Image

# 引入我们定义的学生模型和 SwiftFormer
# 确保 image_encoder.py 在当前目录
from image_encoder import CLIPSwiftFormerEncoder

# ==========================================
# 1. 模型定义部分
# ==========================================

class StudentCLIP(nn.Module):
    """ 定义如上所示的学生模型 """
    def __init__(self, teacher_model_name='openai/clip-vit-base-patch32', embed_dim=512):
        super().__init__()
        self.vision_model = CLIPSwiftFormerEncoder(projection_dim=embed_dim, model_variant='L1')
        original_clip = CLIPModel.from_pretrained(teacher_model_name)
        self.text_model = original_clip.text_model
        self.text_projection = original_clip.text_projection
        self.logit_scale = nn.Parameter(original_clip.logit_scale.clone())
        del original_clip

    def forward(self, images, input_ids, attention_mask=None):
        image_embeds = self.vision_model(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.text_projection(text_outputs.pooler_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        
        return logits_per_image, image_embeds, text_embeds

class TeacherCLIP(nn.Module):
    """ 教师模型封装 (用于加载 LoRA) """
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.logit_scale = self.clip.logit_scale

    def forward(self, images, input_ids, attention_mask=None):
        with torch.no_grad(): # 教师模型永远不需要梯度
            vision_outputs = self.clip.vision_model(pixel_values=images)
            image_embeds = self.clip.visual_projection(vision_outputs.pooler_output)
            image_features = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
            text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = self.clip.text_projection(text_outputs.pooler_output)
            text_features = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
            
            return logits_per_image, image_features, text_features

# ==========================================
# 2. 数据处理部分 (复用之前的逻辑)
# ==========================================
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return default_collate(batch)

class RSJsonDataset(Dataset):
    def __init__(self, json_path, img_dir, processor, transform):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.processor = processor 
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        corrected_file_name = item['image'].replace('\\', '/')
        img_path = os.path.join(self.img_dir, corrected_file_name) 
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return None
        img_tensor = self.transform(img)
        text_inputs = self.processor.tokenizer(
            item['text'], padding='max_length', truncation=True, max_length=77, return_tensors='pt'
        )
        return {'image': img_tensor, 'input_ids': text_inputs['input_ids'].squeeze(0), 'attention_mask': text_inputs['attention_mask'].squeeze(0)}

# ==========================================
# 3. 损失函数 (KL Divergence for Distillation)
# ==========================================
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """ 计算 KL 散度损失 """
    # Softmax with temperature
    p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL Divergence: sum(p_t * (log(p_t) - log(p_s)))
    # PyTorch KLDivLoss expects input as log_softmax and target as probability
    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
    return loss

def contrastive_loss(logits):
    """ 标准 CLIP 损失 """
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return (loss_i + loss_t) / 2

# ==========================================
# 4. 主训练循环
# ==========================================
def main():
    # --- 配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64 # 可以尝试更大，因为学生模型更小
    EPOCHS = 8
    LR = 5e-4 # SwiftFormer 也是 Transformer-based，但从头训练 Vision Encoder 需要较大的 LR
    WEIGHT_DECAY = 0.05
    
    # 路径
    TEACHER_LORA_PATH = './clip_lora_rs_finetuned.pth' # LoRA 权重路径
    JSON_FILE = './remote_sensing_clip_data.json'
    IMAGE_ROOT_DIR = r'/root/mnist-clip/RS_images_2800/RS_images_2800'
    SAVE_PATH = './student_swiftformer_clip.pth'
    
    # --- 1. 准备 Teacher Model ---
    print("正在加载 LoRA 教师模型...")
    base_teacher = TeacherCLIP().to(DEVICE)
    # 针对 CLIP 模型，我们将 LoRA 注入到 Attention 的 q_proj 和 v_proj 层
    # 这里的配置必须与训练 Teacher 时完全一致
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1, bias="none", modules_to_save=["logit_scale"]
    )
    # 加载 LoRA 权重
    # 注意：PeftModel.from_pretrained 通常用于保存后的文件夹
    # 如果您保存的是 state_dict (如上一个脚本)，我们需要先初始化再 load_state_dict
    teacher_model = PeftModel(base_teacher, peft_config)
    # 加载您微调后的权重
    if os.path.exists(TEACHER_LORA_PATH):
        state_dict = torch.load(TEACHER_LORA_PATH, map_location=DEVICE)
        # 过滤 key 以匹配 PEFT 模型结构 (通常可以直接加载，取决于保存方式)
        teacher_model.load_state_dict(state_dict, strict=False) 
        print("✅ 教师模型权重加载成功。")
    else:
        print("⚠️ 警告：未找到 LoRA 权重，将使用原始 CLIP 作为教师。")
    
    teacher_model.eval() # 教师模型始终处于 eval 模式
    
    # --- 2. 准备 Student Model ---
    print("正在初始化学生模型 (SwiftFormer Vision + CLIP Text)...")
    student_model = StudentCLIP().to(DEVICE)
    
    # --- 3. 数据准备 ---
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    # 使用与 Teacher 相同的预处理
    img_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
    ])
    
    full_dataset = RSJsonDataset(JSON_FILE, IMAGE_ROOT_DIR, processor, img_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    # --- 4. 优化器 ---
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # --- 5. 训练循环 ---
    best_val_loss = float('inf')
    
    print(f"开始蒸馏训练... (Params: {sum(p.numel() for p in student_model.parameters())/1e6:.2f}M)")
    
    for epoch in range(EPOCHS):
        student_model.train()
        train_loss_total = 0
        distill_loss_total = 0
        contrastive_loss_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            if batch is None: continue
            
            imgs = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            # 1. Teacher Forward (No Grad)
            with torch.no_grad():
                t_logits_img, _, _ = teacher_model(imgs, input_ids, mask)
            
            # 2. Student Forward
            s_logits_img, _, _ = student_model(imgs, input_ids, mask)
            
            # 3. Calculate Losses
            # A. Distillation Loss (让学生 Logits 像老师)
            loss_distill = distillation_loss(s_logits_img, t_logits_img, temperature=3.0)
            
            # B. Contrastive Loss (标准 Ground Truth 监督)
            loss_contrast = contrastive_loss(s_logits_img)
            
            # C. Combined Loss (可以调整权重，例如 0.5 : 0.5)
            alpha = 0.5
            loss = alpha * loss_distill + (1 - alpha) * loss_contrast
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            distill_loss_total += loss_distill.item()
            contrastive_loss_total += loss_contrast.item()
            
            pbar.set_postfix({'L_Total': loss.item(), 'L_Dist': loss_distill.item()})
        
        scheduler.step()
        
        # --- Validation ---
        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                imgs = batch['image'].to(DEVICE)
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                
                s_logits, _, _ = student_model(imgs, input_ids, mask)
                val_loss += contrastive_loss(s_logits).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss_total / len(train_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Dst: {distill_loss_total/len(train_loader):.4f}) | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student_model.state_dict(), SAVE_PATH)
            print(f" 模型已保存: {SAVE_PATH}")

if __name__ == "__main__":
    main()