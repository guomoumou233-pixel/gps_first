import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip  # OpenAI CLIP
from transformers import CLIPModel, CLIPConfig, CLIPTokenizer # HuggingFace Transformers
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm

# ================= 配置路径 =================
UCM_IMG_DIR = "/root/mnist-clip/UCM/imgs/UCM"
UCM_JSON_PATH = "/root/mnist-clip/UCM/UCM-en.json"
REMOTECLIP_WEIGHTS = "/root/mnist-clip/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
STUDENT_WEIGHTS = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# PART 1: 嵌入你的视觉编码器代码 (来自 image_encoder.py)
# ==============================================================================
class SwiftFormerMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.bn = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAdditiveAttention(nn.Module):
    def __init__(self, dim=512, key_dim=512, num_heads=8, act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (key_dim // num_heads) ** -0.5
        self.to_q = nn.Conv2d(dim, key_dim, 1)
        self.to_k = nn.Conv2d(dim, key_dim, 1)
        self.w_a = nn.Parameter(torch.randn(1, key_dim, 1, 1))
        self.to_out = nn.Conv2d(key_dim, dim, 1)
        
    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        B, C, H, W = q.shape
        attn_logits = (q * self.w_a).sum(dim=1, keepdim=True)
        attn_logits = attn_logits * self.scale
        attn = attn_logits.view(B, 1, -1)
        attn = F.softmax(attn, dim=-1)
        attn = attn.view(B, 1, H, W)
        global_query = (q * attn).sum(dim=(2, 3), keepdim=True)
        context = k * global_query
        out = self.to_out(context)
        return out + q

class SwiftFormerLocalRepresentation(nn.Module):
    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        return x

class SwiftFormerEncoderBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        self.local_representation = SwiftFormerLocalRepresentation(dim)
        self.attn = EfficientAdditiveAttention(dim=dim, key_dim=dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = SwiftFormerMlp(in_features=dim, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)
        self.drop_path = nn.Identity() 

    def forward(self, x):
        local_feat = self.local_representation(x)
        x_attn = self.attn(local_feat)
        x = local_feat + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x_attn)
        x_mlp = self.mlp(x)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x_mlp)
        return x

class ConvEncoderBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = 4*dim
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x + shortcut
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=3, embed_dim=48):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.proj(x)

class SwiftFormer(nn.Module):
    def __init__(self, layers=[3, 3, 6, 4], embed_dims=[48, 96, 192, 384], downsamples=[True, True, True, True], num_classes=1000, use_conv_encoder_in_stage=[True, True, True, True]):
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dims[0])
        self.stage1 = self._make_stage(embed_dims[0], num_conv=3, num_swift=1)
        self.downsample2 = self._make_downsample(embed_dims[0], embed_dims[1])
        self.stage2 = self._make_stage(embed_dims[1], num_conv=2, num_swift=1)
        self.downsample3 = self._make_downsample(embed_dims[1], embed_dims[2])
        self.stage3 = self._make_stage(embed_dims[2], num_conv=9, num_swift=1)
        self.downsample4 = self._make_downsample(embed_dims[2], embed_dims[3])
        self.stage4 = self._make_stage(embed_dims[3], num_conv=4, num_swift=1)
        self.final_dim = embed_dims[3]

    def _make_downsample(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim)
        )

    def _make_stage(self, dim, num_conv, num_swift):
        blocks = []
        for _ in range(num_conv):
            blocks.append(ConvEncoderBlock(dim))
        for _ in range(num_swift):
            blocks.append(SwiftFormerEncoderBlock(dim))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.downsample2(x)
        x = self.stage2(x)
        x = self.downsample3(x)
        x = self.stage3(x)
        x = self.downsample4(x)
        x = self.stage4(x)
        return x

class CLIPSwiftFormerEncoder(nn.Module):
    def __init__(self, projection_dim=512, model_variant='L1'):
        super().__init__()
        if model_variant == 'L1':
            self.backbone = SwiftFormer(embed_dims=[48, 96, 192, 384])
            prev_dim = 384
        else:
            raise NotImplementedError
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection_head = nn.Linear(prev_dim, projection_dim, bias=False)

    def forward(self, x):
        x = self.backbone(x) 
        x = self.global_pool(x) 
        x = x.flatten(1)     
        x = self.projection_head(x) 
        return x

# ==============================================================================
# PART 2: 嵌入你的学生模型代码 (来自 tiny_student_model.py)
# ==============================================================================
class LightweightStudentCLIP(nn.Module):
    def __init__(self, 
                 vision_variant='L1', 
                 projection_dim=512, 
                 tinyclip_model_name="wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"):
        super().__init__()
        print(f"Initializing Custom Vision Encoder (SwiftFormer-{vision_variant})...")
        self.vision_model = CLIPSwiftFormerEncoder(
            projection_dim=projection_dim,
            model_variant=vision_variant
        )
        print(f"Loading Text Encoder from {tinyclip_model_name}...")
        try:
            pretrained_clip = CLIPModel.from_pretrained(tinyclip_model_name)
            self.text_model = pretrained_clip.text_model
            self.text_projection = pretrained_clip.text_projection
            self.logit_scale = pretrained_clip.logit_scale
            del pretrained_clip.vision_model
            del pretrained_clip
        except OSError:
            print("Warning: 无法连接 HuggingFace，使用随机初始化的文本编码器作为占位符(用于加载本地权重)。")
            config = CLIPConfig()
            dummy_model = CLIPModel(config)
            self.text_model = dummy_model.text_model
            self.text_projection = dummy_model.text_projection
            self.logit_scale = dummy_model.logit_scale

    def encode_image(self, image):
        return self.vision_model(image)

    def encode_text(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features

# ==============================================================================
# PART 3: 数据集定义 (Dataset) - 适配两种模型
# ==============================================================================
class UCMDataset(Dataset):
    def __init__(self, img_dir, json_path, preprocess):
        self.img_dir = img_dir
        self.preprocess = preprocess
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.valid_data = []
        for item in self.data:
            path = os.path.join(self.img_dir, item['imged_id'])
            if os.path.exists(path):
                self.valid_data.append(item)
            else:
                print(f"Warning: Image not found {path}")

    def __len__(self):
        return len(self.valid_data)

    def _clean_caption(self, caption_str):
        # 简单清洗：去除多余空格
        return caption_str.strip()

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        image_path = os.path.join(self.img_dir, item['imged_id'])
        
        # 1. 处理图片 (使用 CLIP 标准预处理: Resize -> CenterCrop -> Normalize)
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        
        # 2. 处理文本 (返回原始文本字符串，后续分别进行 Tokenize)
        caption = self._clean_caption(item['caption'])
        
        return image, caption

# ==============================================================================
# PART 4: 评估函数 (Metrics)
# ==============================================================================
def calculate_metrics(image_emb, text_emb):
    # 余弦相似度
    logits = image_emb @ text_emb.t()
    n_samples = logits.shape[0]
    metrics = {}
    
    # Image-to-Text
    ranks_i2t = logits.argsort(dim=1, descending=True)
    gt_i2t = torch.arange(n_samples, device=logits.device).view(-1, 1)
    
    metrics['I2T_R@1'] = (ranks_i2t[:, :1] == gt_i2t).sum().item() / n_samples * 100
    metrics['I2T_R@5'] = (ranks_i2t[:, :5] == gt_i2t).any(dim=1).sum().item() / n_samples * 100
    metrics['I2T_R@10'] = (ranks_i2t[:, :10] == gt_i2t).any(dim=1).sum().item() / n_samples * 100
    
    # Text-to-Image
    ranks_t2i = logits.argsort(dim=0, descending=True)
    gt_t2i = torch.arange(n_samples, device=logits.device).view(1, -1)
    
    metrics['T2I_R@1'] = (ranks_t2i[:1, :] == gt_t2i).sum().item() / n_samples * 100
    metrics['T2I_R@5'] = (ranks_t2i[:5, :] == gt_t2i).any(dim=0).sum().item() / n_samples * 100
    metrics['T2I_R@10'] = (ranks_t2i[:10, :] == gt_t2i).any(dim=0).sum().item() / n_samples * 100
    
    metrics['Mean_Recall'] = (metrics['I2T_R@1'] + metrics['I2T_R@5'] + metrics['I2T_R@10'] +
                              metrics['T2I_R@1'] + metrics['T2I_R@5'] + metrics['T2I_R@10']) / 6
    return metrics

def print_results(name, m):
    print(f"\n===== {name} Evaluation Results =====")
    print(f"Image-to-Text: R@1: {m['I2T_R@1']:.2f} | R@5: {m['I2T_R@5']:.2f} | R@10: {m['I2T_R@10']:.2f}")
    print(f"Text-to-Image: R@1: {m['T2I_R@1']:.2f} | R@5: {m['T2I_R@5']:.2f} | R@10: {m['T2I_R@10']:.2f}")
    print(f"Mean Recall  : {m['Mean_Recall']:.2f}%")
    print("=========================================")

# ==============================================================================
# PART 5: 主程序
# ==============================================================================
if __name__ == "__main__":
    print(f"Running evaluation on {DEVICE}...")
    
    # --- 1. 加载 RemoteCLIP (Teacher) ---
    print("\n[1/4] Loading RemoteCLIP (Teacher)...")
    teacher_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    try:
        checkpoint = torch.load(REMOTECLIP_WEIGHTS, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # 移除 'module.' 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        teacher_model.load_state_dict(state_dict)
        print("RemoteCLIP weights loaded successfully.")
    except Exception as e:
        print(f"Error loading RemoteCLIP weights: {e}")
        exit()

    # --- 2. 加载 Student Model ---
    print("\n[2/4] Loading Student Model...")
    student_model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    student_model.to(DEVICE)
    
    # 准备 Tokenizer (用于 Student)
    # 你的 Student 文本部分使用了 TinyCLIP，通常兼容 openai/clip-vit-base-patch32 的分词器
    hf_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    try:
        student_ckpt = torch.load(STUDENT_WEIGHTS, map_location=DEVICE)
        # 兼容性处理
        if isinstance(student_ckpt, dict) and 'model_state_dict' in student_ckpt:
            # 有时 PyTorch 保存为 {'model_state_dict': ..., 'optimizer': ...}
            student_model.load_state_dict(student_ckpt['model_state_dict'])
        elif isinstance(student_ckpt, dict) and 'state_dict' in student_ckpt:
            student_model.load_state_dict(student_ckpt['state_dict'])
        elif isinstance(student_ckpt, dict):
             student_model.load_state_dict(student_ckpt)
        else:
            student_model = student_ckpt
        
        student_model.eval()
        print("Student model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading Student Model: {e}")
        exit()

    # --- 3. 准备数据 ---
    print("\n[3/4] Preparing Dataset...")
    dataset = UCMDataset(UCM_IMG_DIR, UCM_JSON_PATH, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Total samples: {len(dataset)}")

    # --- 4. 提取特征并评估 ---
    print("\n[4/4] Extracting Features & Comparing...")

    # 容器
    t_img_feats, t_txt_feats = [], []
    s_img_feats, s_txt_feats = [], []

    with torch.no_grad():
        for images, raw_texts in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            
            # === Teacher Forward ===
            # Teacher 需要 clip.tokenize 处理后的 tensor
            teacher_tokens = clip.tokenize(raw_texts, truncate=True).to(DEVICE)
            
            t_img = teacher_model.encode_image(images)
            t_txt = teacher_model.encode_text(teacher_tokens)
            
            # 归一化
            t_img = t_img / t_img.norm(dim=-1, keepdim=True)
            t_txt = t_txt / t_txt.norm(dim=-1, keepdim=True)
            
            t_img_feats.append(t_img.cpu())
            t_txt_feats.append(t_txt.cpu())

            # === Student Forward ===
            # Student 需要 HF Tokenizer 生成的 input_ids
            inputs = hf_tokenizer(raw_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            
            s_img = student_model.encode_image(images)
            s_txt = student_model.encode_text(inputs['input_ids'], inputs['attention_mask'])
            
            # 归一化 (重要！Student 的 encode 方法未归一化)
            s_img = s_img / s_img.norm(dim=-1, keepdim=True)
            s_txt = s_txt / s_txt.norm(dim=-1, keepdim=True)
            
            s_img_feats.append(s_img.cpu())
            s_txt_feats.append(s_txt.cpu())

    # 拼接
    t_img_feats = torch.cat(t_img_feats, dim=0).to(DEVICE)
    t_txt_feats = torch.cat(t_txt_feats, dim=0).to(DEVICE)
    s_img_feats = torch.cat(s_img_feats, dim=0).to(DEVICE)
    s_txt_feats = torch.cat(s_txt_feats, dim=0).to(DEVICE)

    # 计算指标
    t_metrics = calculate_metrics(t_img_feats, t_txt_feats)
    s_metrics = calculate_metrics(s_img_feats, s_txt_feats)

    # 打印结果
    print_results("RemoteCLIP (Teacher)", t_metrics)
    print_results("Lightweight Student", s_metrics)

    diff = s_metrics['Mean_Recall'] - t_metrics['Mean_Recall']
    print(f"\n>>> Performance Gap: {diff:.2f}% Mean Recall")
    if diff > -5.0:
        print(">>> Result: Excellent! The distillation is very effective.")
    elif diff > -10.0:
        print(">>> Result: Good. The student model is competitive.")
    else:
        print(">>> Result: Needs improvement. The gap is significant.")