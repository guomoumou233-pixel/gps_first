# file: clip_visual_word_discovery_FIXED.py
# 任务：给定一张遥感图片，从 CLIP 词汇表中找出视觉上最匹配的 Top-5 英文单词

import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from tiny_student_model import LightweightStudentCLIP
import os
import re

# ==================== 配置 ====================
image_path = "/root/mnist-clip/test_image/airport_1.jpg"

# 使用你蒸馏后的最强模型（推荐！遥感专用）
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
ckpt = torch.load("/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt", map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

model.eval().cuda()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==================== 图像预处理 ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).cuda()

# ==================== 构建完整英文单词词汇表 ====================
vocab = tokenizer.get_vocab()
valid_words = []

# 正则匹配纯字母单词（过滤 subword 和特殊符号）
pattern = re.compile(r'^[a-zA-Z]+$')

for word, idx in vocab.items():
    if pattern.match(word) and len(word) >= 3:
        valid_words.append((word, idx))

print(f"从 {len(vocab)} 个 token 中筛选出 {len(valid_words)} 个完整英文单词")

# 取前 20000 个常见单词（加速 + 更合理）
valid_words = valid_words[:20000]
word_list = [w[0] for w in valid_words]
token_ids = [w[1] for w in valid_words]

# ==================== 批量编码所有单词 ====================
print("正在批量编码单词并计算相似度（约30秒）...")
with torch.no_grad():
    # 图像特征
    img_feat = model.encode_image(img_tensor)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # [1,512]

    # 分批编码单词（避免爆显存）
    batch_size = 512
    all_text_feats = []

    for i in range(0, len(token_ids), batch_size):
        batch_tokens = torch.tensor(token_ids[i:i+batch_size], device='cuda')  # [B]
        
        # 构造 [CLS] word [EOS]
        seq_len = 3
        batch_input = torch.full((len(batch_tokens), seq_len), tokenizer.eos_token_id, 
                                dtype=torch.long, device='cuda')
        batch_input[:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id
        batch_input[:, 1] = batch_tokens
        
        # attention_mask
        attention_mask = (batch_input != tokenizer.pad_token_id)

        # 编码
        text_feat = model.encode_text(batch_input, attention_mask)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        all_text_feats.append(text_feat)

    all_text_feats = torch.cat(all_text_feats, dim=0)  # [N,512]

    # 计算相似度（全部在 GPU 上完成）
    scores = (img_feat @ all_text_feats.t()).squeeze(0)  # [N]

    # 应用模型自带的温度缩放（关键：保持在同一设备）
    if hasattr(model, 'logit_scale'):
        scores = scores * model.logit_scale.exp()

    # 最后才转 CPU
    scores = scores.cpu()

# ==================== 输出 Top-5 最匹配单词 ====================
topk = 5
top_scores, top_indices = torch.topk(scores, topk)

print(f"\n图像：{os.path.basename(image_path)}")
print(f"模型：你的蒸馏学生模型（RemoteCLIP 蒸馏）\n")
print(f"Top-{topk} 视觉最匹配的单词：\n")
print("-" * 55)
for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
    word = word_list[idx]
    print(f"Top {i+1:2d} | {score.item():6.3f} | →  \"{word}\"")

print("\n" + "="*55)
print("="*55)