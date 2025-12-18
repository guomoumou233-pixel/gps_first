# file: infer_single_image_finetuned.py
# 使用你单独训练好的模型权重：/root/mnist-clip/tiny_student_finetuned.pt

import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from tiny_student_model import LightweightStudentCLIP
import os

# ==================== 1. 配置路径 ====================
image_path = "/root/mnist-clip/data/RSICD_images/airport_3.jpg"
model_weight_path = "/root/mnist-clip/tiny_student_finetuned.pt"   # ← 你的单独训练权重

# ==================== 2. 加载模型 ====================
print("正在加载你的微调模型...")
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)

# 加载你单独训练的权重
ckpt = torch.load(model_weight_path, map_location='cpu')
# 如果你保存的是 state_dict 直接加载；如果是带 optimizer 的 dict，取 model_state_dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)  # 直接是 state_dict

model.eval()
model.cuda()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==================== 3. 图像预处理（和训练时完全一致）===================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# 加载并处理单张图片
img_pil = Image.open(image_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).cuda()  # [1, 3, 224, 224]

# ==================== 4. 五段文本描述 ====================
texts = [
    "an airport with many planes",
    "a parking lot with many cars",
    "a river passing through the city",
    "green farmland and fields",
    "a large bridge over water"
]

# ==================== 5. 推理 ====================
with torch.no_grad():
    # 图像特征
    img_feat = model.encode_image(img_tensor)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # L2 归一化

    # 文本特征（一次性编码5段）
    text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
    
    text_feat = model.encode_text(text_inputs['input_ids'], text_inputs['attention_mask'])
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # 计算余弦相似度并使用模型自带的 logit_scale 进行温度缩放
    logits = img_feat @ text_feat.t()           # [1, 5]
    logits = logits.squeeze(0)                  # [5]
    logit_scale = model.logit_scale.exp()       # 训练时学到的温度参数
    logits = logits * logit_scale

    # Softmax 得到概率分布
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

# ==================== 6. 结果输出 ====================
print(f"\n图像路径: {image_path}")
print(f"模型权重: {model_weight_path}\n")
print("匹配得分（Softmax 概率）：")
print("-" * 70)
for text, prob in zip(texts, probs):
    print(f"{prob*100:6.2f}%  →  {text}")

best_idx = probs.argmax()
print("\n" + "="*70)
print(f"最终预测最匹配的描述（第 {best_idx+1} 条）：")
print(f"\"{texts[best_idx]}\"")
print(f"置信度：{probs[best_idx]*100:.2f}%")
print("="*70)