# file: eval_bridge_matching.py
# 测试：给定 "a large bridge over water"，从5张图中选出最匹配的一张

import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from tiny_student_model import LightweightStudentCLIP
import os
from pathlib import Path

# ==================== 1. 配置 ====================
text_query = "a large bridge over water"

image_folder = "/root/mnist-clip/test_image"          # ← 你的5张测试图所在文件夹
model_path   = "/root/mnist-clip/tiny_student_finetuned.pt"   # ← 你单独训练/蒸馏的最佳权重

# ==================== 2. 加载模型 ====================
print("正在加载你的学生模型...")
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
ckpt = torch.load(model_path, map_location='cpu')

if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.eval().cuda()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==================== 3. 图像预处理（训练时一致） ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# ==================== 4. 读取所有测试图片 ====================
image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
if len(image_paths) == 0:
    raise FileNotFoundError(f"未在 {image_folder} 中找到 jpg/png 图片！")
print(f"找到 {len(image_paths)} 张测试图片")

# ==================== 5. 编码文本（只编码一次） ====================
with torch.no_grad():
    text_inputs = tokenizer([text_query], padding=True, truncation=True, 
                            max_length=77, return_tensors="pt").to("cuda")
    text_feat = model.encode_text(text_inputs.input_ids, text_inputs.attention_mask)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)   # [1,512]

# ==================== 6. 逐图推理 ====================
results = []

with torch.no_grad():
    for img_path in image_paths:
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"跳过无法打开的图片 {img_path}: {e}")
            continue

        img_tensor = transform(pil_img).unsqueeze(0).cuda()   # [1,3,224,224]
        
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # [1,512]

        # 余弦相似度 + 温度缩放
        logit = (img_feat @ text_feat.t()).squeeze()                # [1]
        logit = logit * model.logit_scale.exp()                     # 使用模型学到的温度

        results.append({
            "path": str(img_path),
            "filename": img_path.name,
            "logit": logit.item()
        })

# ==================== 7. Softmax 转概率 ====================
if len(results) == 0:
    raise ValueError("没有成功加载任何图片！")

logits = torch.tensor([r["logit"] for r in results])
probs = torch.softmax(logits, dim=0).numpy()

# ==================== 8. 输出结果 ====================
print(f"\n文本描述： \"{text_query}\"")
print(f"共对 {len(probs)} 张图片进行匹配\n")
print("-" * 80)
for i, (prob, item) in enumerate(zip(probs, results)):
    print(f"{prob*100:6.2f}%  →  {item['filename']}")

best_idx = probs.argmax()
best_prob = probs[best_idx]
best_file = results[best_idx]["filename"]

print("\n" + "="*80)
print(f"最终预测最匹配的图片（第 {best_idx+1} 张）：")
print(f"文件名： {best_file}")
print(f"置信度： {best_prob*100:.2f}%")
print(f"完整路径： {results[best_idx]['path']}")
print("="*80)