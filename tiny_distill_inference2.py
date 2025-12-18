# file: eval_bridge_matching_BEST_DISTILLED.py
# 专用于你蒸馏后保存的最好模型：BEST_student_model.pt

import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from tiny_student_model import LightweightStudentCLIP
from pathlib import Path
import os

# ==================== 配置 ====================
text_query = "a large bridge over water"

image_folder = "/root/mnist-clip/test_image"          # 你的5张测试图所在文件夹
best_model_path = "/root/mnist-clip/remoteclip_student_with_val/BEST_student_model.pt"

# ==================== 加载最佳蒸馏模型 ====================
print(f"正在加载蒸馏最佳模型：\n{best_model_path}")
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)

# 加载 BEST_student_model.pt（自动兼容两种保存格式）
ckpt = torch.load(best_model_path, map_location='cpu')
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)   # 直接就是 state_dict

model.eval().cuda()
print("模型加载完成！")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==================== 图像预处理（与训练完全一致） ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# ==================== 读取测试图片 ====================
image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
if len(image_paths) == 0:
    raise FileNotFoundError(f"在 {image_folder} 中未找到任何 jpg/png 图片！")

print(f"发现 {len(image_paths)} 张测试图片，开始推理...\n")

# ==================== 文本编码（只算一次） ====================
with torch.no_grad():
    text_inputs = tokenizer([text_query], padding=True, truncation=True, 
                            max_length=77, return_tensors="pt").to("cuda")
    text_feat = model.encode_text(text_inputs.input_ids, text_inputs.attention_mask)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)   # [1,512]

# ==================== 逐图推理 ====================
results = []
with torch.no_grad():
    for img_path in image_paths:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"跳过损坏图片 {img_path}: {e}")
            continue

        img_tensor = transform(img_pil).unsqueeze(0).cuda()   # [1,3,224,224]
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # 相似度 + 温度缩放
        logit = (img_feat @ text_feat.t()).squeeze(0) * model.logit_scale.exp()
        results.append({
            "path": str(img_path),
            "filename": img_path.name,
            "logit": logit.item()
        })

# ==================== Softmax 转概率 ====================
logits = torch.tensor([r["logit"] for r in results])
probs = torch.softmax(logits, dim=0).cpu().numpy()

# ==================== 输出结果 ====================
print(f"文本描述： \"{text_query}\"")
print(f"共评估 {len(probs)} 张图片\n")
print("=" * 85)
for i, (prob, item) in enumerate(sorted(zip(probs, results), reverse=True, key=lambda x: x[0])):
    rank = i + 1
    mark = " ← 最匹配！" if i == 0 else ""
    print(f"Top{rank:2d} | {prob*100:6.2f}%  →  {item['filename']}{mark}")

best_idx = probs.argmax()
best_file = results[best_idx]["filename"]
best_prob = probs[best_idx]

print("\n" + "=" * 85)
print(f"最终预测：最匹配的图片是 →  {best_file}")
print(f"置信度：{best_prob*100:.2f}%")
print(f"完整路径：{results[best_idx]['path']}")
print("=" * 85)