# file: infer_single_image.py
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from tiny_student_model import LightweightStudentCLIP
import os

# ==================== 1. 路径与模型加载 ====================
image_path = "/root/mnist-clip/data/RSICD_images/airport_3.jpg"
best_ckpt   = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"   # ← 你训练完后自动保存的那个

# 加载你的最佳学生模型
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
ckpt = torch.load(best_ckpt, map_location='cpu')
model.load_state_dict(ckpt)
model.eval().cuda()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==================== 2. 图像预处理（和你训练时完全一致） ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711]
    ),
])

# 加载单张图片
pil_img = Image.open(image_path).convert('RGB')
img_tensor = transform(pil_img).unsqueeze(0).cuda()   # [1,3,224,224]

# ==================== 3. 五段文本描述 ====================
texts = [
    "an airport with many planes",
    "a parking lot with many cars",
    "a river passing through the city",
    "green farmland and fields",
    "a large bridge over water"
]

# ==================== 4. 推理 ====================
with torch.no_grad():
    # 图像特征
    image_feat = model.encode_image(img_tensor)                    # [1,512]
    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

    # 文本特征（batch 编码）
    text_inputs = tokenizer(texts, padding=True, truncation=True, 
                            max_length=77, return_tensors="pt").to("cuda")
    text_feat = model.encode_text(
        text_inputs.input_ids,
        text_inputs.attention_mask
    )                                                              # [5,512]
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # 余弦相似度 → logits
    logits = image_feat @ text_feat.t()          # [1,5]
    logits = logits.squeeze(0)                   # [5]

    # 温度缩放（使用模型自带的 logit_scale，和训练时完全一致）
    temperature = model.logit_scale.exp()
    logits = logits * temperature

    # Softmax 得到概率
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

# ==================== 5. 输出结果 ====================
print(f"图像: {os.path.basename(image_path)}\n")
print("描述 → Softmax 概率（越接近 1 越匹配）")
print("-" * 60)
for text, prob in zip(texts, probs):
    print(f"{prob*100:6.2f}%  →  {text}")

# 最终预测
best_idx = probs.argmax()
print("\n" + "="*60)
print(f"最终预测最匹配的描述（第 {best_idx+1} 条）：")
print(f"\"{texts[best_idx]}\"")
print(f"置信度: {probs[best_idx]*100:.2f}%")
print("="*60)