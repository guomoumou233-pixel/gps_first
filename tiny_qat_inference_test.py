# tiny_qat_inference_test_final_fixed.py
# 关键修复：在任何 import torch 之前设置量化引擎

# ==================== 关键修复：在最顶部设置量化引擎 ====================
import os
# 禁用 oneDNN 的某些优化（可选，但有时有助于强制使用 QuantizedCPU）
os.environ['USE_ONEDNN_QUANTIZER'] = '1'  # 可选

# 必须在 import torch 之前设置（这是最关键的一步！）
os.environ["TORCH_QUANTIZATION_BACKEND"] = "x86"  # 强制使用 x86 (oneDNN) 后端
# 或者使用旧方式（如果 x86 不行）：
# os.environ["TORCH_QUANTIZATION_BACKEND"] = "fbgemm"

# ==================== 现在才开始 import torch 和其他模块 ====================
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from types import MethodType

from tiny_student_model import LightweightStudentCLIP

# 检查是否生效
print(f"当前量化引擎: {torch.backends.quantized.engine}")

# ==================== 配置 ====================
IMAGE_PATH = "/root/mnist-clip/data/RSICD_images/airport_3.jpg"
INT8_CKPT_PATH = "./quantized_models/qat_quantized_student.pt"
DEVICE = torch.device("cpu")

print(f"INT8 QAT 推理设备: {DEVICE}")
print(f"加载量化权重: {INT8_CKPT_PATH}")

# ==================== 1. 实例化模型 ====================
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
model.to('cpu')

# ==================== 2. 插入 QuantStub / DeQuantStub ====================
model.vision_quant = torch.ao.quantization.QuantStub()
model.vision_dequant = torch.ao.quantization.DeQuantStub()
model.text_quant = torch.ao.quantization.QuantStub()
model.text_dequant = torch.ao.quantization.DeQuantStub()

# ==================== 3. 配置量化策略 ====================
backend = torch.backends.quantized.engine  # 使用已设置的引擎
print(f"使用量化后端: {backend}")

model.qconfig = None
linear_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        linear_count += 1
print(f"共配置 {linear_count} 个 nn.Linear 层进行量化")

# ==================== 4. prepare_qat ====================
model.train()
torch.quantization.prepare_qat(model, inplace=True)
print("prepare_qat 完成")

# ==================== 5. convert ====================
model.eval()
quantized_model = torch.quantization.convert(model, inplace=False)
print("convert 完成")

# ==================== 6. 加载权重 ====================
ckpt = torch.load(INT8_CKPT_PATH, map_location='cpu')
quantized_model.load_state_dict(ckpt, strict=True)
print("量化权重加载成功！")

quantized_model.to(DEVICE)

# ==================== 7. 修补 encode 函数 ====================
def quantized_encode_image(self, image):
    x = self.vision_quant(image)
    x = self.vision_model.backbone(x)
    x = self.vision_model.global_pool(x)
    x = torch.flatten(x, 1)
    x = self.vision_model.projection_head(x)
    return self.vision_dequant(x)

def quantized_encode_text(self, input_ids, attention_mask=None):
    embeddings = self.text_model.embeddings(input_ids)
    q_embeddings = self.text_quant(embeddings)
    encoder_outputs = self.text_model.encoder(inputs_embeds=q_embeddings, attention_mask=attention_mask)
    sequence_output = encoder_outputs.last_hidden_state
    pooled = sequence_output[:, 0]  # [CLS] token
    projected = self.text_projection(pooled)
    return self.text_dequant(projected)

quantized_model.encode_image = MethodType(quantized_encode_image, quantized_model)
quantized_model.encode_text = MethodType(quantized_encode_text, quantized_model)

# ==================== 8. 数据准备 ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

pil_img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

texts = [
    "an airport with many planes",
    "a parking lot with many cars",
    "a river passing through the city",
    "green farmland and fields",
    "a large bridge over water"
]

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
input_ids = text_inputs['input_ids'].to(DEVICE)
attention_mask = text_inputs['attention_mask'].to(DEVICE)

# ==================== 9. 推理 ====================
quantized_model.eval()
with torch.no_grad():
    image_feat = quantized_model.encode_image(img_tensor)
    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    text_feat = quantized_model.encode_text(input_ids, attention_mask)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    logits = quantized_model.logit_scale.exp() * image_feat @ text_feat.t()
    probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()

# ==================== 10. 输出 ====================
print("\n" + "="*70)
print("INT8 QAT 量化模型推理成功！")
for text, prob in zip(texts, probs):
    print(f"{prob*100:6.2f}%  →  {text}")
print("="*70)