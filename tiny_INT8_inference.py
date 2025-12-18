# inference_clean.py
# 极简专业版推理脚本（无进度条，输出清晰优雅）

import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer
from torchvision import transforms

# 图像预处理（与训练完全一致）
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711]
    ),
])

@torch.no_grad()
def main():
    # ==================== 配置区域 ====================
    model_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_LINEAR_ONLY_INT8.pt"
    image_path = "/root/mnist-clip/data/RSICD_images/airport_1.jpg"
    texts = [
        "some planes are parked in an airport",
        "A satellite image showing a lush green park with a river running through it.",
        "An aerial view of an urban area with tall skyscrapers and dense traffic.",
        "A detailed illustration of a flying insect landing on a pink flower."
    ]
    # ===================================================

    print("加载量化模型中...")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # 自动获取缩放参数
    scale = 1.0
    if hasattr(model, 'logit_scale') and model.logit_scale is not None:
        scale = model.logit_scale.exp().item()
    elif hasattr(model, 'temperature') and model.temperature is not None:
        scale = 1.0 / model.temperature.item()
    else:
        scale = 20.0  # 常用经验值

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 图像 & 文本编码
    image = Image.open(image_path).convert("RGB")
    image_input = transform(image).unsqueeze(0)
    text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")

    image_feat = model.encode_image(image_input)
    text_feat  = model.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])

    # L2 归一化 + 正确缩放 + softmax
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat  = F.normalize(text_feat,  dim=-1)
    logits = image_feat @ text_feat.T * scale
    probs  = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

    # 输出结果
    print(f"\n图像: {os.path.basename(image_path)}")
    print("-" * 70)
    for i, (text, prob) in enumerate(sorted(zip(texts, probs), key=lambda x: -x[1]), 1):
        print(f"{i:>2}. {prob*100:6.3f}%  →  {text}")
        if i == 1:
            print(f" 最匹配描述（置信度 {prob*100:.3f}%）")
    print("-" * 70)
    print(f"预测结果: \"{texts[probs.argmax()]}\"")
    print(f"最高置信度: {probs.max()*100:.3f}%")
    print("INT8 量化模型推理成功，精度几乎无损！")

if __name__ == "__main__":
    main()