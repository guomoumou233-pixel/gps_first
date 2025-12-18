# zero_shot_evaluation.py
# 对 INT8 量化模型进行严格零样本分类评测
# 数据集：/root/mnist-clip/RS_images_2800/RS_images_2800
# 类别：aGrass, bField, cIndustry, dRiverLake, eForest, fResident, gParking

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# ------------------- 图像预处理 -------------------
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711]
    ),
])

# ------------------- 类别定义（请按需调整描述，越自然越好） -------------------
class_templates = [
    "a remote sensing image of {}",
    "an aerial view showing {}",
    "a satellite photo of {}",
    "a high-resolution image of {}",
]

class_names = {
    "aGrass":     "grassland",
    "bField":     "farmland",
    "cIndustry":  "industrial area",
    "dRiverLake": "river and lake",
    "eForest":    "dense forest",
    "fResident":  "residential area",
    "gParking":   "parking lot",
}

# 生成所有提示文本
def generate_prompts():
    prompts = []
    labels = []
    for folder, name in class_names.items():
        for tmp in class_templates:
            prompts.append(tmp.format(name))
        labels.append(folder)  # 真实标签
    return prompts, labels

# ------------------- 主评测函数 -------------------
@torch.no_grad()
def zero_shot_eval():
    model_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_LINEAR_ONLY_INT8.pt"
    data_root  = "/root/mnist-clip/RS_images_2800/RS_images_2800"

    print("正在加载 INT8 量化模型...")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # 自动获取 logit_scale
    scale = 20.0
    if hasattr(model, 'logit_scale'):
        scale = model.logit_scale.exp().item()
    print(f"使用 logit_scale: {scale:.4f}")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_prompts, true_labels = generate_prompts()
    print(f"共生成 {len(text_prompts)} 条文本提示（每类 {len(class_templates)} 条）")

    # 编码所有文本提示（只做一次）
    text_inputs = tokenizer(text_prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    text_feat = model.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])
    text_feat = F.normalize(text_feat, dim=-1) * scale  # 提前乘上 scale

    # 统计
    correct = 0
    total   = 0
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    print("开始零样本分类评测...")
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path) or folder_name not in class_names:
            continue

        true_class = folder_name
        class_total[true_class] += 1

        for img_name in tqdm(os.listdir(folder_path), desc=f"Testing {folder_name}", leave=False):
            img_path = os.path.join(folder_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_input = transform(image).unsqueeze(0)

                image_feat = model.encode_image(image_input)
                image_feat = F.normalize(image_feat, dim=-1)

                # 计算相似度
                logits = image_feat @ text_feat.T  # [1, N]
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

                # 每类取最高分（ensemble）
                class_scores = {}
                for i, prompt in enumerate(text_prompts):
                    class_key = true_labels[i // len(class_templates)]
                    if class_key not in class_scores:
                        class_scores[class_key] = probs[i]
                    else:
                        class_scores[class_key] = max(class_scores[class_key], probs[i])

                pred_class = max(class_scores, key=class_scores.get)

                if pred_class == true_class:
                    correct += 1
                    class_correct[true_class] += 1
                total += 1

            except Exception as e:
                print(f"处理图像失败 {img_path}: {e}")

    # ------------------- 输出结果 -------------------
    print("\n" + "="*60)
    print("           ZERO-SHOT 分类评测结果 (INT8 模型)")
    print("="*60)
    print(f"{'类别':<15} {'准确率':<10} {'正确/总数'}")
    print("-"*60)
    for cls in class_names.keys():
        acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        print(f"{cls:<15} {acc*100:6.2f}%    {class_correct[cls]:>4}/{class_total[cls]:<4}")
    print("-"*60)
    overall_acc = correct / total if total > 0 else 0
    print(f"{'总体准确率':<15} {overall_acc*100:6.2f}%    {correct}/{total}")
    print("="*60)
    print(f"INT8 量化模型零样本 Top-1 准确率：{overall_acc*100:.2f}%")
    print("评测完成！")

if __name__ == "__main__":
    zero_shot_eval()