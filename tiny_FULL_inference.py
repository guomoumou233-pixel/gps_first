import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer

# 导入您提供的模型结构 (必须导入，因为 torch.load 需要类定义)
from tiny_student_model import LightweightStudentCLIP 
from image_encoder import CLIPSwiftFormerEncoder # 确保 image_encoder 也被导入

# --- 路径和配置 ---
IMAGE_PATH = "/root/mnist-clip/data/RSICD_images/airport_1.jpg" # 图像路径

# 注意：使用您上一轮保存的 "量化 Linear + Embedding 并保存完整对象" 的文件路径
MODEL_PATH = "/root/mnist-clip/remoteclip_student_with_val2/quantized_FULL_OBJECT_INT8.pt" 

CANDIDATE_TEXTS = [ # 文本描述
    "some planes are parked in an airport",
    "A detailed illustration of a flying insect landing on a pink flower.",
    "A satellite image showing a lush green park with a river running through it.",
    "An aerial view of an urban area with tall skyscrapers and dense traffic.",
]
# CLIP 标准 Tokenizer
TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# --- 图像预处理 ---
image_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), 
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


def load_model():
    """
    加载完整保存的量化模型对象，强制关闭安全模式。
    """
    print(f"🚀 步骤 1: 尝试加载完整量化模型对象 (强制 weights_only=False)...")
    try:
        # **关键修改点：显式设置 weights_only=False**
        quantized_model = torch.load(
            MODEL_PATH, 
            map_location="cpu",
            weights_only=False  # 禁用安全检查，允许加载自定义类
        )
        quantized_model.eval()
        print("完整量化模型对象加载成功！")
        return quantized_model.cpu()
    except Exception as e:
        print(f"\n⚠️ 最终错误: 无法加载模型对象，请检查 MODEL_PATH。")
        print(f"原始错误: {e}")
        return None


def run_inference(model: nn.Module, image_path: str, texts: list):
    """
    执行图像-文本相似度匹配推理。
    """
    if model is None:
        return
        
    # 1. 图像预处理
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"\n⚠️ 错误: 找不到图像文件: {image_path}")
        return
        
    image_input = image_transform(image).unsqueeze(0)

    # 2. 文本预处理
    text_inputs = TOKENIZER(
        texts, 
        padding=True, 
        return_tensors="pt", 
        max_length=77, 
        truncation=True
    )

    # 3. 特征编码与相似度计算
    with torch.no_grad():
        try:
            # model.forward() 返回 logits_per_image, logits_per_text
            logits_per_image, _ = model(
                image=image_input,
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask']
            )
        except AttributeError as e:
            # 捕获之前警告的错误
            print(f"\n❌ 推理失败！检测到 AttributeError: {e}")
            print("这很可能就是对 HuggingFace 文本编码器中的 nn.Embedding 层进行量化导致的兼容性问题。")
            print("请使用只量化 Linear 层的模型 (`quantized_LINEAR_ONLY_INT8.pt`) 来进行推理。")
            return

        # 4. Softmax 转换为置信度
        probs = F.softmax(logits_per_image, dim=-1)
        
    return probs.squeeze(0).tolist()


def display_results(probs: list, texts: list, image_name: str):
    """
    格式化输出结果，模仿附件照片效果。
    """
    results = sorted(zip(probs, texts), key=lambda x: x[0], reverse=True)
    
    print("\n" + "="*70)
    print(f"图 像: {image_name}")
    print("-" * 70)
    
    best_match_text = results[0][1]
    best_match_prob = results[0][0] * 100
    
    for i, (prob, text) in enumerate(results, 1):
        prob_percent = prob * 100
        print(f"{i}. {prob_percent:.3f}% → {text}")
        
    print("-" * 70)
    print(f"最匹配描述 (置信度 {best_match_prob:.3f}%)")
    print(f"预测结果: \"{best_match_text}\"")
    print("="*70)


if __name__ == "__main__":
    # 1. 加载模型
    quantized_model = load_model()
    
    # 2. 执行推理
    if quantized_model:
        probabilities = run_inference(quantized_model, IMAGE_PATH, CANDIDATE_TEXTS)
    
        # 3. 展示结果
        if probabilities:
            display_results(probabilities, CANDIDATE_TEXTS, os.path.basename(IMAGE_PATH))