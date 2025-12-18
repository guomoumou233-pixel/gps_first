import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from torchvision import transforms
from PIL import Image
import os
import sys

# 必须导入您的 StudentCLIP 类，它依赖于 image_encoder.py
# 确保 StudentCLIP.py 和 image_encoder.py 文件存在于当前目录
try:
    from StudentCLIP import StudentCLIP
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 StudentCLIP.py 文件在当前目录下，并且它能正确导入 image_encoder。")
    sys.exit(1)


# ----------------------------------------------------------------------
# 1. 推理主函数
# ----------------------------------------------------------------------
@torch.no_grad()
def inference():
    # === 配置区域 (与您的要求和先前脚本保持一致) ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEACHER_MODEL_NAME = 'openai/clip-vit-base-patch32'
    
    # 权重路径
    MODEL_WEIGHTS_PATH = '/root/mnist-clip/student_clip_best_model.pt' 
    
    # 零样本分类输入
    IMAGE_PATH = '/root/mnist-clip/RS_images_2800/RS_images_2800/fResident/f003.jpg'
    LABELS = [
        "grass", "field", "Industry", "riverlake", 
        "forest", "resident", "parking"
    ]
    # 使用上下文 Prompt 提高准确性
    TEXT_PROMPTS = [f"a remote sensing image of {label}." for label in LABELS]
    
    print("-" * 60)
    print("🚀 正在加载训练完成的学生模型...")
    print(f"🔄 当前设备: {DEVICE}")

    # === 1. 初始化模型与处理器 ===
    model = StudentCLIP(teacher_model_name=TEACHER_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(TEACHER_MODEL_NAME)
    
    # === 2. 加载权重 ===
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
            print(f"✅ 成功加载学生模型权重: {MODEL_WEIGHTS_PATH}")
        except RuntimeError as e:
            print(f"❌ 错误: 权重文件加载失败 (结构可能不匹配): {e}")
            return
    else:
        print(f"❌ 错误: 未找到权重文件 {MODEL_WEIGHTS_PATH}。请检查路径。")
        return
        
    model.eval()

    # === 3. 数据预处理 ===
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到图像文件 {IMAGE_PATH}")
        return
    
    # 图像预处理
    image_inputs = processor.image_processor(image, return_tensors="pt")
    image_tensor = image_inputs.pixel_values.to(DEVICE)
    
    # 文本处理
    text_inputs = processor.tokenizer(
        TEXT_PROMPTS, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(DEVICE)
    attention_mask = text_inputs.attention_mask.to(DEVICE)

    # === 4. 执行推理 (修正了 ValueError 和 IndentationError) ===
    # 确保此行相对于其上一个逻辑行（如 text_inputs = ...）正确缩进
    logits_per_image, _, _, _ = model(image_tensor, input_ids, attention_mask)

    # === 5. 输出结果 ===
    # 相似度 Logits 转为概率
    probs = logits_per_image.softmax(dim=-1).squeeze(0)
    
    # 获取最高概率的索引
    best_match_index = probs.argmax().item()
    predicted_label = LABELS[best_match_index]
    
    # 格式化输出
    print("-" * 60)
    print(f"推理图像路径: {IMAGE_PATH}")
    print(f"最终预测结果: 【{predicted_label}】")
    print("-" * 60)
    print("标签 Softmax 相似度得分:")
    
    # 打印每个标签的概率，并按概率降序排列
    results = sorted(zip(LABELS, probs.tolist()), key=lambda x: x[1], reverse=True)
    
    for label, prob in results:
        print(f"  {label:<10}: {prob:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    inference()