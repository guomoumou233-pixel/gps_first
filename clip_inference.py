import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np

# ----------------------------------------------------------------------
# 教师模型封装 (必须与训练时的结构完全一致)
# ----------------------------------------------------------------------

class CLIPTeacherModel(nn.Module):
    """
    封装 Hugging Face CLIP 模型，用于加载微调后的权重。
    """
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        # 1. 加载完整的 CLIP 模型
        self.clip = CLIPModel.from_pretrained(model_name)
        self.logit_scale = self.clip.logit_scale

    # 仅保留推理所需的特征提取方法
    
    def get_image_features(self, images):
        """ 仅计算图像特征 (归一化后的嵌入) """
        vision_outputs = self.clip.vision_model(pixel_values=images)
        image_embeds = self.clip.visual_projection(vision_outputs.pooler_output)
        # CLIP 特征必须归一化
        image_features = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_features
    
    def get_text_features(self, input_ids, attention_mask=None):
        """ 仅计算文本特征 (归一化后的嵌入) """
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.clip.text_projection(text_outputs.pooler_output)
        # CLIP 特征必须归一化
        text_features = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_features


# ----------------------------------------------------------------------
# 推理主函数
# ----------------------------------------------------------------------

@torch.no_grad()
def inference_clip(
    image_path: str,
    candidate_texts: list,
    model_weights_path: str,
    model_name: str = 'openai/clip-vit-base-patch32'
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型和处理器
    teacher_model = CLIPTeacherModel(model_name=model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 2. 加载微调后的权重
    try:
        # map_location 确保权重文件可以在任何设备上加载
        teacher_model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        print(f"✅ 成功加载微调权重: {model_weights_path}")
    except Exception as e:
        # 如果加载失败，模型将使用预训练的 CLIP 默认权重
        print(f"❌ 错误: 无法加载权重文件。请检查路径或文件是否完整。错误信息: {e}")
        return None
    
    teacher_model.eval()
    
    # 3. 图像预处理 (与训练时保持一致)
    
    # 修正 CLIPProcessor 属性访问（兼容新版本 transformers）
    try:
        image_size = processor.image_processor.size['shortest_edge']
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
    except AttributeError:
        # 兼容旧版本 transformers
        print("警告: 使用旧版 processor 属性访问。")
        image_size = processor.size['shortest_edge']
        image_mean = processor.image_mean
        image_std = processor.image_std
    
    img_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    try:
        img = Image.open(image_path).convert("RGB")
        image_tensor = img_transform(img).unsqueeze(0).to(DEVICE) # [1, 3, H, W]
    except FileNotFoundError:
        return f"错误: 找不到图像文件: {image_path}"
    except Exception as e:
        return f"错误: 加载或处理图像失败: {e}"

    # 4. 文本 tokenization
    text_inputs = processor.tokenizer(
        candidate_texts, 
        padding='max_length', 
        truncation=True, 
        max_length=77, 
        return_tensors='pt'
    )
    input_ids = text_inputs['input_ids'].to(DEVICE)
    attention_mask = text_inputs['attention_mask'].to(DEVICE)

    # 5. 特征提取
    image_features = teacher_model.get_image_features(image_tensor)
    text_features = teacher_model.get_text_features(input_ids, attention_mask)

    # 6. 计算相似度 (余弦相似度)
    # 相似度矩阵: [1, N_text]，范围在 [-1, 1]
    similarity_scores = (image_features @ text_features.T) 
    
    # 结果转为 NumPy 数组并挤压维度
    similarity_scores = similarity_scores.squeeze(0).cpu().numpy()

    # 7. 找出最佳匹配
    best_match_index = np.argmax(similarity_scores)
    best_match_text = candidate_texts[best_match_index]
    
    # 8. 格式化输出
    results = {
        "image_path": image_path,
        "best_match_class": best_match_text,
        "similarity_results": {}
    }
    
    for i, text in enumerate(candidate_texts):
        # 将相似度转换为百分比形式，或仅保留小数点后四位
        similarity_value = float(similarity_scores[i])
        results["similarity_results"][text] = f"{similarity_value:.4f}"

    return results

# ----------------------------------------------------------------------
# 运行示例
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # --- 配置参数 ---
    # 根据您的要求，设置权重文件的路径
    PTH_PATH = '/root/mnist-clip/fine_tuned_clip_teacher.pth'
    
    # 🚨 请将此路径替换为您要推理的实际图像文件路径
    TEST_IMAGE = '/root/mnist-clip/RS_images_2800/RS_images_2800/dRiverLake/d012.jpg' 
    
    # 遥感图像的候选类别文本描述
    CANDIDATE_CLASSES = [
        "Grass",
        "Field",
        "Industry",
        "RiverLake",
        "Forest", # 对应 aForest
        "Resident",
        "Parking",
    ]
    
    # --- 运行推理 ---
    print(f"正在对图像 {TEST_IMAGE} 进行推理...")
    
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ 找不到测试图像 {TEST_IMAGE}。请替换为您的实际图像路径。")
    elif not os.path.exists(PTH_PATH):
        print(f"\n❌ 找不到权重文件 {PTH_PATH}。请检查路径是否正确。")
    else:
        results = inference_clip(
            image_path=TEST_IMAGE,
            candidate_texts=CANDIDATE_CLASSES,
            model_weights_path=PTH_PATH
        )

        # --- 打印结果 ---
        if results is not None:
            print("\n--- 推理结果 ---")
            print(f"图像路径: {results['image_path']}")
            print(f"预测最佳类别: {results['best_match_class']}")
            print("\n与各候选文本的相似度 (余弦相似度, 范围 -1.0000 到 1.0000):")
            
            # 对相似度进行排序，以便更清晰地看到最佳匹配
            sorted_scores = sorted(
                results['similarity_results'].items(), 
                key=lambda item: float(item[1]), 
                reverse=True
            )
            
            for text, score in sorted_scores:
                print(f"  [相似度: {score}] - {text}")