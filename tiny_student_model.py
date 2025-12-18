import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPConfig
from transformers import CLIPTokenizer
# 导入你提供的轻量化视觉编码器
# 假设 image_encoder.py 在同一目录下
from image_encoder import CLIPSwiftFormerEncoder

class LightweightStudentCLIP(nn.Module):
    """
    轻量化 Student CLIP 模型
    - 视觉编码器: 自定义 SwiftFormer (来自 image_encoder.py)
    - 文本编码器: TinyCLIP-ViT-61M (来自 Hugging Face)
    """
    def __init__(self, 
                 vision_variant='L1', 
                 projection_dim=512, 
                 tinyclip_model_name="wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"):
        super().__init__()

        # ----------------------------------------------------------------------
        # 1. 初始化视觉编码器 (Visual Encoder)
        # ----------------------------------------------------------------------
        print(f"Initializing Custom Vision Encoder (SwiftFormer-{vision_variant})...")
        self.vision_model = CLIPSwiftFormerEncoder(
            projection_dim=projection_dim,
            model_variant=vision_variant
        )

        # ----------------------------------------------------------------------
        # 2. 初始化文本编码器 (Text Encoder)
        # ----------------------------------------------------------------------
        print(f"Loading Text Encoder from {tinyclip_model_name}...")
        # 我们加载整个 CLIP 模型，然后只提取文本部分，丢弃它的视觉部分以节省显存
        try:
            # 尝试从 Hugging Face 加载预训练权重
            pretrained_clip = CLIPModel.from_pretrained(tinyclip_model_name)
            
            # 提取文本 Transformer 部分
            self.text_model = pretrained_clip.text_model
            # 提取文本投影层 (Text Projection: hidden_size -> projection_dim)
            self.text_projection = pretrained_clip.text_projection
            # 提取 Logit Scale (温度系数)
            self.logit_scale = pretrained_clip.logit_scale
            
            # 检查维度是否匹配
            if self.text_projection.out_features != projection_dim:
                raise ValueError(f"维度不匹配: 文本编码器输出 {self.text_projection.out_features} "
                                 f"与预设投影维度 {projection_dim} 不一致。")
            
            # 清理不再需要的原始视觉模型，释放内存
            del pretrained_clip.vision_model
            del pretrained_clip
            
        except OSError:
            print("Warning: 无法连接 HuggingFace 或找不到模型，使用随机初始化的文本编码器作为占位符（仅用于调试）。")
            config = CLIPConfig()
            dummy_model = CLIPModel(config)
            self.text_model = dummy_model.text_model
            self.text_projection = dummy_model.text_projection
            self.logit_scale = dummy_model.logit_scale

    def encode_image(self, image):
        """仅提取图像特征 (未归一化)"""
        return self.vision_model(image)

    def encode_text(self, input_ids, attention_mask=None):
        """仅提取文本特征 (未归一化)"""
        # 1. Transformer 提取特征
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 2. 获取 [EOS] token 的特征 (pooled_output)
        # HuggingFace CLIP 实现中 text_model 输出的 pooler_output 已经是选好的特征
        pooled_output = text_outputs.pooler_output
        
        # 3. 投影到联合空间
        text_features = self.text_projection(pooled_output)
        return text_features

    def forward(self, image, input_ids, attention_mask=None, return_loss=False):
        """
        标准 CLIP 前向传播
        返回: logits_per_image, logits_per_text
        """
        # 1. 获取特征
        image_features = self.encode_image(image)       # [batch_size, 512]
        text_features = self.encode_text(input_ids, attention_mask) # [batch_size, 512]

        # 2. 特征归一化 (L2 Normalize)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 3. 计算余弦相似度 (Cosine Similarity) * Temperature
        # logit_scale 通常是 learnable parameter，需要 exp() 保证为正
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

# ----------------------------------------------------------------------
# 测试代码
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import CLIPTokenizer

    # 1. 实例化模型
    student_model = LightweightStudentCLIP()
    
    # 将模型移动到 GPU (如果可用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(device)
    student_model.train() # 设置为训练模式

    # 2. 准备 Dummy 数据
    # 图像: [Batch=4, Channel=3, Height=224, Width=224]
    dummy_images = torch.randn(4, 3, 224, 224).to(device)
    
    # 文本: 使用 Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    texts = ["a photo of a cat", "a photo of a dog", "a car", "a swift transformer"]
    text_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    # 3. 前向传播
    print("\n--- Running Forward Pass ---")
    logits_img, logits_txt = student_model(
        image=dummy_images,
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask']
    )

    print(f"Logits Image Shape: {logits_img.shape}") # Should be [4, 4]
    print(f"Logits Text Shape:  {logits_txt.shape}") # Should be [4, 4]

    # 4. 打印参数统计
    print("\n--- Parameter Stats ---")
    total_params = sum(p.numel() for p in student_model.parameters())
    vis_params = sum(p.numel() for p in student_model.vision_model.parameters())
    txt_params = sum(p.numel() for p in student_model.text_model.parameters()) + \
                 sum(p.numel() for p in student_model.text_projection.parameters())
    
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    print(f"Vision Encoder (SwiftFormer): {vis_params / 1e6:.2f}M")
    print(f"Text Encoder (TinyCLIP): {txt_params / 1e6:.2f}M")