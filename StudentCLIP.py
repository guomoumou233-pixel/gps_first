import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig
# 假设您的 image_encoder.py 文件名为 image_encoder.py 且在同一目录下
from image_encoder import CLIPSwiftFormerEncoder 

class StudentCLIP(nn.Module):
    def __init__(self, 
                 teacher_model_name='openai/clip-vit-base-patch32', 
                 embed_dim=512):
        super().__init__()
        
        # 1. 视觉编码器：您的轻量化 SwiftFormer
        # 注意：确保 SwiftFormer 的 projection_dim 与 CLIP 的 embed_dim 一致 (512)
        self.vision_model = CLIPSwiftFormerEncoder(projection_dim=embed_dim, model_variant='L1')
        
        # 2. 文本编码器：从预训练 CLIP 加载 (作为初始化)
        # 我们只加载 text_model 和 text_projection 部分
        original_clip = CLIPModel.from_pretrained(teacher_model_name)
        self.text_model = original_clip.text_model
        self.text_projection = original_clip.text_projection
        
        # 3. Logit Scale (可学习的温度系数)
        self.logit_scale = nn.Parameter(original_clip.logit_scale.clone())
        
        # 清理原始 CLIP 以节省内存 (只保留了 text 部分的引用)
        del original_clip

    def forward(self, images, input_ids, attention_mask=None):
        # --- Image Path ---
        # SwiftFormer 输出已经是投影后的 [B, 512]
        image_embeds = self.vision_model(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # --- Text Path ---
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # 取 pooler_output (通常是 EOS token 的特征)
        text_embeds = self.text_projection(text_outputs.pooler_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # --- Logits ---
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text, image_embeds, text_embeds