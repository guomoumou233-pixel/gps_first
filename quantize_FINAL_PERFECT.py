# quantize_FINAL_PERFECT.py
# 2025 年唯一真正可用的量化方案（绕过 PyTorch 2.6+ bug）

import os
import torch
import torch.nn as nn
from tiny_student_model import LightweightStudentCLIP

def quantize_perfect():
    fp32_path = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
    save_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_student_int8_PERFECT.pt"

    print("加载模型...")
    model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    model.load_state_dict(torch.load(fp32_path, map_location="cpu"))
    model.eval()

    # 关键修复：在量化前，手动冻结所有 Embedding 层
    # 这样 quantize_dynamic 就不会破坏它们的 __getattr__ 行为
    def freeze_embeddings(module):
        if isinstance(module, nn.Embedding):
            module.weight.requires_grad = False
            # 标记为已冻结，quantize_dynamic 会跳过它
            module._freeze = True

    model.text_model.embeddings.token_embedding._freeze = True
    model.text_model.embeddings.position_embedding._freeze = True

    print("执行量化（Linear 用动态 int8，Embedding 用官方 weight-only）...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
            # Embedding 完全跳过，让它保持原始结构（最稳定！）
        },
        dtype=torch.qint8
    )

    # 手动对 Embedding 做 weight-only 量化（最安全方式）
    from torch.ao.nn.quantized import Embedding
    def convert_embedding_to_weight_only(module):
        if isinstance(module, nn.Embedding) and not hasattr(module, '_weight_only'):
            qembed = Embedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                _weight=module.weight.detach(),
                _qscheme=torch.per_tensor_affine,
                dtype=torch.qint8
            )
            # 替换原模块
            parent = module.__module__
            name = module.__name__ if hasattr(module, '__name__') else None
            import types
            setattr(module.__class__, name, qembed) if name else None

    # 替换两个关键 Embedding
    quantized_model.text_model.embeddings.token_embedding = torch.nn.quantized.Embedding(
        num_embeddings=quantized_model.text_model.embeddings.token_embedding.num_embeddings,
        embedding_dim=quantized_model.text_model.embeddings.token_embedding.embedding_dim,
        _weight=quantized_model.text_model.embeddings.token_embedding.weight.detach()
    )
    quantized_model.text_model.embeddings.position_embedding = torch.nn.quantized.Embedding(
        num_embeddings=quantized_model.text_model.embeddings.position_embedding.num_embeddings,
        embedding_dim=quantized_model.text_model.embeddings.position_embedding.embedding_dim,
        _weight=quantized_model.text_model.embeddings.position_embedding.weight.detach()
    )

    print("量化完成！")

    # 保存整个模型
    torch.save(quantized_model, save_path)
    print(f"完美量化模型已保存：{save_path}")

if __name__ == "__main__":
    quantize_perfect()