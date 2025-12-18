# quantize_full_model_and_save_object.py
# 目标：量化 Linear 和 Embedding，并保存完整模型对象 (可能存在兼容性风险)

import os
import torch
import torch.nn as nn
from tiny_student_model import LightweightStudentCLIP

def quantize_full_and_save_object():
    # --- 路径配置 ---
    fp32_path = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
    # 保存完整模型对象，因此使用 .pt 后缀
    save_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_FULL_OBJECT_INT8.pt"

    print("🚀 步骤 1: 加载 FP32 模型...")
    model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    model.load_state_dict(torch.load(fp32_path, map_location="cpu"))
    model.eval()

    # --- 关键修改点：量化 Linear 和 Embedding ---
    print("🚀 步骤 2: 执行 Weight-Only INT8 动态量化 (包含 Linear 和 Embedding)...")
    
    # 采用 quantize_and_save_model.py 中的配置
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            nn.Linear:    torch.ao.quantization.default_dynamic_qconfig,
            nn.Embedding: torch.ao.quantization.float_qparams_weight_only_qconfig
        },
        dtype=torch.qint8
    )
    print("量化完成！")

    # --- 关键修改点：保存完整模型对象 ---
    print(f"🚀 步骤 3: 保存完整的量化模型对象 (可直接加载) 到 {save_path}...")
    
    # 直接保存完整的模型对象，包含结构和权重
    torch.save(quantized_model, save_path)
    print("完整量化模型对象已保存！")

    # --- 大小对比 ---
    if os.path.exists(fp32_path):
        orig = os.path.getsize(fp32_path) / 1024 / 1024
        q    = os.path.getsize(save_path) / 1024 / 1024
        print(f"\n模型大小对比:")
        print(f"原始 FP32: {orig:.1f} MB")
        print(f"量化 INT8: {q:.1f} MB (压缩 {orig/q:.2f}x)")
    else:
        print(f"\n警告: 找不到原始模型文件 {fp32_path}，跳过大小对比。")

if __name__ == "__main__":
    quantize_full_and_save_object()