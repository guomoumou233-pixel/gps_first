# quantize_linear_and_save_state_dict.py
# 结合了 '只量化 Linear' 和 '只保存 state_dict' 的最佳实践

import os
import torch
import torch.nn as nn
from tiny_student_model import LightweightStudentCLIP # 假设这个文件在同目录下

def quantize_and_save():
    # --- 路径配置 (从您的脚本中获取) ---
    fp32_path = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
    # 使用新的保存路径来区分文件
    save_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_LINEAR_ONLY_state_dict.pt"

    print("🚀 步骤 1: 加载 FP32 模型...")
    model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    # 确保加载的是原始 FP32 模型的权重
    model.load_state_dict(torch.load(fp32_path, map_location="cpu"))
    model.eval()

    # --- 关键修改点：只量化 Linear 层 ---
    print("🚀 步骤 2: 执行 Weight-Only INT8 动态量化 (仅 Linear 层)...")
    
    # 使用 qconfig_spec 明确指定只对 nn.Linear 进行动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            # 仅包含 Linear 层的配置
            nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
        },
        dtype=torch.qint8
    )
    print("量化完成！")

    # --- 关键修改点：只保存 state_dict ---
    print(f"🚀 步骤 3: 仅保存量化后的模型权重字典 (state_dict) 到 {save_path}...")
    
    # 仅保存 state_dict 是 PyTorch 2.6+ 兼容的最安全方式
    torch.save(quantized_model.state_dict(), save_path)
    print("量化权重已保存！")

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
    quantize_and_save()