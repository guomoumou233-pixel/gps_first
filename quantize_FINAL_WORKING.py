# quantize_FINAL_WORKING.py
import os
import torch
import torch.nn as nn
from tiny_student_model import LightweightStudentCLIP

def quantize_and_save_whole_model():
    fp32_path = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
    save_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_student_int8_WHOLE_MODEL.pt"

    print("正在加载 FP32 模型...")
    model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    model.load_state_dict(torch.load(fp32_path, map_location="cpu"))
    model.eval()

    print("正在执行 Weight-Only INT8 量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            nn.Linear:    torch.ao.quantization.default_dynamic_qconfig,
            nn.Embedding: torch.ao.quantization.float_qparams_weight_only_qconfig
        },
        dtype=torch.qint8
    )
    print("量化完成！")

    # 关键：保存整个量化模型对象（不是 state_dict）
    torch.save(quantized_model, save_path)
    print(f"完整量化模型已保存：{save_path}")

    orig = os.path.getsize(fp32_path) / 1024 / 1024
    q    = os.path.getsize(save_path) / 1024 / 1024
    print(f"原始 {orig:.1f} MB → 量化 {q:.1f} MB (压缩 {orig/q:.2f}x)")

if __name__ == "__main__":
    quantize_and_save_whole_model()