# quantize_to_int4_quanto_0_2_0_fixed.py
# 专为 quanto==0.2.0 定制：添加 axis=0 参数

import os
import torch
import torch.nn as nn
from tiny_student_model import LightweightStudentCLIP
import quanto
from quanto import qint4  # 0.2.0 中存在

# 1. 加载原始模型
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
model.load_state_dict(torch.load(
    "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt",
    map_location="cpu"
))
model.eval()

print("正在将所有 Linear 层量化为 INT4 (quanto 0.2.0 专用方案)...")

# 0.2.0 官方推荐写法：先 quantize_weight → 再 QLinear.from_linear
replaced = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"量化: {name} -> {module.in_features}x{module.out_features}")

        # Step 1: 只量化权重（INT4） - 关键修复：添加 axis=0
        quanto.quantize_weight(module, qint4, axis=0, group_size=64)  # ← 修复：添加 axis=0

        # Step 2: 替换为 quanto 的 QLinear（完成包装）
        qlinear = quanto.QLinear.from_linear(module)  # 0.2.0 正确用法
        # 手动替换到父模块
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, qlinear)
        replaced += 1

print(f"成功完成！共替换 {replaced} 个 Linear 层为 INT4")

# 保存完整量化模型
save_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_INT4_quanto_0_2_0_fixed.pt"
torch.save(model, save_path)

# 大小对比
fp32_path = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
fp32_size = os.path.getsize(fp32_path) / 1024 / 1024
int4_size = os.path.getsize(save_path) / 1024 / 1024
print(f"原始 FP32 : {fp32_size:.1f} MB")
print(f"INT4 模型 : {int4_size:.1f} MB")
print(f"总压缩比  : {fp32_size/int4_size:.2f}x")
print(f"相比 INT8 再压缩 : {62.4/int4_size:.2f}x (假设你 INT8 为 62MB)")
print("INT4 量化大功告成！可直接用于 CPU 推理！")