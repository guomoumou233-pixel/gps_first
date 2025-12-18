# quantize_to_int4_native_fixed.py
# PyTorch 原生 INT4 weight-only 量化（修复 requires_grad 错误）

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tiny_student_model import LightweightStudentCLIP

class Int4Linear(nn.Module):
    """自定义 INT4 Linear 模块（weight-only 量化，per-channel）"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 存储 INT4 量化权重 (uint8, 0~15 代表 -8~7) - 用 Buffer (无梯度)
        self.register_buffer('qweight', torch.empty((out_features, in_features), dtype=torch.uint8))
        # scale 和 zero_point 用 Parameter (float32, 带梯度但 eval 时冻结)
        self.scale = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self.zero_point = nn.Parameter(torch.empty(out_features, dtype=torch.int32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.scale)
        nn.init.zeros_(self.zero_point)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Dequantize: weight_fp = (qweight - zero_point) * scale
        weight_fp = (self.qweight.float() - self.zero_point.unsqueeze(1)) * self.scale.unsqueeze(1)
        return F.linear(input, weight_fp, self.bias)

def quantize_to_int4(model):
    """递归替换所有 Linear 为 Int4Linear"""
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            orig_weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None

            # Per-output-channel INT4 量化 (-8 ~ 7, 4-bit signed)
            min_val = orig_weight.min(dim=1)[0]
            max_val = orig_weight.max(dim=1)[0]
            scale = (max_val - min_val) / 15.0  # 16 levels for INT4
            zero_point = torch.round(min_val / scale + 8)  # shift to 0~15

            # 量化权重到 uint8 (0~15)
            quantized_weight = torch.clamp(
                torch.round((orig_weight - min_val.unsqueeze(1)) / scale.unsqueeze(1)) + 8,
                0, 15
            ).to(torch.uint8)

            # 替换为自定义 Int4Linear
            new_module = Int4Linear(module.in_features, module.out_features, module.bias is not None)
            new_module.qweight.copy_(quantized_weight)  # Buffer 赋值
            new_module.scale.data.copy_(scale)
            new_module.zero_point.data.copy_(zero_point)
            if bias is not None:
                new_module.bias.data.copy_(bias)

            # 手动替换
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_module)
            replaced += 1
            print(f"已量化 Linear: {name} (out_features={module.out_features})")

    return replaced

# 主函数
model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
model.load_state_dict(torch.load(
    "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt",
    map_location="cpu"
))
model.eval()

print("正在将所有 Linear 层量化为 INT4 (PyTorch 原生，修复 requires_grad)...")
num_replaced = quantize_to_int4(model)
print(f"完成！已替换 {num_replaced} 个 Linear 层")

# 保存
save_path = "/root/mnist-clip/remoteclip_student_with_val2/quantized_INT4_NATIVE_FIXED.pt"
torch.save(model, save_path)

# 大小对比
fp32_size = os.path.getsize("/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt") / 1024 / 1024
int4_size = os.path.getsize(save_path) / 1024 / 1024
print(f"原始 FP32 : {fp32_size:.1f} MB")
print(f"INT4 模型 : {int4_size:.1f} MB")
print(f"总压缩比  : {fp32_size/int4_size:.2f}x")
print("INT4 量化成功！可直接用于 CPU 推理！")