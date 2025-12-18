import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer
import open_clip
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy

# --------------------------- 导入你的模型定义 ---------------------------
# 必须确保 tiny_student_model.py 和 image_encoder.py 在同一路径下
from tiny_student_model import LightweightStudentCLIP

# --------------------------- 路径配置 ---------------------------
TEACHER_CHECKPOINT_PATH = "/root/mnist-clip/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = "/root/mnist-clip/data/RSICD_images"
JSON_PATH = "/root/mnist-clip/data/RSICD-en_cleaned.json"
PRETRAINED_STUDENT_PATH = "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt"
SAVE_DIR = "./quantized_models"

# --------------------------- 工具类定义 ---------------------------
class RSICDDataset(Dataset):
    def __init__(self, data_list, img_dir):
        self.data_list = data_list
        self.img_dir = img_dir
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = os.path.join(self.img_dir, item["imged_id"]) 
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return None
        return img, item["caption"]

def compute_distillation_loss(s_logits, t_logits, alpha=0.1, temp=4.0, device="cuda"):
    hard_loss = F.cross_entropy(s_logits, torch.arange(s_logits.size(0), device=device))
    soft_targets = F.softmax(t_logits / temp, dim=-1)
    kd_loss = F.kl_div(F.log_softmax(s_logits / temp, dim=-1), soft_targets, reduction='batchmean') * (temp**2)
    return alpha * hard_loss + (1.0 - alpha) * kd_loss

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return [b[0] for b in batch], [b[1] for b in batch]

# --------------------------- 主函数 ---------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # QAT 的准备工作通常建议在 CPU 上进行，训练时再转 GPU
    # 最后 Convert 必须回 CPU (除非使用 nightly build 的 GPU 量化后端)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # ==================== 1. 数据准备 (随机采样 1000) ====================
    print(f"正在加载数据集: {JSON_PATH}")
    with open(JSON_PATH) as f:
        full_data = json.load(f)
    
    # 随机选取 1000 个样本
    random.seed(42)
    qat_data = random.sample(full_data, 1000)
    print(f"已随机选取 {len(qat_data)} 张图片用于 QAT 训练。")

    dataset = RSICDDataset(qat_data, IMG_DIR)
    
    # QAT 推荐使用较小的 BatchSize，确保每个 Batch 统计稳定
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                            num_workers=4, collate_fn=collate_fn)

    # 预处理：QAT 阶段通常不建议做剧烈的增强，使用验证集的标准化处理即可
    # 这样可以让量化参数（scale/zero_point）更稳定地收敛
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ==================== 2. 加载 Teacher (用于监督) ====================
    print("正在加载 Teacher 模型...")
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=TEACHER_CHECKPOINT_PATH, device=device
    )
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False
    teacher_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # ==================== 3. 加载 Student 并准备 QAT ====================
    print("正在初始化 Student 模型...")
    # 注意：这里我们先在 CPU 上初始化，配置好量化规则后再移入 GPU
    student_model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    
    # 加载蒸馏后的 FP32 权重
    print(f"加载预训练权重: {PRETRAINED_STUDENT_PATH}")
    state_dict = torch.load(PRETRAINED_STUDENT_PATH, map_location='cpu')
    student_model.load_state_dict(state_dict)
    
    student_model.train() # QAT 需要在 train 模式下运行以更新 observer

    # ------------------ QAT 核心配置 (关键步骤) ------------------
    # 1. 设置后端 (x86 CPU 用 fbgemm, ARM 用 qnnpack)
    backend = 'fbgemm'
    student_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # 2. 定制化量化策略：只量化 Linear 层，其他层设为 None (保持 FP32)
    # 先将全局 qconfig 设为 None
    student_model.qconfig = None 
    
    # 遍历所有子模块，只给 Linear 层加上 qconfig
    print("正在配置量化规则：仅针对 nn.Linear 层进行 INT8 量化...")
    quantized_layers_count = 0
    for name, module in student_model.named_modules():
        if isinstance(module, nn.Linear):
            # 使用针对 QAT 优化的默认配置
            module.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            quantized_layers_count += 1
            
    print(f"共发现 {quantized_layers_count} 个 Linear 层将被量化。")

    # 3. 准备 QAT 模型 (插入 FakeQuantize 节点)
    # prepare_qat 会原地修改模型结构
    torch.quantization.prepare_qat(student_model, inplace=True)
    print("QAT 准备完成 (Fake Quantization 节点已插入)。")

    # 将准备好的模型移动到 GPU 进行训练
    student_model.to(device)
    student_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # ==================== 4. QAT 训练循环 (微调) ====================
    # QAT 只需要很少的 epoch (通常 1-5 轮) 和很小的学习率
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-6, weight_decay=1e-4)
    
    epochs = 1 # 1000 张图微调 1-2 轮通常足够校准 observer
    print(f"\n开始 QAT 微调，共 {epochs} 轮...")

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0
        
        for pil_images, captions in tqdm(dataloader, desc=f"QAT Epoch {epoch+1}"):
            if pil_images is None: continue
            
            # 数据准备
            img_tensor = torch.stack([preprocess(img) for img in pil_images]).to(device)
            
            # --- Teacher Forward (无梯度) ---
            with torch.no_grad():
                t_tokens = teacher_tokenizer(captions).to(device)
                img_f = teacher_model.encode_image(img_tensor)
                txt_f = teacher_model.encode_text(t_tokens)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                logit_scale = teacher_model.logit_scale.exp()
                t_logits_i = logit_scale * img_f @ txt_f.t()
                t_logits_t = t_logits_i.t()

            # --- Student Forward (QAT 模式) ---
            # 此时前向传播是模拟 INT8 行为的 (Fake Quantization)
            text_inputs = student_tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            s_logits_i, s_logits_t = student_model(img_tensor, text_inputs.input_ids, text_inputs.attention_mask)

            # --- Loss ---
            loss = (compute_distillation_loss(s_logits_i, t_logits_i, device=device, alpha=0.1, temp=4.0) +
                    compute_distillation_loss(s_logits_t, t_logits_t, device=device, alpha=0.1, temp=4.0)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    # ==================== 5. 模型转换 (Convert) ====================
    print("\n正在将 QAT 模型转换为推理模型 (CPU)...")
    
    # 转换必须在 CPU 上进行
    student_model.cpu()
    student_model.eval()
    
    # 转换：这步操作会将 FakeQuantize + Linear 融合成真正的 QuantizedLinear
    # 权重的 dtype 会变成 qint8
    quantized_model = torch.quantization.convert(student_model, inplace=False)
    
    print("转换完成！")
    
    # ==================== 6. 验证与保存 ====================
    # 简单验证一下转换后的模型层
    print("\n检查 Text Encoder 的某个层是否已量化:")
    # 打印其中一个层看看是否变成了 QuantizedLinear
    try:
        # 这里的路径取决于 HuggingFace 具体的层命名，仅作为示例检查
        sample_layer = quantized_model.text_model.encoder.layers[0].self_attn.q_proj
        print(f"Text Encoder Layer 0 Q-Proj type: {type(sample_layer)}")
        # 预期输出: <class 'torch.ao.nn.quantized.modules.linear.Linear'>
    except:
        print("无法精确定位层，打印模型结构摘要...")
        # print(quantized_model) # 结构太长，此处省略

    # 保存
    save_path = os.path.join(SAVE_DIR, "qat_quantized_student.pt")
    # 保存 state_dict (这是最稳妥的方式，但在推理时需要先实例化同样结构的模型然后 convert 才能 load)
    # 或者保存整个模型对象 (Script 方式通常对 Quantized Model 支持有限，Pickle 方式依赖代码结构)
    
    torch.save(quantized_model.state_dict(), save_path)
    print(f"\n[Success] 量化模型权重已保存至: {save_path}")
    
    # 保存一个脚本化的版本（可选，视 PyTorch 版本支持情况而定）
    # try:
    #     scripted_model = torch.jit.script(quantized_model)
    #     torch.jit.save(scripted_model, os.path.join(SAVE_DIR, "qat_quantized_student_scripted.pt"))
    #     print("Scripted 模型已保存。")
    # except Exception as e:
    #     print(f"Script 保存跳过 (这在量化 Transformer 时很常见): {e}")

    # 打印文件大小对比
    fp32_size = os.path.getsize(PRETRAINED_STUDENT_PATH) / (1024 * 1024)
    int8_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n文件大小对比:")
    print(f"FP32 Model: {fp32_size:.2f} MB")
    print(f"INT8 Model: {int8_size:.2f} MB")
    print("注意：由于只量化了 Linear 层，其他 Conv 层仍为 FP32，压缩率取决于 Linear 层占比。")

if __name__ == "__main__":
    main()