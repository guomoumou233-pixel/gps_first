import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip
from transformers import CLIPTokenizer
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy

# 引入您的学生模型定义 (假设 tiny_student_model.py 存在)
from tiny_student_model import LightweightStudentCLIP

# 引入量化库
from torch.ao.quantization import (
    QuantStub, DeQuantStub, prepare_qat, convert,
    QConfig, MinMaxObserver, FakeQuantize,
)

# =========================== 1. INT8 模型所需的关键辅助类 (来自量化训练脚本) ===========================

# 1.1 彻底绕过 Embedding 量化的实现 (用于重现 INT8 模型的结构)
class FakeQuantDisabledEmbedding(nn.Module):
    """
    用于替换 nn.Embedding，确保 QAT 流程跳过文本嵌入层。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, weight=None):
        super().__init__()
        # 在这里，我们只需要确保结构匹配，以便加载 state_dict
        if weight is not None:
            self.weight = nn.Parameter(weight.detach().clone())
        else:
            self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        # 手动实现 embedding lookup，确保是 FP32 运算
        out = self.weight.index_select(0, input_ids.reshape(-1))
        out = out.view(*input_ids.shape, self.embedding_dim)
        return out

# 1.2 QAT 模型 Wrapper
class QuantizableStudentWrapper(nn.Module):
    """
    用于包裹学生模型，添加 QuantStub/DeQuantStub
    """
    def __init__(self, student_model):
        super().__init__()
        self.student = student_model
        self.quant_img = QuantStub()
        self.dequant_i = DeQuantStub()
        self.dequant_t = DeQuantStub()

    def forward(self, img, input_ids, attention_mask):
        img = self.quant_img(img)
        s_logits_i, s_logits_t = self.student(img, input_ids, attention_mask)
        s_logits_i = self.dequant_i(s_logits_i)
        s_logits_t = self.dequant_t(s_logits_t)
        return s_logits_i, s_logits_t

# 1.3 QConfig 定义 (用于重现 INT8 模型的结构)
def create_custom_qat_qconfig():
    w_observer = MinMaxObserver.with_args(dtype=torch.qint8, quant_min=-127, quant_max=127)
    a_observer = MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255)
    return QConfig(activation=FakeQuantize.with_args(observer=a_observer),
                   weight=FakeQuantize.with_args(observer=w_observer))

# ================= 2. 配置路径 =================
MODEL_PATHS = {
    "RemoteCLIP (Teacher)": "/root/mnist-clip/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt",
    "Student (FP32)": "/root/mnist-clip/remoteclip_student_with_val2/BEST_student_model.pt",
    "Student (INT8)": "/root/mnist-clip/remoteclip_qat_student_fixed/BEST_student_int8.pt"
}

DATASET_DIR = "/root/mnist-clip/RS_images_2800/RS_images_2800"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = "cpu"
Q_BACKEND = "fbgemm" # 假设训练时使用的后端

# ================= 3. 数据集定义 =================
class RSZeroShotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        self.class_to_idx = {}
        for idx, subdir in enumerate(subdirs):
            # 类别解析：aGrass -> Grass
            class_name = subdir[1:] 
            self.class_names.append(class_name)
            self.class_to_idx[idx] = class_name
            
            class_dir = os.path.join(root_dir, subdir)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ================= 4. 模型加载函数 =================

def get_common_preprocess():
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def load_remoteclip_teacher():
    print(">>> Loading RemoteCLIP Teacher...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=MODEL_PATHS["RemoteCLIP (Teacher)"], device=DEVICE
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer, DEVICE

def load_student_fp32():
    print(">>> Loading Student FP32...")
    model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)
    state_dict = torch.load(MODEL_PATHS["Student (FP32)"], map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return model, get_common_preprocess(), tokenizer, DEVICE

def load_student_int8():
    print(">>> Loading Student INT8 (Reconstructing Quantization Graph)...")
    
    torch.backends.quantized.engine = Q_BACKEND
    base_model = LightweightStudentCLIP(vision_variant='L1', projection_dim=512)

    # 1. 替换 Embedding (重现结构)
    for name, module in list(base_model.named_modules()):
        if isinstance(module, nn.Embedding):
            new_emb = FakeQuantDisabledEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                weight=module.weight
            ).to(CPU_DEVICE) 
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = base_model.get_submodule(parent_name) if parent_name else base_model
            setattr(parent, child_name, new_emb)
            
    # 2. 设置 QConfig 并禁用不需要量化的模块
    custom_qconfig = create_custom_qat_qconfig()
    base_model.qconfig = custom_qconfig

    def disable_quant_for_module(model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, FakeQuantDisabledEmbedding)):
                module.qconfig = None
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                module.qconfig = None
    
    disable_quant_for_module(base_model)
    
    # 3. 包装、Prepare、Convert (在 CPU 上进行)
    qat_model = QuantizableStudentWrapper(base_model)
    qat_model.eval() 
    qat_model.to(CPU_DEVICE)
    
    # 注意：这里调用 prepare_qat 时模型必须是 train()，
    # 但我们加载 INT8 模型不需要再进行 QAT，我们只需要构建结构。
    # 我们用 copy.deepcopy 来保证原始模型状态不变
    int8_model_prepared = prepare_qat(copy.deepcopy(qat_model), inplace=False)

    int8_model = convert(int8_model_prepared, inplace=False)
    
    # 4. 加载权重
    state_dict = torch.load(MODEL_PATHS["Student (INT8)"], map_location=CPU_DEVICE)
    int8_model.load_state_dict(state_dict)
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # INT8 模型运行在 CPU 上
    return int8_model, get_common_preprocess(), tokenizer, CPU_DEVICE

# ================= 5. 测评核心逻辑 =================

def run_evaluation(model_name, model_loader_func):
    try:
        model, preprocess, tokenizer, device = model_loader_func()
    except Exception as e:
        print(f"❌ 加载模型 {model_name} 失败: {e}")
        return model_name, 0.0

    dataset = RSZeroShotDataset(DATASET_DIR, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 1. 生成 Zero-shot 文本特征
    print(f"Generating Zero-shot Classifier for {len(dataset.class_names)} classes...")
    prompts = [f"a satellite photo of {c}." for c in dataset.class_names]
    
    with torch.no_grad():
        if "Teacher" in model_name:
            # Teacher model uses OpenCLIP tokenizer
            text_tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(text_tokens)
            
        else: # Student model (FP32 or INT8) uses HuggingFace tokenizer
            text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            
            if "INT8" in model_name:
                # INT8 模型，Text model 是 FP32，直接调用 internal student's text encoder
                text_features = model.student.encode_text(text_inputs.input_ids, text_inputs.attention_mask)
            else:
                text_features = model.encode_text(text_inputs.input_ids, text_inputs.attention_mask)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 2. 推理循环
    correct = 0
    total = 0
    
    print(f"Start Evaluating {model_name} on {device}...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取图像特征
            if "Teacher" in model_name:
                image_features = model.encode_image(images)
                logit_scale = model.logit_scale.exp()
            
            elif "INT8" in model_name:
                # INT8 推理流程：手动触发 QuantStub -> 视觉模型 (量化) -> DeQuantStub
                # 图像特征提取
                q_img = model.quant_img(images)
                img_emb = model.student.vision_model(q_img) 
                image_features = model.dequant_i(img_emb)
                
                # 获取 Logit Scale
                logit_scale = model.student.logit_scale.exp()
            
            else: # FP32 Student
                image_features = model.encode_image(images)
                logit_scale = model.logit_scale.exp()

            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度 Logits
            logits = logit_scale * image_features @ text_features.t()

            # 预测
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\n=============================================")
    print(f"📊 {model_name} Final Accuracy:")
    print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"=============================================")
    return model_name, accuracy

if __name__ == "__main__":
    results = []
    
    print("==================================================")
    print("   RemoteCLIP & Student Zero-shot Evaluation")
    print("==================================================")
    
    results.append(run_evaluation("RemoteCLIP (Teacher)", load_remoteclip_teacher))
    results.append(run_evaluation("Student (FP32)", load_student_fp32))
    results.append(run_evaluation("Student (INT8)", load_student_int8))

    print("\n\n=============== 最终测评结果 SUMMARY ===============")
    for name, acc in results:
        print(f"{name}: {acc:.2f}%")
    print("=====================================================")