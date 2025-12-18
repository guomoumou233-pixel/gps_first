import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import warnings

# 忽略TypedStorage警告（已知旧PyTorch问题，升级PyTorch可解决）
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# 量化库引入（QAT需要）
from torch.ao.quantization import (
    QuantStub, DeQuantStub, prepare_qat,
    QConfig, MinMaxObserver, FakeQuantize,
)

# 彻底绕过 Embedding 量化的实现
class FakeQuantDisabledEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, weight=None):
        super().__init__()
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

        out = self.weight.index_select(0, input_ids.reshape(-1))
        out = out.view(*input_ids.shape, self.embedding_dim)
        return out

    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

# 从 tiny_student_model.py 导入
from tiny_student_model import LightweightStudentCLIP

class QuantizableStudentWrapper(nn.Module):
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

# 自定义 QConfig
def create_custom_qat_qconfig():
    w_observer = MinMaxObserver.with_args(dtype=torch.qint8, quant_min=-127, quant_max=127)
    a_observer = MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255)
    return QConfig(activation=FakeQuantize.with_args(observer=a_observer),
                   weight=FakeQuantize.with_args(observer=w_observer))

if __name__ == "__main__":
    # 参数设置（从训练脚本中复制默认值）
    vision_variant = 'L1'
    model_path = '/root/mnist-clip/remoteclip_qat_student_fixed/BEST_student_qat.pt'
    image_path = '/root/mnist-clip/data/RSICD_images/airport_3.jpg'
    texts = [
        "an airport with many planes",
        "a parking lot with many cars",
        "a river passing through the city",
        "green farmland and fields",
        "a large bridge over water"
    ]

    # 设备设置（QAT模型可用CUDA或CPU，优先CUDA）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建学生模型
    print("Building Student model...")
    original_student = LightweightStudentCLIP(vision_variant=vision_variant, projection_dim=512)

    # 替换 Embedding
    replaced = 0
    for name, module in list(original_student.named_modules()):
        if isinstance(module, nn.Embedding):
            print(f"Replacing Embedding: {name}")
            new_emb = FakeQuantDisabledEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                weight=module.weight
            ).to(device)
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = original_student.get_submodule(parent_name) if parent_name else original_student
            setattr(parent, child_name, new_emb)
            replaced += 1
    print(f"Successfully replaced {replaced} Embedding layers.")

    # 排除分组卷积
    for name, module in original_student.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            print(f"Excluding grouped conv: {name} (groups={module.groups})")

    # 设置 QConfig 并禁用某些模块的量化
    custom_qconfig = create_custom_qat_qconfig()
    original_student.qconfig = custom_qconfig

    def disable_quant_for_module(model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, FakeQuantDisabledEmbedding)):
                module.qconfig = None
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                module.qconfig = None

    disable_quant_for_module(original_student)

    # 包装模型
    qat_model = QuantizableStudentWrapper(original_student)
    qat_model.to(device)

    # 准备 QAT（插入fake quant节点）
    qat_model = prepare_qat(qat_model, inplace=False)

    # 加载 QAT 权重
    try:
        qat_model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Load error: {e}")
        raise

    qat_model.to(device)
    qat_model.eval()  # 设置为eval模式（QAT推理使用fake quant模拟量化）

    # Tokenizer
    student_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 图像预处理（使用验证预处理）
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Tokenize 文本
    text_inputs = student_tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

    # 推理
    with torch.no_grad():
        s_logits_i, s_logits_t = qat_model(img_tensor, text_inputs.input_ids, text_inputs.attention_mask)

    # 获取相似度 (logits_per_image [1, 5])
    similarities = F.softmax(s_logits_i[0], dim=0).cpu().numpy()

    # 输出结果
    print("Similarity Scores (Softmax):")
    for text, score in zip(texts, similarities):
        print(f"{text}: {score:.4f}")