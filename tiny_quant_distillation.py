# tiny_quant_distillation_fixed.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
from tqdm import tqdm
from transformers import CLIPTokenizer
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import open_clip
import copy
import traceback

# =========================== 量化库引入 ===========================
from torch.ao.quantization import (
    QuantStub, DeQuantStub, prepare_qat, convert,
    QConfig, MinMaxObserver, FakeQuantize,
)

# =========================== 彻底绕过 Embedding 量化的实现 ===========================
class FakeQuantDisabledEmbedding(nn.Module):
    """
    完全不触发 PyTorch 量化系统 Embedding 检查的实现
    使用 index_select 手动实现 lookup，避免 F.embedding 和 nn.Embedding
    """
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

        # 手动实现 embedding lookup
        out = self.weight.index_select(0, input_ids.reshape(-1))
        out = out.view(*input_ids.shape, self.embedding_dim)
        return out

    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


# --------------------------- 你的学生模型 ---------------------------
from tiny_student_model import LightweightStudentCLIP 

# --------------------------- 路径 --------------------------
TEACHER_CHECKPOINT_PATH = "/root/mnist-clip/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
IMG_DIR = "/root/mnist-clip/data/RSICD_images"
JSON_PATH = "/root/mnist-clip/data/RSICD-en_cleaned.json"


# --------------------------- 数据集 & 损失函数 ---------------------------
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


# --------------------------- 自定义 QConfig ---------------------------
def create_custom_qat_qconfig():
    w_observer = MinMaxObserver.with_args(dtype=torch.qint8, quant_min=-127, quant_max=127)
    a_observer = MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255)
    return QConfig(activation=FakeQuantize.with_args(observer=a_observer),
                   weight=FakeQuantize.with_args(observer=w_observer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--vision_variant', type=str, default='L1')
    parser.add_argument('--save_dir', type=str, default='./remoteclip_qat_student_fixed')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--distill_T', type=float, default=4.0)
    parser.add_argument('--distill_alpha', type=float, default=0.1)
    parser.add_argument('--q_backend', type=str, default='fbgemm', choices=['fbgemm', 'qnnpack'])

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 1. Teacher ====================
    print("Loading Teacher model...")
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=TEACHER_CHECKPOINT_PATH, device=device
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # ==================== 2. 预处理 ====================
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ==================== 3. Student + 彻底修复 Embedding & 分组卷积 ====================
    print("Building and patching Student model for QAT...")
    original_student = LightweightStudentCLIP(vision_variant=args.vision_variant, projection_dim=512)

    # --- 步骤1：替换所有 nn.Embedding 为完全不可量化的实现 ---
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

            # 动态替换子模块
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = original_student.get_submodule(parent_name) if parent_name else original_student
            setattr(parent, child_name, new_emb)
            replaced += 1
    print(f"Successfully replaced {replaced} Embedding layers with quantization-safe version.\n")

    # --- 步骤2：排除所有分组卷积 (groups > 1) ---
    grouped_conv_paths = []
    for name, module in original_student.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            grouped_conv_paths.append(name)
            print(f"Excluding grouped conv from quantization: {name} (groups={module.groups})")

    # --- 步骤3：设置 QConfig 并手动禁用不需要量化的模块 ---
    custom_qconfig = create_custom_qat_qconfig()
    original_student.qconfig = custom_qconfig

    def disable_quant_for_module(model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, FakeQuantDisabledEmbedding)):
                module.qconfig = None
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                module.qconfig = None

    disable_quant_for_module(original_student)

    # --- 步骤4：包装 + prepare_qat ---
    qat_model = QuantizableStudentWrapper(original_student)
    qat_model.to(device)
    qat_model.train()

    # 关键：不使用 propagate_qconfig_to_module，直接 prepare_qat
    qat_model = prepare_qat(qat_model, inplace=False)
    print(f"QAT model ready. Backend: {args.q_backend}\n")

    # ==================== 5. Tokenizer & Optimizer ====================
    student_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    optimizer = torch.optim.AdamW(
        [p for p in qat_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6
    )

    # ==================== 6. 数据加载 ====================
    with open(JSON_PATH) as f:
        data = json.load(f)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = RSICDDataset(train_data, IMG_DIR)
    val_dataset   = RSICDDataset(val_data,   IMG_DIR)

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0: return None
        return [b[0] for b in batch], [b[1] for b in batch]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # ==================== 7. 训练循环 ====================
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ----- Train -----
        qat_model.train()
        train_loss = 0.0
        for pil_images, captions in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            if pil_images is None: continue

            img_tensor = torch.stack([train_preprocess(img) for img in pil_images]).to(device)
            text_tokens = teacher_tokenizer(captions).to(device)
            text_inputs = student_tokenizer(captions, padding=True, truncation=True,
                                              max_length=77, return_tensors="pt").to(device)

            with torch.no_grad():
                img_f = teacher_model.encode_image(img_tensor)
                txt_f = teacher_model.encode_text(text_tokens)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                logit_scale = teacher_model.logit_scale.exp()
                t_logits_i = logit_scale * img_f @ txt_f.t()
                t_logits_t = t_logits_i.t()

            s_logits_i, s_logits_t = qat_model(img_tensor, text_inputs.input_ids, text_inputs.attention_mask)

            loss = (compute_distillation_loss(s_logits_i, t_logits_i, alpha=args.distill_alpha, temp=args.distill_T, device=device) +
                    compute_distillation_loss(s_logits_t, t_logits_t, alpha=args.distill_alpha, temp=args.distill_T, device=device)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ----- Valid -----
        qat_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pil_images, captions in val_loader:
                if pil_images is None: continue
                img_tensor = torch.stack([val_preprocess(img) for img in pil_images]).to(device)
                text_tokens = teacher_tokenizer(captions).to(device)
                text_inputs = student_tokenizer(captions, padding=True, truncation=True,
                                                  max_length=77, return_tensors="pt").to(device)

                img_f = teacher_model.encode_image(img_tensor)
                txt_f = teacher_model.encode_text(text_tokens)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                logit_scale = teacher_model.logit_scale.exp()
                t_logits_i = logit_scale * img_f @ txt_f.t()
                t_logits_t = t_logits_i.t()

                s_logits_i, s_logits_t = qat_model(img_tensor, text_inputs.input_ids, text_inputs.attention_mask)

                loss = (compute_distillation_loss(s_logits_i, t_logits_i, alpha=args.distill_alpha, temp=args.distill_T, device=device) +
                        compute_distillation_loss(s_logits_t, t_logits_t, alpha=args.distill_alpha, temp=args.distill_T, device=device)) / 2
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end=" ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(qat_model.state_dict(), f"{args.save_dir}/BEST_student_qat.pt")

            # ===== 关键：安全导出 INT8 模型 =====
            try:
                torch.backends.quantized.engine = args.q_backend
                model_to_convert = copy.deepcopy(qat_model).to(device)
                model_to_convert.eval()
                print(" Converting QAT → INT8...", end="")
                int8_model = convert(model_to_convert, inplace=False)
                int8_model = int8_model.cpu()
                torch.save(int8_model.state_dict(), f"{args.save_dir}/BEST_student_int8.pt")
                print(" Success! INT8 model saved.")
            except Exception as e:
                print(f" Failed! INT8 conversion error: {e}")
                traceback.print_exc()
        else:
            patience_counter += 1
            print(f"(patience {patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping! Best epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}")
            break

        if epoch % 5 == 0:
            torch.save(qat_model.state_dict(), f"{args.save_dir}/student_qat_epoch{epoch}.pt")

    print(f"\nTraining finished! Best QAT model: {args.save_dir}/BEST_student_qat.pt")
    print(f"Best INT8 model: {args.save_dir}/BEST_student_int8.pt")


if __name__ == "__main__":
    main()