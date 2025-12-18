import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiftFormerMlp(nn.Module):
    """
    SwiftFormer Encoder 中的 Linear Block (MLP)。
    结构: Conv1x1 -> BN -> GeLU -> Conv1x1
    参考论文 Figure 3 (Bottom Right) "Linear" 部分
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.bn = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAdditiveAttention(nn.Module):
    """
    论文核心组件: Efficient Additive Attention
    参考论文 Section 3.2 和 Figure 2(d)
    
    不同于标准的 Self-Attention (Q*K^T * V)，这里使用了:
    1. Query 与可学习参数 w_a 进行点积，计算 attention map
    2. 基于 attention map 对 Query 进行池化得到全局 Query (global_query)
    3. Global Query 与 Key 进行 element-wise 乘法，得到全局上下文 (Global Context)
    4. 最终输出融合了原始 Query 和 经过变换的全局上下文
    """
    def __init__(self, dim=512, key_dim=512, num_heads=8, act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (key_dim // num_heads) ** -0.5
        
        # 将输入映射为 Query 和 Key
        # 论文提到: "Input embedding matrix x is transformed into query (Q) and key (K)"
        # 这里为了保持空间维度，使用 1x1 卷积代替 Linear
        self.to_q = nn.Conv2d(dim, key_dim, 1)
        self.to_k = nn.Conv2d(dim, key_dim, 1)
        
        # Learnable parameter vector w_a (Eq 4)
        # 形状为 (1, key_dim, 1, 1) 以便于广播
        self.w_a = nn.Parameter(torch.randn(1, key_dim, 1, 1))
        
        # 最后的线性变换 T (Eq 6)
        self.to_out = nn.Conv2d(key_dim, dim, 1)
        
    def forward(self, x):
        # x: [Batch, C, H, W]
        
        q = self.to_q(x) # [B, key_dim, H, W]
        k = self.to_k(x) # [B, key_dim, H, W]
        
        # --- 计算 Global Query (Eq 4 & 5) ---
        # Eq 4: alpha = Q * w_a / sqrt(d)
        # 实际上是计算 Q 和 w_a 的相似度。
        # 这里我们直接将 q 与 w_a 进行 element-wise 乘法然后求和? 
        # 论文 Eq 4 写的是点积。考虑到卷积操作的特性，我们可以将 w_a 视为权重。
        # 但 w_a 是一个参数向量。
        # 实现方式：Q (B, C, H, W) * w_a (1, C, 1, 1) -> (B, C, H, W) -> Sum over C -> (B, 1, H, W)
        # 或者 按照图 2(d)，是 Q matrix multiplied by learnable weights and pooled.
        # 标准实现通常是将 w_a 作为一个 1x1 卷积核应用在 Q 上，或者直接点乘。
        
        B, C, H, W = q.shape
        
        # 计算注意力分数 alpha
        # Q * w_a (element-wise multiplication broadcasted) 
        # 然后在通道维度求和得到每个空间位置的分数? 
        # 论文 Eq 4: alpha = Q . w_a
        # 最符合文意的是：对 Q 的每个 token (u_i) 与 w_a 做点积。
        # 这等价于一个 out_channels=1 的 1x1 卷积，权重为 w_a
        
        # 1. 计算每个位置的 Attention Logits
        attn_logits = (q * self.w_a).sum(dim=1, keepdim=True) # [B, 1, H, W]
        attn_logits = attn_logits * self.scale
        
        # 2. Softmax 归一化 (空间维度)
        # Flatten spatial dims for softmax
        attn = attn_logits.view(B, 1, -1) # [B, 1, H*W]
        attn = F.softmax(attn, dim=-1)
        attn = attn.view(B, 1, H, W) # [B, 1, H, W]
        
        # 3. 计算 Global Query vector q_g (Eq 5)
        # q_g = sum(alpha_i * Q_i)
        # 广播乘法后在空间维度求和
        global_query = (q * attn).sum(dim=(2, 3), keepdim=True) # [B, C, 1, 1]
        
        # --- Global Context Modeling ---
        # Eq: context = K * global_query (element-wise)
        # 将全局查询广播回所有位置，并与 Key 进行交互
        context = k * global_query # [B, C, H, W]
        
        # --- Output Projection ---
        # Eq 6: x_hat = Q_norm + T(K * q)
        # 这里的 Q_norm 或者是原始 Q 的一个变换。
        # 简单的实现通常是将 context 经过线性层变换后与 Q 相加 (ResNet style within module) 
        # 或者直接输出 T(context)，残差连接在 Block 外部处理。
        # 论文 Eq 6 显示 EAA 模块的输出包含了 Q^hat。
        # 结合 Eq 8 (Block level): X_new = QK(X) + X
        # 我们这里让 EAA 输出 T(Context) + Q。
        
        out = self.to_out(context)
        
        # 论文 Eq 6 中有一项 \hat{Q}。通常为了梯度流更顺畅，Q 分支也会加上。
        # 但在 SwiftFormer Encoder Block 中，输入 X 已经有一个残差连接。
        # 如果这里再加 Q，相当于加了两次（一次作为 Residual X，一次作为 Q）。
        # 然而 Q 是经过 1x1 变换的。
        # 参考官方实现或图示，通常输出是 Context 的变换。
        # 但 Eq 6 明确写了 \hat{Q} + ...
        # 我们这里返回 out + q (假设 q 是变换后的特征)
        return out + q

class SwiftFormerLocalRepresentation(nn.Module):
    """
    SwiftFormer Encoder 中的 Local Representation 部分。
    结构: DWConv 3x3 -> BN -> PointConv 1x1 -> BN -> GeLU
    参考论文 Figure 3 (Bottom Right)
    """
    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        # 注意：图中 Local Representation 输出后紧接着就是 Attention，没有再次 GeLU
        # 但通常 Conv 后面会跟 Activation。根据 Eq 8:
        # X = Conv_1(DWConv_BN(X))，这里 Conv_1 是 point-wise。
        # 我们的实现包含了 BN 和 Act 以确保稳定性。
        return x

class SwiftFormerEncoderBlock(nn.Module):
    """
    SwiftFormer Encoder Block
    包含:
    1. Local Representation (Conv based)
    2. Efficient Additive Attention
    3. Linear (MLP)
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        
        # 1. Local Representation
        self.local_representation = SwiftFormerLocalRepresentation(dim)
        
        # 2. Efficient Additive Attention
        self.attn = EfficientAdditiveAttention(dim=dim, key_dim=dim)
        
        # 3. Linear / MLP
        hidden_features = int(dim * mlp_ratio)
        self.mlp = SwiftFormerMlp(in_features=dim, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
        
        # Layer Scale (Optional but recommended for Deep ViTs, though not explicitly detailed in SwiftFormer diagram, likely standard)
        self.layer_scale_1 = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)
        
        self.drop_path = nn.Identity() # Placeholder for DropPath if needed

    def forward(self, x):
        # 论文 Eq 8
        
        # Part 1: Local Representation
        # X = LocalRep(X) (论文公式中没有残差，但通常 LocalRep 是对特征的提取)
        # Figure 3 显示 Local Representation 是串联的，输入到 Attention
        local_feat = self.local_representation(x)
        
        # Part 2: Attention with Residual
        # X = QK(X) + X
        # 注意：这里的 X 是 LocalRep 的输出还是原始输入？
        # Figure 3 显示： Input -> Local Rep -> Attention -> Linear. 
        # 并没有显示跨越 Local Rep 的大残差。
        # 但 Attention 模块本身有 "+ X" (Eq 8)。
        # 我们假设输入到 Attention 的是 LocalRep 的输出。
        
        x_attn = self.attn(local_feat)
        x = local_feat + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x_attn)
        
        # Part 3: MLP with Residual
        # X = MLP(X) + X
        x_mlp = self.mlp(x)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x_mlp)
        
        return x

class ConvEncoderBlock(nn.Module):
    """
    Conv Encoder Block
    用于早期阶段，纯卷积结构。
    结构: DWConv 3x3 -> BN -> Conv 1x1 -> GeLU -> Conv 1x1
    参考 Eq 7
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = 4*dim
        
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
        self.pw_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x + shortcut
        return x

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer
    论文 Section 3.3: "Two 3x3 convolutions with a stride of 2"
    """
    def __init__(self, in_chans=3, embed_dim=48):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.proj(x)

class SwiftFormer(nn.Module):
    """
    SwiftFormer 主干网络
    实现论文 Table 4 中的配置 (默认为 SwiftFormer-L1)
    """
    def __init__(self, 
                 layers=[3, 3, 6, 4],    # Blocks per stage (Example L1 config roughly)
                 embed_dims=[48, 96, 192, 384], 
                 downsamples=[True, True, True, True],
                 num_classes=1000,
                 use_conv_encoder_in_stage=[True, True, True, True]): # 哪些阶段包含 ConvEncoder
        super().__init__()
        
        self.patch_embed = PatchEmbedding(embed_dim=embed_dims[0])
        
        self.network = nn.ModuleList()
        
        for i in range(len(layers)):
            stage = nn.Sequential()
            
            # 1. Downsampling (except for first stage which uses PatchEmbed)
            # 论文: "Between two consecutive stages... downsampling layer"
            # 第一阶段直接接 PatchEmbed 输出，不需要额外下采样
            if i > 0:
                # Downsample: Increase channels, reduce resolution (stride 2)
                downsample = nn.Sequential(
                    nn.Conv2d(embed_dims[i-1], embed_dims[i], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dims[i])
                )
                stage.add_module(f"downsample_{i}", downsample)
            
            # 2. Blocks
            # 论文提到每个 Stage 由 Conv Encoder 和 SwiftFormer Encoder 组成。
            # 具体配置在 Table 4。
            # 这里简化逻辑：我们混合堆叠 ConvEncoderBlock 和 SwiftFormerEncoderBlock
            # 实际上 SwiftFormer 的设计是先放 ConvEncoder 再放 SwiftFormerEncoder
            # 为了通用性，我们假设 layers[i] 是总块数，我们可以按比例分配或全部使用 SwiftFormerEncoder
            # 根据 Table 4 (L1):
            # Stage 1: 48 ch, 3 ConvEnc, 1 SwiftEnc
            # Stage 2: 96 ch, 2 ConvEnc, 1 SwiftEnc
            # Stage 3: 192 ch, 9 ConvEnc, 1 SwiftEnc (NOTE: Table 4 says 9 Conv, 1 Swift)
            # Stage 4: 384 ch, 4 ConvEnc, 1 SwiftEnc
            
            # 这里我们硬编码 L1 的结构作为默认，或者通过参数传递具体的 block 类型列表
            # 为了代码简洁，我们假定 layers 参数仅仅是 SwiftFormerEncoder 的数量，
            # 而 ConvEncoder 的数量在代码中预设，或者简化为只使用 SwiftFormerEncoder (如果不严格遵循 L1)
            # 为了准确复现 L1，我们手动构建 stages。
            pass 

        # 重写构建逻辑以严格匹配 SwiftFormer-L1
        # Stage 1
        self.stage1 = self._make_stage(embed_dims[0], num_conv=3, num_swift=1)
        # Stage 2 (Downsample + Blocks)
        self.downsample2 = self._make_downsample(embed_dims[0], embed_dims[1])
        self.stage2 = self._make_stage(embed_dims[1], num_conv=2, num_swift=1)
        # Stage 3
        self.downsample3 = self._make_downsample(embed_dims[1], embed_dims[2])
        self.stage3 = self._make_stage(embed_dims[2], num_conv=9, num_swift=1)
        # Stage 4
        self.downsample4 = self._make_downsample(embed_dims[2], embed_dims[3])
        self.stage4 = self._make_stage(embed_dims[3], num_conv=4, num_swift=1)
        
        self.final_dim = embed_dims[3]

    def _make_downsample(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim)
        )

    def _make_stage(self, dim, num_conv, num_swift):
        blocks = []
        for _ in range(num_conv):
            blocks.append(ConvEncoderBlock(dim))
        for _ in range(num_swift):
            blocks.append(SwiftFormerEncoderBlock(dim))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        
        x = self.downsample2(x)
        x = self.stage2(x)
        
        x = self.downsample3(x)
        x = self.stage3(x)
        
        x = self.downsample4(x)
        x = self.stage4(x)
        return x

class CLIPSwiftFormerEncoder(nn.Module):
    """
    针对 CLIP 封装的 Image Encoder。
    输入: Images [B, 3, H, W]
    输出: Embeddings [B, projection_dim] (即 CLIP 论文中的 image latent vector)
    """
    def __init__(self, projection_dim=512, model_variant='L1'):
        super().__init__()
        
        # 默认使用 SwiftFormer-L1 配置
        # Table 4: L1
        # Embed Dims: 48, 96, 192, 384
        # Conv/Swift Blocks: (3,1), (2,1), (9,1), (4,1)
        if model_variant == 'L1':
            self.backbone = SwiftFormer(
                embed_dims=[48, 96, 192, 384]
            )
            prev_dim = 384
        else:
            # 可以扩展 XS, S, L3 等配置
            raise NotImplementedError("Currently only L1 variant is hardcoded for demo.")
            
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection Head (Mapping to shared embedding space)
        # CLIP 通常包含一个 Linear 层将视觉特征投影到与文本特征相同的维度
        self.projection_head = nn.Linear(prev_dim, projection_dim, bias=False)

    def forward(self, x):
        # 1. Backbone Features
        x = self.backbone(x) # [B, 384, H/32, W/32]
        
        # 2. Pooling
        x = self.global_pool(x) # [B, 384, 1, 1]
        x = x.flatten(1)        # [B, 384]
        
        # 3. Projection
        x = self.projection_head(x) # [B, projection_dim]
        
        return x


# 使用示例
if __name__ == "__main__":
    # 模拟输入 (Batch=2, RGB, 224x224)
    dummy_img = torch.randn(2, 3, 224, 224)
    
    # 初始化 CLIP Image Encoder (假设输出维度为 512)
    clip_img_encoder = CLIPSwiftFormerEncoder(projection_dim=512)
    
    # 前向传播
    embedding = clip_img_encoder(dummy_img)
    
    print(f"Input shape: {dummy_img.shape}")
    print(f"Output embedding shape: {embedding.shape}") # Should be [2, 512]
    
    # 打印参数量以验证 L1 规模 (~12M params)
    total_params = sum(p.numel() for p in clip_img_encoder.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")