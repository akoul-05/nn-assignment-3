"""
Implement the following models for classification.

This homework extends HW2 by adding advanced neural network architectures.
The ClassificationLoss implementation is provided from HW2.
Implement the two new models below.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        You need to implement this for your Vision Transformer.

        Hint: A transformer block typically consists of:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_dropout = nn.Dropout(dropout)
        #raise NotImplementedError("TransformerBlock.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """

        x_norm1 = self.norm1(x)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + self.attn_dropout(attn_output)

        x_norm2 = self.norm2(x)
        x = x + self.mlp_dropout(self.mlp(x_norm2))
        return x
        #raise NotImplementedError("TransformerBlock.forward() is not implemented")


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 256):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer.
        You can use this directly in your ViTClassifier implementation.

        Args:
            img_size: size of input image (assumed square)
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H//p, p, W//p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        x = x.reshape(B, self.num_patches, C * p * p)

        # Linear projection
        return self.projection(x)


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.2,
    ):
        """
        An MLP with multiple hidden layers and residual connections

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        dim = h * w * 3  # input dimension
        self.flatten = nn.Flatten()
        self.input_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),)
        
        # Used ChatGPT to help with the implementation of this class
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.fc1s  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2s  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        #raise NotImplementedError("MLPClassifierDeepResidual.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        #raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")
        x = self.flatten(x)
        x = self.input_proj(x)
        for ln, fc1, fc2 in zip(self.norms, self.fc1s, self.fc2s):
            y = ln(x)
            y = fc1(y)
            y = self.act(y)
            y = self.drop(y)
            y = fc2(y)
            x = x + self.drop(y)
        return self.head(x)


class ViTClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):

        """
        Vision Transformer (ViT) classifier

        Args:
            h: int, height of input image
            w: int, width of input image
            num_classes: int, number of classes

        Hint - you can add more arguments to the constructor such as:
            patch_size: int, size of image patches
            embed_dim: int, embedding dimension
            num_layers: int, number of transformer layers
            num_heads: int, number of attention heads

        Note: You can use the provided PatchEmbedding class. You'll need to implement the TransformerBlock class.
        """
        super().__init__()
        img_size = h
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=3, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)



        #raise NotImplementedError("ViTClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) input image

        Returns:
            tensor (b, num_classes) logits
        """
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, 1+N, D)
        x = x + self.pos_embed                          # add learnable positions
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                               # (B, D)
        logits = self.head(cls_out)                     # (B, C)
        return logits
    
        #raise NotImplementedError("ViTClassifier.forward() is not implemented")


model_factory = {
    "mlp_deep_residual": MLPClassifierDeepResidual,
    "vit": ViTClassifier,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r