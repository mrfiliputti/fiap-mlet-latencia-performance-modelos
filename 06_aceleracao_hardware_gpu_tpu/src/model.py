"""
Módulo de definição de modelo: Vision Transformer simplificado.

Implementa um Vision Transformer (ViT) compacto como caso de estudo para
demonstração de técnicas de aceleração por hardware, conforme mencionado
na seção 'Hands On' do material da Aula 06:

    "Trabalharemos com um Vision Transformer como caso de estudo,
     analisando operações compute-bound e memory-bound."

O modelo incorpora suporte nativo a:
- Mixed Precision Training via AMP (Micikevicius et al., 2018)
- Gradient Checkpointing para economia de memória (Chen et al., 2016)
- Dimensões alinhadas a múltiplos de 8/64 para utilização ótima de Tensor Cores

Referências:
    Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
    Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
    Dao, T. (2023). FlashAttention-2. NeurIPS.
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Converte imagem em sequência de patch embeddings.

    Divide a imagem de entrada em patches não-sobrepostos e projeta cada
    patch em um vetor de embedding. O tamanho do embedding (embed_dim) é
    escolhido como múltiplo de 64 para utilização ótima de Tensor Cores,
    conforme discutido na seção 'Tensor Cores: Hardware Especializado
    para Deep Learning' do material da aula.

    Args:
        img_size: Tamanho da imagem (quadrada).
        patch_size: Tamanho de cada patch.
        in_channels: Canais de entrada (3 para RGB).
        embed_dim: Dimensão do embedding (múltiplo de 64 recomendado).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Projeção linear via convolução — operação compute-bound
        # que se beneficia de Tensor Cores (alta intensidade aritmética)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, H, W) -> (B, n_patches, embed_dim)."""
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Atenção multi-cabeça para Vision Transformer.

    Implementa self-attention padrão. Em GPUs com Tensor Cores, as
    multiplicações de matrizes Q×K^T e Attention×V são operações
    compute-bound com alta intensidade aritmética (~100 FLOP/byte),
    conforme Tabela 1 do material da aula.

    A operação softmax, por outro lado, é memory-bound (~1 FLOP/byte),
    motivando otimizações como FlashAttention (Dao et al., 2022) que
    reorganizam computações para maximizar uso de SRAM.

    Args:
        embed_dim: Dimensão do embedding.
        n_heads: Número de cabeças de atenção.
        dropout: Taxa de dropout.

    Referência:
        Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient
        Exact Attention with IO-Awareness. NeurIPS.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim deve ser divisível por n_heads"

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, N, D) -> (B, N, D)."""
        B, N, D = x.shape

        # Projeção QKV — operação compute-bound (matmul denso)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Atenção: Q×K^T — compute-bound, beneficia Tensor Cores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax — operação memory-bound (~1 FLOP/byte)
        # Conforme Tabela 1 do material da aula
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Attention × V — compute-bound
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Bloco Transformer com suporte a gradient checkpointing.

    Implementa um bloco padrão Pre-LayerNorm Transformer. A LayerNorm
    é uma operação memory-bound (~3 FLOP/byte), conforme classificação
    do modelo roofline discutido na seção 'O Modelo Roofline:
    Caracterizando Workloads'.

    O bloco suporta gradient checkpointing conforme proposto por
    Chen et al. (2016), permitindo trade-off entre memória e computação.

    Args:
        embed_dim: Dimensão do embedding.
        n_heads: Número de cabeças de atenção.
        mlp_ratio: Multiplicador para dimensão do MLP.
        dropout: Taxa de dropout.

    Referência:
        Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # LayerNorm — operação memory-bound (~3 FLOP/byte)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),  # Compute-bound (matmul)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),  # Compute-bound (matmul)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com conexões residuais."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """Vision Transformer simplificado para experimentos de aceleração.

    Implementa ViT compacto como caso de estudo para demonstração de:
    - Mixed Precision Training (Micikevicius et al., 2018)
    - Gradient Checkpointing (Chen et al., 2016)
    - Profiling de operações compute-bound vs memory-bound

    Todas as dimensões são escolhidas como múltiplos de 8 para
    alinhamento com Tensor Cores, conforme recomendado na seção
    'Tensor Cores: Hardware Especializado para Deep Learning':
    "dimensões de embedding múltiplas de 64 garante utilização ótima
    em todas as gerações de Tensor Cores."

    Args:
        img_size: Tamanho da imagem de entrada.
        patch_size: Tamanho de cada patch.
        in_channels: Número de canais de entrada.
        n_classes: Número de classes de saída.
        embed_dim: Dimensão do embedding (múltiplo de 64).
        depth: Número de blocos Transformer.
        n_heads: Número de cabeças de atenção.
        mlp_ratio: Multiplicador para dimensão do MLP.
        dropout: Taxa de dropout.
        use_checkpointing: Se True, aplica gradient checkpointing.

    Referências:
        Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
        Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        n_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token e positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Blocos Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Cabeça de classificação
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Inicialização de pesos
        self._init_weights()

        logger.info(
            "SimpleViT criado: img=%d, patch=%d, embed=%d, depth=%d, heads=%d, "
            "params=%.2fM, checkpointing=%s",
            img_size, patch_size, embed_dim, depth, n_heads,
            self.count_parameters() / 1e6, use_checkpointing,
        )

    def _init_weights(self) -> None:
        """Inicializa pesos com distribuição truncated normal."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass do Vision Transformer.

        Args:
            x: Tensor de entrada (B, C, H, W).

        Returns:
            Logits de classificação (B, n_classes).
        """
        B = x.shape[0]

        # Patch embedding — operação compute-bound (convolução)
        x = self.patch_embed(x)

        # Adiciona class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional embedding — operação element-wise (memory-bound)
        x = self.pos_drop(x + self.pos_embed)

        # Blocos Transformer — com ou sem gradient checkpointing
        # Implementa conceito de Chen et al. (2016) conforme seção
        # 'Otimização de Memória GPU: Gradient Checkpointing'
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                # Gradient checkpointing: recomputa ativações no backward
                # Reduz memória de O(n) para O(√n) ao custo de ~33%
                # computação adicional (Chen et al., 2016)
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Classificação a partir do class token
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

    def count_parameters(self) -> int:
        """Conta total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_footprint(self) -> dict:
        """Estima footprint de memória do modelo.

        Calcula memória ocupada por parâmetros e buffers, relevante
        para análise de hierarquia de memória GPU discutida na seção
        'Hierarquia de Memória: O Gargalo Crítico'.

        Returns:
            Dicionário com tamanhos em MB para diferentes precisões.
        """
        param_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in self.buffers()
        )
        total_bytes = param_bytes + buffer_bytes

        return {
            "params_count": self.count_parameters(),
            "fp32_mb": total_bytes / (1024 * 1024),
            "fp16_mb": total_bytes / (1024 * 1024) / 2,
            "int8_mb": total_bytes / (1024 * 1024) / 4,
        }

    def fit(
        self,
        train_loader: "torch.utils.data.DataLoader",
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        n_epochs: int = 1,
    ) -> list:
        """Interface simplificada de treinamento.

        Args:
            train_loader: DataLoader de treino.
            optimizer: Otimizador.
            criterion: Função de perda.
            device: Dispositivo (CPU/GPU).
            n_epochs: Número de épocas.

        Returns:
            Lista de losses por época.
        """
        self.to(device)
        self.train()
        epoch_losses = []

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        return epoch_losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predição com o modelo.

        Args:
            x: Tensor de entrada (B, C, H, W).

        Returns:
            Classes preditas (B,).
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return logits.argmax(dim=-1)

    def score(
        self,
        data_loader: "torch.utils.data.DataLoader",
        device: torch.device,
    ) -> float:
        """Calcula acurácia no dataset.

        Args:
            data_loader: DataLoader para avaliação.
            device: Dispositivo.

        Returns:
            Acurácia (0.0 a 1.0).
        """
        self.to(device)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = self.predict(inputs)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1)


def create_model(
    img_size: int = 32,
    patch_size: int = 4,
    in_channels: int = 3,
    n_classes: int = 10,
    embed_dim: int = 128,
    depth: int = 4,
    n_heads: int = 4,
    use_checkpointing: bool = False,
) -> SimpleViT:
    """Factory function para criar modelo ViT.

    Parâmetros padrão escolhidos para alinhamento com Tensor Cores:
    embed_dim=128 (múltiplo de 64), n_heads=4 (head_dim=32, múltiplo de 8).

    Args:
        img_size: Tamanho da imagem.
        patch_size: Tamanho do patch.
        in_channels: Canais de entrada.
        n_classes: Número de classes.
        embed_dim: Dimensão do embedding.
        depth: Profundidade (número de blocos).
        n_heads: Cabeças de atenção.
        use_checkpointing: Ativar gradient checkpointing.

    Returns:
        Instância de SimpleViT.
    """
    return SimpleViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        use_checkpointing=use_checkpointing,
    )
