# Datasets — Aula 4: Otimização de Modelos: Pruning e Quantização

## Dataset Principal: CIFAR-10

### Descrição

O **CIFAR-10** (Canadian Institute For Advanced Research) é um benchmark
amplamente utilizado em visão computacional, composto por imagens coloridas
de 32×32 pixels distribuídas em 10 classes. Ele é baixado automaticamente
pela biblioteca `torchvision` na primeira execução dos notebooks.

### Estrutura

| Atributo | Valor |
|----------|-------|
| Imagens de treino | 50 000 |
| Imagens de teste | 10 000 |
| Resolução | 32 × 32 × 3 (RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Balanceamento | Uniforme — 5 000 imagens/classe (treino), 1 000/classe (teste) |
| Formato | Tensores PyTorch via `torchvision.datasets.CIFAR10` |

### Estatísticas de Normalização

| Canal | Média | Desvio-padrão |
|-------|-------|---------------|
| R | 0.4914 | 0.2470 |
| G | 0.4822 | 0.2435 |
| B | 0.4465 | 0.2616 |

### Fonte

- **Referência:** Krizhevsky, A. (2009). *Learning Multiple Layers of
  Features from Tiny Images.* Technical Report, University of Toronto.
- **Download:** automático via `torchvision.datasets.CIFAR10(root="data/raw", download=True)`

### Reprodução

```bash
# O download é feito automaticamente ao executar o notebook 01.
# Alternativamente:
python -c "import torchvision; torchvision.datasets.CIFAR10(root='data/raw', download=True)"
```

### Por que CIFAR-10?

Conforme discutido na seção *Saiba Mais* do documento 04, a escolha de um
dataset de tamanho pedagógico permite iterar rapidamente sobre técnicas de
pruning e quantização, mantendo ciclos de treinamento curtos (minutos) e
resultados visualmente interpretáveis.
