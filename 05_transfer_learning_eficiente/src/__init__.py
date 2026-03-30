"""
Aula 05 — Transfer Learning Eficiente e Adaptação de Modelos.

Módulos:
    data_preprocessing: Pipeline de pré-processamento textual e geração de dataset
    model: Wrapper PEFT com suporte a LoRA, QLoRA e Adapters
    training: Treinamento com Hugging Face Trainer e comparação de eficiência
    evaluation: Métricas de classificação e benchmarking de latência/memória
    utils: Utilitários para serialização, configuração e instrumentação

Referências:
    Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
    Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
"""

__version__ = "1.0.0"
