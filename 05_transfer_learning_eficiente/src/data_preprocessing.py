"""
Pré-processamento de dados para Transfer Learning Eficiente.

Implementa pipeline de preparação de dados textuais para fine-tuning
de modelos pré-treinados, conforme discutido na seção 'Fundamentos do
Transfer Learning' do material da Aula 5 (Pan & Yang, 2010).

Inclui geração de dataset sintético de classificação de domínio com
5 categorias temáticas (jurídico, tecnologia, financeiro, saúde, educação),
projetado para demonstrar adaptação de domínio com LoRA (Hu et al., 2022).

Referências:
    Pan, S.J., Yang, Q. (2010). A Survey on Transfer Learning.
        IEEE TKDE, 22(10), 1345-1359.
    Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large
        Language Models. ICLR.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Sintético
# ---------------------------------------------------------------------------

# Vocabulários temáticos por domínio para geração de dados sintéticos.
# Cada domínio possui templates, adjetivos, verbos e substantivos
# representativos do campo de conhecimento.
DOMAIN_VOCABULARIES: Dict[str, dict] = {
    "juridico": {
        "templates": [
            "O {adj} contrato estabelece que as partes devem {verbo} as cláusulas de {subst} conforme a legislação vigente.",
            "Nos termos do {subst}, o réu apresentou {adj} recurso contra a decisão que determinou {verbo} a sentença.",
            "A {adj} jurisprudência do tribunal superior consolidou entendimento sobre {subst} no âmbito do direito {adj2}.",
            "Conforme disposto no artigo sobre {subst}, as partes acordaram em {verbo} os termos do {adj} acordo bilateral.",
            "O poder judiciário decidiu {verbo} a {adj} petição referente ao caso de {subst} apresentado pela defesa.",
        ],
        "adj": ["referido", "supracitado", "mencionado", "presente", "competente", "devido"],
        "adj2": ["civil", "penal", "tributário", "trabalhista", "constitucional", "administrativo"],
        "verbo": ["cumprir", "observar", "ratificar", "homologar", "deferir", "indeferir"],
        "subst": ["mandado", "embargo", "habeas corpus", "litígio", "jurisdição", "competência"],
    },
    "tecnologia": {
        "templates": [
            "O {adj} sistema utiliza {subst} para {verbo} os dados em tempo real com alta performance.",
            "A implementação do {subst} permite {verbo} a latência do {adj} pipeline de processamento distribuído.",
            "Utilizando {adj} técnicas de {subst}, conseguimos {verbo} a escalabilidade da aplicação em nuvem.",
            "O {adj} framework oferece suporte a {subst} que possibilita {verbo} microsserviços em containers.",
            "A arquitetura baseada em {subst} foi projetada para {verbo} requisições com {adj} throughput.",
        ],
        "adj": ["novo", "escalável", "distribuído", "robusto", "otimizado", "resiliente"],
        "adj2": ["novo", "escalável", "distribuído", "robusto", "otimizado", "resiliente"],
        "verbo": ["processar", "otimizar", "escalar", "deployar", "monitorar", "orquestrar"],
        "subst": ["microserviço", "API REST", "container Docker", "cluster Kubernetes", "cache Redis", "message broker"],
    },
    "financeiro": {
        "templates": [
            "O {adj} relatório trimestral indica que o {subst} apresentou {verbo} em relação ao período anterior.",
            "A análise do {subst} sugere que investidores devem {verbo} suas posições no {adj} mercado de capitais.",
            "Os indicadores de {subst} demonstram {adj} tendência de {verbo} nos próximos trimestres do ano fiscal.",
            "O {adj} portfólio diversificado conseguiu {verbo} os riscos associados ao {subst} do mercado atual.",
            "Segundo analistas, o {subst} deve {verbo} com base nas {adj} projeções macroeconômicas divulgadas.",
        ],
        "adj": ["consolidado", "volátil", "estável", "crescente", "decrescente", "moderado"],
        "adj2": ["consolidado", "volátil", "estável", "crescente", "decrescente", "moderado"],
        "verbo": ["crescer", "valorizar", "diversificar", "mitigar", "reavaliar", "capitalizar"],
        "subst": ["índice Bovespa", "taxa Selic", "câmbio", "spread bancário", "CDI", "IPCA"],
    },
    "saude": {
        "templates": [
            "O {adj} estudo clínico demonstrou que o {subst} pode {verbo} os sintomas em pacientes crônicos.",
            "A equipe médica recomenda {verbo} o {subst} como parte do {adj} protocolo terapêutico hospitalar.",
            "Resultados do {adj} ensaio clínico indicam que {subst} consegue {verbo} biomarcadores inflamatórios.",
            "O {adj} tratamento combina {subst} com terapia complementar para {verbo} a qualidade de vida.",
            "Pesquisadores observaram que o {subst} pode {verbo} a resposta imunológica em {adj} condições clínicas.",
        ],
        "adj": ["randomizado", "multicêntrico", "controlado", "longitudinal", "promissor", "experimental"],
        "adj2": ["randomizado", "multicêntrico", "controlado", "longitudinal", "promissor", "experimental"],
        "verbo": ["reduzir", "atenuar", "monitorar", "prevenir", "tratar", "diagnosticar"],
        "subst": ["biomarcador", "anticorpo monoclonal", "vacina mRNA", "terapia gênica", "protocolo clínico", "exame laboratorial"],
    },
    "educacao": {
        "templates": [
            "A {adj} metodologia de ensino propõe {verbo} o {subst} como ferramenta pedagógica inovadora.",
            "Estudantes que utilizaram {subst} conseguiram {verbo} seu desempenho em {adj} avaliações escolares.",
            "O {adj} currículo incorpora {subst} para {verbo} competências essenciais do século XXI.",
            "Professores relatam que o {subst} permite {verbo} a aprendizagem de forma {adj} e engajadora.",
            "A {adj} plataforma de {subst} foi desenvolvida para {verbo} o acesso equitativo à educação.",
        ],
        "adj": ["ativa", "híbrida", "adaptativa", "colaborativa", "personalizada", "inclusiva"],
        "adj2": ["ativa", "híbrida", "adaptativa", "colaborativa", "personalizada", "inclusiva"],
        "verbo": ["potencializar", "avaliar", "desenvolver", "democratizar", "gamificar", "personalizar"],
        "subst": [
            "aprendizagem baseada em projetos", "sala de aula invertida",
            "tutoria inteligente", "avaliação formativa", "plataforma EAD", "recurso digital",
        ],
    },
}


def generate_synthetic_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gera dataset sintético de classificação de domínio textual.

    Cria documentos em português distribuídos em 5 domínios temáticos para
    demonstrar Transfer Learning e fine-tuning com PEFT, conforme discutido
    na seção 'Fundamentos do Transfer Learning' do material da Aula 5.

    A adaptação de domínio é um caso clássico de Transfer Learning onde o
    modelo pré-treinado em corpus genérico é adaptado a terminologias
    especializadas (Pan & Yang, 2010).

    Args:
        n_samples: Número total de amostras a gerar.
        random_state: Seed para reprodutibilidade.
        output_path: Caminho para salvar CSV. Se None, não salva.

    Returns:
        DataFrame com colunas ['text', 'label', 'domain_id'].

    Referência:
        Pan, S.J., Yang, Q. (2010). A Survey on Transfer Learning.
        IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.
    """
    rng = np.random.RandomState(random_state)
    domain_names = sorted(DOMAIN_VOCABULARIES.keys())
    samples_per_domain = n_samples // len(domain_names)

    records: List[dict] = []
    for domain_name in domain_names:
        vocab = DOMAIN_VOCABULARIES[domain_name]
        for _ in range(samples_per_domain):
            template = rng.choice(vocab["templates"])
            text = template.format(
                adj=rng.choice(vocab["adj"]),
                adj2=rng.choice(vocab.get("adj2", vocab["adj"])),
                verbo=rng.choice(vocab["verbo"]),
                subst=rng.choice(vocab["subst"]),
            )
            records.append({
                "text": text,
                "label": domain_name,
                "domain_id": domain_names.index(domain_name),
            })

    # Preencher amostras restantes aleatoriamente
    remaining = n_samples - len(records)
    for _ in range(remaining):
        domain_name = rng.choice(domain_names)
        vocab = DOMAIN_VOCABULARIES[domain_name]
        template = rng.choice(vocab["templates"])
        text = template.format(
            adj=rng.choice(vocab["adj"]),
            adj2=rng.choice(vocab.get("adj2", vocab["adj"])),
            verbo=rng.choice(vocab["verbo"]),
            subst=rng.choice(vocab["subst"]),
        )
        records.append({
            "text": text,
            "label": domain_name,
            "domain_id": domain_names.index(domain_name),
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %d samples to %s", len(df), output_path)

    return df


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """
    Pipeline de pré-processamento textual para fine-tuning eficiente.

    Prepara dados de classificação de domínio para treino com LoRA/QLoRA.
    O pré-processamento adequado é pré-requisito para garantir convergência
    ao treinar adaptadores de baixo rank (Hu et al., 2022).

    A divisão estratificada preserva o balanceamento de classes, importante
    para evitar viés quando o número de parâmetros treináveis é pequeno
    — como ocorre em técnicas PEFT (Dettmers et al., 2023).

    Referências:
        Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large
            Language Models. ICLR.
        Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of
            Quantized LLMs. NeurIPS.
    """

    def __init__(
        self,
        max_length: int = 128,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self._label_map: Optional[Dict[str, int]] = None
        self._data: Optional[pd.DataFrame] = None

    # ----- load / clean / prepare / split ----------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dataset CSV.

        Args:
            filepath: Caminho para o arquivo CSV.

        Returns:
            DataFrame carregado.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset não encontrado: {filepath}")
        self._data = pd.read_csv(filepath)
        logger.info("Carregou %d amostras de %s", len(self._data), filepath)
        return self._data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Limpa dados textuais: remove nulos, espaços extras e linhas vazias.

        Args:
            df: DataFrame a limpar. Se None, usa dados carregados anteriormente.

        Returns:
            DataFrame limpo.
        """
        if df is None:
            df = self._data
        if df is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")

        df = df.copy()
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].str.strip()
        df = df[df["text"].str.len() > 0]
        df = df.reset_index(drop=True)
        self._data = df
        logger.info("Dados limpos: %d amostras restantes", len(df))
        return df

    def prepare_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepara features: codifica labels e calcula estatísticas textuais.

        A codificação de labels como inteiros é necessária para o
        CrossEntropyLoss utilizado no fine-tuning de classificadores
        baseados em Transformer (Devlin et al., 2019).

        Args:
            df: DataFrame a processar. Se None, usa dados internos.

        Returns:
            DataFrame com colunas adicionais label_id, text_length, word_count.

        Referência:
            Devlin, J. et al. (2019). BERT: Pre-training of Deep
            Bidirectional Transformers. NAACL.
        """
        if df is None:
            df = self._data
        if df is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")

        df = df.copy()
        unique_labels = sorted(df["label"].unique())
        self._label_map = {label: idx for idx, label in enumerate(unique_labels)}
        df["label_id"] = df["label"].map(self._label_map)
        df["text_length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        self._data = df
        return df

    def split_data(
        self, df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide dados em train / validation / test com estratificação.

        A estratificação garante proporções consistentes de classes em cada
        split, essencial para avaliação justa de métodos PEFT que treinam
        poucos parâmetros (Hu et al., 2022).

        Args:
            df: DataFrame a dividir. Se None, usa dados internos.

        Returns:
            Tupla (train_df, val_df, test_df).
        """
        if df is None:
            df = self._data
        if df is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["label"],
        )
        val_ratio = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_df["label"],
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # ----- Propriedades -----------------------------------------------------

    @property
    def label_map(self) -> Optional[Dict[str, int]]:
        """Mapa label_string → label_id."""
        return self._label_map

    @property
    def num_labels(self) -> int:
        """Número de classes distintas."""
        if self._label_map is None:
            raise ValueError("Labels não preparados. Execute prepare_features().")
        return len(self._label_map)

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """DataFrame interno atual."""
        return self._data
