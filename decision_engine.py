"""
decision_engine.py
==================
Motor de decisão analítica orientado a prioridade.
Responsabilidade: classificar fabricantes, gerar scores e justificativas textuais.

Blocos de análise:
  1. Bloco Financeiro  — variação de receita, lucro e margem
  2. Bloco Estoque     — nível de encalhe e variação
  3. Bloco Eficiência  — GMROII e variação

Score final: soma dos três blocos (0–6)
Prioridade: Baixa (0-1) | Média (2-3) | Alta (≥4 ou regra forçada)
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Thresholds configuráveis
# ---------------------------------------------------------------------------

# Bloco Financeiro
THRESHOLD_MELHORA = 0.10        # > +10% → melhora relevante
THRESHOLD_PIORA = -0.05         # < -5%  → piora relevante

# Bloco Estoque
THRESHOLD_ENCALHE_LEVE = 0.10   # 10%–25% → encalhe presente
THRESHOLD_ENCALHE_ALTO = 0.25   # > 25%  → encalhe elevado

# Bloco Eficiência
THRESHOLD_GMROII_MELHORA = 0.10
THRESHOLD_GMROII_PIORA = -0.05

# Prioridade
SCORE_ALTA = 4
SCORE_MEDIA = 2


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticoFabricante:
    fabricante: str

    # KPIs
    receita_liquida: float = 0.0
    lucro: float = 0.0
    margem: float = 0.0
    estoque_atual: float = 0.0
    pct_encalhe: float = 0.0
    gmroii: float = 0.0
    estoque_encalhe: float = 0.0
    n_skus: int = 0
    n_skus_encalhe: int = 0

    # Variações (None quando não há mês anterior)
    var_receita: Optional[float] = None
    var_lucro: Optional[float] = None
    var_margem: Optional[float] = None
    var_encalhe: Optional[float] = None
    var_gmroii: Optional[float] = None

    # Classificações por bloco
    status_financeiro: str = "sem_dados"
    status_estoque: str = "sem_dados"
    status_eficiencia: str = "sem_dados"

    # Score (0–6)
    score_financeiro: int = 0
    score_estoque: int = 0
    score_eficiencia: int = 0
    score_total: int = 0

    # Prioridade final
    prioridade: str = "Indefinida"
    vetores: list = field(default_factory=list)
    justificativa: str = ""

    # Flag de regra forçada
    forcado_alta: bool = False


# ---------------------------------------------------------------------------
# Bloco 1 — Financeiro
# ---------------------------------------------------------------------------

def _score_financeiro(var_receita, var_lucro, var_margem) -> tuple[int, str]:
    """
    Avalia os três indicadores financeiros e retorna (score, status).
    Usa o pior indicador como driver do score.
    """
    if var_receita is None or pd.isna(var_receita):
        return 0, "sem_dados"

    # Classifica cada indicador individualmente
    def _classifica(v):
        if v > THRESHOLD_MELHORA:
            return 0, "melhora"
        elif v < THRESHOLD_PIORA:
            return 2, "piora"
        else:
            return 1, "estavel"

    scores = [_classifica(v) for v in [var_receita, var_lucro, var_margem] if not pd.isna(v)]
    if not scores:
        return 0, "sem_dados"

    score = max(s[0] for s in scores)

    if score == 2:
        status = "piora_relevante"
    elif score == 1:
        status = "estavel"
    else:
        status = "melhora_relevante"

    return score, status


# ---------------------------------------------------------------------------
# Bloco 2 — Estoque
# ---------------------------------------------------------------------------

def _score_estoque(pct_encalhe, var_encalhe) -> tuple[int, str]:
    """
    Avalia nível e variação do encalhe.
    """
    if pct_encalhe >= THRESHOLD_ENCALHE_ALTO:
        # Encalhe elevado é sempre score 2
        return 2, "encalhe_elevado"
    elif pct_encalhe >= THRESHOLD_ENCALHE_LEVE:
        # Encalhe presente: agrava se crescente
        if var_encalhe is not None and not pd.isna(var_encalhe) and var_encalhe > 0.10:
            return 2, "encalhe_crescente"
        return 1, "encalhe_presente"
    else:
        return 0, "sem_encalhe"


# ---------------------------------------------------------------------------
# Bloco 3 — Eficiência
# ---------------------------------------------------------------------------

def _score_eficiencia(var_gmroii) -> tuple[int, str]:
    """
    Avalia variação do GMROII.
    """
    if var_gmroii is None or pd.isna(var_gmroii):
        return 0, "sem_dados"

    if var_gmroii > THRESHOLD_GMROII_MELHORA:
        return 0, "melhora"
    elif var_gmroii < THRESHOLD_GMROII_PIORA:
        return 2, "piora"
    else:
        return 1, "estavel"


# ---------------------------------------------------------------------------
# Gerador de justificativa textual
# ---------------------------------------------------------------------------

_TEXTOS_FINANCEIRO = {
    "melhora_relevante": "Os indicadores financeiros mostram crescimento relevante no período.",
    "estavel": "A performance financeira permanece estável, sem variações significativas.",
    "piora_relevante": "Houve deterioração relevante nos indicadores financeiros (receita, lucro ou margem).",
    "sem_dados": "Dados financeiros disponíveis apenas para o período atual — sem comparativo.",
}

_TEXTOS_ESTOQUE = {
    "sem_encalhe": "O nível de encalhe é baixo, sem comprometimento relevante do capital em estoque.",
    "encalhe_presente": "Há encalhe moderado no portfólio, demandando atenção aos itens parados.",
    "encalhe_elevado": "O encalhe é elevado, com volume significativo de capital imobilizado em itens sem giro.",
    "encalhe_crescente": "O encalhe está crescendo, indicando acúmulo progressivo de itens sem saída.",
}

_TEXTOS_EFICIENCIA = {
    "melhora": "A eficiência do estoque (GMROII) melhorou — o capital investido está gerando mais retorno.",
    "estavel": "A eficiência do estoque permanece estável.",
    "piora": "A eficiência do estoque (GMROII) piorou — o capital alocado está gerando menos retorno.",
    "sem_dados": "GMROII calculado apenas para o período atual — sem referência comparativa.",
}

_COMBOS_ESPECIAIS = {
    ("melhora_relevante", "encalhe_elevado"): (
        "Crescimento de receita acompanhado por aumento de itens em encalhe, "
        "indicando crescimento com acúmulo de estoque e potencial risco de capital imobilizado."
    ),
    ("melhora_relevante", "encalhe_crescente"): (
        "A receita cresce, mas o encalhe avança simultaneamente — "
        "sinal de que o crescimento não está sendo seletivo."
    ),
    ("piora_relevante", "encalhe_elevado"): (
        "Queda nos indicadores financeiros combinada com alto encalhe: "
        "situação crítica de capital imobilizado e resultado em deterioração."
    ),
    ("piora_relevante", "encalhe_crescente"): (
        "Piora financeira e crescimento do encalhe simultaneamente — "
        "ação corretiva urgente é necessária."
    ),
}


def _gerar_justificativa(
    status_financeiro: str,
    status_estoque: str,
    status_eficiencia: str,
    prioridade: str,
    forcado_alta: bool,
    var_receita: Optional[float],
    var_lucro: Optional[float],
    pct_encalhe: float,
    gmroii: float,
) -> str:
    partes = []

    # Verifica combinação especial
    combo = _COMBOS_ESPECIAIS.get((status_financeiro, status_estoque))
    if combo:
        partes.append(combo)
    else:
        partes.append(_TEXTOS_FINANCEIRO.get(status_financeiro, ""))
        partes.append(_TEXTOS_ESTOQUE.get(status_estoque, ""))

    partes.append(_TEXTOS_EFICIENCIA.get(status_eficiencia, ""))

    # Complemento quantitativo
    if pct_encalhe > 0:
        partes.append(f"O encalhe representa {pct_encalhe:.1%} do valor de estoque do fabricante.")

    if gmroii != 0:
        partes.append(f"GMROII atual: {gmroii:.2f} (lucro por unidade monetária de estoque).")

    if forcado_alta:
        partes.append(
            "⚠️ Prioridade elevada para ALTA pela combinação de encalhe relevante e queda de lucro."
        )

    return " ".join(p for p in partes if p)


# ---------------------------------------------------------------------------
# Classificação final de prioridade
# ---------------------------------------------------------------------------

def _classificar_prioridade(
    score_total: int,
    status_estoque: str,
    status_financeiro: str,
) -> tuple[str, bool]:
    """
    Retorna (prioridade, forcado_alta).
    Regra forçada: encalhe relevante + queda de lucro → Alta.
    """
    forcado = False

    encalhe_relevante = status_estoque in ("encalhe_elevado", "encalhe_crescente")
    queda_lucro = status_financeiro == "piora_relevante"

    if encalhe_relevante and queda_lucro:
        forcado = True
        return "Alta", forcado

    if score_total >= SCORE_ALTA:
        return "Alta", forcado
    elif score_total >= SCORE_MEDIA:
        return "Média", forcado
    else:
        return "Baixa", forcado


def _extrair_vetores(
    score_financeiro: int,
    score_estoque: int,
    score_eficiencia: int,
) -> list[str]:
    """Retorna lista dos vetores que contribuíram para o score."""
    vetores = []
    if score_financeiro >= 1:
        vetores.append("Financeiro")
    if score_estoque >= 1:
        vetores.append("Estoque")
    if score_eficiencia >= 1:
        vetores.append("Eficiência")
    return vetores


# ---------------------------------------------------------------------------
# API principal do motor
# ---------------------------------------------------------------------------

def diagnosticar_fabricante(row: pd.Series) -> DiagnosticoFabricante:
    """
    Recebe uma linha do DataFrame de KPIs (com ou sem variação) e
    retorna um DiagnosticoFabricante completo.
    """
    def _get(col, default=None):
        val = row.get(col, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return val

    d = DiagnosticoFabricante(
        fabricante=_get("fabricante", ""),
        receita_liquida=_get("receita_liquida", 0.0),
        lucro=_get("lucro", 0.0),
        margem=_get("margem", 0.0),
        estoque_atual=_get("estoque_atual", 0.0),
        pct_encalhe=_get("pct_encalhe", 0.0),
        gmroii=_get("gmroii", 0.0),
        estoque_encalhe=_get("estoque_encalhe", 0.0),
        n_skus=int(_get("n_skus", 0)),
        n_skus_encalhe=int(_get("n_skus_encalhe", 0)),
        var_receita=_get("var_receita"),
        var_lucro=_get("var_lucro"),
        var_margem=_get("var_margem"),
        var_encalhe=_get("var_encalhe"),
        var_gmroii=_get("var_gmroii"),
    )

    # Scores por bloco
    d.score_financeiro, d.status_financeiro = _score_financeiro(
        d.var_receita, d.var_lucro, d.var_margem
    )
    d.score_estoque, d.status_estoque = _score_estoque(d.pct_encalhe, d.var_encalhe)
    d.score_eficiencia, d.status_eficiencia = _score_eficiencia(d.var_gmroii)

    d.score_total = d.score_financeiro + d.score_estoque + d.score_eficiencia

    # Prioridade
    d.prioridade, d.forcado_alta = _classificar_prioridade(
        d.score_total, d.status_estoque, d.status_financeiro
    )

    # Vetores de atenção
    d.vetores = _extrair_vetores(d.score_financeiro, d.score_estoque, d.score_eficiencia)

    # Justificativa textual
    d.justificativa = _gerar_justificativa(
        d.status_financeiro,
        d.status_estoque,
        d.status_eficiencia,
        d.prioridade,
        d.forcado_alta,
        d.var_receita,
        d.var_lucro,
        d.pct_encalhe,
        d.gmroii,
    )

    return d


def diagnosticar_todos(kpis_df: pd.DataFrame) -> list[DiagnosticoFabricante]:
    """
    Aplica o motor de decisão a todos os fabricantes no DataFrame de KPIs.

    Retorna lista de DiagnosticoFabricante ordenada por score (maior primeiro).
    """
    diagnosticos = [diagnosticar_fabricante(row) for _, row in kpis_df.iterrows()]
    return sorted(diagnosticos, key=lambda d: d.score_total, reverse=True)


def diagnosticos_to_dataframe(diagnosticos: list[DiagnosticoFabricante]) -> pd.DataFrame:
    """Converte lista de diagnósticos em DataFrame para visualização/exportação."""
    rows = []
    for d in diagnosticos:
        rows.append({
            "Fabricante": d.fabricante,
            "Prioridade": d.prioridade,
            "Score": d.score_total,
            "Vetores": " + ".join(d.vetores) if d.vetores else "—",
            "Receita Líquida": d.receita_liquida,
            "Lucro": d.lucro,
            "Margem (%)": d.margem,
            "Estoque Atual": d.estoque_atual,
            "% Encalhe": d.pct_encalhe,
            "GMROII": d.gmroii,
            "Var. Receita": d.var_receita,
            "Var. Lucro": d.var_lucro,
            "Var. Margem": d.var_margem,
            "Var. Encalhe": d.var_encalhe,
            "Var. GMROII": d.var_gmroii,
            "Justificativa": d.justificativa,
            "N° SKUs": d.n_skus,
            "SKUs em Encalhe": d.n_skus_encalhe,
        })
    return pd.DataFrame(rows)
