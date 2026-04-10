"""
processor.py
============
Camada de processamento e construção das camadas analíticas.
Responsabilidade: agregar, integrar e calcular KPIs — sem lógica de decisão.

Camadas construídas:
  1. vendas_raw      → dados originais
  2. vendas_sku      → agregação por SKU (N - D)
  3. giro_raw        → estoque original
  4. base_integrada  → JOIN vendas_sku + giro_raw
  5. kpis_fabricante → KPIs consolidados por fabricante
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Camada 2 — vendas_sku
# ---------------------------------------------------------------------------

def build_vendas_sku(vendas_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega vendas por SKU, subtraindo devoluções (Tipo=D) das vendas (Tipo=N).

    Campos produzidos:
      - receita_liquida  : soma de venda_liquida (N positivo, D negativo)
      - lucro_total      : soma de lucro
      - quantidade_total : soma de quantidade (D conta negativo)
      - margem_media     : média ponderada de margem por receita_liquida
      - fabricante       : primeiro fabricante associado ao SKU
      - descricao        : primeira descrição associada ao SKU
    """
    df = vendas_raw.copy()

    # Devoluções viram negativo para subtração correta
    sinal = df["tipo"].map({"N": 1, "D": -1}).fillna(0)
    df["_venda_ajustada"] = df["venda_liquida"] * sinal
    df["_lucro_ajustado"] = df["lucro"] * sinal
    df["_qtd_ajustada"] = df["quantidade"] * sinal
    df["_margem_pond"] = df["margem"] * df["_venda_ajustada"].abs()

    grp = df.groupby("sku", as_index=False)

    vendas_sku = grp.agg(
        receita_liquida=("_venda_ajustada", "sum"),
        lucro_total=("_lucro_ajustado", "sum"),
        quantidade_total=("_qtd_ajustada", "sum"),
        _margem_pond_sum=("_margem_pond", "sum"),
        _margem_abs_sum=("_venda_ajustada", lambda x: x.abs().sum()),
        fabricante=("fabricante", "first"),
        descricao=("descricao", "first"),
    )

    # Margem média ponderada por receita absoluta
    vendas_sku["margem_media"] = np.where(
        vendas_sku["_margem_abs_sum"] > 0,
        vendas_sku["_margem_pond_sum"] / vendas_sku["_margem_abs_sum"],
        0,
    )
    vendas_sku.drop(columns=["_margem_pond_sum", "_margem_abs_sum"], inplace=True)

    return vendas_sku


# ---------------------------------------------------------------------------
# Camada 4 — base_integrada
# ---------------------------------------------------------------------------

def build_base_integrada(
    vendas_sku: pd.DataFrame,
    giro_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    JOIN entre vendas_sku e giro_raw por SKU.
    SKUs sem vendas mas com estoque são mantidos (left join da perspectiva do giro).
    """
    base = pd.merge(
        giro_raw,
        vendas_sku[["sku", "receita_liquida", "lucro_total", "quantidade_total", "margem_media"]],
        on="sku",
        how="left",
    )

    # Preenche SKUs que existem no estoque mas não tiveram venda no período
    base["receita_liquida"] = base["receita_liquida"].fillna(0)
    base["lucro_total"] = base["lucro_total"].fillna(0)
    base["quantidade_total"] = base["quantidade_total"].fillna(0)
    base["margem_media"] = base["margem_media"].fillna(0)

    # Flag de encalhe individual (somente ESA = "Encalhe" é problema crítico)
    base["is_encalhe"] = base["classificacao_esa"].str.lower() == "encalhe"

    return base


# ---------------------------------------------------------------------------
# Camada 5 — kpis_fabricante
# ---------------------------------------------------------------------------

def build_kpis_fabricante(base_integrada: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida KPIs por fabricante a partir da base integrada.

    KPIs produzidos:
      - receita_liquida     : soma de receita_liquida
      - lucro               : soma de lucro_total
      - margem              : margem ponderada por receita
      - estoque_atual       : soma de estoque_atual (em valor $)
      - estoque_encalhe     : soma de estoque dos itens classificados como Encalhe
      - pct_encalhe         : estoque_encalhe / estoque_atual
      - gmroii              : lucro / estoque_atual (retorno financeiro do estoque)
    """
    df = base_integrada.copy()

    grp = df.groupby("fabricante", as_index=False)

    kpis = grp.agg(
        receita_liquida=("receita_liquida", "sum"),
        lucro=("lucro_total", "sum"),
        estoque_atual=("estoque_atual", "sum"),
        _marg_pond=("margem_media", lambda x: (x * df.loc[x.index, "receita_liquida"].abs()).sum()),
        _receita_abs=("receita_liquida", lambda x: x.abs().sum()),
        n_skus=("sku", "count"),
        n_skus_encalhe=("is_encalhe", "sum"),
    )

    # Estoque em valor dos itens em Encalhe
    encalhe_df = df[df["is_encalhe"]].groupby("fabricante", as_index=False)["estoque_atual"].sum()
    encalhe_df.rename(columns={"estoque_atual": "estoque_encalhe"}, inplace=True)
    kpis = kpis.merge(encalhe_df, on="fabricante", how="left")
    kpis["estoque_encalhe"] = kpis["estoque_encalhe"].fillna(0)

    # Margem ponderada
    kpis["margem"] = np.where(
        kpis["_receita_abs"] > 0,
        kpis["_marg_pond"] / kpis["_receita_abs"],
        0,
    )
    kpis.drop(columns=["_marg_pond", "_receita_abs"], inplace=True)

    # % Encalhe = estoque_encalhe / estoque_atual total do fabricante
    kpis["pct_encalhe"] = np.where(
        kpis["estoque_atual"] > 0,
        kpis["estoque_encalhe"] / kpis["estoque_atual"],
        0,
    )

    # GMROII = Lucro / Estoque Atual (quanto lucro o estoque gera)
    kpis["gmroii"] = np.where(
        kpis["estoque_atual"] > 0,
        kpis["lucro"] / kpis["estoque_atual"],
        0,
    )

    return kpis.sort_values("receita_liquida", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Variação entre períodos
# ---------------------------------------------------------------------------

def build_variacao(
    kpis_atual: pd.DataFrame,
    kpis_anterior: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula variação percentual dos KPIs entre dois períodos.

    Retorna DataFrame com colunas de variação para:
      receita_liquida, lucro, margem, pct_encalhe, gmroii
    """
    anterior = kpis_anterior[
        ["fabricante", "receita_liquida", "lucro", "margem", "pct_encalhe", "gmroii"]
    ].rename(
        columns={
            "receita_liquida": "receita_liquida_ant",
            "lucro": "lucro_ant",
            "margem": "margem_ant",
            "pct_encalhe": "pct_encalhe_ant",
            "gmroii": "gmroii_ant",
        }
    )

    merged = kpis_atual.merge(anterior, on="fabricante", how="left")

    def _var(atual, anterior):
        """Variação percentual segura."""
        return np.where(
            anterior.abs() > 0,
            (atual - anterior) / anterior.abs(),
            np.where(atual > 0, 1.0, 0.0),
        )

    merged["var_receita"] = _var(merged["receita_liquida"], merged["receita_liquida_ant"])
    merged["var_lucro"] = _var(merged["lucro"], merged["lucro_ant"])
    merged["var_margem"] = _var(merged["margem"], merged["margem_ant"])
    merged["var_encalhe"] = _var(merged["pct_encalhe"], merged["pct_encalhe_ant"])
    merged["var_gmroii"] = _var(merged["gmroii"], merged["gmroii_ant"])

    # Limpa colunas auxiliares
    merged.drop(
        columns=["receita_liquida_ant", "lucro_ant", "margem_ant", "pct_encalhe_ant", "gmroii_ant"],
        inplace=True,
        errors="ignore",
    )

    return merged


# ---------------------------------------------------------------------------
# Pipeline completo (helper para uso no app)
# ---------------------------------------------------------------------------

def run_pipeline(
    vendas_raw: pd.DataFrame,
    giro_raw: pd.DataFrame,
    vendas_raw_anterior: Optional[pd.DataFrame] = None,
    giro_raw_anterior: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Executa o pipeline completo e retorna todas as camadas.

    Retorna
    -------
    dict com chaves:
      vendas_sku, base_integrada, kpis_fabricante,
      kpis_anterior (se disponível), kpis_com_variacao (se disponível)
    """
    vendas_sku = build_vendas_sku(vendas_raw)
    base_integrada = build_base_integrada(vendas_sku, giro_raw)
    kpis = build_kpis_fabricante(base_integrada)

    result = {
        "vendas_sku": vendas_sku,
        "base_integrada": base_integrada,
        "kpis_fabricante": kpis,
        "kpis_anterior": None,
        "kpis_com_variacao": None,
    }

    if vendas_raw_anterior is not None and giro_raw_anterior is not None:
        vendas_sku_ant = build_vendas_sku(vendas_raw_anterior)
        base_ant = build_base_integrada(vendas_sku_ant, giro_raw_anterior)
        kpis_ant = build_kpis_fabricante(base_ant)
        kpis_var = build_variacao(kpis, kpis_ant)

        result["kpis_anterior"] = kpis_ant
        result["kpis_com_variacao"] = kpis_var

    return result
