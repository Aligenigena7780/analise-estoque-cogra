"""
processor.py
============
Camada de processamento e construção das camadas analíticas.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


MESES_PT = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}


def listar_periodos_disponiveis(vendas_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame único com ano/mês disponíveis no arquivo.
    """
    periodos = (
        vendas_raw[["ano", "mes"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["ano", "mes"])
        .reset_index(drop=True)
    )
    periodos["ano"] = periodos["ano"].astype(int)
    periodos["mes"] = periodos["mes"].astype(int)
    periodos["mes_nome"] = periodos["mes"].map(MESES_PT)
    periodos["chave"] = periodos["ano"].astype(str) + "-" + periodos["mes"].astype(str).str.zfill(2)
    return periodos


def obter_periodo_padrao(vendas_raw: pd.DataFrame) -> tuple[int, int, int]:
    """
    Retorna ano, mês início e mês fim do período padrão:
    mês mais recente disponível.
    """
    periodos = listar_periodos_disponiveis(vendas_raw)
    if periodos.empty:
        raise ValueError("Nenhum período válido encontrado no relatório de vendas.")
    ultimo = periodos.iloc[-1]
    return int(ultimo["ano"]), int(ultimo["mes"]), int(ultimo["mes"])


def filtrar_vendas_periodo(vendas_raw: pd.DataFrame, ano: int, mes_inicio: int, mes_fim: int) -> pd.DataFrame:
    """
    Filtra vendas dentro do range selecionado no ano informado.
    """
    if mes_inicio > mes_fim:
        raise ValueError("Mês início não pode ser maior que mês fim.")

    mask = (
        (vendas_raw["ano"] == ano) &
        (vendas_raw["mes"] >= mes_inicio) &
        (vendas_raw["mes"] <= mes_fim)
    )
    return vendas_raw.loc[mask].copy()


def construir_periodo_anterior(ano: int, mes_inicio: int, mes_fim: int) -> tuple[int, int, int]:
    """
    Constrói o período anterior equivalente em quantidade de meses.

    Exemplo:
    2026, 1, 3 -> 2025, 10, 12
    """
    inicio_total = ano * 12 + mes_inicio - 1
    fim_total = ano * 12 + mes_fim - 1
    tamanho = fim_total - inicio_total + 1

    inicio_ant = inicio_total - tamanho
    fim_ant = fim_total - tamanho

    ano_ant_inicio = inicio_ant // 12
    mes_ant_inicio = (inicio_ant % 12) + 1

    ano_ant_fim = fim_ant // 12
    mes_ant_fim = (fim_ant % 12) + 1

    if ano_ant_inicio != ano_ant_fim:
        # Mantemos restrição simples da v1: comparação dentro de um único ano anterior.
        # Como o período selecionado já é filtrado por ano único, o período anterior equivalente
        # pode cruzar ano. Nesse caso, a filtragem será feita por data completa em helper abaixo.
        pass

    return ano_ant_inicio, mes_ant_inicio, (ano_ant_fim * 100 + mes_ant_fim)


def filtrar_vendas_periodo_anterior(vendas_raw: pd.DataFrame, ano: int, mes_inicio: int, mes_fim: int) -> pd.DataFrame:
    """
    Cria a janela anterior equivalente com base em data mensal real.
    """
    periodo_atual = pd.period_range(
        start=f"{ano}-{mes_inicio:02d}",
        end=f"{ano}-{mes_fim:02d}",
        freq="M",
    )
    periodo_anterior = periodo_atual - len(periodo_atual)

    chaves = {str(p) for p in periodo_anterior}
    return vendas_raw.loc[vendas_raw["ano_mes"].isin(chaves)].copy()


def build_vendas_sku(vendas_raw: pd.DataFrame) -> pd.DataFrame:
    df = vendas_raw.copy()

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

    vendas_sku["margem_media"] = np.where(
        vendas_sku["_margem_abs_sum"] > 0,
        vendas_sku["_margem_pond_sum"] / vendas_sku["_margem_abs_sum"],
        0,
    )
    vendas_sku.drop(columns=["_margem_pond_sum", "_margem_abs_sum"], inplace=True)

    return vendas_sku


def build_base_integrada(vendas_sku: pd.DataFrame, giro_raw: pd.DataFrame) -> pd.DataFrame:
    base = pd.merge(
        giro_raw,
        vendas_sku[["sku", "receita_liquida", "lucro_total", "quantidade_total", "margem_media"]],
        on="sku",
        how="left",
    )

    base["receita_liquida"] = base["receita_liquida"].fillna(0)
    base["lucro_total"] = base["lucro_total"].fillna(0)
    base["quantidade_total"] = base["quantidade_total"].fillna(0)
    base["margem_media"] = base["margem_media"].fillna(0)

    base["is_encalhe"] = base["classificacao_esa"].str.lower() == "encalhe"

    return base


def build_kpis_fabricante(base_integrada: pd.DataFrame) -> pd.DataFrame:
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

    encalhe_df = df[df["is_encalhe"]].groupby("fabricante", as_index=False)["estoque_atual"].sum()
    encalhe_df.rename(columns={"estoque_atual": "estoque_encalhe"}, inplace=True)
    kpis = kpis.merge(encalhe_df, on="fabricante", how="left")
    kpis["estoque_encalhe"] = kpis["estoque_encalhe"].fillna(0)

    kpis["margem"] = np.where(
        kpis["_receita_abs"] > 0,
        kpis["_marg_pond"] / kpis["_receita_abs"],
        0,
    )
    kpis.drop(columns=["_marg_pond", "_receita_abs"], inplace=True)

    kpis["pct_encalhe"] = np.where(
        kpis["estoque_atual"] > 0,
        kpis["estoque_encalhe"] / kpis["estoque_atual"],
        0,
    )

    kpis["gmroii"] = np.where(
        kpis["estoque_atual"] > 0,
        kpis["lucro"] / kpis["estoque_atual"],
        0,
    )

    return kpis.sort_values("receita_liquida", ascending=False).reset_index(drop=True)


def build_variacao(kpis_atual: pd.DataFrame, kpis_anterior: pd.DataFrame) -> pd.DataFrame:
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

    merged.drop(
        columns=["receita_liquida_ant", "lucro_ant", "margem_ant", "pct_encalhe_ant", "gmroii_ant"],
        inplace=True,
        errors="ignore",
    )
    return merged


def run_pipeline(
    vendas_raw_historico: pd.DataFrame,
    giro_raw_atual: pd.DataFrame,
    ano: int,
    mes_inicio: int,
    mes_fim: int,
    giro_raw_anterior: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Executa o pipeline completo usando:
    - vendas histórico + filtro por período
    - giro atual obrigatório
    - giro anterior opcional
    """
    vendas_atual = filtrar_vendas_periodo(vendas_raw_historico, ano, mes_inicio, mes_fim)
    if vendas_atual.empty:
        raise ValueError("O período selecionado não possui vendas.")

    vendas_anterior = filtrar_vendas_periodo_anterior(vendas_raw_historico, ano, mes_inicio, mes_fim)

    vendas_sku = build_vendas_sku(vendas_atual)
    base_integrada = build_base_integrada(vendas_sku, giro_raw_atual)
    kpis = build_kpis_fabricante(base_integrada)

    result = {
        "vendas_filtradas_atual": vendas_atual,
        "vendas_filtradas_anterior": vendas_anterior,
        "vendas_sku": vendas_sku,
        "base_integrada": base_integrada,
        "kpis_fabricante": kpis,
        "kpis_anterior": None,
        "kpis_com_variacao": None,
        "periodo_atual": f"{ano}-{mes_inicio:02d} até {ano}-{mes_fim:02d}",
    }

    if not vendas_anterior.empty and giro_raw_anterior is not None:
        vendas_sku_ant = build_vendas_sku(vendas_anterior)
        base_ant = build_base_integrada(vendas_sku_ant, giro_raw_anterior)
        kpis_ant = build_kpis_fabricante(base_ant)
        kpis_var = build_variacao(kpis, kpis_ant)

        result["kpis_anterior"] = kpis_ant
        result["kpis_com_variacao"] = kpis_var

    return result
