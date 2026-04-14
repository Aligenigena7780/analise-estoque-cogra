"""
data_loader.py
==============
Camada de ingestão de dados brutos (Excel).
Responsabilidade: carregar, validar e expor os dados sem transformações de negócio.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Union

import pandas as pd

Source = Union[str, Path, io.BytesIO, Any]

VENDAS_REQUIRED_COLS: dict[str, list[str]] = {
    "sku": [
        "SKU", "Sku", "sku",
        "Código", "Codigo", "codigo",
        "Cod. Produto", "Cod.Produto",
    ],
    "fabricante": [
        "Fabricante", "fabricante", "FABRICANTE",
        "Fornecedor", "fornecedor",
    ],
    "descricao": [
        "Descrição", "Descricao", "descricao",
        "Produto", "produto",
        "Desc. Produto",
    ],
    "tipo": [
        "Tipo", "tipo", "TIPO",
    ],
    "venda_liquida": [
        "Venda Líquida", "Venda Liquida",
        "VendaLiquida", "venda_liquida",
        "Vlr. Líquido", "Vlr Liquido",
    ],
    "lucro": [
        "Lucro", "lucro", "LUCRO",
        "Lucro Bruto", "lucro_bruto",
    ],
    "margem": [
        "Margem c/ Reb.",
        "Margem c/ Rebate",
        "Margem com Rebate",
        "Margem c/Rebate",
        "Margem c/Reb.",
        "Margem",
        "margem",
        "Margem %",
    ],
    "quantidade": [
        "Qtd.",
        "Quantidade", "quantidade",
        "Qtd", "Qtde", "qtd",
        "Qty",
    ],
    "data": [
        "calendarioData",   # nome real confirmado
        "Data",
        "Data Emissão",
        "Data Emissao",
        "Data Faturamento",
        "Emissão",
        "Emissao",
    ],
}

VENDAS_OPTIONAL_COLS: dict[str, list[str]] = {
    "documento": [
        "Documento", "documento",
    ],
}

GIRO_REQUIRED_COLS: dict[str, list[str]] = {
    "sku": [
        "SKU", "Sku", "sku",
        "Código", "Codigo", "codigo",
        "Cod. Produto",
    ],
    "fabricante": [
        "Fabricante", "fabricante", "FABRICANTE",
        "Fornecedor", "fornecedor",
    ],
    "descricao": [
        "Descrição", "Descricao", "descricao",
        "Produto", "produto",
        "Desc. Produto",
    ],
    "estoque_atual": [
        "Estoque Atual",
        "EstoqueAtual", "estoque_atual",
        "Estoque", "estoque",
        "Saldo Estoque",
    ],
    "media_vendas": [
        "Média Vendas",
        "Media Vendas",
        "Média de Vendas",
        "Media de Vendas",
        "MediaVendas",
        "media_vendas",
        "Méd. Vendas",
    ],
    "indice_giro": [
        "Indice Giro",
        "Índice Giro",
        "Indice de Giro",
        "Índice de Giro",
        "IndiceGiro",
        "indice_giro",
        "Giro",
    ],
    "doi": [
        "Dias Estoque + Pedidos Mês",
        "Dias Estoque + Pedidos Mes",
        "DOI", "doi",
        "Dias de Inventário", "Dias Inventario",
    ],
    "dias_sem_venda": [
        "Dias Última Venda",
        "Dias Ultima Venda",
        "Dias sem Venda",
        "Dias Sem Venda",
        "DiasSemVenda",
        "dias_sem_venda",
        "D. Sem Venda",
    ],
    "classificacao_esa": [
        "ESA Atual",
        "ESA",
        "esa",
        "esa_atual",
        "Classificação ESA",
        "Classificacao ESA",
        "Class. ESA",
    ],
}

GIRO_OPTIONAL_COLS: dict[str, list[str]] = {
    "clientes": [
        "Clientes", "clientes",
    ],
    "linha": [
        "Linha", "linha",
    ],
    "grupo": [
        "Grupo", "grupo",
    ],
}

ESA_VALIDAS: set[str] = {
    "Sem",
    "Novo",
    "Giro",
    "Abaixo",
    "Normal",
    "Aging",
    "Slow",
    ">=120d",
    "Encalhe",
}

_ESA_CANONICAL_MAP = {
    "sem": "Sem",
    "novo": "Novo",
    "giro": "Giro",
    "abaixo": "Abaixo",
    "normal": "Normal",
    "aging": "Aging",
    "slow": "Slow",
    ">=120d": ">=120d",
    "encalhe": "Encalhe",
}


def _resolve_source(source: Source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return path
    return source


def _normalize_column(df: pd.DataFrame, candidates: list[str], canonical: str) -> pd.DataFrame:
    for candidate in candidates:
        if candidate in df.columns:
            if candidate != canonical:
                df = df.rename(columns={candidate: canonical})
            return df

    raise KeyError(
        f"\n  Coluna obrigatória '{canonical}' não encontrada.\n"
        f"  Nomes aceitos:\n    {candidates}\n"
        f"  Colunas presentes:\n    {sorted(df.columns.tolist())}"
    )


def _normalize_columns(df: pd.DataFrame, col_map: dict[str, list[str]]) -> pd.DataFrame:
    for canonical, candidates in col_map.items():
        df = _normalize_column(df, candidates, canonical)
    return df


def _try_normalize_optional(df: pd.DataFrame, col_map: dict[str, list[str]]) -> pd.DataFrame:
    for canonical, candidates in col_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                if candidate != canonical:
                    df = df.rename(columns={candidate: canonical})
                break
    return df


def _is_blankish(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "-", "--"}


def _parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({
        "": None,
        "-": None,
        "--": None,
        "nan": None,
        "None": None,
        "none": None,
    })

    def _convert_one(x):
        if x is None or pd.isna(x):
            return None

        x = str(x).strip()
        if x == "":
            return None

        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x and "." not in x:
            x = x.replace(",", ".")

        try:
            return float(x)
        except ValueError:
            return None

    out = s.apply(_convert_one)
    return pd.to_numeric(out, errors="coerce").fillna(0.0)


def _parse_pct(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("%", "", regex=False).str.strip()
    return _parse_numeric(s)


def _normalize_esa_value(value: str) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip()
    if raw == "":
        return ""
    return _ESA_CANONICAL_MAP.get(raw.lower(), raw)


def load_vendas(source: Source) -> pd.DataFrame:
    source = _resolve_source(source)

    df = pd.read_excel(source, dtype=str).copy()
    df.columns = df.columns.astype(str).str.strip()

    df = _normalize_columns(df, VENDAS_REQUIRED_COLS)
    df = _try_normalize_optional(df, VENDAS_OPTIONAL_COLS)

    df["_blank_venda_liquida"] = df["venda_liquida"].apply(_is_blankish)
    df["_blank_lucro"] = df["lucro"].apply(_is_blankish)
    df["_blank_margem"] = df["margem"].apply(_is_blankish)
    df["_blank_quantidade"] = df["quantidade"].apply(_is_blankish)
    df["_blank_data"] = df["data"].apply(_is_blankish)

    df["venda_liquida"] = _parse_numeric(df["venda_liquida"])
    df["lucro"] = _parse_numeric(df["lucro"])
    df["margem"] = _parse_pct(df["margem"])
    df["quantidade"] = _parse_numeric(df["quantidade"])

    df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip()
    df["fabricante"] = df["fabricante"].astype(str).str.strip()
    df["descricao"] = df["descricao"].astype(str).str.strip()

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)

    if "documento" in df.columns:
        df["documento"] = df["documento"].astype(str).str.strip()

    return df


def load_giro(source: Source) -> pd.DataFrame:
    source = _resolve_source(source)

    df = pd.read_excel(source, dtype=str).copy()
    df.columns = df.columns.astype(str).str.strip()

    df = _normalize_columns(df, GIRO_REQUIRED_COLS)
    df = _try_normalize_optional(df, GIRO_OPTIONAL_COLS)

    df["_blank_estoque_atual"] = df["estoque_atual"].apply(_is_blankish)
    df["_blank_media_vendas"] = df["media_vendas"].apply(_is_blankish)
    df["_blank_indice_giro"] = df["indice_giro"].apply(_is_blankish)
    df["_blank_doi"] = df["doi"].apply(_is_blankish)
    df["_blank_dias_sem_venda"] = df["dias_sem_venda"].apply(_is_blankish)

    df["estoque_atual"] = _parse_numeric(df["estoque_atual"])
    df["media_vendas"] = _parse_numeric(df["media_vendas"])
    df["indice_giro"] = _parse_numeric(df["indice_giro"])
    df["doi"] = _parse_numeric(df["doi"])
    df["dias_sem_venda"] = _parse_numeric(df["dias_sem_venda"])

    df["sku"] = df["sku"].astype(str).str.strip()
    df["fabricante"] = df["fabricante"].astype(str).str.strip()
    df["descricao"] = df["descricao"].astype(str).str.strip()
    df["classificacao_esa"] = df["classificacao_esa"].astype(str).str.strip().apply(_normalize_esa_value)

    if "clientes" in df.columns:
        df["clientes"] = _parse_numeric(df["clientes"])
    if "linha" in df.columns:
        df["linha"] = df["linha"].astype(str).str.strip()
    if "grupo" in df.columns:
        df["grupo"] = df["grupo"].astype(str).str.strip()

    return df


def load_vendas_multi_mes(sources: dict[str, Source]) -> dict[str, pd.DataFrame]:
    return {label: load_vendas(src) for label, src in sources.items()}


def validate_vendas(df: pd.DataFrame) -> list[str]:
    warns: list[str] = []

    tipos_validos = {"N", "D"}
    tipos_invalidos = set(df["tipo"].dropna().unique()) - tipos_validos
    if tipos_invalidos:
        warns.append(
            f"Valores inesperados na coluna 'Tipo': {sorted(tipos_invalidos)}. "
            f"Somente 'N' e 'D' são processados."
        )

    for col_flag, label in [
        ("_blank_venda_liquida", "Venda Líquida"),
        ("_blank_lucro", "Lucro"),
        ("_blank_margem", "Margem"),
        ("_blank_quantidade", "Quantidade"),
        ("_blank_data", "Data"),
    ]:
        n = int(df.get(col_flag, pd.Series(dtype=bool)).sum())
        if n > 0:
            warns.append(f"{n} linha(s) com '{label}' vazia/inválida.")

    n_data_invalida = int(df["data"].isna().sum())
    if n_data_invalida > 0:
        warns.append(f"{n_data_invalida} linha(s) com data inválida após parsing.")

    n_sku_vazio = int(df["sku"].astype(str).str.strip().eq("").sum())
    if n_sku_vazio > 0:
        warns.append(f"{n_sku_vazio} linha(s) com SKU vazio nas vendas.")

    return warns


def validate_giro(df: pd.DataFrame) -> list[str]:
    warns: list[str] = []

    for col_flag, label in [
        ("_blank_estoque_atual", "Estoque Atual"),
        ("_blank_media_vendas", "Média Vendas"),
        ("_blank_indice_giro", "Indice Giro"),
        ("_blank_doi", "DOI"),
        ("_blank_dias_sem_venda", "Dias sem Venda"),
    ]:
        n = int(df.get(col_flag, pd.Series(dtype=bool)).sum())
        if n > 0:
            warns.append(f"{n} linha(s) com '{label}' vazio/inválido.")

    n_esa_vazia = int(df["classificacao_esa"].astype(str).str.strip().eq("").sum())
    if n_esa_vazia > 0:
        warns.append(f"{n_esa_vazia} linha(s) com ESA vazia.")

    esas_encontradas = set(
        df["classificacao_esa"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique()
    )
    esas_inesperadas = esas_encontradas - ESA_VALIDAS
    if esas_inesperadas:
        warns.append(
            f"Classificações ESA fora da política atual: {sorted(esas_inesperadas)}."
        )

    n_sku_vazio = int(df["sku"].astype(str).str.strip().eq("").sum())
    if n_sku_vazio > 0:
        warns.append(f"{n_sku_vazio} linha(s) com SKU vazio no giro.")

    return warns
