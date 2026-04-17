from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Análise de Produtos",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# CONFIG VISUAL
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

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


# =========================================================
# HELPERS
# =========================================================
def fmt_brl(valor: float) -> str:
    if pd.isna(valor):
        return "R$ 0"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_brl_int(valor: float) -> str:
    if pd.isna(valor):
        return "R$ 0"
    return f"R$ {valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(valor: float) -> str:
    if pd.isna(valor):
        return "0,00%"
    return f"{valor * 100:.2f}%".replace(".", ",")


def fmt_var(valor: float) -> str:
    if pd.isna(valor):
        return "—"
    sinal = "+" if valor > 0 else ""
    return f"{sinal}{valor * 100:.2f}%".replace(".", ",")


def fmt_num(valor: float) -> str:
    if pd.isna(valor):
        return "0,00"
    return f"{valor:.2f}".replace(".", ",")


def parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    def convert(x):
        if x in ["", "nan", "None", "none", "-", "--"]:
            return np.nan

        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x and "." not in x:
            x = x.replace(",", ".")

        try:
            return float(x)
        except Exception:
            return np.nan

    return s.apply(convert)


def localizar_coluna(df: pd.DataFrame, nomes_possiveis: list[str], obrigatoria: bool = True) -> str | None:
    for nome in nomes_possiveis:
        if nome in df.columns:
            return nome
    if obrigatoria:
        raise KeyError(f"Coluna não encontrada. Esperado um destes nomes: {nomes_possiveis}")
    return None


# =========================================================
# LOADERS
# =========================================================
@st.cache_data(show_spinner=False)
def carregar_vendas(arquivo_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(arquivo_bytes), dtype=str).copy()
    df.columns = df.columns.astype(str).str.strip()

    col_data = localizar_coluna(df, ["calendarioData", "Data", "Data Emissão", "Data Emissao"])
    col_bruto = localizar_coluna(df, ["Venda", "Vendas", "Venda Bruta", "Faturamento Bruto"])
    col_liquido = localizar_coluna(df, ["Venda Líquida", "Venda Liquida"])
    col_lucro = localizar_coluna(df, ["Lucro"])
    col_tipo = localizar_coluna(df, ["Tipo"])
    col_sku = localizar_coluna(df, ["SKU", "Sku", "sku"])
    col_fabricante = localizar_coluna(df, ["Fabricante"])

    df = df.rename(columns={
        col_data: "data",
        col_bruto: "vendas",
        col_liquido: "venda_liquida",
        col_lucro: "lucro",
        col_tipo: "tipo",
        col_sku: "sku",
        col_fabricante: "fabricante",
    })

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia"] = df["data"].dt.day
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)

    df["vendas"] = parse_numeric(df["vendas"]).fillna(0)
    df["venda_liquida"] = parse_numeric(df["venda_liquida"]).fillna(0)
    df["lucro"] = parse_numeric(df["lucro"]).fillna(0)
    df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
    df["fabricante"] = df["fabricante"].astype(str).str.strip()

    # No seu relatório, devoluções já vêm negativas.
    df["vendas_aj"] = df["vendas"]
    df["venda_liquida_aj"] = df["venda_liquida"]
    df["lucro_aj"] = df["lucro"]

    return df


@st.cache_data(show_spinner=False)
def carregar_giro(arquivo_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(arquivo_bytes), dtype=str).copy()
    df.columns = df.columns.astype(str).str.strip()

    col_custo = localizar_coluna(df, ["Custo"])
    col_fabricante = localizar_coluna(df, ["Fabricante"])
    col_sku = localizar_coluna(df, ["SKU", "Sku", "sku"], obrigatoria=False)

    rename_map = {
        col_custo: "estoque_total",
        col_fabricante: "fabricante",
    }
    if col_sku:
        rename_map[col_sku] = "sku"

    df = df.rename(columns=rename_map)
    df["estoque_total"] = parse_numeric(df["estoque_total"]).fillna(0)
    df["fabricante"] = df["fabricante"].astype(str).str.strip()

    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str).str.strip()

    return df


# =========================================================
# CÁLCULOS BLOCO 1
# =========================================================
def obter_mes_mais_recente(df_vendas: pd.DataFrame) -> tuple[int, int]:
    base = df_vendas.dropna(subset=["data"]).copy()
    ano = int(base["ano"].max())
    mes = int(base.loc[base["ano"] == ano, "mes"].max())
    return ano, mes


def filtrar_mes(df: pd.DataFrame, ano: int, mes: int) -> pd.DataFrame:
    return df[(df["ano"] == ano) & (df["mes"] == mes)].copy()


def obter_mes_anterior(ano: int, mes: int) -> tuple[int, int]:
    if mes == 1:
        return ano - 1, 12
    return ano, mes - 1


def calcular_kpis(df_vendas_mes: pd.DataFrame, estoque_total: float) -> dict:
    bruto = df_vendas_mes["vendas_aj"].sum()
    liquido = df_vendas_mes["venda_liquida_aj"].sum()
    lucro = df_vendas_mes["lucro_aj"].sum()
    margem = (lucro / liquido) if liquido != 0 else 0
    gmroii = (lucro / estoque_total) if estoque_total != 0 else 0

    return {
        "faturamento_bruto": bruto,
        "faturamento_liquido": liquido,
        "lucro": lucro,
        "margem": margem,
        "gmroii": gmroii,
        "estoque_total": estoque_total,
    }


def calcular_variacao(atual: float, anterior: float) -> float:
    if anterior == 0:
        if atual == 0:
            return 0
        return 1.0
    return (atual - anterior) / abs(anterior)


def grafico_diario_mes(df_vendas_mes: pd.DataFrame) -> pd.DataFrame:
    graf = (
        df_vendas_mes.groupby("dia", as_index=False)["vendas_aj"]
        .sum()
        .rename(columns={"vendas_aj": "faturamento_bruto"})
        .sort_values("dia")
    )
    return graf


def grafico_ultimos_6_meses(df_vendas: pd.DataFrame, ano_ref: int, mes_ref: int) -> pd.DataFrame:
    periodo_ref = pd.Period(f"{ano_ref}-{mes_ref:02d}", freq="M")
    periodos = [periodo_ref - i for i in range(5, -1, -1)]
    chaves = [str(p) for p in periodos]

    base = (
        df_vendas[df_vendas["ano_mes"].isin(chaves)]
        .groupby("ano_mes", as_index=False)["vendas_aj"]
        .sum()
        .rename(columns={"vendas_aj": "faturamento_bruto"})
    )

    df_base = pd.DataFrame({"ano_mes": chaves})
    base = df_base.merge(base[["ano_mes", "faturamento_bruto"]], on="ano_mes", how="left").fillna(0)
    base["label"] = pd.to_datetime(base["ano_mes"]).dt.strftime("%m/%Y")
    return base


# =========================================================
# CÁLCULOS BLOCO 2
# =========================================================
def calcular_kpis_fabricante(df_vendas_mes: pd.DataFrame, df_giro: pd.DataFrame, fabricante: str) -> dict:
    vendas_fab = df_vendas_mes[df_vendas_mes["fabricante"] == fabricante].copy()
    giro_fab = df_giro[df_giro["fabricante"] == fabricante].copy()

    estoque_total = giro_fab["estoque_total"].sum()
    return calcular_kpis(vendas_fab, estoque_total)


def grafico_diario_fabricante(df_vendas_mes: pd.DataFrame, fabricante: str) -> pd.DataFrame:
    base = (
        df_vendas_mes[df_vendas_mes["fabricante"] == fabricante]
        .groupby("dia", as_index=False)["vendas_aj"]
        .sum()
        .rename(columns={"vendas_aj": "faturamento_bruto"})
        .sort_values("dia")
    )
    return base


def grafico_6m_fabricante(df_vendas: pd.DataFrame, fabricante: str, ano_ref: int, mes_ref: int) -> pd.DataFrame:
    periodo_ref = pd.Period(f"{ano_ref}-{mes_ref:02d}", freq="M")
    periodos = [periodo_ref - i for i in range(5, -1, -1)]
    chaves = [str(p) for p in periodos]

    base = (
        df_vendas[
            (df_vendas["fabricante"] == fabricante) &
            (df_vendas["ano_mes"].isin(chaves))
        ]
        .groupby("ano_mes", as_index=False)["vendas_aj"]
        .sum()
        .rename(columns={"vendas_aj": "faturamento_bruto"})
    )

    df_base = pd.DataFrame({"ano_mes": chaves})
    base = df_base.merge(base[["ano_mes", "faturamento_bruto"]], on="ano_mes", how="left").fillna(0)
    base["label"] = pd.to_datetime(base["ano_mes"]).dt.strftime("%m/%Y")
    return base


def ordem_fabricantes(df_mes_atual: pd.DataFrame) -> list[str]:
    base = (
        df_mes_atual.groupby("fabricante", as_index=False)["venda_liquida_aj"]
        .sum()
        .sort_values("venda_liquida_aj", ascending=False)
    )
    return base["fabricante"].dropna().astype(str).tolist()


# =========================================================
# INTERFACE
# =========================================================
st.markdown('<div class="main-title">Análise de Produtos</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Bloco 1 — Análise da Cogra | Bloco 2 — Análise por Fabricante</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Arquivos")
    vendas_file = st.file_uploader("Relatório de Vendas", type=["xlsx", "xls"], key="vendas")
    giro_file = st.file_uploader("Relatório de Giro Atual", type=["xlsx", "xls"], key="giro")

if not vendas_file or not giro_file:
    st.info("Faça o upload do relatório de vendas e do relatório de giro atual para continuar.")
    st.stop()

try:
    df_vendas = carregar_vendas(vendas_file.read())
    df_giro = carregar_giro(giro_file.read())
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()

ano_atual, mes_atual = obter_mes_mais_recente(df_vendas)
ano_anterior, mes_anterior = obter_mes_anterior(ano_atual, mes_atual)

df_mes_atual = filtrar_mes(df_vendas, ano_atual, mes_atual)
df_mes_anterior = filtrar_mes(df_vendas, ano_anterior, mes_anterior)

estoque_total_cogra = df_giro["estoque_total"].sum()

kpis_atual = calcular_kpis(df_mes_atual, estoque_total_cogra)
kpis_anterior = calcular_kpis(df_mes_anterior, estoque_total_cogra)

variacoes = {
    "faturamento_bruto": calcular_variacao(kpis_atual["faturamento_bruto"], kpis_anterior["faturamento_bruto"]),
    "faturamento_liquido": calcular_variacao(kpis_atual["faturamento_liquido"], kpis_anterior["faturamento_liquido"]),
    "lucro": calcular_variacao(kpis_atual["lucro"], kpis_anterior["lucro"]),
    "margem": calcular_variacao(kpis_atual["margem"], kpis_anterior["margem"]),
    "gmroii": calcular_variacao(kpis_atual["gmroii"], kpis_anterior["gmroii"]),
}

# =========================================================
# BLOCO 1 — ANÁLISE DA COGRA
# =========================================================
st.markdown('<div class="section-title">Análise da Cogra</div>', unsafe_allow_html=True)
st.caption(f"Mês analisado: {MESES_PT[mes_atual]}/{ano_atual} | Comparação: {MESES_PT[mes_anterior]}/{ano_anterior}")

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric("Faturamento Bruto", fmt_brl_int(kpis_atual["faturamento_bruto"]), fmt_var(variacoes["faturamento_bruto"]))
c2.metric("Faturamento Líquido", fmt_brl_int(kpis_atual["faturamento_liquido"]), fmt_var(variacoes["faturamento_liquido"]))
c3.metric("Lucro", fmt_brl_int(kpis_atual["lucro"]), fmt_var(variacoes["lucro"]))
c4.metric("Margem", fmt_pct(kpis_atual["margem"]), fmt_var(variacoes["margem"]))
c5.metric("GMROII", fmt_num(kpis_atual["gmroii"]), fmt_var(variacoes["gmroii"]))
c6.metric("Estoque Total", fmt_brl_int(kpis_atual["estoque_total"]))

st.markdown("### Performance de vendas dentro do mês")
graf_dia = grafico_diario_mes(df_mes_atual)
st.bar_chart(graf_dia.set_index("dia")["faturamento_bruto"])

st.markdown("### Performance de vendas dos últimos 6 meses")
graf_6m = grafico_ultimos_6_meses(df_vendas, ano_atual, mes_atual)
st.line_chart(graf_6m.set_index("label")["faturamento_bruto"])

# =========================================================
# BLOCO 2 — ANÁLISE POR FABRICANTE
# =========================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Análise por Fabricante</div>', unsafe_allow_html=True)
st.caption("Painéis expansíveis por fabricante, com resumo no cabeçalho e detalhamento ao abrir.")

fabricantes = ordem_fabricantes(df_mes_atual)

if not fabricantes:
    st.warning("Nenhum fabricante encontrado no mês analisado.")
else:
    for fabricante in fabricantes:
        kpi_fab_atual = calcular_kpis_fabricante(df_mes_atual, df_giro, fabricante)
        kpi_fab_anterior = calcular_kpis_fabricante(df_mes_anterior, df_giro, fabricante)

        var_fab = calcular_variacao(
            kpi_fab_atual["faturamento_liquido"],
            kpi_fab_anterior["faturamento_liquido"]
        )

        expander_label = (
            f"{fabricante} | "
            f"Líq.: {fmt_brl_int(kpi_fab_atual['faturamento_liquido'])} | "
            f"Lucro: {fmt_brl_int(kpi_fab_atual['lucro'])} | "
            f"Margem: {fmt_pct(kpi_fab_atual['margem'])} | "
            f"Var.: {fmt_var(var_fab)} | "
            f"GMROII: {fmt_num(kpi_fab_atual['gmroii'])}"
        )

        with st.expander(expander_label, expanded=False):
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric(
                "Faturamento Bruto",
                fmt_brl_int(kpi_fab_atual["faturamento_bruto"]),
                fmt_var(calcular_variacao(kpi_fab_atual["faturamento_bruto"], kpi_fab_anterior["faturamento_bruto"]))
            )
            col2.metric(
                "Faturamento Líquido",
                fmt_brl_int(kpi_fab_atual["faturamento_liquido"]),
                fmt_var(calcular_variacao(kpi_fab_atual["faturamento_liquido"], kpi_fab_anterior["faturamento_liquido"]))
            )
            col3.metric(
                "Lucro",
                fmt_brl_int(kpi_fab_atual["lucro"]),
                fmt_var(calcular_variacao(kpi_fab_atual["lucro"], kpi_fab_anterior["lucro"]))
            )
            col4.metric(
                "Margem",
                fmt_pct(kpi_fab_atual["margem"]),
                fmt_var(calcular_variacao(kpi_fab_atual["margem"], kpi_fab_anterior["margem"]))
            )
            col5.metric(
                "GMROII",
                fmt_num(kpi_fab_atual["gmroii"]),
                fmt_var(calcular_variacao(kpi_fab_atual["gmroii"], kpi_fab_anterior["gmroii"]))
            )
            col6.metric(
                "Estoque Total",
                fmt_brl_int(kpi_fab_atual["estoque_total"])
            )

            st.markdown("#### Performance diária no mês")
            graf_dia_fab = grafico_diario_fabricante(df_mes_atual, fabricante)
            if graf_dia_fab.empty:
                st.info("Sem movimentação no mês para este fabricante.")
            else:
                st.bar_chart(graf_dia_fab.set_index("dia")["faturamento_bruto"])

            st.markdown("#### Performance dos últimos 6 meses")
            graf_6m_fab = grafico_6m_fabricante(df_vendas, fabricante, ano_atual, mes_atual)
            st.line_chart(graf_6m_fab.set_index("label")["faturamento_bruto"])
