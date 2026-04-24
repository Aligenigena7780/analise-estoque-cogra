from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        font-size: 60px;
        font-weight: 900;
    
        background: linear-gradient(
            90deg,
            #A80014,
            #E20A13,
            #FF4A1A
        );
    
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #A80014;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 40px;
        font-weight: 700;
        margin-top: 2rem;
        color: #F2F2F2;
        margin-bottom: 1rem;
    }
    .section-divider {
        height: 4px;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.55), transparent);
        margin: 7px 0;
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

ORDEM_ESA = [
    "8 - Sem",
    "0 - Novo",
    "4 - Giro",
    "1 - Abaixo",
    "2 - Normal",
    "3 - Aging",
    "5 - Slow",
    "6 - >=120d",
    "7 - Encalhe",
    "-"
]

CORES_ESA = {
    "8 - Sem": "#757575",
    "0 - Novo": "#9E9E9E",
    "4 - Giro": "#03A9F4",
    "1 - Abaixo": "#8BC34A",
    "2 - Normal": "#4CAF50",
    "3 - Aging": "#FFC107",
    "5 - Slow": "#FF9800",
    "6 - >=120d": "#F44336",
    "7 - Encalhe": "#B71C1C",
    "-": "#616161",
    "Sem classificação": "#BDBDBD",
}

GRUPO_ESA = {
    "0 - Novo": "Saudável",
    "4 - Giro": "Saudável",
    "1 - Abaixo": "Saudável",
    "2 - Normal": "Saudável",

    "3 - Aging": "Atenção",
    "5 - Slow": "Atenção",

    "6 - >=120d": "Crítico",
    "7 - Encalhe": "Crítico",

    "8 - Sem": "Sem classificação",
    "-": "Sem classificação",
    "Sem classificação": "Sem classificação",
}

def ordenar_esa(df: pd.DataFrame, coluna_esa: str = "ESA Atual") -> pd.DataFrame:
    ordem_map = {esa: i for i, esa in enumerate(ORDEM_ESA)}
    df = df.copy()
    df["ordem"] = df[coluna_esa].map(ordem_map).fillna(999)
    return df.sort_values("ordem").drop(columns="ordem")

def cores_por_esa(series_esa: pd.Series) -> list[str]:
    return [CORES_ESA.get(v, "#BDBDBD") for v in series_esa]
    
# =========================================================
# HELPERS
# =========================================================
def fmt_brl(valor: float) -> str:
    if pd.isna(valor):
        return "R$ 0"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pp(valor: float) -> str:
    if pd.isna(valor):
        return "—"
    sinal = "+" if valor > 0 else ""
    return f"{sinal}{valor * 100:.2f} p.p.".replace(".", ",")

def fmt_delta_indice(valor: float) -> str:
    if pd.isna(valor):
        return "—"
    sinal = "+" if valor > 0 else ""
    return f"{sinal}{valor:.2f}".replace(".", ",")

def fmt_brl_int(valor: float) -> str:
    if pd.isna(valor):
        return "R$ 0"
    return f"R$ {valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_brl_int_label(valor: float) -> str:
    if pd.isna(valor):
        return r"R\$ 0"
    texto = f"R$ {valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return texto.replace("$", r"\$")


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
    col_sku = localizar_coluna(
    df,
    ["SKU", "Sku", "sku", "Código", "Codigo", "Cod Produto", "Cod. Produto"],
    obrigatoria=False
)

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
st.markdown('<div class="sub-title">Desenvolvido por Lucas Rodrigues</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Análise da Cogra</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Arquivos")
    vendas_file = st.file_uploader("Relatório de Vendas", type=["xlsx", "xls"], key="vendas")
    giro_file = st.file_uploader("Relatório de Giro Atual", type=["xlsx", "xls"], key="giro")
    giro_anterior_file = st.file_uploader(
        "Relatório de Giro Anterior (opcional)",
        type=["xlsx", "xls"],
        key="giro_anterior"
    )

if not vendas_file or not giro_file:
    st.info("Faça o upload do relatório de vendas e do relatório de giro atual para continuar.")
    st.stop()

try:
    df_vendas = carregar_vendas(vendas_file.read())
    df_giro = carregar_giro(giro_file.read())
    df_giro_anterior = carregar_giro(giro_anterior_file.read()) if giro_anterior_file else None
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()

ano_atual, mes_atual = obter_mes_mais_recente(df_vendas)
ano_anterior, mes_anterior = obter_mes_anterior(ano_atual, mes_atual)

df_mes_atual = filtrar_mes(df_vendas, ano_atual, mes_atual)
df_mes_anterior = filtrar_mes(df_vendas, ano_anterior, mes_anterior)

estoque_total_cogra = df_giro["estoque_total"].sum()
estoque_total_cogra_anterior = df_giro_anterior["estoque_total"].sum() if df_giro_anterior is not None else None

kpis_atual = calcular_kpis(df_mes_atual, estoque_total_cogra)
kpis_anterior = calcular_kpis(df_mes_anterior, estoque_total_cogra_anterior) if estoque_total_cogra_anterior is not None else None

variacoes = {
    "faturamento_bruto": (
        calcular_variacao(
            kpis_atual["faturamento_bruto"],
            kpis_anterior["faturamento_bruto"]
        )
        if kpis_anterior is not None else np.nan
    ),
    "faturamento_liquido": (
        calcular_variacao(
            kpis_atual["faturamento_liquido"],
            kpis_anterior["faturamento_liquido"]
        )
        if kpis_anterior is not None else np.nan
    ),
    "lucro": (
        calcular_variacao(
            kpis_atual["lucro"],
            kpis_anterior["lucro"]
        )
        if kpis_anterior is not None else np.nan
    ),
    "margem_pp": (
        kpis_atual["margem"] - kpis_anterior["margem"]
        if kpis_anterior is not None else np.nan
    ),
    "gmroii_delta": (
        kpis_atual["gmroii"] - kpis_anterior["gmroii"]
        if kpis_anterior is not None else np.nan
    ),
}

# =========================================================
# BLOCO 1 — ANÁLISE DA COGRA
# =========================================================

st.caption(f"Mês analisado: {MESES_PT[mes_atual]}/{ano_atual} | Comparação: {MESES_PT[mes_anterior]}/{ano_anterior}")

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric("Faturamento Bruto", fmt_brl_int(kpis_atual["faturamento_bruto"]), fmt_var(variacoes["faturamento_bruto"]))
c2.metric("Faturamento Líquido", fmt_brl_int(kpis_atual["faturamento_liquido"]), fmt_var(variacoes["faturamento_liquido"]))
c3.metric("Lucro", fmt_brl_int(kpis_atual["lucro"]), fmt_var(variacoes["lucro"]))
c4.metric(
    "Margem",
    fmt_pct(kpis_atual["margem"]),
    fmt_pp(variacoes["margem_pp"])
)

c5.metric(
    "GMROII",
    fmt_num(kpis_atual["gmroii"]),
    fmt_delta_indice(variacoes["gmroii_delta"])
)
c6.metric("Estoque Total", fmt_brl_int(kpis_atual["estoque_total"]))

st.markdown("### Performance de vendas dentro do mês")
graf_dia = grafico_diario_mes(df_mes_atual)

fig_dia = go.Figure()

graf_dia["hover_brl"] = graf_dia["faturamento_bruto"].apply(fmt_brl_int)
graf_dia["hover_dia"] = graf_dia["dia"].apply(lambda x: f"Dia {x}")

fig_dia.add_trace(go.Scatter(
    x=graf_dia["dia"],
    y=graf_dia["faturamento_bruto"],
    mode="lines+markers",
    line=dict(shape="spline", width=3, color="#E20A13"),
    marker=dict(size=6),
    customdata=np.stack([graf_dia["hover_dia"], graf_dia["hover_brl"]], axis=-1),
    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
))

fig_dia.update_layout(
    title="Performance de vendas dentro do mês",
    xaxis_title="Dia",
    yaxis_title="Faturamento",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_dia, use_container_width=True, key="grafico_dia_cogra")

st.markdown("### Performance de vendas dos últimos 6 meses")
graf_6m = grafico_ultimos_6_meses(df_vendas, ano_atual, mes_atual)

fig = go.Figure()

graf_6m["hover_brl"] = graf_6m["faturamento_bruto"].apply(fmt_brl_int)

fig.add_trace(go.Scatter(
    x=graf_6m["label"],
    y=graf_6m["faturamento_bruto"],
    mode="lines+markers",
    line=dict(shape="spline", width=3, color="#E20A13"),
    marker=dict(size=6),
    customdata=np.stack([graf_6m["label"], graf_6m["hover_brl"]], axis=-1),
    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
))

fig.update_layout(
    title="Performance dos últimos 6 meses",
    xaxis_title="Mês",
    yaxis_title="Faturamento",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True, key="grafico_6m_cogra")

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

        kpi_fab_anterior = (
            calcular_kpis_fabricante(df_mes_anterior, df_giro_anterior, fabricante)
            if df_giro_anterior is not None else None
        )

        expander_label = (
            f"{fabricante} | "
            f"Bruto: {fmt_brl_int_label(kpi_fab_atual['faturamento_bruto'])} | "
            f"Lucro: {fmt_brl_int_label(kpi_fab_atual['lucro'])} | "
            f"Margem: {fmt_pct(kpi_fab_atual['margem'])} | "
            f"GMROII: {fmt_num(kpi_fab_atual['gmroii'])}"
        )

        with st.expander(expander_label, expanded=False):
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            if kpi_fab_anterior is not None:
                delta_bruto = fmt_var(calcular_variacao(
                    kpi_fab_atual["faturamento_bruto"],
                    kpi_fab_anterior["faturamento_bruto"]
                ))
                delta_liq = fmt_var(calcular_variacao(
                    kpi_fab_atual["faturamento_liquido"],
                    kpi_fab_anterior["faturamento_liquido"]
                ))
                delta_lucro = fmt_var(calcular_variacao(
                    kpi_fab_atual["lucro"],
                    kpi_fab_anterior["lucro"]
                ))
                delta_margem = fmt_pp(
                    kpi_fab_atual["margem"] - kpi_fab_anterior["margem"]
                )
                delta_gmroii = fmt_delta_indice(
                    kpi_fab_atual["gmroii"] - kpi_fab_anterior["gmroii"]
                )
            else:
                delta_bruto = "—"
                delta_liq = "—"
                delta_lucro = "—"
                delta_margem = "—"
                delta_gmroii = "—"

            col1.metric(
                "Faturamento Bruto",
                fmt_brl_int(kpi_fab_atual["faturamento_bruto"]),
                delta_bruto
            )
            col2.metric(
                "Faturamento Líquido",
                fmt_brl_int(kpi_fab_atual["faturamento_liquido"]),
                delta_liq
            )
            col3.metric(
                "Lucro",
                fmt_brl_int(kpi_fab_atual["lucro"]),
                delta_lucro
            )

            col4.metric(
                "Margem",
                fmt_pct(kpi_fab_atual["margem"]),
                delta_margem
            )
            col5.metric(
                "GMROII",
                fmt_num(kpi_fab_atual["gmroii"]),
                delta_gmroii
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
                fig_dia_fab = go.Figure()
            
                graf_dia_fab["hover_brl"] = graf_dia_fab["faturamento_bruto"].apply(fmt_brl_int)
                graf_dia_fab["hover_dia"] = graf_dia_fab["dia"].apply(lambda x: f"Dia {x}")
                
                fig_dia_fab.add_trace(go.Scatter(
                    x=graf_dia_fab["dia"],
                    y=graf_dia_fab["faturamento_bruto"],
                    mode="lines+markers",
                    line=dict(shape="spline", width=3, color="#E20A13"),
                    marker=dict(size=6),
                    customdata=np.stack([graf_dia_fab["hover_dia"], graf_dia_fab["hover_brl"]], axis=-1),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
                ))
            
                fig_dia_fab.update_layout(
                    title="Performance diária no mês",
                    xaxis_title="Dia",
                    yaxis_title="Faturamento",
                    template="plotly_dark",
                    height=400
                )
            
                st.plotly_chart(
                    fig_dia_fab,
                    use_container_width=True,
                    key=f"grafico_dia_fabricante_{fabricante}"
                )
            
            st.markdown("#### Performance dos últimos 6 meses")
            graf_6m_fab = grafico_6m_fabricante(df_vendas, fabricante, ano_atual, mes_atual)
            
            fig_fab = go.Figure()
            
            graf_6m_fab["hover_brl"] = graf_6m_fab["faturamento_bruto"].apply(fmt_brl_int)
            
            fig_fab.add_trace(go.Scatter(
                x=graf_6m_fab["label"],
                y=graf_6m_fab["faturamento_bruto"],
                mode="lines+markers",
                line=dict(shape="spline", width=3, color="#E20A13"),
                marker=dict(size=6),
                customdata=np.stack([graf_6m_fab["label"], graf_6m_fab["hover_brl"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
            ))
            
            fig_fab.update_layout(
                title="Performance dos últimos 6 meses",
                xaxis_title="Mês",
                yaxis_title="Faturamento",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(
                fig_fab,
                use_container_width=True,
                key=f"grafico_6m_fabricante_{fabricante}"
            )

# ============================================================
# BLOCO 3 — ANÁLISE DE ESTOQUE
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Análise de Estoque</div>', unsafe_allow_html=True)

# ----------------------------
# KPI 1 — Custo Total Atual
# ----------------------------
custo_total_atual = df_giro["estoque_total"].sum()

# ----------------------------
# KPI 2 — Custo Total Anterior
# ----------------------------
if df_giro_anterior is not None:
    custo_total_anterior = df_giro_anterior["estoque_total"].sum()

    # Variação %
    if custo_total_anterior != 0:
        variacao_estoque = (custo_total_atual - custo_total_anterior) / custo_total_anterior
        delta_estoque = fmt_var(variacao_estoque)
    else:
        delta_estoque = "—"
else:
    custo_total_anterior = None
    delta_estoque = "—"

# ----------------------------
# Exibição dos KPIs
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Custo Total Atual",
    fmt_brl_int(custo_total_atual)
)

col2.metric(
    "Custo Mês Anterior",
    fmt_brl_int(custo_total_anterior) if custo_total_anterior is not None else "—"
)

col3.metric(
    "Variação do Estoque",
    delta_estoque
)


# ----------------------------
# PARTE 2 — CUSTO POR ESA
# ----------------------------

st.markdown("### Distribuição do Estoque por ESA")

df_esa = (
    df_giro
    .groupby("ESA Atual", as_index=False)["estoque_total"]
    .sum()
)

df_esa = df_esa[df_esa["estoque_total"] > 0].copy()
df_esa = ordenar_esa(df_esa)

# gráfico
df_esa["hover_brl"] = df_esa["estoque_total"].apply(fmt_brl_int)

fig_esa = go.Figure()

fig_esa.add_trace(go.Bar(
    x=df_esa["estoque_total"],
    y=df_esa["ESA Atual"],
    orientation="h",
    customdata=df_esa["hover_brl"],
    hovertemplate="%{customdata}<extra></extra>",
    marker=dict(color=cores_por_esa(df_esa["ESA Atual"]))
))

fig_esa.update_layout(
    title="Custo de Estoque por Classificação ESA",
    xaxis_title="Custo (R$)",
    yaxis_title="Classificação ESA",
    template="plotly_dark",
    yaxis=dict(
        categoryorder="array",
        categoryarray=list(reversed(ORDEM_ESA))
    )
)

st.plotly_chart(fig_esa, use_container_width=True, key="grafico_estoque_esa")

# ----------------------------
# PARTE 3.1 — FATURAMENTO POR ESA (GERAL)
# ----------------------------

st.markdown("### Faturamento por ESA")

if df_giro_anterior is None:
    st.warning("Envie o relatório de giro anterior para visualizar o faturamento por ESA.")
else:
    # base vendas
    df_vendas_esa = df_mes_atual[["sku", "vendas_aj"]].copy()

    # base giro anterior (ESA)
    col_sku_giro_ant = "sku" if "sku" in df_giro_anterior.columns else None
    col_esa_giro_ant = "ESA Atual" if "ESA Atual" in df_giro_anterior.columns else None
    
    if col_sku_giro_ant is None or col_esa_giro_ant is None:
        st.error(
            f"Não foi possível montar o faturamento por ESA. "
            f"Colunas encontradas no giro anterior: {list(df_giro_anterior.columns)}"
        )
        st.stop()
    
    df_esa_lookup = df_giro_anterior[[col_sku_giro_ant, col_esa_giro_ant]].copy()
    df_esa_lookup = df_esa_lookup.rename(columns={
        col_sku_giro_ant: "sku",
        col_esa_giro_ant: "ESA Atual"
    })

    # merge
    df_merge = df_vendas_esa.merge(
        df_esa_lookup,
        on="sku",
        how="left"
    )

    # tratar não encontrados
    df_merge["ESA Atual"] = df_merge["ESA Atual"].fillna("Sem classificação")

    # agrupamento
    df_fat_esa = (
        df_merge
        .groupby("ESA Atual", as_index=False)["vendas_aj"]
        .sum()
        .rename(columns={"vendas_aj": "faturamento_bruto"})
    )

    df_fat_esa = df_fat_esa[df_fat_esa["faturamento_bruto"] > 0].copy()
    df_fat_esa = ordenar_esa(df_fat_esa)

    # hover formatado
    df_fat_esa["hover_brl"] = df_fat_esa["faturamento_bruto"].apply(fmt_brl_int)

    # gráfico
    fig_fat_esa = go.Figure()

    fig_fat_esa.add_trace(go.Bar(
        x=df_fat_esa["faturamento_bruto"],
        y=df_fat_esa["ESA Atual"],
        orientation="h",
        customdata=df_fat_esa["hover_brl"],
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(color=cores_por_esa(df_fat_esa["ESA Atual"]))
    ))

    fig_fat_esa.update_layout(
        title="Faturamento por Classificação ESA",
        xaxis_title="Faturamento (R$)",
        yaxis_title="Classificação ESA",
        template="plotly_dark",
        height=400,
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(ORDEM_ESA + ["Sem classificação"]))
        )
    )

    st.plotly_chart(fig_fat_esa, use_container_width=True, key="grafico_faturamento_esa")

# ----------------------------
# PARTE 3.2 — FATURAMENTO POR ESA POR FABRICANTE
# ----------------------------

st.markdown("### Faturamento por ESA por Fabricante")

if df_giro_anterior is None:
    st.warning("Envie o relatório de giro anterior para visualizar o faturamento por ESA por fabricante.")
else:
    fabricantes_fat_esa = (
        df_mes_atual.groupby("fabricante", as_index=False)["vendas_aj"]
        .sum()
        .sort_values("vendas_aj", ascending=False)
    )

    for _, row in fabricantes_fat_esa.iterrows():
        fabricante = row["fabricante"]
        faturamento_total_fab = row["vendas_aj"]

        expander_label = (
            f"{fabricante} | "
            f"Faturamento Bruto: {fmt_brl_int_label(faturamento_total_fab)}"
        )

        with st.expander(expander_label, expanded=False):
            # vendas da fabricante no mês
            df_vendas_fab = df_mes_atual.loc[
                df_mes_atual["fabricante"] == fabricante,
                ["sku", "vendas_aj"]
            ].copy()

            # lookup ESA do giro anterior
            df_esa_lookup_fab = df_giro_anterior[[col_sku_giro_ant, col_esa_giro_ant]].copy()
            df_esa_lookup_fab = df_esa_lookup_fab.rename(columns={
                col_sku_giro_ant: "sku",
                col_esa_giro_ant: "ESA Atual"
            })

            # merge
            df_merge_fab = df_vendas_fab.merge(
                df_esa_lookup_fab,
                on="sku",
                how="left"
            )

            # tratar não encontrados
            df_merge_fab["ESA Atual"] = df_merge_fab["ESA Atual"].fillna("Sem classificação")

            # agrupamento
            df_fat_esa_fab = (
                df_merge_fab
                .groupby("ESA Atual", as_index=False)["vendas_aj"]
                .sum()
                .rename(columns={"vendas_aj": "faturamento_bruto"})
            )

            df_fat_esa_fab = df_fat_esa_fab[df_fat_esa_fab["faturamento_bruto"] > 0].copy()
            df_fat_esa_fab = ordenar_esa(df_fat_esa_fab)

            # hover formatado
            df_fat_esa_fab["hover_brl"] = df_fat_esa_fab["faturamento_bruto"].apply(fmt_brl_int)

            # gráfico
            fig_fat_esa_fab = go.Figure()

            fig_fat_esa_fab.add_trace(go.Bar(
                x=df_fat_esa_fab["faturamento_bruto"],
                y=df_fat_esa_fab["ESA Atual"],
                orientation="h",
                customdata=df_fat_esa_fab["hover_brl"],
                hovertemplate="%{customdata}<extra></extra>",
                marker=dict(color=cores_por_esa(df_fat_esa_fab["ESA Atual"]))
            ))

            fig_fat_esa_fab.update_layout(
                title=f"Faturamento por ESA — {fabricante}",
                xaxis_title="Faturamento (R$)",
                yaxis_title="Classificação ESA",
                template="plotly_dark",
                height=400,
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=list(reversed(ORDEM_ESA + ["Sem classificação"]))
                )
            )

            st.plotly_chart(
                fig_fat_esa_fab,
                use_container_width=True,
                key=f"grafico_faturamento_esa_fabricante_{fabricante}"
            )

            st.markdown("#### Estoque por ESA")

            df_estoque_esa_fab = (
                df_giro[df_giro["fabricante"] == fabricante]
                .groupby("ESA Atual", as_index=False)["estoque_total"]
                .sum()
            )

            df_estoque_esa_fab = df_estoque_esa_fab[df_estoque_esa_fab["estoque_total"] > 0].copy()
            df_estoque_esa_fab = ordenar_esa(df_estoque_esa_fab)

            df_estoque_esa_fab["hover_brl"] = df_estoque_esa_fab["estoque_total"].apply(fmt_brl_int)
            
            fig_estoque_esa_fab = go.Figure()

            fig_estoque_esa_fab.add_trace(go.Bar(
                x=df_estoque_esa_fab["estoque_total"],
                y=df_estoque_esa_fab["ESA Atual"],
                orientation="h",
                customdata=df_estoque_esa_fab["hover_brl"],
                hovertemplate="%{customdata}<extra></extra>",
                marker=dict(color=cores_por_esa(df_estoque_esa_fab["ESA Atual"]))
            ))

            fig_estoque_esa_fab.update_layout(
                title=f"Estoque por ESA — {fabricante}",
                xaxis_title="Estoque (R$)",
                yaxis_title="Classificação ESA",
                template="plotly_dark",
                height=400,
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=list(reversed(ORDEM_ESA))
                )
            )

            st.plotly_chart(
                fig_estoque_esa_fab,
                use_container_width=True,
                key=f"grafico_estoque_esa_fabricante_{fabricante}"
            )

# ----------------------------
# PARTE 4 — COMPARAÇÃO DE ESTOQUE POR ESA
# ----------------------------

st.markdown("### Resumo do Estoque por Grupo de Risco")

df_giro["grupo_esa"] = df_giro["ESA Atual"].map(GRUPO_ESA).fillna("Sem classificação")

estoque_saudavel = df_giro.loc[df_giro["grupo_esa"] == "Saudável", "estoque_total"].sum()
estoque_atencao = df_giro.loc[df_giro["grupo_esa"] == "Atenção", "estoque_total"].sum()
estoque_critico = df_giro.loc[df_giro["grupo_esa"] == "Crítico", "estoque_total"].sum()

total = custo_total_atual if custo_total_atual != 0 else 1

pct_saudavel = estoque_saudavel / total
pct_atencao = estoque_atencao / total
pct_critico = estoque_critico / total

g1, g2, g3 = st.columns(3)

g1.metric(
    "Estoque Saudável",
    fmt_brl_int(estoque_saudavel),
    f"{fmt_pct(pct_saudavel)} do total"
)
g2.metric("Estoque em Atenção", fmt_brl_int(estoque_atencao), fmt_pct(pct_atencao))
g3.metric("Estoque Crítico", fmt_brl_int(estoque_critico), fmt_pct(pct_critico))

st.markdown("### Comparação de Estoque por ESA (Atual vs Anterior)")

if df_giro_anterior is None:
    st.warning("Envie o relatório de giro anterior para visualizar a comparação de estoque por ESA.")
else:
    opcoes_fabricante = ["Cogra"] + sorted(df_giro["fabricante"].dropna().unique().tolist())

    filtro_comp_esa = st.selectbox(
        "Visualizar comparação de estoque por ESA para:",
        options=opcoes_fabricante,
        key="filtro_comp_esa"
    )

    if filtro_comp_esa == "Cogra":
        df_giro_base = df_giro.copy()
        df_giro_anterior_base = df_giro_anterior.copy()
    else:
        df_giro_base = df_giro[df_giro["fabricante"] == filtro_comp_esa].copy()
        df_giro_anterior_base = df_giro_anterior[df_giro_anterior["fabricante"] == filtro_comp_esa].copy()

    df_esa_atual_comp = (
        df_giro_base
        .groupby("ESA Atual", as_index=False)["estoque_total"]
        .sum()
        .rename(columns={"estoque_total": "Estoque Atual"})
    )

    df_esa_anterior_comp = (
        df_giro_anterior_base
        .groupby("ESA Atual", as_index=False)["estoque_total"]
        .sum()
        .rename(columns={"estoque_total": "Estoque Anterior"})
    )

    df_comp_esa = df_esa_atual_comp.merge(
        df_esa_anterior_comp,
        on="ESA Atual",
        how="outer"
    ).fillna(0)

    df_comp_esa = df_comp_esa[
        (df_comp_esa["Estoque Atual"] > 0) | (df_comp_esa["Estoque Anterior"] > 0)
    ].copy()

    def calc_var_esa(row):
        anterior = row["Estoque Anterior"]
        atual = row["Estoque Atual"]
        if anterior == 0:
            if atual == 0:
                return 0
            return np.nan
        return (atual - anterior) / anterior

    df_comp_esa["Variação"] = df_comp_esa.apply(calc_var_esa, axis=1)

    ordem_map = {esa: i for i, esa in enumerate(ORDEM_ESA)}
    df_comp_esa["ordem"] = df_comp_esa["ESA Atual"].map(ordem_map).fillna(999)
    df_comp_esa = df_comp_esa.sort_values("ordem").drop(columns="ordem")

    df_comp_esa["Estoque Atual"] = df_comp_esa["Estoque Atual"].apply(fmt_brl_int)
    df_comp_esa["Estoque Anterior"] = df_comp_esa["Estoque Anterior"].apply(fmt_brl_int)
    df_comp_esa["Variação"] = df_comp_esa["Variação"].apply(fmt_var)

    st.dataframe(
        df_comp_esa.rename(columns={"ESA Atual": "ESA"}),
        use_container_width=True,
        hide_index=True
    )
