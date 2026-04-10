"""
app.py
======
Interface Streamlit do sistema analítico orientado à decisão.

Fluxo de estado correto:
  1. Usuário faz upload dos arquivos
  2. Usuário clica em "Processar Dados"
  3. app.py lê os bytes UMA VEZ e salva em st.session_state["bytes_*"]
  4. Chama processar_dados() com os bytes salvos (cacheado por @st.cache_data)
  5. Resultado salvo em st.session_state["resultado"] e st.session_state["diagnosticos"]
  6. Rerenders subsequentes (filtros etc.) apenas leem session_state — não releem arquivos
  7. Upload de novos arquivos + clique em "Processar" recomeça o ciclo

Isso garante que .read() nos objetos UploadedFile só seja chamado uma vez por ciclo.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_giro, load_vendas, validate_giro, validate_vendas
from decision_engine import DiagnosticoFabricante, diagnosticar_todos, diagnosticos_to_dataframe
from processor import run_pipeline


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sistema Analítico de Decisão",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS customizado
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f12;
    color: #e8e8e2;
}
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
.stApp { background: #0d0f12; }

section[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2128;
}

.main-header { padding: 2rem 0 1.5rem; border-bottom: 1px solid #1e2128; margin-bottom: 2rem; }
.main-header h1 { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #e8e8e2; margin: 0; letter-spacing: -0.03em; }
.main-header p { color: #6b7280; font-size: 0.9rem; margin: 0.4rem 0 0; font-family: 'DM Mono', monospace; }

.badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }
.badge-alta { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-media { background: rgba(234,179,8,0.12); color: #facc15; border: 1px solid rgba(234,179,8,0.25); }
.badge-baixa { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }

.fab-card { background: #111318; border: 1px solid #1e2128; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; position: relative; overflow: hidden; }
.fab-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; }
.fab-card.alta::before { background: #ef4444; }
.fab-card.media::before { background: #eab308; }
.fab-card.baixa::before { background: #22c55e; }

.fab-name { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #e8e8e2; margin-bottom: 0.5rem; }
.fab-meta { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #6b7280; }
.fab-justificativa { background: #0d0f12; border: 1px solid #1e2128; border-radius: 6px; padding: 0.9rem 1rem; margin-top: 1rem; font-size: 0.85rem; color: #9ca3af; line-height: 1.6; font-style: italic; }

.vetor-tag { display: inline-block; background: rgba(99,102,241,0.12); color: #818cf8; border: 1px solid rgba(99,102,241,0.2); border-radius: 3px; padding: 0.15rem 0.5rem; font-family: 'DM Mono', monospace; font-size: 0.65rem; margin-right: 0.3rem; text-transform: uppercase; letter-spacing: 0.06em; }

.metric-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: 0.8rem; }
.metric-item .label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #4b5563; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-item .value { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 600; color: #d1d5db; }

.delta-up { color: #22c55e; }
.delta-down { color: #ef4444; }

hr { border-color: #1e2128 !important; margin: 1.5rem 0 !important; }

.stFileUploader label { font-family: 'DM Mono', monospace; font-size: 0.8rem; }
.stSelectbox label, .stMultiselect label { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #6b7280; }
div[data-testid="stMetric"] { background: #111318; border: 1px solid #1e2128; border-radius: 8px; padding: 1rem; }
div[data-testid="stMetricLabel"] p { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #6b7280; }
div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; font-weight: 700; }
.stAlert { border-radius: 6px; font-family: 'DM Mono', monospace; font-size: 0.8rem; }
.stButton button { font-family: 'DM Mono', monospace; font-size: 0.8rem; }

.ready-banner { background: #111318; border: 1px solid #1e2128; border-left: 3px solid #6366f1; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers de formatação
# ---------------------------------------------------------------------------

def fmt_brl(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"R$ {value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_var(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if value == 0:
        return '<span>• 0,0%</span>'
    sinal = "▲" if value > 0 else "▼"
    cls = "delta-up" if value > 0 else "delta-down"
    return f'<span class="{cls}">{sinal} {abs(value) * 100:.1f}%</span>'


def badge_prioridade(prioridade: str) -> str:
    cls = {
        "Alta": "badge-alta",
        "Média": "badge-media",
        "Baixa": "badge-baixa",
    }.get(prioridade, "badge-baixa")
    return f'<span class="badge {cls}">{prioridade}</span>'


def card_class(prioridade: str) -> str:
    return {"Alta": "alta", "Média": "media", "Baixa": "baixa"}.get(prioridade, "baixa")


def clear_processing_state() -> None:
    """
    Remove estado derivado de processamentos anteriores.
    Deve ser chamado antes de um novo processamento.
    """
    for key in [
        "resultado",
        "diagnosticos",
        "warnings",
        "tem_anterior",
    ]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar — Upload e configuração
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem;">
            <span style="font-family:'DM Mono',monospace; font-size:0.65rem;
                         color:#4b5563; text-transform:uppercase; letter-spacing:0.1em;">
                Sistema Analítico v1.1
            </span>
            <div style="font-family:'Syne',sans-serif; font-weight:800;
                        font-size:1.1rem; margin-top:0.2rem;">
                Motor de Decisão
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("**📂 Mês Atual**")
        vendas_atual = st.file_uploader("Relatório de Vendas", type=["xlsx", "xls"], key="fu_vendas_atual")
        giro_atual = st.file_uploader("Relatório de Giro / Estoque", type=["xlsx", "xls"], key="fu_giro_atual")

        st.divider()

        st.markdown("**📂 Mês Anterior** *(opcional — habilita comparativos)*")
        vendas_anterior = st.file_uploader("Relatório de Vendas", type=["xlsx", "xls"], key="fu_vendas_ant")
        giro_anterior = st.file_uploader("Relatório de Giro / Estoque", type=["xlsx", "xls"], key="fu_giro_ant")

        st.divider()

        st.markdown("**🔧 Filtros**")
        filtro_prioridade = st.multiselect(
            "Prioridade",
            options=["Alta", "Média", "Baixa"],
            default=["Alta", "Média", "Baixa"],
        )
        encalhe_minimo = st.slider(
            "% Encalhe mínimo",
            min_value=0, max_value=100, value=0, step=5, format="%d%%",
        )

        st.divider()

        processar = st.button("🚀 Processar Dados", use_container_width=True, type="primary")

        if st.session_state.get("resultado") is not None:
            if st.button("🗑 Limpar resultado", use_container_width=True):
                for key in [
                    "resultado",
                    "diagnosticos",
                    "warnings",
                    "tem_anterior",
                    "bytes_vendas",
                    "bytes_giro",
                    "bytes_vendas_ant",
                    "bytes_giro_ant",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

        return {
            "vendas_atual": vendas_atual,
            "giro_atual": giro_atual,
            "vendas_anterior": vendas_anterior,
            "giro_anterior": giro_anterior,
            "filtro_prioridade": filtro_prioridade,
            "encalhe_minimo": encalhe_minimo / 100,
            "processar": processar,
        }


# ---------------------------------------------------------------------------
# Pipeline com cache — recebe bytes, não objetos de arquivo
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def processar_dados(
    vendas_bytes: bytes,
    giro_bytes: bytes,
    vendas_ant_bytes,
    giro_ant_bytes,
) -> tuple:
    """
    Executa o pipeline completo.
    Recebe bytes (serializáveis) para funcionar corretamente com @st.cache_data.
    """
    warns = []

    vendas_raw = load_vendas(io.BytesIO(vendas_bytes))
    giro_raw = load_giro(io.BytesIO(giro_bytes))

    warns.extend(validate_vendas(vendas_raw))
    warns.extend(validate_giro(giro_raw))

    vendas_ant = None
    giro_ant = None

    if vendas_ant_bytes and giro_ant_bytes:
        vendas_ant = load_vendas(io.BytesIO(vendas_ant_bytes))
        giro_ant = load_giro(io.BytesIO(giro_ant_bytes))
        warns.extend(["[Mês Anterior] " + w for w in validate_vendas(vendas_ant)])
        warns.extend(["[Mês Anterior] " + w for w in validate_giro(giro_ant)])

    resultado = run_pipeline(vendas_raw, giro_raw, vendas_ant, giro_ant)
    return resultado, warns


# ---------------------------------------------------------------------------
# Render — Tela de boas-vindas (sem arquivos carregados)
# ---------------------------------------------------------------------------

def render_welcome():
    steps = [
        ("01", "Carregue o <b>Relatório de Vendas</b> do mês atual na sidebar"),
        ("02", "Carregue o <b>Relatório de Giro / Estoque</b> do mês atual na sidebar"),
        ("03", "Opcionalmente, carregue os mesmos relatórios do <b>mês anterior</b> para ativar comparativos"),
        ("04", "Clique em <b>Processar Dados</b>"),
        ("05", "O sistema consolida, interpreta e prioriza os fabricantes automaticamente"),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div style="display:flex; gap:1rem; align-items:flex-start; background:#111318;
                    border:1px solid #1e2128; border-radius:8px; padding:0.9rem 1rem; margin-bottom:0.5rem;">
            <div style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#4b5563; min-width:2rem;">{num}</div>
            <div style="font-size:0.9rem; color:#9ca3af;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem; padding:1rem 1.2rem; background:#111318;
                border:1px solid #1e2128; border-radius:8px; max-width:760px;">
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#4b5563;
                    text-transform:uppercase; margin-bottom:0.6rem;">Colunas esperadas nos arquivos</div>
        <div style="font-size:0.82rem; color:#6b7280; line-height:1.9;">
            <b style="color:#9ca3af;">Vendas:</b>
            SKU · Fabricante · Tipo (N/D) · Venda Líquida · Lucro ·
            <span style="color:#818cf8;">Margem c/ Reb.</span> ·
            <span style="color:#818cf8;">Qtd.</span><br>
            <b style="color:#9ca3af;">Giro:</b>
            SKU · Fabricante · Estoque Atual ·
            <span style="color:#818cf8;">Média Vendas</span> ·
            <span style="color:#818cf8;">Indice Giro</span> ·
            <span style="color:#818cf8;">Dias Estoque + Pedidos Mês</span> ·
            <span style="color:#818cf8;">Dias Última Venda</span> ·
            <span style="color:#818cf8;">ESA Atual</span>
        </div>
        <div style="font-size:0.75rem; color:#4b5563; margin-top:0.5rem;">
            Os nomes em roxo são os confirmados no layout real.
            Variações próximas também são aceitas automaticamente.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render — Banner "pronto para processar"
# ---------------------------------------------------------------------------

def render_ready_banner(tem_anterior: bool):
    msg_ant = (
        "✅ Mês anterior carregado — comparativos serão gerados."
        if tem_anterior
        else "ℹ️ Mês anterior não carregado — modo período único."
    )
    st.markdown(f"""
    <div class="ready-banner">
        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; margin-bottom:0.4rem;">
            Arquivos carregados
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:0.8rem; color:#6b7280;">
            Clique em <b style="color:#818cf8;">Processar Dados</b> na sidebar para iniciar o diagnóstico.<br>
            {msg_ant}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render — KPIs gerais
# ---------------------------------------------------------------------------

def render_kpis_gerais(kpis_df: pd.DataFrame):
    total_receita = kpis_df["receita_liquida"].sum()
    total_lucro = kpis_df["lucro"].sum()
    total_estoque = kpis_df["estoque_atual"].sum()
    margem_media = total_lucro / total_receita if total_receita > 0 else 0
    pct_enc_total = kpis_df["estoque_encalhe"].sum() / total_estoque if total_estoque > 0 else 0
    gmroii_medio = total_lucro / total_estoque if total_estoque > 0 else 0
    n_fabricantes = len(kpis_df)

    st.markdown("### Visão Geral do Período")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Receita Líquida", fmt_brl(total_receita))
    c2.metric("Lucro", fmt_brl(total_lucro))
    c3.metric("Margem Média", f"{margem_media:.1%}")
    c4.metric("Estoque Total", fmt_brl(total_estoque))
    c5.metric("% Encalhe", f"{pct_enc_total:.1%}")
    c6.metric("GMROII", f"{gmroii_medio:.2f}")
    c7.metric("Fabricantes", str(n_fabricantes))
    st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render — Distribuição de prioridades
# ---------------------------------------------------------------------------

def render_distribuicao(diagnosticos: list[DiagnosticoFabricante]):
    alta = sum(1 for d in diagnosticos if d.prioridade == "Alta")
    media = sum(1 for d in diagnosticos if d.prioridade == "Média")
    baixa = sum(1 for d in diagnosticos if d.prioridade == "Baixa")
    total = len(diagnosticos) or 1

    st.markdown(f"""
    <div style="background:#111318; border:1px solid #1e2128; border-radius:10px;
                padding:1.2rem 1.5rem; margin-bottom:1.5rem;">
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#4b5563;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;">
            Distribuição de Prioridade
        </div>
        <div style="display:flex; gap:2rem; align-items:center;">
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem;
                            font-weight:800; color:#f87171;">{alta}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#6b7280;">Alta</div>
            </div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem;
                            font-weight:800; color:#facc15;">{media}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#6b7280;">Média</div>
            </div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem;
                            font-weight:800; color:#4ade80;">{baixa}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#6b7280;">Baixa</div>
            </div>
            <div style="flex:1; margin-left:1rem;">
                <div style="height:8px; background:#1e2128; border-radius:4px;
                            overflow:hidden; display:flex;">
                    <div style="width:{alta/total*100:.1f}%; background:#ef4444;"></div>
                    <div style="width:{media/total*100:.1f}%; background:#eab308;"></div>
                    <div style="width:{baixa/total*100:.1f}%; background:#22c55e;"></div>
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                            color:#4b5563; margin-top:0.4rem;">{total} fabricantes analisados</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render — Card individual de fabricante
# ---------------------------------------------------------------------------

def render_card_fabricante(d: DiagnosticoFabricante, tem_anterior: bool):
    cls = card_class(d.prioridade)
    vetores_html = "".join(
        f'<span class="vetor-tag">{v}</span>' for v in d.vetores
    ) if d.vetores else ""

    var_receita_html = fmt_var(d.var_receita) if tem_anterior else "—"
    var_lucro_html = fmt_var(d.var_lucro) if tem_anterior else "—"
    var_encalhe_html = fmt_var(d.var_encalhe) if tem_anterior else "—"

    cor_encalhe = (
        "#f87171" if d.pct_encalhe > 0.25
        else "#facc15" if d.pct_encalhe > 0.10
        else "#d1d5db"
    )

    st.markdown(f"""
    <div class="fab-card {cls}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="fab-name">{d.fabricante}</div>
                <div class="fab-meta">{d.n_skus} SKUs · {d.n_skus_encalhe} em Encalhe</div>
            </div>
            <div style="text-align:right;">
                {badge_prioridade(d.prioridade)}
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                            color:#4b5563; margin-top:0.4rem;">score {d.score_total}/6</div>
            </div>
        </div>

        {"<div style='margin-top:0.6rem;'>" + vetores_html + "</div>" if vetores_html else ""}

        <div class="metric-row">
            <div class="metric-item">
                <div class="label">Receita Líquida</div>
                <div class="value">{fmt_brl(d.receita_liquida)}</div>
                <div style="font-size:0.7rem; margin-top:0.1rem;">{var_receita_html}</div>
            </div>
            <div class="metric-item">
                <div class="label">Lucro</div>
                <div class="value">{fmt_brl(d.lucro)}</div>
                <div style="font-size:0.7rem; margin-top:0.1rem;">{var_lucro_html}</div>
            </div>
            <div class="metric-item">
                <div class="label">Margem</div>
                <div class="value">{d.margem:.1f}%</div>
            </div>
            <div class="metric-item">
                <div class="label">Estoque Atual</div>
                <div class="value">{fmt_brl(d.estoque_atual)}</div>
            </div>
            <div class="metric-item">
                <div class="label">% Encalhe</div>
                <div class="value" style="color:{cor_encalhe};">{d.pct_encalhe:.1%}</div>
                <div style="font-size:0.7rem; margin-top:0.1rem;">{var_encalhe_html}</div>
            </div>
            <div class="metric-item">
                <div class="label">GMROII</div>
                <div class="value">{d.gmroii:.2f}</div>
            </div>
        </div>

        <div class="fab-justificativa">
            💬 {d.justificativa}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render — Tabela e exportação
# ---------------------------------------------------------------------------

def render_tabela_export(df_export: pd.DataFrame):
    with st.expander("📋 Tabela Completa / Exportar"):
        st.dataframe(df_export, use_container_width=True, height=400)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="Diagnósticos")
        buf.seek(0)

        st.download_button(
            label="⬇️ Baixar Excel",
            data=buf,
            file_name="diagnostico_fabricantes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    config = render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>🎯 Sistema Analítico de Decisão</h1>
        <p>consolidação · diagnóstico automático · priorização por score</p>
    </div>
    """, unsafe_allow_html=True)

    vendas_atual = config["vendas_atual"]
    giro_atual = config["giro_atual"]
    vendas_anterior = config["vendas_anterior"]
    giro_anterior = config["giro_anterior"]
    processar = config["processar"]

    # Estado 1: sem arquivos
    if not vendas_atual or not giro_atual:
        render_welcome()
        return

    # Estado 2: processar explicitamente ao clicar
    if processar:
        clear_processing_state()

        with st.spinner("🔄 Lendo arquivos..."):
            try:
                st.session_state["bytes_vendas"] = vendas_atual.read()
                st.session_state["bytes_giro"] = giro_atual.read()
                st.session_state["bytes_vendas_ant"] = vendas_anterior.read() if vendas_anterior else None
                st.session_state["bytes_giro_ant"] = giro_anterior.read() if giro_anterior else None
            except Exception as e:
                st.error(f"❌ Erro ao ler arquivos: {e}")
                return

        with st.spinner("⚙️ Processando pipeline..."):
            try:
                resultado, warns = processar_dados(
                    st.session_state["bytes_vendas"],
                    st.session_state["bytes_giro"],
                    st.session_state["bytes_vendas_ant"],
                    st.session_state["bytes_giro_ant"],
                )

                kpis_com_variacao = resultado.get("kpis_com_variacao")
                kpis_df = kpis_com_variacao if kpis_com_variacao is not None else resultado["kpis_fabricante"]

                diagnosticos = diagnosticar_todos(kpis_df)
                tem_anterior = resultado["kpis_anterior"] is not None

                st.session_state["resultado"] = resultado
                st.session_state["diagnosticos"] = diagnosticos
                st.session_state["warnings"] = warns
                st.session_state["tem_anterior"] = tem_anterior

            except KeyError as e:
                clear_processing_state()
                st.error(f"❌ Coluna não encontrada: {e}")
                return
            except Exception as e:
                clear_processing_state()
                st.error(f"❌ Erro durante o processamento: {e}")
                st.exception(e)
                return

    # Estado 3: arquivos carregados, aguardando clique
    if "resultado" not in st.session_state:
        render_ready_banner(bool(vendas_anterior and giro_anterior))
        return

    # Estado 4: resultado disponível
    resultado = st.session_state["resultado"]
    diagnosticos = st.session_state["diagnosticos"]
    warns = st.session_state.get("warnings", [])
    tem_anterior = st.session_state.get("tem_anterior", False)

    if warns:
        with st.expander(f"⚠️ {len(warns)} aviso(s) de qualidade dos dados", expanded=False):
            for w in warns:
                st.markdown(f"- {w}")

    render_kpis_gerais(resultado["kpis_fabricante"])
    render_distribuicao(diagnosticos)

    filtrados = [
        d for d in diagnosticos
        if d.prioridade in config["filtro_prioridade"]
        and d.pct_encalhe >= config["encalhe_minimo"]
    ]

    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#4b5563; margin-bottom:1rem;">
        Exibindo {len(filtrados)} de {len(diagnosticos)} fabricantes
        {" · ⚡ Comparativo com mês anterior ativado" if tem_anterior
         else " · Modo período único (sem comparativo)"}
    </div>
    """, unsafe_allow_html=True)

    for d in filtrados:
        render_card_fabricante(d, tem_anterior)

    st.divider()

    df_export = diagnosticos_to_dataframe(filtrados)
    render_tabela_export(df_export)


if __name__ == "__main__":
    main()
