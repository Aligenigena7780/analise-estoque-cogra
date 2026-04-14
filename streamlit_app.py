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
from processor import (
    listar_periodos_disponiveis,
    obter_periodo_padrao,
    run_pipeline,
)

st.set_page_config(
    page_title="Sistema Analítico de Decisão",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, .stApp { background:#0d0f12; color:#e8e8e2; }
.ready-banner { background:#111318; border:1px solid #1e2128; border-left:3px solid #6366f1; border-radius:8px; padding:1.2rem 1.5rem; margin:1rem 0; }
.fab-card { background:#111318; border:1px solid #1e2128; border-radius:10px; padding:1rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


def fmt_brl(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"R$ {value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_var(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    sinal = "▲" if value > 0 else "▼"
    return f"{sinal} {abs(value) * 100:.1f}%"


def clear_processing_state() -> None:
    for key in [
        "resultado",
        "diagnosticos",
        "warnings",
        "tem_anterior",
        "bytes_vendas",
        "bytes_giro",
        "bytes_giro_ant",
        "vendas_loaded_preview",
        "warnings_vendas",
    ]:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def processar_dados(
    vendas_bytes: bytes,
    giro_bytes: bytes,
    giro_ant_bytes,
    ano: int,
    mes_inicio: int,
    mes_fim: int,
):
    warns = []

    vendas_raw = load_vendas(io.BytesIO(vendas_bytes))
    giro_raw = load_giro(io.BytesIO(giro_bytes))

    warns.extend(validate_vendas(vendas_raw))
    warns.extend(validate_giro(giro_raw))

    giro_ant = None
    if giro_ant_bytes:
        giro_ant = load_giro(io.BytesIO(giro_ant_bytes))
        warns.extend(["[Giro Comparação] " + w for w in validate_giro(giro_ant)])

    resultado = run_pipeline(
        vendas_raw_historico=vendas_raw,
        giro_raw_atual=giro_raw,
        ano=ano,
        mes_inicio=mes_inicio,
        mes_fim=mes_fim,
        giro_raw_anterior=giro_ant,
    )
    return resultado, warns


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## Motor de Decisão")

        st.markdown("### Vendas")
        vendas_historico = st.file_uploader(
            "Relatório de Vendas (histórico)",
            type=["xlsx", "xls"],
            key="fu_vendas_historico",
        )

        st.markdown("### Giro")
        giro_atual = st.file_uploader(
            "Giro Atual",
            type=["xlsx", "xls"],
            key="fu_giro_atual",
        )
        giro_comparacao = st.file_uploader(
            "Giro de Comparação (opcional)",
            type=["xlsx", "xls"],
            key="fu_giro_comp",
        )

        st.markdown("### Filtros")
        filtro_prioridade = st.multiselect(
            "Prioridade",
            options=["Alta", "Média", "Baixa"],
            default=["Alta", "Média", "Baixa"],
        )
        encalhe_minimo = st.slider(
            "% Encalhe mínimo",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%",
        )

        processar = st.button("🚀 Processar Dados", use_container_width=True, type="primary")

        if st.session_state.get("resultado") is not None:
            if st.button("🗑 Limpar resultado", use_container_width=True):
                clear_processing_state()
                st.rerun()

        return {
            "vendas_historico": vendas_historico,
            "giro_atual": giro_atual,
            "giro_comparacao": giro_comparacao,
            "filtro_prioridade": filtro_prioridade,
            "encalhe_minimo": encalhe_minimo / 100,
            "processar": processar,
        }


def render_period_selector(vendas_file) -> tuple[int, int, int] | None:
    if not vendas_file:
        return None

    if "bytes_vendas" not in st.session_state:
        st.session_state["bytes_vendas"] = vendas_file.read()
        vendas_file.seek(0)

    if "vendas_loaded_preview" not in st.session_state:
        vendas_preview = load_vendas(io.BytesIO(st.session_state["bytes_vendas"]))
        st.session_state["vendas_loaded_preview"] = vendas_preview
        st.session_state["warnings_vendas"] = validate_vendas(vendas_preview)

    vendas_preview = st.session_state["vendas_loaded_preview"]
    periodos = listar_periodos_disponiveis(vendas_preview)

    if periodos.empty:
        st.error("Nenhum período válido encontrado no relatório de vendas.")
        return None

    ano_padrao, mes_inicio_padrao, mes_fim_padrao = obter_periodo_padrao(vendas_preview)
    anos_disponiveis = sorted(periodos["ano"].unique().tolist())

    st.markdown("### Período de Análise")
    with st.expander("Selecionar período", expanded=True):
        ano = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=anos_disponiveis.index(ano_padrao),
        )

        periodos_ano = periodos[periodos["ano"] == ano].copy()
        meses_disponiveis = periodos_ano["mes"].tolist()
        meses_nomes = [MESES_PT[m] for m in meses_disponiveis]

        idx_inicio = meses_disponiveis.index(mes_inicio_padrao) if ano == ano_padrao else 0
        idx_fim = meses_disponiveis.index(mes_fim_padrao) if ano == ano_padrao else len(meses_disponiveis) - 1

        col1, col2 = st.columns(2)
        with col1:
            mes_inicio_nome = st.selectbox("Mês início", options=meses_nomes, index=idx_inicio)
        with col2:
            mes_fim_nome = st.selectbox("Mês fim", options=meses_nomes, index=idx_fim)

        mes_inicio = meses_disponiveis[meses_nomes.index(mes_inicio_nome)]
        mes_fim = meses_disponiveis[meses_nomes.index(mes_fim_nome)]

        if mes_inicio > mes_fim:
            st.warning("O mês início não pode ser maior que o mês fim.")
            return None

        return ano, mes_inicio, mes_fim


def render_ready_banner(tem_giro_comparacao: bool):
    msg = "✅ Giro de comparação carregado." if tem_giro_comparacao else "ℹ️ Sem giro de comparação."
    st.markdown(f"""
    <div class="ready-banner">
        Arquivos carregados. Clique em <b>Processar Dados</b> na barra lateral.<br>{msg}
    </div>
    """, unsafe_allow_html=True)


def render_kpis_gerais(kpis_df: pd.DataFrame):
    total_receita = kpis_df["receita_liquida"].sum()
    total_lucro = kpis_df["lucro"].sum()
    total_estoque = kpis_df["estoque_atual"].sum()
    margem_media = total_lucro / total_receita if total_receita > 0 else 0
    pct_enc_total = kpis_df["estoque_encalhe"].sum() / total_estoque if total_estoque > 0 else 0
    gmroii_medio = total_lucro / total_estoque if total_estoque > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Receita Líquida", fmt_brl(total_receita))
    c2.metric("Lucro", fmt_brl(total_lucro))
    c3.metric("Margem Média", f"{margem_media:.1%}")
    c4.metric("Estoque Total", fmt_brl(total_estoque))
    c5.metric("% Encalhe", f"{pct_enc_total:.1%}")
    c6.metric("GMROII", f"{gmroii_medio:.2f}")


def render_card_fabricante(d: DiagnosticoFabricante, tem_anterior: bool):
    st.markdown(f"""
    <div class="fab-card">
        <b>{d.fabricante}</b> — {d.prioridade}<br>
        Receita: {fmt_brl(d.receita_liquida)} |
        Lucro: {fmt_brl(d.lucro)} |
        Margem: {d.margem:.1f}% |
        Estoque: {fmt_brl(d.estoque_atual)} |
        % Encalhe: {d.pct_encalhe:.1%} |
        GMROII: {d.gmroii:.2f}
        <br><br>
        {'Var. Receita: ' + fmt_var(d.var_receita) + ' | ' if tem_anterior else ''}
        {'Var. Lucro: ' + fmt_var(d.var_lucro) + ' | ' if tem_anterior else ''}
        {'Var. Encalhe: ' + fmt_var(d.var_encalhe) if tem_anterior else ''}
        <br><br>
        {d.justificativa}
    </div>
    """, unsafe_allow_html=True)


def main():
    config = render_sidebar()

    st.title("🎯 Sistema Analítico de Decisão")

    vendas_historico = config["vendas_historico"]
    giro_atual = config["giro_atual"]
    giro_comparacao = config["giro_comparacao"]
    processar = config["processar"]

    if not vendas_historico or not giro_atual:
        st.info("Carregue o relatório de vendas histórico e o giro atual para começar.")
        return

    periodo = render_period_selector(vendas_historico)
    if periodo is None:
        return

    ano, mes_inicio, mes_fim = periodo

    if "warnings_vendas" in st.session_state and st.session_state["warnings_vendas"]:
        with st.expander("⚠️ Avisos prévios de vendas", expanded=False):
            for w in st.session_state["warnings_vendas"]:
                st.markdown(f"- {w}")

    if processar:
        clear_processing_state()
        st.session_state["bytes_vendas"] = vendas_historico.read()
        st.session_state["bytes_giro"] = giro_atual.read()
        st.session_state["bytes_giro_ant"] = giro_comparacao.read() if giro_comparacao else None

        with st.spinner("Processando..."):
            try:
                resultado, warns = processar_dados(
                    st.session_state["bytes_vendas"],
                    st.session_state["bytes_giro"],
                    st.session_state["bytes_giro_ant"],
                    ano,
                    mes_inicio,
                    mes_fim,
                )

                kpis_com_variacao = resultado.get("kpis_com_variacao")
                kpis_df = kpis_com_variacao if kpis_com_variacao is not None else resultado["kpis_fabricante"]

                diagnosticos = diagnosticar_todos(kpis_df)

                st.session_state["resultado"] = resultado
                st.session_state["diagnosticos"] = diagnosticos
                st.session_state["warnings"] = warns
                st.session_state["tem_anterior"] = resultado["kpis_anterior"] is not None

            except Exception as e:
                clear_processing_state()
                st.error(f"Erro durante o processamento: {e}")
                st.exception(e)
                return

    if "resultado" not in st.session_state:
        render_ready_banner(bool(giro_comparacao))
        return

    resultado = st.session_state["resultado"]
    diagnosticos = st.session_state["diagnosticos"]
    warns = st.session_state.get("warnings", [])
    tem_anterior = st.session_state.get("tem_anterior", False)

    if warns:
        with st.expander(f"⚠️ {len(warns)} aviso(s) de qualidade dos dados)", expanded=False):
            for w in warns:
                st.markdown(f"- {w}")

    st.caption(f"Período selecionado: {ano}-{mes_inicio:02d} até {ano}-{mes_fim:02d}")
    render_kpis_gerais(resultado["kpis_fabricante"])

    filtrados = [
        d for d in diagnosticos
        if d.prioridade in config["filtro_prioridade"]
        and d.pct_encalhe >= config["encalhe_minimo"]
    ]

    for d in filtrados:
        render_card_fabricante(d, tem_anterior)

    df_export = diagnosticos_to_dataframe(filtrados)
    st.dataframe(df_export, use_container_width=True)


if __name__ == "__main__":
    main()
