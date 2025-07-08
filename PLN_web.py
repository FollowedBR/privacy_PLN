import streamlit as st
import spacy
import re
from pathlib import Path
from collections import defaultdict

CUSTOM_CATEGORIES_VISUAL = {
    "SAUDE_ALTO": {"label": "🔴 ALTA SENSIBILIDADE (Saúde)", "color": "#FF4B4B"},
    "FINANCEIRO_ALTO": {"label": "🔴 ALTA SENSIBILIDADE (Financeiro)", "color": "#FF4B4B"},
    "IDENTIFICACAO_ALTO": {"label": "🔴 ALTA SENSIBILIDADE (Identificação)", "color": "#FF4B4B"},
    "LOCATION_HIGH": {"label": "🔴 ALTA SENSIBILIDADE (Localização)", "color": "#FF4B4B"},
    "CONTATO_MEDIO": {"label": "🟡 MÉDIA SENSIBILIDADE (Contato)", "color": "#FFC700"},
    "DEMOGRAFICO_MEDIO": {"label": "🟡 MÉDIA SENSIBILIDADE (Demográfico)", "color": "#FFC700"},
    "NAVEGACAO_MEDIO": {"label": "🟡 MÉDIA SENSIBILIDADE (Navegação)", "color": "#FFC700"},
    "PERFIL_MEDIO": {"label": "🟡 MÉDIA SENSIBILIDADE (Perfil)", "color": "#FFC700"},
    "SOCIAL_MEDIO": {"label": "🟡 MÉDIA SENSIBILIDADE (Social)", "color": "#FFC700"},
    "IDENTIFICACAO_MEDIO": {"label": "🟡 MÉDIA SENSIBILIDADE (Identificação)", "color": "#FFC700"},
    "TECNICO_BAIXO": {"label": "🟢 BAIXA SENSIBILIDADE (Técnico)", "color": "#28A745"},
    "OUTRO_BAIXO": {"label": "🟢 BAIXA SENSIBILIDADE (Não especificado)", "color": "#28A745"}
}

@st.cache_resource
def carregar_modelo_base():
    """Carrega o modelo base do spaCy para segmentação de sentenças."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Modelo base 'en_core_web_sm' não encontrado. Por favor, execute: python -m spacy download en_core_web_sm")
        return None

@st.cache_resource
def carregar_modelo_customizado(modelo_path):
    """Carrega um modelo customizado treinado."""
    try:
        return spacy.load(modelo_path)
    except OSError:
        st.error(f"Modelo customizado não encontrado em '{modelo_path}'. Certifique-se de que o modelo foi treinado e está no diretório correto.")
        return None


def analisar_texto(texto_politica, nlp_base, nlp_custom):
    """
    Analisa uma política de privacidade e retorna as predições.
    """
    if not texto_politica or not texto_politica.strip():
        return None

    texto_politica = re.sub(r'\s+', ' ', texto_politica.replace('\n', ' '))
    
    doc_completo = nlp_base(texto_politica)
    
    resultados = defaultdict(list)
    
    for sent in doc_completo.sents:
        if len(sent.text.strip()) < 15:
            continue

        doc_sentenca = nlp_custom(sent.text)
        scores = doc_sentenca.cats
        
        for categoria, score in scores.items():
            if score > 0.95:
                sentenca_normalizada = re.sub(r'\s+', ' ', sent.text.strip())
                resultados[categoria].append((sentenca_normalizada, score))
    
    return resultados


def main():
    st.set_page_config(page_title="Analisador de Políticas de Privacidade", layout="wide")

    nlp_base = carregar_modelo_base()
    if not nlp_base:
        return

    st.title("🔎 Analisador de Políticas de Privacidade")
    st.markdown("Cole o texto de uma política de privacidade em inglês abaixo para identificar os tipos de dados coletados e sua sensibilidade.")

    modelos_disponiveis = [p.name for p in Path(".").iterdir() if p.is_dir() and p.name.startswith("modelo_drop")]
    
    if not modelos_disponiveis:
        st.warning("Nenhum modelo treinado foi encontrado. Por favor, execute o script de treinamento primeiro.")
        return

    modelo_selecionado = st.selectbox(
        "Selecione o modelo para a análise:",
        sorted(modelos_disponiveis)
    )

    nlp_custom = carregar_modelo_customizado(modelo_selecionado)
    if not nlp_custom:
        return

    texto_input = st.text_area(
        "Cole o texto da política de privacidade aqui:",
        height=300,
        placeholder="Example: We collect your name, email address, and credit card information to process your order..."
    )

    if st.button("Analisar Texto", type="primary"):
        with st.spinner("Analisando o texto... Isso pode levar um momento."):
            resultados = analisar_texto(texto_input, nlp_base, nlp_custom)

        st.subheader("Resultados da Análise")

        if not resultados:
            st.info("Nenhuma categoria de dado foi identificada com o limiar de confiança definido (80%).")
        else:
            sorted_results = sorted(
                resultados.items(), 
                key=lambda item: list(CUSTOM_CATEGORIES_VISUAL.keys()).index(item[0])
            )

            for categoria_key, sentencas_com_score in sorted_results:
                visual_info = CUSTOM_CATEGORIES_VISUAL.get(categoria_key)
                if visual_info:
                    label = visual_info["label"]
                    color = visual_info["color"]

                    with st.expander(f"{label} ({len(sentencas_com_score)} ocorrências)"):
                        sentencas_unicas = sorted(list(set(sentencas_com_score)), key=lambda x: x[1], reverse=True)
                        
                        for sentenca, score in sentencas_unicas:
                            st.markdown(f"""
                            <div style="border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                                <p><em>"{sentenca}"</em></p>
                                <p style="text-align: right; font-size: 0.9em;"><strong>Confiança: {score:.2%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
