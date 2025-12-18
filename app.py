import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração da Página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# Caminhos locais
MODEL_PATH = 'modelo_obesidade.pkl'
DATA_PATH = 'Obesity.csv'

# Carregar o modelo salvo
@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Erro: Arquivo '{MODEL_PATH}' não encontrado na pasta local.")
        return None
    return joblib.load(MODEL_PATH)

pipeline = carregar_modelo()

# Título
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

# Abas
tab1, tab2 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico"])

with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Coleta de Dados
    with col1:
        genero = st.selectbox("Gênero", ['Male', 'Female'])
        idade = st.number_input("Idade", 1, 120, 25)
        altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
        peso = st.number_input("Peso (kg)", 10.0, 300.0, 70.0)
        historia_fam = st.selectbox("Histórico Familiar de Sobrepeso?", ['yes', 'no'])

    with col2:
        favc = st.selectbox("Consome comida calórica frequentemente?", ['yes', 'no'])
        fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais", 1, 4, 3)
        caec = st.selectbox("Come entre refeições?", ['Sometimes', 'Frequently', 'Always', 'no'])
        smoke = st.selectbox("Fumante?", ['yes', 'no'])

    with col3:
        ch2o = st.slider("Consumo de água diário (1-3L)", 1, 3, 2)
        scc = st.selectbox("Monitora calorias ingeridas?", ['yes', 'no'])
        faf = st.slider("Frequência de atividade física (0-3)", 0, 3, 1)
        tue = st.slider("Tempo usando dispositivos (0-2)", 0, 2, 1)
        calc = st.selectbox("Consumo de álcool", ['Sometimes', 'Frequently', 'Always', 'no'])
        mtrans = st.selectbox("Meio de transporte principal", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    # Botão de Predição
    if st.button("Realizar Diagnóstico"):
        if pipeline:
            # Engenharia de Features: Calcular IMC
            imc = peso / (altura ** 2)
            
            # DataFrame com os dados (nomes das colunas IGUAIS ao notebook)
            dados_input = pd.DataFrame({
                'genero': [genero],
                'idade': [idade],
                'altura_m': [altura],
                'peso_kg': [peso],
                'historia_familiar_sobrepeso': [historia_fam],
                'come_comida_calorica_freq': [favc],
                'freq_consumo_vegetais': [fcvc],
                'num_refeicoes_principais': [ncp],
                'come_entre_refeicoes': [caec],
                'fumante': [smoke],
                'consumo_agua_litros': [ch2o],
                'monitora_calorias': [scc],
                'freq_atividade_fisica': [faf],
                'tempo_uso_dispositivos': [tue],
                'freq_consumo_alcool': [calc],
                'meio_transporte': [mtrans],
                'imc': [imc]
            })

            try:
                predicao = pipeline.predict(dados_input)[0]
                st.success(f"### Resultado Previsto: {predicao}")
                st.info(f"IMC Calculado: {imc:.2f}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")

with tab2:
    st.header("Insights da Base de Dados")
    
    if os.path.exists(DATA_PATH):
        df_dash = pd.read_csv(DATA_PATH)
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Pacientes", len(df_dash))
        c2.metric("Média de Peso", f"{df_dash['Weight'].mean():.1f} kg")
        c3.metric("Média de Idade", f"{df_dash['Age'].mean():.1f} anos")

        # Gráficos
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Distribuição de Obesidade")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(y='Obesity', data=df_dash, order=df_dash['Obesity'].value_counts().index, palette='viridis', ax=ax1)
            st.pyplot(fig1)
            
        with col_g2:
            st.subheader("Peso vs Altura")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='Weight', y='Height', hue='Obesity', data=df_dash, alpha=0.6, ax=ax2)
            st.pyplot(fig2)
    else:
        st.warning(f"Arquivo '{DATA_PATH}' não encontrado. Coloque-o na mesma pasta do script.")