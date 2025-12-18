import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Carregar dados
try:
    df_original = pd.read_csv('Obesity.csv')
    print("✅ Arquivo Obesity.csv carregado.")
except FileNotFoundError:
    print("❌ Erro: Obesity.csv não encontrado na pasta.")
    exit()

# 2. Renomear colunas (igual ao seu notebook)
mapa_colunas = {
    'Gender': 'genero', 'Age': 'idade', 'Height': 'altura_m', 'Weight': 'peso_kg',
    'family_history': 'historia_familiar_sobrepeso', 'FAVC': 'come_comida_calorica_freq',
    'FCVC': 'freq_consumo_vegetais', 'NCP': 'num_refeicoes_principais',
    'CAEC': 'come_entre_refeicoes', 'SMOKE': 'fumante', 'CH2O': 'consumo_agua_litros',
    'SCC': 'monitora_calorias', 'FAF': 'freq_atividade_fisica', 'TUE': 'tempo_uso_dispositivos',
    'CALC': 'freq_consumo_alcool', 'MTRANS': 'meio_transporte', 'Obesity': 'nivel_obesidade'
}
df = df_original.rename(columns=mapa_colunas)

# 3. Engenharia de Features (Básica para garantir funcionamento)
# Convertendo tipos para evitar erros
colunas_int = ['freq_consumo_vegetais', 'num_refeicoes_principais', 'consumo_agua_litros', 'freq_atividade_fisica', 'tempo_uso_dispositivos']
for col in colunas_int:
    df[col] = df[col].astype(int)

# 4. Separar X e y
X = df.drop('nivel_obesidade', axis=1)
y = df['nivel_obesidade']

# 5. Configurar Pipeline
colunas_categoricas = [
    'genero', 'historia_familiar_sobrepeso', 'come_comida_calorica_freq',
    'come_entre_refeicoes', 'fumante', 'monitora_calorias',
    'freq_consumo_alcool', 'meio_transporte'
]
colunas_numericas = [col for col in X.columns if col not in colunas_categoricas]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), colunas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Treinar
print("⏳ Treinando modelo localmente...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 7. Validar
acc = accuracy_score(y_test, model_pipeline.predict(X_test))
print(f"✅ Modelo retreinado! Acurácia local: {acc:.2%}")

# 8. Salvar (Sobrescreve o arquivo antigo do Colab)
joblib.dump(model_pipeline, 'modelo_obesidade.pkl')
print("💾 Arquivo 'modelo_obesidade.pkl' atualizado com sucesso!")