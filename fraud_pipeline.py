# =============================================================================
# Pipeline de Detecção de Fraudes — Cartão de Crédito
# Stack: Python · Pandas · pd.read_csv · DataFrame Storage
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import hashlib

# ── 0. CONFIGURAÇÕES GLOBAIS ──────────────────────────────────────────────────

SEED           = 42
N_TRANSACOES   = 5_000        # volume do dataset sintético
TAXA_FRAUDE    = 0.03         # 3% de transações fraudulentas
DATA_DIR       = "data"       # pasta de armazenamento dos DataFrames
RAW_CSV        = f"{DATA_DIR}/transacoes_raw.csv"
PROCESSED_CSV  = f"{DATA_DIR}/transacoes_processadas.csv"
FEATURES_CSV   = f"{DATA_DIR}/features_engenharia.csv"
ALERTAS_CSV    = f"{DATA_DIR}/alertas_fraude.csv"
REPORT_JSON    = f"{DATA_DIR}/relatorio_pipeline.json"

os.makedirs(DATA_DIR, exist_ok=True)
np.random.seed(SEED)

print("=" * 65)
print("  PIPELINE DE DETECÇÃO DE FRAUDES — CARTÃO DE CRÉDITO")
print("=" * 65)


# =============================================================================
# ETAPA 1 — GERAÇÃO DO DATASET SINTÉTICO (simula extração de banco)
# =============================================================================
print("\n[ETAPA 1] Gerando dataset sintético de transações...")

n_normal  = int(N_TRANSACOES * (1 - TAXA_FRAUDE))
n_fraude  = N_TRANSACOES - n_normal

# Datas distribuídas nos últimos 90 dias
data_base = datetime(2025, 1, 1)
datas_normal = [
    (data_base + timedelta(seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S")
    for s in np.random.randint(0, 90 * 86400, n_normal)
]
datas_fraude = [
    (data_base + timedelta(seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S")
    for s in np.random.randint(0, 90 * 86400, n_fraude)
]

# Categorias de estabelecimento
CATEGORIAS = [
    "supermercado", "farmácia", "restaurante", "postos_combustivel",
    "eletrônicos", "vestuário", "viagens", "e-commerce", "saques_atm",
]
CAT_FRAUDE = ["eletrônicos", "viagens", "e-commerce", "saques_atm"]

def gerar_cartao():
    return "****" + "".join([str(np.random.randint(0, 10)) for _ in range(4)])

# ── Transações normais
df_normal = pd.DataFrame({
    "transaction_id":   [hashlib.md5(f"n{i}".encode()).hexdigest()[:12].upper() for i in range(n_normal)],
    "timestamp":        datas_normal,
    "cartao_id":        [gerar_cartao() for _ in range(n_normal)],
    "valor":            np.round(np.random.lognormal(mean=4.5, sigma=1.0, size=n_normal), 2),
    "categoria":        np.random.choice(CATEGORIAS, n_normal, p=[.20,.12,.15,.10,.08,.10,.08,.12,.05]),
    "pais":             np.random.choice(["BR","BR","BR","BR","US","AR","CL"], n_normal),
    "cidade":           np.random.choice(["São Paulo","Rio de Janeiro","Belo Horizonte","Curitiba","Brasília"], n_normal),
    "dispositivo":      np.random.choice(["mobile","web","pos","atm"], n_normal, p=[.45,.30,.20,.05]),
    "parcelas":         np.random.choice([1,1,1,2,3,6,12], n_normal),
    "hora_dia":         [int(d[11:13]) for d in datas_normal],
    "fraude":           0
})

# ── Transações fraudulentas (padrões distintos)
df_fraude = pd.DataFrame({
    "transaction_id":   [hashlib.md5(f"f{i}".encode()).hexdigest()[:12].upper() for i in range(n_fraude)],
    "timestamp":        datas_fraude,
    "cartao_id":        [gerar_cartao() for _ in range(n_fraude)],
    "valor":            np.round(np.random.lognormal(mean=6.5, sigma=1.3, size=n_fraude), 2),  # valores maiores
    "categoria":        np.random.choice(CAT_FRAUDE, n_fraude),
    "pais":             np.random.choice(["NG","RO","CN","RU","US","BR"], n_fraude, p=[.20,.15,.15,.15,.25,.10]),
    "cidade":           np.random.choice(["Lagos","Bucarest","Shanghai","Moscow","Online"], n_fraude),
    "dispositivo":      np.random.choice(["web","atm","mobile"], n_fraude, p=[.55,.30,.15]),
    "parcelas":         np.random.choice([1,1,1], n_fraude),
    "hora_dia":         np.random.choice(list(range(0, 6)) + list(range(22, 24)), n_fraude),  # madrugada
    "fraude":           1
})

# ── Concatena e embaralha
df_raw = pd.concat([df_normal, df_fraude], ignore_index=True)
df_raw = df_raw.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Salva CSV bruto
df_raw.to_csv(RAW_CSV, index=False, encoding="utf-8")
print(f"  ✓ Dataset bruto salvo: {RAW_CSV}")
print(f"  ✓ Shape: {df_raw.shape[0]:,} transações × {df_raw.shape[1]} colunas")
print(f"  ✓ Distribuição de fraudes: {df_raw['fraude'].value_counts().to_dict()}")


# =============================================================================
# ETAPA 2 — LEITURA COM pd.read_csv (ponto central da requisição)
# =============================================================================
print("\n[ETAPA 2] Leitura do CSV com pd.read_csv...")

df = pd.read_csv(
    RAW_CSV,
    parse_dates=["timestamp"],          # converte para datetime automaticamente
    dtype={
        "transaction_id": str,
        "cartao_id":       str,
        "valor":           float,
        "categoria":       "category",  # economiza memória
        "pais":            "category",
        "cidade":          "category",
        "dispositivo":     "category",
        "parcelas":        int,
        "hora_dia":        int,
        "fraude":          int,
    },
    low_memory=False,
)

print(f"  ✓ DataFrame carregado: {df.shape}")
print(f"  ✓ Uso de memória: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("\n  Primeiras linhas:")
print(df.head(3).to_string(index=False))
print("\n  Tipos de dados:")
print(df.dtypes.to_string())


# =============================================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO
# =============================================================================
print("\n[ETAPA 3] Pré-processamento do DataFrame...")

# 3.1 — Verificação de nulos
nulos = df.isnull().sum()
print(f"  Valores nulos por coluna:\n{nulos[nulos > 0].to_string() if nulos.sum() > 0 else '  Nenhum valor nulo encontrado.'}")

# 3.2 — Remoção de duplicatas
dupl_antes = len(df)
df = df.drop_duplicates(subset=["transaction_id"])
print(f"  Duplicatas removidas: {dupl_antes - len(df)}")

# 3.3 — Filtragem de valores inválidos
df = df[df["valor"] > 0]
df = df[df["hora_dia"].between(0, 23)]

# 3.4 — Normalização de strings
df["categoria"] = df["categoria"].astype(str).str.lower().str.strip()
df["pais"]      = df["pais"].astype(str).str.upper().str.strip()

# 3.5 — Colunas de tempo derivadas
df["data"]           = df["timestamp"].dt.date
df["ano"]            = df["timestamp"].dt.year
df["mes"]            = df["timestamp"].dt.month
df["dia_semana"]     = df["timestamp"].dt.dayofweek          # 0=seg, 6=dom
df["fim_de_semana"]  = (df["dia_semana"] >= 5).astype(int)
df["madrugada"]      = ((df["hora_dia"] >= 0) & (df["hora_dia"] < 6)).astype(int)

print(f"  ✓ DataFrame pré-processado: {df.shape}")
df.to_csv(PROCESSED_CSV, index=False)
print(f"  ✓ Salvo em: {PROCESSED_CSV}")


# =============================================================================
# ETAPA 4 — FEATURE ENGINEERING (criação das variáveis preditoras)
# =============================================================================
print("\n[ETAPA 4] Feature Engineering...")

df_feat = df.copy()

# 4.1 — Codificação de variáveis categóricas
PAISES_ALTO_RISCO = {"NG", "RO", "CN", "RU", "UA", "PK"}
df_feat["pais_alto_risco"]      = df_feat["pais"].isin(PAISES_ALTO_RISCO).astype(int)
df_feat["pais_estrangeiro"]     = (~df_feat["pais"].isin(["BR"])).astype(int)
df_feat["dispositivo_atm"]      = (df_feat["dispositivo"] == "atm").astype(int)
df_feat["dispositivo_web"]      = (df_feat["dispositivo"] == "web").astype(int)
df_feat["categoria_alto_risco"] = df_feat["categoria"].isin(
    ["eletrônicos", "viagens", "e-commerce", "saques_atm"]
).astype(int)

# 4.2 — Estatísticas por cartão (aggregações comportamentais)
stats_cartao = df_feat.groupby("cartao_id").agg(
    total_transacoes     = ("valor", "count"),
    valor_total          = ("valor", "sum"),
    valor_medio          = ("valor", "mean"),
    valor_max            = ("valor", "max"),
    valor_std            = ("valor", "std"),
    qtd_paises_distintos = ("pais", "nunique"),
    qtd_categorias       = ("categoria", "nunique"),
    qtd_madrugada        = ("madrugada", "sum"),
).reset_index()
stats_cartao.columns.name = None

df_feat = df_feat.merge(stats_cartao, on="cartao_id", how="left")

# 4.3 — Score de risco por regra (heurística)
df_feat["score_risco"] = (
    df_feat["pais_alto_risco"]      * 30 +
    df_feat["madrugada"]            * 20 +
    df_feat["categoria_alto_risco"] * 15 +
    df_feat["pais_estrangeiro"]     * 10 +
    df_feat["dispositivo_atm"]      * 10 +
    (df_feat["valor"] > 3000).astype(int) * 25 +
    (df_feat["qtd_paises_distintos"] > 2).astype(int) * 15
)

# 4.4 — Normalização do valor (z-score por categoria)
df_feat["valor_zscore"] = df_feat.groupby("categoria")["valor"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-6)
)

# 4.5 — Flag: valor muito acima da média do cartão
df_feat["valor_acima_media"] = (
    df_feat["valor"] > df_feat["valor_medio"] * 3
).astype(int)

# Preenche NaN de desvio padrão (cartões com 1 transação)
df_feat["valor_std"] = df_feat["valor_std"].fillna(0)

print(f"  ✓ Features criadas: {df_feat.shape[1]} colunas")
df_feat.to_csv(FEATURES_CSV, index=False)
print(f"  ✓ Salvo em: {FEATURES_CSV}")


# =============================================================================
# ETAPA 5 — DETECÇÃO DE FRAUDES (modelo baseado em regras + scoring)
# =============================================================================
print("\n[ETAPA 5] Aplicando modelo de detecção de fraudes...")

# Threshold de risco
THRESHOLD_ALTO   = 70   # alerta imediato / bloqueio
THRESHOLD_MEDIO  = 40   # revisão manual

# Classificação por score
def classificar_risco(score):
    if score >= THRESHOLD_ALTO:
        return "ALTO"
    elif score >= THRESHOLD_MEDIO:
        return "MEDIO"
    else:
        return "BAIXO"

df_feat["nivel_risco"]       = df_feat["score_risco"].apply(classificar_risco)
df_feat["alerta_fraude"]     = (df_feat["nivel_risco"] == "ALTO").astype(int)
df_feat["revisao_necessaria"]= (df_feat["nivel_risco"] == "MEDIO").astype(int)

# Razões do alerta (auditoria)
def razoes_alerta(row):
    razoes = []
    if row["pais_alto_risco"]:          razoes.append("País de alto risco")
    if row["madrugada"]:                razoes.append("Horário suspeito (madrugada)")
    if row["categoria_alto_risco"]:     razoes.append("Categoria de risco")
    if row["valor"] > 3000:             razoes.append(f"Valor elevado (R$ {row['valor']:,.2f})")
    if row["pais_estrangeiro"]:         razoes.append("Transação internacional")
    if row["qtd_paises_distintos"] > 2: razoes.append("Múltiplos países")
    if row["valor_acima_media"]:        razoes.append("Valor 3x acima da média")
    return " | ".join(razoes) if razoes else "Nenhuma"

print("  Calculando razões dos alertas (pode levar alguns segundos)...")
df_feat["razoes_alerta"] = df_feat.apply(razoes_alerta, axis=1)

print(f"\n  Distribuição de níveis de risco:")
print(df_feat["nivel_risco"].value_counts().to_frame("contagem").to_string())


# =============================================================================
# ETAPA 6 — GERAÇÃO DE ALERTAS E ARMAZENAMENTO FINAL
# =============================================================================
print("\n[ETAPA 6] Gerando relatório de alertas e armazenando DataFrames...")

# DataFrame de alertas (apenas transações de risco ALTO)
COLUNAS_ALERTA = [
    "transaction_id", "timestamp", "cartao_id", "valor",
    "categoria", "pais", "dispositivo", "hora_dia",
    "score_risco", "nivel_risco", "razoes_alerta", "fraude"
]
df_alertas = df_feat[df_feat["nivel_risco"] == "ALTO"][COLUNAS_ALERTA].copy()
df_alertas = df_alertas.sort_values("score_risco", ascending=False).reset_index(drop=True)
df_alertas.to_csv(ALERTAS_CSV, index=False)
print(f"  ✓ Alertas salvos: {ALERTAS_CSV} ({len(df_alertas):,} alertas)")

# Salva DataFrame completo com features
df_feat.to_csv(FEATURES_CSV, index=False)
print(f"  ✓ DataFrame completo salvo: {FEATURES_CSV}")

# =============================================================================
# ETAPA 7 — MÉTRICAS DE AVALIAÇÃO DO PIPELINE
# =============================================================================
print("\n[ETAPA 7] Métricas de performance do pipeline...")

# Confusão: fraude real vs alerta gerado
tp = len(df_feat[(df_feat["fraude"] == 1) & (df_feat["alerta_fraude"] == 1)])
fp = len(df_feat[(df_feat["fraude"] == 0) & (df_feat["alerta_fraude"] == 1)])
fn = len(df_feat[(df_feat["fraude"] == 1) & (df_feat["alerta_fraude"] == 0)])
tn = len(df_feat[(df_feat["fraude"] == 0) & (df_feat["alerta_fraude"] == 0)])

precisao    = tp / (tp + fp + 1e-10)
recall      = tp / (tp + fn + 1e-10)
f1          = 2 * precisao * recall / (precisao + recall + 1e-10)
acuracia    = (tp + tn) / len(df_feat)
falsos_pos  = fp / (fp + tn + 1e-10)

metricas = {
    "total_transacoes":      int(len(df_feat)),
    "fraudes_reais":         int(df_feat["fraude"].sum()),
    "alertas_gerados":       int(df_feat["alerta_fraude"].sum()),
    "revisoes_geradas":      int(df_feat["revisao_necessaria"].sum()),
    "verdadeiro_positivo":   int(tp),
    "falso_positivo":        int(fp),
    "falso_negativo":        int(fn),
    "verdadeiro_negativo":   int(tn),
    "precisao_pct":          round(precisao * 100, 2),
    "recall_pct":            round(recall * 100, 2),
    "f1_score":              round(f1, 4),
    "acuracia_pct":          round(acuracia * 100, 2),
    "taxa_falso_positivo":   round(falsos_pos * 100, 2),
    "gerado_em":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\n  {'Métrica':<30} {'Valor':>10}")
print(f"  {'-'*42}")
print(f"  {'Total de transações':<30} {metricas['total_transacoes']:>10,}")
print(f"  {'Fraudes reais':<30} {metricas['fraudes_reais']:>10,}")
print(f"  {'Alertas gerados (ALTO)':<30} {metricas['alertas_gerados']:>10,}")
print(f"  {'Revisões necessárias':<30} {metricas['revisoes_geradas']:>10,}")
print(f"  {'Verdadeiro Positivo (TP)':<30} {metricas['verdadeiro_positivo']:>10,}")
print(f"  {'Falso Positivo (FP)':<30} {metricas['falso_positivo']:>10,}")
print(f"  {'Falso Negativo (FN)':<30} {metricas['falso_negativo']:>10,}")
print(f"  {'Precisão':<30} {metricas['precisao_pct']:>9.1f}%")
print(f"  {'Recall (Sensibilidade)':<30} {metricas['recall_pct']:>9.1f}%")
print(f"  {'F1-Score':<30} {metricas['f1_score']:>10.4f}")
print(f"  {'Acurácia':<30} {metricas['acuracia_pct']:>9.1f}%")
print(f"  ✓ Relatório salvo: {REPORT_JSON}")


# =============================================================================
# ETAPA 8 — ANÁLISE EXPLORATÓRIA RÁPIDA
# =============================================================================
print("\n[ETAPA 8] Análise exploratória...")

print("\n  Top 5 categorias com mais alertas:")
top_cat = (df_feat[df_feat["alerta_fraude"] == 1]
           .groupby("categoria")["alerta_fraude"]
           .count()
           .sort_values(ascending=False)
           .head(5))
print(top_cat.to_string())

print("\n  Top 5 países com mais alertas:")
top_pais = (df_feat[df_feat["alerta_fraude"] == 1]
            .groupby("pais")["alerta_fraude"]
            .count()
            .sort_values(ascending=False)
            .head(5))
print(top_pais.to_string())

print("\n  Valor médio por nível de risco:")
print(df_feat.groupby("nivel_risco")["valor"].mean().round(2).to_string())

print("\n  Alertas por hora do dia:")
print(df_feat[df_feat["alerta_fraude"] == 1]["hora_dia"].value_counts().sort_index().to_string())


# =============================================================================
# RESUMO FINAL — ARQUIVOS GERADOS
# =============================================================================
print("\n" + "=" * 65)
print("  PIPELINE CONCLUÍDO — ARQUIVOS GERADOS")
print("=" * 65)
arquivos = [
    (RAW_CSV,        "Dataset bruto (CSV original)"),
    (PROCESSED_CSV,  "DataFrame pré-processado"),
    (FEATURES_CSV,   "DataFrame com features de engenharia"),
    (ALERTAS_CSV,    "Alertas de fraude (risco ALTO)"),
    (REPORT_JSON,    "Relatório de métricas (JSON)"),
]
for path, desc in arquivos:
    tamanho = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
    print(f"  {path:<45} {tamanho:>6.1f} KB  — {desc}")

print("\n  DataFrames em memória:")
for nome, dframe in [("df (processado)", df), ("df_feat (features)", df_feat), ("df_alertas (alertas)", df_alertas)]:
    kb = dframe.memory_usage(deep=True).sum() / 1024
    print(f"  {nome:<30} shape={str(dframe.shape):<20} {kb:>6.1f} KB")

print("\n  Pipeline finalizado com sucesso.")
print("=" * 65)
