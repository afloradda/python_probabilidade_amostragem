import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

df = pd.read_csv("populacao_brasileira.csv")

st.title("Análise de Probabilidade sobre o documento 'População Brasileira'.")

st.header("Visualização dos Dados")
head = df.head()

st.dataframe(head)

# Questão 1: Probabilidade de não ser fluente em inglês.
st.subheader("Questão 1: Probabilidade Complementar de Fluência em Inglês")

total_pessoas = len(df)
fluentes = df[df["nível de proficiência em inglês"] == "Avançado"].shape[0]
p_fluente = fluentes / total_pessoas
p_nao_fluente = 1 - p_fluente

st.write(f"A probabilidade de uma pessoa aleatória **não ser fluente** em inglês é **{p_nao_fluente:.2%}**")

# Questão 2: Probabilidade de renda > 5 mil em AL ou PA
st.subheader("Questão 2: Probabilidade de Renda Superior a 5 Mil em AL ou PA")

df_al_pa = df[df["estado"].isin(["AL", "PA"])]
total_al_pa = len(df_al_pa)
renda_acima_5k = df_al_pa[df_al_pa["renda"] > 5000].shape[0]
p_renda_acima_5k = renda_acima_5k / total_al_pa

st.write(f"A probabilidade de uma pessoa de **Alagoas ou Pará** ter renda superior a **5 mil reais** é **{p_renda_acima_5k:.2%}**")

# Questão 3: Probabilidade de ensino = 'Superior' p/ pessoa em AM
st.subheader("Questão 3: Probabilidade de pessoa com Ensino Superior em Amazonas")
df_am = df[df["estado"].isin(["AM"])]
total_am = len(df_am)
ensino_superior = df_am[df_am["escolaridade"] == 'Superior'].shape[0]
p_com_superior = ensino_superior / total_am

k = 5
p_geom = ((1 - p_com_superior) ** (k - 1) * p_com_superior)

st.write(f"A probabilidade de uma pessoa do **Amazonas** ter ensino superior completo é **{p_com_superior:.2%}**")
st.write(f"A probabilidade da **5ª pessoa** amazonense que você conversar ser a primeira com ensino superior completo é **{p_geom:.2%}**")

# Questão 4: Faixa de renda predominante e função densidade de probabilidade
st.subheader("Questão 4: Faixa de Renda Predominante e Densidade de Probabilidade")

faixa = 1500
df["faixa_renda"] = (df["renda"] // faixa) * faixa

# Determinar a faixa de renda com mais pessoas
faixa_mais_comum = df["faixa_renda"].value_counts().idxmax()

st.write(f"A faixa de renda mais comum é entre **{faixa_mais_comum} e {faixa_mais_comum + faixa} reais**")

# Criar gráfico da distribuição de renda
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["renda"], bins=30, kde=True, stat="density", ax=ax)
ax.set_title("Distribuição de Renda com Densidade de Probabilidade")
ax.set_xlabel("Renda (R$)")
ax.set_ylabel("Densidade")
st.pyplot(fig)

# Questão 5: Distribuição normal baseando-se na média e variância da amostra.
st.subheader("Questão 5: Distribuição normal de renda baseando-se na média e variância da amostra")

media_renda = df["renda"].mean()
variancia_renda = df["renda"].var()

desvio_padrao = np.sqrt(variancia_renda)

st.write(f"Média da renda: **R$ {media_renda:.2f}**")
st.write(f"Variância da renda: **{variancia_renda:.2f}**")

# Criar a distribuição normal
eixo_x = np.linspace(df["renda"].min(), df["renda"].max(), 1000)
distribuicao_normal = norm.pdf(eixo_x, media_renda, desvio_padrao)

# Plotando a distribuição normal
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(eixo_x, distribuicao_normal, color='red', label='Distribuição Normal')
sns.histplot(df["renda"], bins=30, kde=True, stat="density", ax=ax, color='blue', alpha=0.5)
ax.set_title("Distribuição Normal da Renda e Histograma")
ax.set_xlabel("Renda (R$)")
ax.set_ylabel("Densidade")
ax.legend()
st.pyplot(fig)

# Questão 6: Probabilidade de encontrar 243 mil pessoas com pós-graduação em 1 milhão de habitantes
st.subheader("Questão 6: Probabilidade de encontrar 243 mil pessoas com pós-graduação")

# Contar a proporção de pessoas com pós-graduação na amostra
total_amostra = len(df)
total_pos_graduacao = df[df["escolaridade"] == "Pós-graduação"].shape[0]
proporcao_pos_graduacao = total_pos_graduacao / total_amostra

# Calcular a expectativa em uma amostra de 1 milhão
populacao_amostra = 1_000_000
esperado_pos_graduacao = proporcao_pos_graduacao * populacao_amostra

# Calcular a probabilidade usando distribuição normal
desvio_padrao = np.sqrt(populacao_amostra * proporcao_pos_graduacao * (1 - proporcao_pos_graduacao))
probabilidade = norm.cdf(243_000, esperado_pos_graduacao, desvio_padrao)

# Exibir resultados
st.write(f"Proporção de pessoas com pós-graduação na amostra: **{proporcao_pos_graduacao:.4f}**")
st.write(f"Número esperado de pessoas com pós-graduação em 1 milhão: **{esperado_pos_graduacao:.0f}**")
st.write(f"Probabilidade de encontrar 243 mil pessoas com pós-graduação: **{probabilidade:.4f}**")

# Plotando a distribuição normal
fig, ax = plt.subplots(figsize=(8, 4))
eixo_x = np.linspace(esperado_pos_graduacao - 3*desvio_padrao, esperado_pos_graduacao + 3*desvio_padrao, 1000)
eixo_y = norm.pdf(eixo_x, esperado_pos_graduacao, desvio_padrao)

ax.plot(eixo_x, eixo_y, color='red', label="Distribuição Normal")
ax.axvline(243_000, color='blue', linestyle='dashed', label="243 mil pessoas")
ax.fill_between(eixo_x, eixo_y, where=(eixo_x <= 243_000), color='blue', alpha=0.3)
ax.set_title("Distribuição Normal da Escolaridade (Pós-Graduação)")
ax.set_xlabel("Número de pessoas com Pós-Graduação")
ax.set_ylabel("Densidade")
ax.legend()
st.pyplot(fig)

# Questão 7: Função de Densidade Acumulada (CDF) para Escolaridade
st.subheader("Questão 7: Função de Densidade Acumulada da Escolaridade")

# Contar frequência de cada nível de escolaridade
freq_escolaridade = df["escolaridade"].value_counts(normalize=True).sort_index()

# Criar a densidade acumulada
cdf_escolaridade = freq_escolaridade.cumsum()

# Exibir tabela de valores
st.write("### Frequência relativa e CDF da escolaridade")
df_cdf = pd.DataFrame({
    "Escolaridade": freq_escolaridade.index,
    "Frequência Relativa": freq_escolaridade.values,
    "Densidade Acumulada": cdf_escolaridade.values
})
st.dataframe(df_cdf)

# Plotar gráfico da CDF
fig, ax = plt.subplots(figsize=(8, 4))
ax.step(cdf_escolaridade.index, cdf_escolaridade.values, where="mid", color='blue', label="CDF")
ax.set_xlabel("Nível de Escolaridade")
ax.set_ylabel("Probabilidade Acumulada")
ax.set_title("Função de Densidade Acumulada (CDF) - Escolaridade")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)


# Questão 8: Margem de erro amostral para proporção de pessoas com inglês intermediário
st.subheader("Questão 8: Margem de erro amostral para inglês intermediário")

# Tamanho da amostra
n = len(df)

# Contagem e proporção de pessoas com inglês intermediário
total_intermediario = df[df["nível de proficiência em inglês"] == "Intermediário"].shape[0]
p_intermediario = total_intermediario / n

# Nível de confiança de 95% -> Z = 1.96
z = 1.96

# Cálculo da margem de erro
margem_erro = z * np.sqrt((p_intermediario * (1 - p_intermediario)) / n)

# Exibir os resultados
st.write(f"Proporção de pessoas com inglês intermediário: **{p_intermediario:.4f}**")
st.write(f"Margem de erro amostral (95% de confiança): **{margem_erro:.4f}**")

# Exibir intervalo de confiança
st.write(f"Intervalo de confiança: **({p_intermediario - margem_erro:.4f}, {p_intermediario + margem_erro:.4f})**")


# Questão 9: Probabilidade de encontrar 60 pessoas com renda 1000 reais acima da média
st.subheader("Questão 9: Probabilidade de encontrar 60 pessoas com renda 1000 reais acima da média")

# Calcular média e desvio padrão da renda
media_renda = df["renda"].mean()
desvio_padrao_renda = df["renda"].std()

# Definir o valor da renda 1000 acima da média
renda_alvo = media_renda + 1000

# Calcular a probabilidade de uma pessoa ter renda maior que esse valor
prob_individual = 1 - norm.cdf(renda_alvo, media_renda, desvio_padrao_renda)

# Tamanho da amostra
n = len(df)

# Aproximação normal para a distribuição binomial
mu = n * prob_individual
sigma = np.sqrt(n * prob_individual * (1 - prob_individual))

# Padronizando a variável para normal padrão
z = (60 - mu) / sigma
prob_60_pessoas = 1 - norm.cdf(z)

# Exibir os resultados
st.write(f"Média da renda: **R$ {media_renda:.2f}**")
st.write(f"Desvio padrão da renda: **R$ {desvio_padrao_renda:.2f}**")
st.write(f"Renda alvo (R$ 1.000 acima da média): **R$ {renda_alvo:.2f}**")
st.write(f"Probabilidade de uma pessoa ter renda maior que R$ {renda_alvo:.2f}: **{prob_individual:.4f}**")
st.write(f"Probabilidade de encontrar pelo menos 60 pessoas com essa renda: **{prob_60_pessoas:.4f}**")

# Criar a distribuição normal com destaque para a renda alvo
fig, ax = plt.subplots(figsize=(8, 4))
eixo_x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
eixo_y = norm.pdf(eixo_x, mu, sigma)

ax.plot(eixo_x, eixo_y, color='red', label="Distribuição Normal Aproximada")
ax.axvline(60, color='blue', linestyle='dashed', label="60 Pessoas")
ax.fill_between(eixo_x, eixo_y, where=(eixo_x >= 60), color='blue', alpha=0.3)
ax.set_title("Distribuição Normal da Quantidade de Pessoas com Renda Alvo")
ax.set_xlabel("Número de Pessoas")
ax.set_ylabel("Densidade")
ax.legend()
st.pyplot(fig)


# Questão 10: Probabilidade de escolher uma pessoa com os critérios específicos
st.subheader("Questão 10: Probabilidade de escolher um homem da região Sudeste com ensino fundamental e renda > R$2.000")

# Definir estados da região Sudeste
regiao_sudeste = {"SP", "MG", "RJ", "ES"}

# Filtrar amostra com os critérios especificados
filtro = (
    (df["estado"].isin(regiao_sudeste)) &
    (df["sexo"] == "Masculino") &
    (df["escolaridade"] == "Fundamental Completo") &
    (df["renda"] > 2000)
)

# Número de pessoas que atendem ao critério
total_selecionados = df[filtro].shape[0]

# Probabilidade calculada
probabilidade = total_selecionados / len(df)

# Exibir os resultados
st.write(f"Total de pessoas na amostra: **{len(df)}**")
st.write(f"Total de pessoas que atendem aos critérios: **{total_selecionados}**")
st.write(f"Probabilidade de selecionar uma pessoa com esses critérios: **{probabilidade:.4f}**")