import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("populacao_brasileira.csv")

st.title("Análise de Probabilidade sobre o documento 'População Brasileira'.")

st.header("Header do Documento")
descriptive_stats = df.describe()

st.dataframe(descriptive_stats)