# python -m streamlit run streamlit_hw.py

import streamlit as st

st.title("calculadora de Média")

st.sidebar.header("escolha os numeros")
num1 = st.sidebar.number_input("Numero 1", value=0)
num2 = st.sidebar.number_input("Numero 2", value=0)

calcular = st.sidebar.button("Calcular Media")

st.write("informações: ")
st.write(f"Número 1: {num1}")
st.write(f"Número 2: {num2}")

if calcular:
    media = (num1 + num2) / 2
    st.write(f"a media é: {media}")

