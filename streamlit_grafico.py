import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Exemplo com grafico")

x = np.linspace(0,10,100)
y = np.sin(x)

data = pd.DataFrame({'x':x, 'y = sin(x)':y})
st.write("Dados gerados:", data)

fig,ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Grafico de seno')
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")

st.pyplot(fig)

freq = st.slider("frequencia",1,10,1)
y_freq = np.sin(freq*x)

fig2,ax2 = plt.subplots()
ax2.plot(x,y_freq)
ax2.set_title(f"grafico de seno com frequencia {freq}")
ax2.set_xlabel("x")
ax2.set_ylabel("sin",freq," x")
st.pyplot(fig2)
