import streamlit as st
import random

@st.cache
def hit_news_api(n):
	st.write("Cache Miss for n:",n)
	return [random.randint(0,1000) for i in range(n)]

results = hit_news_api(5)
st.write(results)