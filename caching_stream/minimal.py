import streamlit as st
import random
def hit_news_api(country, n):
 st.write("Cache Miss for n:",n,"and country:",country)
 return [random.randint(0,1000) for i in range(n)]
results = hit_news_api("USA", 5)
st.write(results)