import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px

'''
# Club and Nationality App

This very simple webapp allows you to select and visualize players from certain clubs and certain nationalities.
'''


df = st.cache(pd.read_csv)("football_data.csv")

"Dataset Preview (fotoball_data.csv)",df

#In the following part the isin function is used however 
clubs = st.sidebar.multiselect('Show Player for clubs?', df['Club'].unique()) #give the multiselect dialogue box in the sidebar to select clubs
nationalities = st.sidebar.multiselect('Show Player from Nationalities?', df['Nationality'].unique())#give the multiselect dialogue box in the sidebar to select nationalities

new_df = df[(df['Club'].isin(clubs)) & (df['Nationality'].isin(nationalities))]

"Dataset Preview with the respective filter of clubs and nationalities"
st.write(new_df)



# Create distplot with custom bin_size
fig = px.scatter(new_df, x ='Overall',y='Age',color='Name')

'''
### Here is a simple chart between player age and overall
'''

st.plotly_chart(fig)
