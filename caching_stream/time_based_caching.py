
import streamlit as st
import random
from datetime import datetime
import math

@st.cache
def hit_news_api(country, n, dummy_time):
	return [random.randint(0,1000) for i in range(n)]

def truncate_time(dt_time,period = 5):
	'''
	Given a period in minute truncates the datetime to the nearest lowest divisor for that minute:
	For Example: 
	Given dt_time='2020-03-27 02:32:19.684443' and period = 5 the function returns '2020-03-27 02:30:00.0'
	Given dt_time='2020-03-27 02:35:19.684443' and period = 5 the function returns '2020-03-27 02:35:00.0'
	'''

	# 1. Find nearest floor of minute to the current time minute
	new_minute_part = math.floor(dt_time.minute/period)*period

	# 2. Replace minute part in curr_time with nearest floored down minute 
	dt_time = dt_time.replace(minute=new_minute_part,second=0, microsecond=0)
	return dt_time

curr_time = datetime.now()
results = hit_news_api("USA", 5, truncate_time(curr_time))
st.write(results)