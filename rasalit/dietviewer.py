
import streamlit as st

from http import server 

st.markdown("# Rasa DIET Explorer")

import pathlib 

path = pathlib.Path("rasalit/html/diet.html")
print(path, path.exists())
page = path.read_text()

st.markdown(page, unsafe_allow_html=True)