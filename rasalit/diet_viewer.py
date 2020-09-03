import pathlib
import streamlit as st
import streamlit.components.v1 as components


blob = pathlib.Path("rasalit/html/diet/index.html").read_text()
print(blob)
components.html(blob, height=400,)
