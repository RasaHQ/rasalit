import pathlib
import streamlit.components.v1 as components


blob = pathlib.Path("rasalit/html/diet/index.html").read_text()

components.html(
    blob,
    height=400,
)
