import pathlib
import argparse

import pandas as pd
import altair as alt
import streamlit as st
from rasa.nlu.training_data import Message

from rasalit.apps.livenlu.common import (
    load_interpreter,
    create_altair_chart,
    create_displacy_chart,
)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder", help="Pass the model folder.")
args = parser.parse_args()

model_folder = args.folder


st.markdown("# Rasa NLU Model Playground")
st.markdown("You can select a model on the left to interact with.")

model_files = [str(p.parts[-1]) for p in pathlib.Path(model_folder).glob("*")]
model_file = st.sidebar.selectbox("What model do you want to use", model_files)

interpreter = load_interpreter(model_folder, model_file)

text_input = st.text_input("Text Input for Model", "Hello")
blob = interpreter.parse(text_input)

msg = Message(text_input)
for i, element in enumerate(interpreter.pipeline):
    element.process(msg)

nlu_dict = msg.as_dict_nlu()
tokens = [t.text for t in nlu_dict["tokens"] if t.text != "__CLS__"]

st.markdown("## Tokens and Entities")
st.write(
    create_displacy_chart(tokens=tokens, entities=nlu_dict["entities"]),
    unsafe_allow_html=True,
)

st.markdown("## Intents")

chart_data = pd.DataFrame(blob["intent_ranking"]).sort_values("name")
p = create_altair_chart(chart_data)
st.altair_chart(p.properties(width=600))

st.markdown("## Feature Sets")

st.write(f"text_sparse_features: {nlu_dict['text_sparse_features'].toarray().shape}")

if "text_dense_features" in nlu_dict:
    st.write(f"text_dense_features: {nlu_dict['text_dense_features'].toarray().shape}")
