import sys
import argparse
import streamlit as st
import pandas as pd
import altair as alt
import pathlib

import nlpaug.augmenter.char as nac
from rasalit.apps.spelling.classifier import RasaClassifier

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_folder", help="Pass the model folder.")
parser.add_argument("--project_folder", help="The folder where you're running from.")
args = parser.parse_args()

model_folder = args.model_folder
sys.path.append(args.project_folder)


st.markdown("# Rasa Spelling Playground")
st.markdown("You can select a model on the left to interact with.")

st.sidebar.markdown("Made with love over at [Rasa](https://rasa.com/).")
st.sidebar.image(
    "https://rasahq.github.io/rasa-nlu-examples/square-logo.svg", width=100
)
model_files = [str(p.parts[-1]) for p in pathlib.Path(model_folder).glob("*.tar.gz")]
model_file = st.sidebar.selectbox("What model do you want to use", model_files)

text_input = st.text_input("Text Input for Model", "Hello")


min_char, max_char = st.sidebar.slider(
    "Number of characters to change.", min_value=0, max_value=10, value=(1, 2), step=1
)
min_word, max_word = st.sidebar.slider(
    "Number of words to change.", min_value=0, max_value=10, value=(1, 2), step=1
)

aug = nac.KeyboardAug(
    aug_char_min=min_char,
    aug_char_max=max_char,
    aug_word_min=2,
    aug_word_max=2,
    include_special_char=False,
    include_numeric=False,
    include_upper_case=False,
)

clf = RasaClassifier(pathlib.Path(model_folder) / model_file)

augs = aug.augment(text_input, n=500)
preds = clf.predict_proba(augs)

probas = clf.predict_proba(augs)
source = pd.DataFrame(
    {clf.class_names_[i]: probas[:, i] for i in range(len(clf.class_names_))}
).melt()


error_bars = (
    alt.Chart(source)
    .mark_errorbar(extent="stdev")
    .encode(x=alt.X("value:Q", scale=alt.Scale(zero=False)), y=alt.Y("variable:N"))
)

points = (
    alt.Chart(source)
    .mark_point(filled=True, color="black")
    .encode(
        x=alt.X("value:Q", aggregate="mean"),
        y=alt.Y("variable:N"),
    )
)

st.markdown("## Simple Line View")
st.altair_chart(error_bars + points, use_container_width=True)


hist = (
    alt.Chart(source)
    .mark_area(opacity=0.3, interpolate="step")
    .encode(
        alt.X("value:Q", bin=alt.Bin(maxbins=100)),
        alt.Y("count()", stack=None),
        alt.Color("variable:N"),
    )
)

with st.beta_expander("Histogram View"):
    st.altair_chart(hist, use_container_width=True)

with st.beta_expander("Generated Examples"):
    st.experimental_show(augs)
