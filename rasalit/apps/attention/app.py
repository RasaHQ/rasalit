import pathlib
import argparse


import spacy
from spacy import displacy
from spacy.tokens import Doc
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import itertools as it

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter

parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder", help="Pass the model folder.")
args = parser.parse_args()

model_folder = args.folder


model_files = [str(p.parts[-1]) for p in pathlib.Path(model_folder).glob("*.tar.gz")]
model_file = st.sidebar.selectbox("What model do you want to use", model_files)


def load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return Interpreter.load(nlu_model)


loaded_nlu = load_interpreter(model_folder, "20201021-122736.tar.gz")

convo = st.sidebar.text_area("Every line is an utterance from the user.", "")
settings_exp = st.sidebar.beta_expander("Show Advanced Settings")
color = settings_exp.text_input("Color in Attention Charts", "purple")

st.markdown("# Rasa Attention Playground")
st.markdown(
    "You can select a trained Rasa model on the left to interact with. Note that the model **must** contain both DIET and TED models for this visual to work."
)


def parse_msg(text):
    msg = Message({TEXT: text})
    for i, element in enumerate(loaded_nlu.pipeline):
        element.process(msg)
    return msg


def matrix_to_plot_df(np_mat, tokens, layer, head):
    data = []
    for i in range(np_mat.shape[0]):
        for j in range(np_mat.shape[1]):
            data.append(
                {
                    "w1": tokens[i],
                    "w2": tokens[j],
                    "attention": np_mat[i, j],
                    "layer": layer,
                    "head": head,
                }
            )
    return pd.DataFrame(data)


def attention_to_plot_df(attention_weights, tokens):
    num_layers, _, num_heads, _, _ = attention_weights.shape
    dataframes = []
    for layer in range(num_layers):
        for head in range(num_heads):
            sub_df = matrix_to_plot_df(
                attention_weights[layer][0][head], tokens, layer=layer, head=head
            )
            dataframes.append(sub_df)
    return pd.concat(dataframes)


def plot_attention_weights(plot_df, title="DIET attention"):
    return (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=alt.X("w1:N", sort=tokens),
            y=alt.Y("w2:N", sort=tokens),
            column="head:Q",
            row="layer:Q",
            color=alt.Color("attention:Q", scale=alt.Scale(range=["white", color])),
            tooltip=[
                alt.Tooltip("w1", title="from"),
                alt.Tooltip("w2", title="to"),
                alt.Tooltip("attention", title="attention"),
            ],
        )
        .properties(title=title)
    )


def _mk_spacy_doc(tokens, entities):
    nlp = spacy.blank("en")
    doc = Doc(nlp.vocab, words=tokens, spaces=[True for _ in tokens])
    for ent in entities:
        span = doc.char_span(ent["start"], ent["end"], label=ent["entity"])
        doc.ents = list(doc.ents) + [span]
    return doc


def create_displacy_chart(tokens, entities):
    doc = _mk_spacy_doc(tokens, entities)
    return displacy.render(doc, style="ent")


def create_intent_chart(dataf):
    return (
        alt.Chart(dataf)
        .mark_bar()
        .encode(
            y="name:N",
            x="confidence:Q",
            tooltip=["name", "confidence"],
        )
    )


def plot_intent_probabilities(plot_df):
    pass


for turn, line in enumerate(convo.split("\n")):
    st.markdown(f"## Utterance {turn + 1}: `{line}`")
    msg = parse_msg(line)
    expander = st.beta_expander("Show Prediction Info")
    expander.markdown(f"**Intent**: `{msg.as_dict()['intent']['name']}`")
    intent_dataf = pd.DataFrame(msg.as_dict()["intent_ranking"]).sort_values("name")
    expander.altair_chart(create_intent_chart(intent_dataf))

    tokens = [t.text for t in msg.as_dict()["text_tokens"]]

    entities = msg.as_dict()["entities"]
    expander.markdown(f"**Entities**: `{[e['value'] for e in entities]}`")
    if len(entities) > 0:
        expander.write(
            create_displacy_chart(tokens, entities=entities),
            unsafe_allow_html=True,
        )
    tokens += ["__CLS__"]

    diet_clf = [e for e in loaded_nlu.pipeline if type(e).__name__ == "DIETClassifier"][
        0
    ]
    diagnostics = diet_clf.process_with_diagnostics(msg)
    diet_dataf = attention_to_plot_df(
        diagnostics["attention_weights"].numpy(), tokens=tokens
    )
    st.altair_chart(plot_attention_weights(diet_dataf))
