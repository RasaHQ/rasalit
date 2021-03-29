import pathlib
import itertools as it
from functools import reduce

import spacy
from spacy import displacy
from spacy.tokens import Doc
import pandas as pd
import altair as alt
import streamlit as st
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT


@st.cache(allow_output_mutation=True)
def load_interpreter(model_dir, model) -> RasaNLUInterpreter:
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return RasaNLUInterpreter(nlu_model)


def fetch_info_from_message(interpreter, text_input):
    msg = Message({TEXT: text_input})
    blob = interpreter.interpreter.parse(text_input)
    nlu_dict = interpreter.featurize_message(msg).as_dict_nlu()
    tokens = [t.text for t in nlu_dict["text_tokens"]]
    return blob, nlu_dict, tokens


def create_altair_chart(dataf):
    return (
        alt.Chart(dataf)
        .mark_bar()
        .encode(
            y="name:N",
            x="confidence:Q",
            tooltip=["name", "confidence"],
        )
    )


def _mk_spacy_doc(tokens, entities):
    nlp = spacy.blank("en")
    doc = Doc(nlp.vocab, words=tokens, spaces=[True for _ in tokens])
    # This is a checking mechanism. Rasa allows for overlapping intents.
    # spaCy totally does not do that.
    taken = []
    warn = False
    for ent in entities:
        if (ent["start"], ent["end"]) not in taken:
            span = doc.char_span(ent["start"], ent["end"], label=ent["entity"])
            doc.ents = list(doc.ents) + [span]
            taken.append((ent["start"], ent["end"]))
        else:
            warn = True
    return doc, warn


def create_displacy_chart(tokens, entities):
    doc, warn = _mk_spacy_doc(tokens, entities)
    if warn:
        st.warning("Overlapping entities detected! ")
        st.json(entities)
    return displacy.render(doc, style="ent")


def parse_token_mat_into_dataframe(mat, tokens):
    data = [
        (tokens[i], tokens[j], mat[i, j])
        for i, j in it.product(range(len(tokens)), range(len(tokens)))
    ]
    return pd.DataFrame(data, columns=["w1", "w2", "score"])


def make_attention_chart(df, tokens, title=""):
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("w1:N", sort=tokens, title="from"),
            y=alt.Y("w2:N", sort=tokens, title="to"),
            color=alt.Color("score:Q", scale=alt.Scale(range=["white", "purple"])),
            tooltip=[
                alt.Tooltip("w1", title="from"),
                alt.Tooltip("w2", title="to"),
                alt.Tooltip("score", title="score"),
            ],
        )
        .properties(width=200, height=200, title=title)
    )


def make_attention_charts(diag_data, tokens):
    n_tf_layer, _, n_heads, _, _ = diag_data.shape

    charts = [[None for i in range(n_tf_layer)] for j in range(n_heads)]
    for j in range(n_heads):
        for i in range(n_tf_layer):
            dataf = parse_token_mat_into_dataframe(diag_data[i][0][j], tokens)
            charts[j][i] = make_attention_chart(
                dataf, tokens, title=f"Layer: {i} Head: {j}"
            )

    return reduce(lambda x, y: x & y, [reduce(lambda x, y: x | y, c) for c in charts])


def fetch_attention_feats(interpreter, text):
    msg = Message({TEXT: text})

    for p in interpreter.interpreter.pipeline:
        p.process(msg)

    diag_data = msg.as_dict()["diagnostic_data"]
    diet_key = [k for k in diag_data.keys() if "DIETClassifier" in k][0]
    return (
        diag_data[diet_key]["attention_weights"],
        [t.text for t in msg.as_dict()["text_tokens"]] + ["<SENT>"],
    )
