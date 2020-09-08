import pathlib

import altair as alt

from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter


import spacy
from spacy import displacy
from spacy.tokens import Doc


def load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return Interpreter.load(nlu_model)


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


def create_displacy_chart(tokens, entities):
    nlp = spacy.blank("en")

    doc = Doc(nlp.vocab, words=tokens, spaces=[True for t in tokens])
    for ent in entities:
        span = doc.char_span(ent["start"], ent["end"], label=ent["entity"])
        doc.ents = list(doc.ents) + [span]
    return displacy.render(doc, style="ent")
