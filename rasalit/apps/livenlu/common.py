import pathlib

import altair as alt

from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT

import spacy
from spacy import displacy
from spacy.tokens import Doc


def load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return Interpreter.load(nlu_model)


def fetch_info_from_message(interpreter, text_input):
    blob = interpreter.parse(text_input)

    msg = Message({TEXT: text_input})
    for i, element in enumerate(interpreter.pipeline):
        element.process(msg)

    nlu_dict = msg.as_dict_nlu()
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
    for ent in entities:
        span = doc.char_span(ent["start"], ent["end"], label=ent["entity"])
        doc.ents = list(doc.ents) + [span]
    return doc


def create_displacy_chart(tokens, entities):
    doc = _mk_spacy_doc(tokens, entities)
    return displacy.render(doc, style="ent")
