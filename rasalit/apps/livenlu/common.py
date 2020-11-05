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


def matrix_to_plot_df(np_mat, tokens, layer, head):
    data = []
    for i in range(np_mat.shape[0]):
        for j in range(np_mat.shape[1]):
            data.append(
                {
                    "w1": tokens[j],
                    "w2": tokens[i],
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


def plot_attention_weights(plot_df, tokens, title="DIET attention"):
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
