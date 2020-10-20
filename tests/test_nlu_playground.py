"""
These tests depend on the availability of the starter project in `tests/demo`. Note
that during the CI a new model is trained at every run to ensure that we've got ourselves
a model file that we can use in these tests.
"""

import pathlib

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT

from rasalit.apps.livenlu.common import _mk_spacy_doc, load_interpreter


def test_load_interpreter_can_load():
    model_folder = "tests/demo/models"
    model_path = list(pathlib.Path(model_folder).glob("*"))[0]
    interpreter = load_interpreter(model_folder, str(model_path.parts[-1]))
    msg = Message({TEXT: "hello world"})
    for i, element in enumerate(interpreter.pipeline):
        element.process(msg)

    nlu_dict = msg.as_dict_nlu()

    assert nlu_dict["text"] == "hello world"
    assert [t.text for t in nlu_dict["tokens"]] == ["hello", "world", "__CLS__"]
    all_intents = [
        "talk_code",
        "bot_challenge",
        "mood_unhappy",
        "mood_great",
        "deny",
        "affirm",
        "goodbye",
        "greet",
    ]
    assert [i["name"] in all_intents for i in nlu_dict["intent_ranking"]]


def test_create_spacy_doc():
    tokens = ["Hello", "python", "and", "golang"]
    entities = [
        {
            "entity": "proglang",
            "start": 6,
            "end": 12,
            "value": "python",
            "extractor": "DIETClassifier",
        },
        {
            "entity": "proglang",
            "start": 17,
            "end": 23,
            "value": "golang",
            "extractor": "DIETClassifier",
        },
    ]
    doc = _mk_spacy_doc(tokens=tokens, entities=entities)
    for ent in doc.ents:
        assert ent.text in ["python", "golang"]
        assert ent.label_ == "proglang"
