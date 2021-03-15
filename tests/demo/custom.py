import os
import typing
from pathlib import Path
from typing import Any, Optional, Text, Dict, List, Type

import numpy as np
from bpemb import BPEmb

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class BytePairFeaturizer(DenseFeaturizer):
    """This component adds BPEmb features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["bpemb"]

    defaults = {
        # specifies the language of the subword segmentation model
        "lang": None,
        # specifies the dimension of the subword embeddings
        "dim": None,
        # specifies the vocabulary size of the segmentation model
        "vs": None,
        # if set to True and the given vocabulary size can't be loaded for the given
        # model, the closest size is chosen
        "vs_fallback": True,
        # specifies the folder in which downloaded BPEmb files will be cached
        "cache_dir": str(Path.home() / Path(".cache/bpemb")),
        # specifies the path to a custom SentencePiece model file
        "model_file": None,
        # specifies the path to a custom embedding file
        "emb_file": None,
    }

    language_list = [
        "mt",
        "sd",
        "cr",
        "ba",
        "ht",
        "scn",
        "bi",
        "stq",
        "sm",
        "diq",
        "no",
        "yi",
        "vec",
        "bug",
        "am",
        "tl",
        "mn",
        "atj",
        "ko",
        "mai",
        "lij",
        "tcy",
        "sl",
        "bn",
        "dv",
        "rm",
        "ng",
        "ml",
        "kg",
        "koi",
        "war",
        "et",
        "mhr",
        "als",
        "bar",
        "ii",
        "sco",
        "got",
        "pnb",
        "ss",
        "bpy",
        "tum",
        "ru",
        "qu",
        "hy",
        "tw",
        "bm",
        "vep",
        "dty",
        "udm",
        "gd",
        "lbe",
        "rmy",
        "azb",
        "kw",
        "ja",
        "wuu",
        "pag",
        "ro",
        "tet",
        "ee",
        "min",
        "su",
        "ha",
        "glk",
        "pcd",
        "tk",
        "nrm",
        "ku",
        "gn",
        "ty",
        "bh",
        "pap",
        "fr",
        "ia",
        "cs",
        "ky",
        "ff",
        "kab",
        "rn",
        "csb",
        "tt",
        "cy",
        "ilo",
        "kaa",
        "hif",
        "ak",
        "pa",
        "crh",
        "ti",
        "myv",
        "ur",
        "se",
        "uz",
        "cdo",
        "lez",
        "srn",
        "kk",
        "pih",
        "de",
        "an",
        "tyv",
        "ext",
        "gan",
        "wo",
        "si",
        "lmo",
        "hak",
        "az",
        "ka",
        "ik",
        "frr",
        "hsb",
        "ho",
        "af",
        "nds",
        "pam",
        "el",
        "fur",
        "cu",
        "hr",
        "my",
        "nl",
        "da",
        "ch",
        "vls",
        "es",
        "as",
        "lt",
        "ny",
        "so",
        "oc",
        "lad",
        "pnt",
        "ms",
        "bcl",
        "os",
        "co",
        "ks",
        "or",
        "ay",
        "wa",
        "nah",
        "fa",
        "pl",
        "mzn",
        "za",
        "th",
        "fj",
        "kbp",
        "be",
        "zh",
        "ce",
        "sh",
        "sr",
        "id",
        "chy",
        "ps",
        "lo",
        "tr",
        "st",
        "he",
        "ang",
        "sah",
        "io",
        "gom",
        "ki",
        "sn",
        "kbd",
        "jam",
        "bo",
        "pms",
        "sk",
        "kv",
        "ckb",
        "nv",
        "dsb",
        "zea",
        "xmf",
        "fi",
        "ltg",
        "ksh",
        "ve",
        "new",
        "na",
        "jv",
        "tn",
        "sw",
        "rw",
        "ln",
        "bs",
        "gag",
        "ab",
        "olo",
        "is",
        "bjn",
        "ceb",
        "om",
        "vi",
        "ast",
        "uk",
        "mg",
        "mwl",
        "arz",
        "li",
        "mrj",
        "yo",
        "frp",
        "gl",
        "la",
        "km",
        "sv",
        "nap",
        "jbo",
        "bxr",
        "gv",
        "br",
        "fo",
        "ug",
        "pi",
        "bg",
        "ie",
        "din",
        "sa",
        "pdc",
        "cho",
        "lb",
        "ig",
        "aa",
        "sc",
        "fy",
        "kj",
        "eo",
        "eu",
        "kl",
        "sq",
        "to",
        "mi",
        "tpi",
        "kr",
        "hi",
        "arc",
        "ga",
        "nov",
        "mdf",
        "vo",
        "pfl",
        "rue",
        "haw",
        "kn",
        "mh",
        "mr",
        "te",
        "ca",
        "ace",
        "cv",
        "zu",
        "it",
        "iu",
        "av",
        "sg",
        "hz",
        "lv",
        "ts",
        "lrc",
        "ar",
        "hu",
        "nn",
        "nso",
        "krc",
        "mk",
        "tg",
        "ne",
        "dz",
        "ta",
        "mus",
        "ady",
        "en",
        "lg",
        "xal",
        "gu",
        "pt",
        "xh",
        "szl",
        "chr",
    ]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

        model_file, emb_file = (
            self.component_config[k] for k in ["model_file", "emb_file"]
        )
        if model_file:
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"BytePair model {model_file} not found. Please check config."
                )
        if emb_file:
            if not os.path.exists(emb_file):
                raise FileNotFoundError(
                    f"BytePair embedding file {emb_file} not found. Please check config."
                )

        if not self.component_config["lang"]:
            raise ValueError(
                "You must specify the `lang` parameter for BytePairEmbedding in `config.yml`."
            )

        if not self.component_config["vs"]:
            raise ValueError(
                "You must specify the `vs` parameter for BytePairEmbedding in `config.yml`."
            )

        if not self.component_config["dim"]:
            raise ValueError(
                "You must specify the `dim` parameter for BytePairEmbedding in `config.yml`."
            )

        self.model = BPEmb(
            lang=self.component_config["lang"],
            dim=self.component_config["dim"],
            vs=self.component_config["vs"],
            vs_fallback=self.component_config["vs_fallback"],
            cache_dir=self.component_config["cache_dir"],
            model_file=self.component_config["model_file"],
            emb_file=self.component_config["emb_file"],
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_bpemb_features(example, attribute)

    def create_word_vector(self, document: Text) -> np.ndarray:
        encoded_ids = self.model.encode_ids(document)
        if encoded_ids:
            return self.model.vectors[encoded_ids[0]]

        return np.zeros((self.component_config["dim"],), dtype=np.float32)

    def set_bpemb_features(self, message: Message, attribute: Text = TEXT) -> None:
        tokens = message.get(TOKENS_NAMES[attribute])

        if not tokens:
            return None

        # We need to reshape here such that the shape is equivalent to that of sparsely
        # generated features. Without it, it'd be a 1D tensor. We need 2D (n_utterance, n_dim).
        text_vector = self.create_word_vector(document=message.get(TEXT)).reshape(1, -1)
        word_vectors = np.array(
            [self.create_word_vector(document=t.text) for t in tokens]
        )

        final_sequence_features = Features(
            word_vectors,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.set_bpemb_features(message)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        if cached_component:
            return cached_component

        return cls(meta)
