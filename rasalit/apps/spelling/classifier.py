import pathlib
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.core.interpreter import RasaNLUInterpreter


def load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return RasaNLUInterpreter(nlu_model)


class RasaClassifier(BaseEstimator, ClassifierMixin):
    """
    The RasaClassifier takes a pretrained Rasa model and turns it into a scikit-learn compatible estimator.
    It expects text as input and it will predict an intent class.

    Usage:

    ```python
    from rasa_nlu_examples.scikit import RasaClassifier

    mod = RasaClassifier(model_path="path/to/model.tar.gz")
    mod.predict(["hello there", "are you a bot?"])
    mod.predict_proba(["hello there", "are you a bot?"])
    ```
    """

    def __init__(self, model_path):
        self.model_path = model_path
        folder = str(pathlib.Path(self.model_path).parents[0])
        file = str(pathlib.Path(self.model_path).parts[-1])
        self.interpreter = load_interpreter(folder, file)
        self.class_names_ = [
            i["name"] for i in self.fetch_info_from_message("hello")["intent_ranking"]
        ]

    def fit(self, X, y):
        return self

    def fetch_info_from_message(self, text_input):
        """
        Fetch all the info from a single text input. Can be used to also retreive entities.

        Usage:

        ```python
        from rasa_nlu_examples.scikit import RasaClassifier

        mod = RasaClassifier(model_path="path/to/model.tar.gz")
        mod.fetch_info_from_message("hello there")
        ```
        """
        return self.interpreter.interpreter.parse(text_input)

    def predict(self, X):
        """
        Makes a class prediction, scikit-style.

        Note that we do not support `predict_proba` because our API allows for
        confidence values that do not lie in the [0, 1] range.

        Usage:

        ```python
        from rasa_nlu_examples.scikit import RasaClassifier

        mod = RasaClassifier(model_path="path/to/model.tar.gz")
        mod.predict(["hello there", "are you a bot?"])
        ```
        """
        return np.array([self.fetch_info_from_message(x)["intent"]["name"] for x in X])
