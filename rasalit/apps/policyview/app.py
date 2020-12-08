import os
import pathlib

import numpy as np
from rich import print

from rasa import model
from rasa.nlu.model import Interpreter
from rasa.shared.core.domain import Domain
from rasa.model import get_model, get_model_subdirectories
from rasa.shared.constants import DEFAULT_DOMAIN_PATH
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.cli.utils import get_validated_path
from rasa.core.policies import PolicyEnsemble


model_folder = "/Users/vincent/Development/rasa/demo-bot-folder/models/"
file = "20201125-140119.tar.gz"

model_path = model.get_model(model_folder + file)
core_path, nlu_path = model.get_model_subdirectories(model_path)

policy = PolicyEnsemble.load(core_path)
domain = Domain.load(os.path.join(core_path, DEFAULT_DOMAIN_PATH))


convo = """
- greet
* utter_greet
"""


events = []
lines = [line for line in convo.split("\n") if len(line) > 0]
for line in lines:
    item = line.replace("*", "").replace("-", "").replace(" ", "")
    event = (
        UserUttered(intent={"name": item})
        if line[0] == "-"
        else ActionExecuted(action_name=item)
    )
    print(event, line, line[0])
    events.append(event)

tracker = DialogueStateTracker.from_events(
    "w",
    evts=events,
)

pred = policy.probabilities_using_best_policy(tracker, domain, loaded_nlu)


def correct_policy_name(polname):
    if "TED" in polname:
        return "TEDPolicy"
    if "Mem" in polname:
        return "MemoizationPolicy"
    if "rule" in polname:
        return "RulePolicy"


from rich.console import Console
from rich.table import Table
from rich import box
from rich.emoji import Emoji


class Overview:
    def __init__(self, nlu_path):
        self.table = Table(title="Conversation Overview", box=box.MINIMAL_DOUBLE_HEAD)
        self.table.add_column("", justify="left", width=3)
        self.table.add_column("utterance", justify="right", style="")
        self.table.add_column("intent/action", justify="right", style="")
        self.table.add_column("proba", style="")
        self.table.add_column("policy", style="")
        self.table.add_column("extra", style="")
        nlu_file = pathlib.Path(nlu_path).parts[-1]
        model_folder = str(nlu_path).replace(nlu_file, "")
        model_path = model.get_model(model_folder + file)
        core_path, nlu_path = model.get_model_subdirectories(model_path)

        self.policy = PolicyEnsemble.load(core_path)
        self.domain = Domain.load(os.path.join(core_path, DEFAULT_DOMAIN_PATH))
        self.loaded_nlu = self._load_interpreter(model_folder, file)
        self.events = []
        print(self.policy.policies)

    def _load_interpreter(self, model_dir, model):
        """Loads the NLU Interpreter"""
        path_str = str(pathlib.Path(model_dir) / model)
        model = get_validated_path(path_str, "model")
        model_path = get_model(model)
        _, nlu_model = get_model_subdirectories(model_path)
        return Interpreter.load(nlu_model)

    def add_usr_row(self, utter, intent, proba, **entities):
        self.table.add_row(
            Emoji("smile"),
            utter,
            intent,
            str(np.round(proba, 4)),
            "",
            "".join(f"{k}={v}" for k, v in entities.items()),
        )

    def add_bot_row(self, utter, action, proba, policy, **slots):
        self.table.add_row(
            Emoji("robot"),
            utter,
            action,
            str(np.round(proba, 4)),
            correct_policy_name(policy),
            "".join(f"{k}={v}" for k, v in slots.items()),
        )

    def show(self):
        console = Console()
        console.print(overview.table)

    def utter(self, text):
        action_name = None
        self.events.append(UserUttered(text=text))
        nlu_pred = self.loaded_nlu.parse(text=text)
        self.add_usr_row(
            text, nlu_pred["intent"]["name"], nlu_pred["intent"]["confidence"]
        )
        while action_name != "action_listen":
            tracker = DialogueStateTracker.from_events("w", evts=self.events)
            policy_pred = policy.probabilities_using_best_policy(
                tracker, self.domain, self.loaded_nlu
            )
            action_name = domain.action_names[np.argmax(policy_pred.probabilities)]
            self.add_bot_row(
                "",
                action=action_name,
                proba=max(pred.probabilities),
                policy=policy_pred.policy_name,
            )
            self.events.append(ActionExecuted(action_name=action_name))

    def refresh(self):
        self.events = []


overview = Overview(
    nlu_path="/Users/vincent/Development/rasa/demo-bot-folder/models/20201125-140119.tar.gz"
)
overview.add_bot_row("greetings", action="utter_greet", proba=0.9999, policy="Mem")
overview.add_bot_row("how can i help", action="utter_help", proba=0.9999, policy="Mem")
overview.add_usr_row("got a joke about code?", intent="joke", proba=0.865, topic="code")
overview.add_bot_row("alas no, i am bot", action="utter_no", proba=0.768, policy="TED")
overview.show()
