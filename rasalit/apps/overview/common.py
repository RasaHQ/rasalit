import json
import pathlib
import pandas as pd


def read_reports(folder, report="intent"):
    files = {
        "entity": "DIETClassifier_report.json",
        "intent": "intent_report.json",
        "response": "response_selection_report.json",
    }
    paths = list(pathlib.Path(folder).glob(f"*/{files[report]}"))
    dicts = [json.loads(p.read_text())["weighted avg"] for p in paths]
    data = [{"config": p.parts[-2], **d} for p, d in zip(paths, dicts)]
    return pd.DataFrame(data).drop(columns=["support"]).melt("config")


def remove(dataf, configs, metrics):
    return dataf.loc[lambda d: d["config"].isin(configs)].loc[
        lambda d: d["variable"].isin(metrics)
    ]


def mk_viewable(dataf):
    return dataf.pivot("config", "variable").droplevel(0, axis=1).reset_index()
