import json
import pathlib
import pandas as pd
import altair as alt


def read_reports(folder, report="intent"):
    files = {
        "entity": "DIETClassifier_report.json",
        "intent": "intent_report.json",
        "response": "response_selection_report.json",
    }
    paths = list(pathlib.Path(folder).glob(f"*/{files[report]}"))
    if paths:
        dicts = [json.loads(p.read_text())["weighted avg"] for p in paths]
        data = [{"config": p.parts[-2], **d} for p, d in zip(paths, dicts)]
        return pd.DataFrame(data).drop(columns=["support"]).melt("config")
    else:
        return pd.DataFrame()


def remove(dataf, configs, metrics):
    if dataf.empty:
        return dataf
    else:
        return dataf.loc[lambda d: d["config"].isin(configs)].loc[
            lambda d: d["variable"].isin(metrics)
        ]


def mk_viewable(dataf):
    return dataf.pivot("config", "variable").droplevel(0, axis=1).reset_index()


def create_altair_chart(dataf):
    return (
        alt.Chart(dataf)
        .mark_bar()
        .encode(
            y="config:N",
            x="value:Q",
            color="config:N",
            row="variable:N",
            tooltip=["config", "value"],
        )
    )
