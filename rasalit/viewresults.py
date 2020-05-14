# to run this please make sure you've got the dependencies
# pip install streamlit altair pandas

import json
import pathlib
import os
import yaml

import streamlit as st
import altair as alt
import pandas as pd


def read_config(results_folder, config_name):
    paths = list(pathlib.Path(results_folder).glob("*/*.yml"))
    config_path = list(str(p) for p in paths if config_name in str(p))

    if len(config_path) == 0 or len(config_path) > 1 or not os.path.exists(config_path[0]):
        return {
            "Error": f"Not able to find config file in results folder '{results_folder}'."
        }

    with open(config_path[0]) as file:
        content = yaml.full_load(file)
        return content


def read_intent_report(path):
    blob = json.loads(path.read_text())
    jsonl = [
        {**v, "config": path.parts[path.parts.index("results") - 1]}
        for k, v in blob.items()
        if "weighted avg" in k
    ]
    return pd.DataFrame(jsonl).drop(columns=["support"])


def read_entity_report(path):
    blob = json.loads(path.read_text())
    jsonl = [
        {**v, "config": path.parts[path.parts.index("results") - 1]}
        for k, v in blob.items()
        if "weighted avg" in k
    ]
    return pd.DataFrame(jsonl).drop(columns=["support"])


def read_response_report(path):
    blob = json.loads(path.read_text())
    jsonl = [
        {**v, "config": path.parts[path.parts.index("results") - 1]}
        for k, v in blob.items()
        if "weighted avg" in k
    ]
    return pd.DataFrame(jsonl).drop(columns=["support"])


def add_zeros(dataf, all_configs):
    for cfg in all_configs:
        if cfg not in list(dataf["config"]):
            dataf = pd.concat(
                [
                    dataf,
                    pd.DataFrame(
                        {
                            "precision": [0],
                            "recall": [0],
                            "f1-score": [0],
                            "config": cfg,
                        }
                    ),
                ]
            )
    return dataf


st.cache()


def read_pandas(results_folder):
    paths = list(pathlib.Path(results_folder).glob("*/*_report.json"))

    if not paths:
        intent_df = entity_df = response_df = pd.DataFrame()
        return intent_df, entity_df, response_df

    configurations = set([p.parts[p.parts.index("results") - 1] for p in paths])

    intent_reports = [p for p in paths if "intent_report" in str(p)]
    if intent_reports:
        intent_df = pd.concat([read_intent_report(p) for p in intent_reports])
    else:
        intent_df = pd.DataFrame()

    response_reports = [p for p in paths if "response_selection_report" in str(p)]
    if response_reports:
        response_df = pd.concat([read_response_report(p) for p in response_reports])
    else:
        response_df = pd.DataFrame()

    entity_reports = []
    for f in paths:
        entity_reports += list(pathlib.Path(f).glob("CRFEntityExtractor_report.json"))
        entity_reports += list(pathlib.Path(f).glob("DIETClassifier_report.json"))

    if entity_reports:
        entity_df = pd.concat([read_entity_report(p) for p in paths]).pipe(
            add_zeros, all_configs=configurations
        )
    else:
        entity_df = pd.DataFrame()

    return intent_df, entity_df, response_df


def show_results_for(result_df, show_raw_data, selected_config, heading):
    if result_df.empty:
        return

    subset_df = result_df.loc[lambda d: d["config"].isin(selected_config)].melt(
        "config"
    )
    st.markdown(f"## {heading}")
    c = (
        alt.Chart(subset_df)
        .mark_bar()
        .encode(y="config:N", x="value:Q", color="config:N", row="variable:N")
    )
    st.altair_chart(c)
    if show_raw_data:
        st.write(result_df.loc[lambda d: d["config"].isin(selected_config)])


def _get_possible_configs(intent_df, entity_df, response_df):
    if not intent_df.empty:
        return sorted(list(intent_df["config"]))
    if not entity_df.empty:
        return sorted(list(intent_df["config"]))
    if not response_df.empty:
        return sorted(list(intent_df["config"]))

    return []


st.markdown("# Rasa GridResults Summary")
st.markdown("Quick Overview of Crossvalidated Runs")

st.sidebar.markdown("### Configure Overview")

results_folder = st.sidebar.text_input(
    "What is your results folder?", value="gridresults"
)

if pathlib.Path(results_folder).exists():
    intent_df, entity_df, response_df = read_pandas(results_folder=results_folder)
    possible_configs = _get_possible_configs(intent_df, entity_df, response_df)
else:
    st.write(f"Are you sure this results folder `{results_folder}` exists?")
    possible_configs = []
    intent_df = entity_df = response_df = pd.DataFrame()

st.sidebar.markdown("Select what you care about.")
selected_config = st.sidebar.multiselect(
    "Select Result Folders", possible_configs, default=possible_configs
)
show_raw_data = st.sidebar.checkbox("Show Raw Data")
if show_raw_data:
    show_markdown = st.sidebar.checkbox("Show Raw as Markdown")

config_to_show = st.sidebar.selectbox("Show content of config:", possible_configs)

show_results_for(
    intent_df, show_raw_data, selected_config, "Intent Summary Overview"
)
show_results_for(
    entity_df, show_raw_data, selected_config, "Entity Summary Overview"
)
show_results_for(
    response_df, show_raw_data, selected_config, "Response Summary Overview"
)
st.markdown("## Content of Config")
st.text(config_to_show)
st.json(read_config(results_folder, config_to_show))
