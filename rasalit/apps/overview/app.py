import argparse
import streamlit as st
import altair as alt
from rasalit.apps.overview.common import read_reports, remove, mk_viewable

parser = argparse.ArgumentParser(description="This app lists animals")
parser.add_argument("--folder", help="Pass the extra folder.")
args = parser.parse_args()

root_folder = args.folder
all_conf_folders = list(read_reports(args.folder, report="intent")["config"].unique())

df_intent = read_reports(root_folder, report="intent")
df_entity = read_reports(root_folder, report="entity")
df_response = read_reports(root_folder, report="response")

st.cache()

st.markdown("# Rasa GridResults Summary")
st.markdown("Quick Overview of Crossvalidated Runs")

st.sidebar.markdown("### Configure Overview")
items = st.sidebar.multiselect(
    "What are you interested in?",
    ("intent", "entity", "response"),
    (
        "intent",
        "entity",
    ),
)
metrics = st.sidebar.multiselect(
    "What metric do you care about?",
    ("f1-score", "precision", "recall"),
    ("f1-score", "precision", "recall"),
)
configs = st.sidebar.multiselect(
    "Which configs do you care about?", all_conf_folders, all_conf_folders
)
show_raw_data = st.sidebar.checkbox("Show Raw Data")

df_intent_subset = df_intent.pipe(remove, configs=configs, metrics=metrics)
df_entity_subset = df_entity.pipe(remove, configs=configs, metrics=metrics)
df_response_subset = df_response.pipe(remove, configs=configs, metrics=metrics)


if "intent" in items:
    st.markdown("## Intent Summary")
    c = (
        alt.Chart(df_intent_subset)
        .mark_bar()
        .encode(y="config:N", x="value:Q", color="config:N", row="variable:N")
    )
    st.altair_chart(c)

    if show_raw_data:
        st.write(df_intent_subset.pipe(mk_viewable))

if "entity" in items:
    st.markdown("## Entity Summary")
    c = (
        alt.Chart(df_entity_subset)
        .mark_bar()
        .encode(y="config:N", x="value:Q", color="config:N", row="variable:N")
    )
    st.altair_chart(c)
    if show_raw_data:
        st.write(df_intent_subset.pipe(mk_viewable))
