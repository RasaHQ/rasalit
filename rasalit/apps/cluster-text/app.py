import pathlib
from io import StringIO

import streamlit as st
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering
from whatlies.language import CountVectorLanguage
from whatlies.transformers import Pca, Umap

uploaded = st.sidebar.file_uploader("Upload a `.txt` File")
if not uploaded:
    txt = pathlib.Path("nlu.md").read_text()
    texts = [
        t.replace(" - ", "") for t in txt.split("\n") if len(t) > 0 and t[0] != "#"
    ]
else:
    bytes_data = uploaded.read()
    stringio = StringIO(bytes_data.decode("utf-8"))
    string_data = stringio.read()
    texts = [
        t.replace(" - ", "")
        for t in string_data.split("\n")
        if len(t) > 0 and t[0] != "#"
    ]

n_svd = st.sidebar.slider(
    "Number of SVD components", min_value=2, max_value=100, step=1
)
min_ngram, max_ngram = st.sidebar.slider(
    "Range of ngrams", min_value=1, max_value=5, step=1, value=(2, 3)
)
reduction = st.sidebar.selectbox("Reduction Method", ("Umap", "Pca"))

st.markdown("# Simple Text Clustering")
st.markdown(
    "This app will attempt to cluster whatever text you give it. It is meant to inspire intents for Rasa `nlu.md` files. Stay tuned for `.yml` support!"
)
lang = CountVectorLanguage(n_svd, ngram_range=(min_ngram, max_ngram))
embset = lang[texts]

cluster = st.sidebar.checkbox("Assign Clusters")
if cluster:
    linkage = st.sidebar.selectbox(
        "Linkage Method", ("average", "complete", "ward", "single")
    )
    n_cluster = st.sidebar.slider(
        "Number of clusters", min_value=2, max_value=100, step=1, value=20
    )

    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_cluster)
    X = embset.to_X()
    clusters = {
        n: str(c) for n, c in zip([e.name for e in embset], model.fit_predict(X))
    }
    embset = embset.assign(group=lambda d: clusters[d.name])

p = (
    embset.transform(Umap(2) if reduction == "Umap" else Pca(2))
    .plot_interactive(annot=False, color="group")
    .properties(width=500, height=500)
)

st.write(p)
