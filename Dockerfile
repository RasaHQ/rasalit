FROM python:3.8 AS rasalit-base

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN useradd --create-home appuser
RUN chown -R appuser $VIRTUAL_ENV
WORKDIR /home/appuser
USER appuser

RUN python -m pip install --upgrade pip

ADD --chown=appuser:appuser . /home/appuser

RUN pip install . && pip install bpemb

FROM rasalit-base AS rasalit-diet-explorer
ENV PORT=8500
CMD python -m rasalit diet-explorer --port $PORT

FROM rasalit-base AS rasalit-live-nlu
RUN python -m spacy download en_core_web_md
ENV PORT=8500
CMD python -m rasalit live-nlu --port $PORT

FROM rasalit-base AS rasalit-nlu-cluster
RUN pip install whatlies[umap]
RUN python -m spacy download en_core_web_md
ENV PORT=8500
CMD python -m rasalit nlu-cluster --port $PORT

FROM rasalit-base AS rasalit-overview
ENV PORT=8500
ENV FOLDER=gridresults
CMD python -m rasalit overview --port $PORT --folder $FOLDER

FROM rasalit-base AS rasalit-spelling
RUN python -m spacy download en_core_web_md
ENV PORT=8500
CMD python -m rasalit spelling --port $PORT
