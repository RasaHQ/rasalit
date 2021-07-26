<img src="docs/logo.png" width=300 align="right">

# RasaLit

A collection of helpful viewers that help with understand Rasa NLU components.
Some of these views are made using
[streamlit](https://github.com/streamlit/streamlit), hence the wink in the name.

Feedback is welcome.

## Installation

You can install via pip by linking to this github repository.

```
python -m pip install git+https://github.com/RasaHQ/rasalit
```

### Compatibility

The focus is to support the most recent version of Rasa. Current we target 2.x. 
We keep older versions around though. You can find rasalit for Rasa 1.10 [here](https://github.com/RasaHQ/rasalit/tree/r1.10).

## Usage

You can directly access the command line app.

```
> python -m rasalit --help
Usage: rasalit [OPTIONS] COMMAND [ARGS]...

  Helper Views for Rasa NLU

Options:
  --help  Show this message and exit.

Commands:
  diet-explorer  Allows you to explore the DIET settings.
  live-nlu       Select a trained Rasa model and interact with it.
  nlu-cluster    Cluster a text file to look for clusters of intents.
  overview       Gives an overview of all `rasa train nlu` results.
  spelling       Check the effect of spelling on NLU predictions.
  version        Prints the current version of rasalit.
```

## Features

The app contains a collection of viewers that each specialize in a seperate task.

### `nlu-cluster`

This command allows you to cluster similar utterances in a text file.

![](docs/cluster.gif)

Note that this app has some extra dependencies. You can install them via;

```
python -m pip install "whatlies[umap]"
```

Example Usage:

```
python -m rasalit nlu-cluster --port 8501
```

This will start a server locally. Internally it is using the [whatlies]() package to
handle the embeddings. This means that while the demo is only in English, you can extend
the code to work for Non-English scenarios too! For more details, as well as a labelling tool,
check out the notebook found [here](https://github.com/RasaHQ/rasalit/blob/main/notebooks/bulk-labelling/bulk-labelling-ui.ipynb).

### `overview`

This command shows an summary of the intent/entity scores from a `rasa train nlu` run.

![](docs/overview.gif)

Example Usage:

```
> python -m rasalit overview --folder gridresults --port 8501
```

This will start a server locally on port that will displace an interactive
dashboard of all your NLU gridsearch data.

To fully benefit from this feature you'll need to run some models first.
You can run cross validation of models in Rasa via the command line:

```
rasa test nlu --config configs/config-light.yml \
              --cross-validation --runs 1 --folds 2 \
              --out gridresults/config-light
rasa test nlu --config configs/config-heavy.yml \
              --cross-validation --runs 1 --folds 2 \
              --out gridresults/config-heavy
```

Then Rasa, in this case, will save the results in `gridresults/config-light` and
`gridresults/config-heavy` respectively.

To get an overview of all the results in subfolders of  `gridresults`,
you can run the `rasalit overview --folder gridresults` command from the same
folder where you ran the `rasa test` command. You'll get some simple charts
 that summarise the intent/entity performance.

### `spelling`

This command let's you predict text with augmented spelling errors to check for robustness.

![](docs/spelling.gif)

```
> python -m rasalit spelling --help
> python -m rasalit spelling --port 8501
```

This will start a server locally on port 8501 that will displace an interactive
playground for your trained Rasa NLU model. You can see the confidence levels change
as you allow for more or less spelling errors.

It's assumed that you run this command from the root of your Rasa project but you
can also make it point to other projects via the command line settings.

### `live-nlu`

This command gives you an interactive gui that lets you see the output of a trained modelling pipeline.

![](docs/nlu-playground.gif)

Example Usage:

```
> python -m rasalit live-nlu --help
> python -m rasalit live-nlu --port 8501
```

This will start a server locally on port 8501 that will displace an interactive
playground for your trained Rasa NLU model. You can see the confidence levels as
well as the detected entities. We also show some shapes of internal featurization
steps.

It's assumed that you run this command from the root of your Rasa project but you
can also make it point to other projects via the command line settings.

#### Attention Charts

If you're using the `DIETClassifier` you'll be able to also use this app to debug
the internals. The app also allows you to inspect all the pipeline settings as well
as the internal attention mechanism.

![](docs/attention.gif)

### `diet-explorer`

This command gives you an interactive visualisation of DIET that allows you to see the available hyperparameters from all the layers in the algorithm.

![](docs/diet-gif.gif)

Example Usage:

```
> rasalit diet-explorer --port 8501
```

This will start a server locally on port 8501 that will display an interactive
visualisation of the DIET architecture.

## Notebooks

This project also hosts a few jupyter notebooks that contain interactive tools.

### Bulk Labelling

The bulk labelling demo found in [this video](https://www.youtube.com/watch?v=YsMoGd7sYMQ)
and [this video](https://www.youtube.com/watch?v=T0dDetqgra4&ab_channel=Rasa) can be found
[here](https://github.com/RasaHQ/rasalit/blob/main/notebooks/bulk-labelling/bulk-labelling-ui.ipynb).

![](docs/bulk.gif)

This notebook allows you to use embeddings and a drawing tool to do some bulk-labelling.

## Running with Docker Compose

You can run all of the above commands at the same time and access them all from the same URL using the provided `docker-compose.yaml`.

To do this, [make sure Docker Compose is installed](https://docs.docker.com/compose/install/) and run:

```bash
# If you haven't already cloned this repo.
git clone git@github.com:RasaHQ/rasalit.git
cd rasalit

RASA_PROJECT_DIR=/path/to/my_rasa_project docker-compose up -d
```

Where `/path/to/my_rasa_project` is the (relative or absolute) path to the Rasa project you want to examine. Wait for a few seconds and you should be able to access all the Rasalit apps by going to http://localhost:8000.

## Contribute

There are many ways you can contribute to this project.

- You can suggest new features.
- You can help review new features.
- You can submit new components.
- You can let us know if there are bugs.
- You can let us know if the components in this library help you.

Feel free to start the discussion by opening an issue on this repository.
Before submitting code to the repository it would help if you first create
an issue so that we can disucss the changes you would like
to contribute. You can ping the maintainer (Github alias: **koaning**) both in
the issues here as well as on the [Rasa forum](https://forum.rasa.com)
if you have any questions.
