# CoBoT

Experiments for Constructive Box Theorizer.

```conda env create -f environment.yml```

will create the `cobot` environment.
Activate it:

```conda activate cobot```

Add lxg/ weakvoncxity to pythonpath:

```export PYTHONPATH="$PYTHONPATH:`pwd`:`pwd`/src"```

Start experiments:

```cd cobot_experiments && python cobot_experiments.py```

It runs the experiments on [abalone, breastw, beans] datasets, it
1. creates a `results` folder, wherde models and computed CoBoT theories are stored.
2. fills the `plots` directory with PNG+PGF files for all 3 datasets. PNG images for abalone are provided here.