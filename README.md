# CoBoT

Experiments for Constructive Box Theorizer.

`conda env create -f environment.yml` will create the `cobot` environment.
`cd cobot_experiments && python -m cobot_experiments.py` runs the experiments on [abalone, breastw, beans] datasets, it
1. creates a `results` folder, where models and computed CoBoT theories are stored.
2. fills the `plots` directory with PNG+PGF files for all 3 datasets. PNG images for abalone are provided here. 