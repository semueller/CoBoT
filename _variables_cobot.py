from pathlib import Path

base_dir = './'

data_path = 'data/cobot'
results_prefix = 'cobot'

def __create_dir(dir):
    dir.mkdir(exist_ok=True, parents=True)
    return


def get_data_dir(task, modelclass=None):
    if modelclass is None:
        dir = Path(base_dir, data_path, task)
    else:
        dir = Path(base_dir, data_path, modelclass, task)
    __create_dir(dir)
    return dir


def get_result_dir(task, custom_prefix=None):
    _results_prefix = results_prefix if custom_prefix is None else custom_prefix
    dir = Path(base_dir, '../results', _results_prefix, task)
    __create_dir(dir)
    return dir


def get_plot_dir(task):
    dir = Path(base_dir, '../plots', results_prefix, task)
    __create_dir(dir)
    return dir
