import os
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from lxg.datasets import __info_dim_classes, NumpyRandomSeed, TorchRandomSeed, _get_dataset_callable
from lxg.models import DNFClassifier, CoBoT, CoBoTExitCode, make_ff
from lxg.util import (load_pkl, dump_pkl, _get_outputs, _get_targets)

from lxg.attribution import integrated_gradients, lime_limetab

from plot_cobot_results import plot_cbx_results


def up_to_top_k_binarization_fn(attribution, k):
    """
    Binarize attribution map by selecting up to the top-k positive values per row.
    If a row has fewer than k positive values, selects all positive values.
    Returns a boolean array of the same shape as attribution.
    """
    if len(attribution.shape) == 1:
        attribution = np.expand_dims(attribution, axis=0)
    if k <= 0:
        raise ValueError("k must be a positive integer, not a cry for help.")

    # Initialize output
    binarized = np.zeros_like(attribution, dtype=bool)

    for i, row in enumerate(attribution):
        # Find indices of strictly positive values
        positive_indices = np.where(row > 0)[0]

        if positive_indices.size == 0:
            continue  # No positives, nothing to do here

        # Get the top-k (or fewer) positive values
        pos_values = row[positive_indices]
        top_k_idx_in_pos = np.argsort(pos_values)[-k:]
        selected_indices = positive_indices[top_k_idx_in_pos]

        binarized[i, selected_indices] = True

    return binarized

def train_nn(model, X_train, y_train, epochs=200, learning_rate=0.001, seed=42):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with TorchRandomSeed(seed=seed):
        model = model.to(device)
        X_train_tensor = torch.Tensor(X_train).to(device)
        y_train_tensor = torch.Tensor(y_train).to(torch.long).to(device)  # Reshape labels

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()  # Binary cross-entropy loss for classification

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if epochs > 2:
                if epoch % int(0.1*epochs) == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    model = model.to('cpu')
    return model


def _test_cobot(cbx: CoBoT, X, Y, A):
    outputs = []
    for i in range(len(X)):  # run dataset through cobot, collect return codes
        x, y, a = X[i], Y[i], A[i]
        outputs.append(
            cbx._explain_sample(sample=x, target=y, attribution=a, no_update=True)[0] # [0]->only keep CoBoT exitcode
        )
    tp = np.sum([o == CoBoTExitCode.SUCCESS for o in outputs]) / len(outputs)  # accuracy
    fp = np.sum([o == CoBoTExitCode.WRONG_LABEL for o in outputs]) / len(outputs)
    missing = np.sum([o in [CoBoTExitCode.NO_BOXES, CoBoTExitCode.ITEMSET_MISSING] for o in outputs]) / len(outputs)
    return {'tp': tp, 'fp': fp, 'missing': missing, 'outputs': outputs}

def online_cobot_examples(task=None):

    # load data
    dataset_name = task
    # X_train -> train NN, X_test -> validation NN + train CoBoT, X_val -> validation NN
    (X_train, y_train), (X_test, y_test), (X_val, y_val), input_size, n_classes = \
        _get_dataset_callable(task)(random_state=11880, as_torch=False, splits=(45, 40, 15))

    if len(X_test) > 10_000: # limit number of samples for CoBoT
        X_test = X_test[:10_000]
        y_test = y_test[:10_000]

    n_dims = X_train.shape[1]
    n_c = len(np.unique(y_train))
    model = make_ff([n_dims] + [32,32,32,32] + [n_c])
    _epochs = dict(beans=20, breastw=2, abalone=15)
    n_epochs = _epochs[task]
    seed = 42
    model = train_nn(model, X_train, y_train, epochs=n_epochs, seed=seed)

    inference_fn = model.predict_batch_softmax  # callable used by attribution methods
    model_fn = lambda x: _get_targets(inference_fn, x, model, 'cpu').detach().numpy() # for cbx
    _to_torch = lambda x: torch.from_numpy(x).to(torch.float)
    test_acc = np.mean(y_test == model_fn(_to_torch(X_test)))
    print(f"black box test accuracy: {test_acc}")

    # different k values for binarization function
    up_to_top_k_k = [i if i < X_train.shape[1] else X_train.shape[1] for i in [2, 4, 6]]

    _fname = f'{dataset_name}.pkl'
    _dir = f'results/cobot/{dataset_name}/{seed}/'
    Path(_dir).mkdir(exist_ok=True, parents=True)
    dump_pkl(model, _dir+'/model.pkl')
    with open(_dir+'/accuracy.txt', 'w') as txtf:
        txtf.writelines(f"black box test accuracy: \\{test_acc}\\ \n")
    cobot_results_fname = Path(_dir, _fname)
    print(f"will save to {str(Path(os.getcwd(), cobot_results_fname))}")

    model.to('cpu')
    _perturb_args = {
        'model': model,
        'inference_fn': inference_fn,
    }
    _grad_expls_args = {
        'model': model,
        'inference_fn': inference_fn,
    }

    _expl_callables = {
        # n_samples: number of steps to approximate integral
        'ig': lambda x, y: integrated_gradients(**_grad_expls_args, data=x, targets=y, n_samples=200,
                                                return_convergence_delta=False, subtract_baseline=True),
        # lime baselines: used to parameterize perturbation method
        'lit': lambda x, y: lime_limetab(**_perturb_args, data=x, targets=y, baselines=np.array(X_train)),
    }




    _results_model = []
    y_val = _get_targets(inference_fn, _to_torch(X_val),
                         model, 'cpu').detach().numpy()
    y_tr = _get_targets(inference_fn, _to_torch(X_test), model, 'cpu').detach().numpy()

    cobot_results = []

    for expl_method in _expl_callables.keys():
        print(f"compute {expl_method} on validation set")
        attr_val = np.array(_expl_callables[expl_method](_to_torch(X_val), torch.from_numpy(y_val)))
        print(f"compute {expl_method} on train set")
        attrs_tr = np.array(_expl_callables[expl_method](_to_torch(X_test), torch.from_numpy(y_tr)))

        for topk in up_to_top_k_k:
            validation_performance = []
            results = dict(binarization=f'top_{topk}',
                           ex=expl_method,
                           task=dataset_name,
                           X_tr=X_test,
                           X_te=X_val,
                           y_tr=y_tr,
                           y_te=y_val,
                           attrs_tr=attr_val,
                           attrrs_te=attr_val,
                           )

            binarization_fn = lambda a: up_to_top_k_binarization_fn(a, topk)

            cobot_coldstart = CoBoT(
                model_fn=model_fn,
                local_explainer_fn=None,
                binarization_fn=binarization_fn,
                binarized_expl_to_itemset_fn=CoBoT._binarized_expl_itemset_id,
            )
            cobot_coldstart.init(n_classes=len(np.unique(y_tr)), n_dims=X_test.shape[1])

            PROGBAR = tqdm(total=len(X_test))
            cbxs = []
            for x, y, a in zip(X_test, y_tr, attrs_tr):
                n_updates = cobot_coldstart._n_updates
                cobot_coldstart._explain_sample(sample=x, target=y, attribution=a)
                if cobot_coldstart._n_updates > n_updates: # performance hit
                    cobot_copy = deepcopy(cobot_coldstart)
                    cobot_copy._deref_callables()
                    cbxs.append(cobot_copy)
                    validation_performance.append(  # only check performance if something changed
                        _test_cobot(cobot_coldstart, X_val, y_val, attr_val)
                    )
                PROGBAR.update(1)
            results['cbxs'] = cbxs
            cobot_coldstart._deref_callables()
            results['final_cbx'] = cobot_coldstart
            results['validation_history'] = validation_performance
            results['bb_testacc'] = test_acc

            # use expls for test data if available
            cobot_results.append(results)
    pd_results = pd.DataFrame(cobot_results)
    print(f"saving, might take a while ...")
    dump_pkl(pd_results, _dir+f'/{dataset_name}_cobot_results.pkl')
    print(f"saved {_dir+f'/{dataset_name}_cobot_results.pkl'}")
    return pd_results


if __name__ == '__main__':
    # results = []
    # for task in ['abalone', 'beans', 'breastw']:
    #     print(f"starting {task}")
    #     pd_task_results = online_cobot_examples(task=task)
    #     results.append(pd_task_results)
    # pd_results = pd.concat(results, ignore_index=True)

    plot_cbx_results(tasks=['abalone', 'beans', 'breastw'])
