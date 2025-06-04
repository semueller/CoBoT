import sklearn.tree
import torch
from torch import nn, device

from itertools import chain, combinations

import numpy as np


import enum

from copy import deepcopy

import weakconvexity.intensional.rectangles as wconv
from weakconvexity.intensional.rectangles import Vec, Rectangle as Rect, convert_lcbs_to_rules

class CoBoTExitCode(enum.Enum):
    SUCCESS = 0
    ATTRIBUTION_ERROR = 1
    NO_BOXES = 2
    AMBIGUOUS = 3
    WRONG_LABEL = 4
    ITEMSET_MISSING = 5

class CoBoT:
    def __init__(self,
                 model_fn,
                 local_explainer_fn,
                 binarization_fn=None,
                 binarized_expl_to_itemset_fn=None,
                 n_classes=-1,
                 track_itemset_history=False):

        self.model_fn = model_fn
        self.local_explainer = local_explainer_fn
        self.explanations = None
        self.binarization_fn = self._default_binarization_fn if binarization_fn is None else binarization_fn
        self.binarized_explanations = None
        self.binarized_expl_to_itemset_fn = CoBoT._binarized_expl_itemset_id \
            if binarized_expl_to_itemset_fn is None else binarized_expl_to_itemset_fn
        # save transformation of binarization->itemset for each explanation
        self.explanations_itemsets = []
        self._track_itemset_history = track_itemset_history

        self.samples = None  # to be initialized in init_fit, to be extended during online learning
        self.labels = None
        self.itemsets = []  # each itemset will be a key in dicts for boxsystems, DNFs, support_idxs
        self.boxsystems = None # k=itemset, val=boxsystem
        self.dnfclassifiers = None # k=itemset, val= DNFClassifier
        # k=itemset, val=idxs of elements from self.samples in support of respective itemset
        # this is NOT boxlevel support!
        self.support_idxs = {}
        self.n_classes = n_classes
        self.n_init_samples = 0
        self.history = []
        self.history_compression = []
        self.history_n_boxes = []
        self.history_n_singletons = []
        self.history_n_itemsets = []
        self.history_itemsets = {}

        self._n_boxes = 0
        self._n_updates = 0
        self._n_singletons = 0

        self.current_update_rate = np.inf
        self.info = {}

    def _deref_callables(self):
        '''Remove all callables '''
        self.model_fn = None
        self.local_explainer = None
        self.binarization_fn = None
        self.binarized_expl_to_itemset_fn = None

    def get_n_boxes(self):
        return self._n_boxes

    def get_n_samples(self):
        return len(self.samples)

    def current_compression_rate(self):
        return 1 - (self.get_n_boxes() / self.get_n_samples())

    def get_n_updates(self):
        return self._n_updates

    def _recompute_n_updates(self):
        return len([i for i in self.history if i[0] != CoBoTExitCode.SUCCESS])

    def current_n_singletons(self):
        n = 0
        for i in self.itemsets:
            for b in self.boxsystems[i]:
                box = b[0]
                n += int(CoBoT.box_is_singleton(box))
        self._n_singletons = n
        return n

    @staticmethod
    def _default_binarization_fn(attribution):
        attribution = attribution/np.expand_dims(np.max(attribution, axis=1), 1)
        attribution = np.array(attribution > 0.01, dtype=bool)
        return attribution

    def convert_boxsystem_to_dnf(self, boxsystem, itemset):
        init_rulemodel = convert_lcbs_to_rules(self.n_classes, boxsystem)
        final_rules = []
        # map dimensions to itemset
        for class_rules in init_rulemodel:
            final_rules.append([])
            for term in class_rules:
                final_rules[-1].append([])
                for literal in term:
                    d, (l, r) = literal
                    d = itemset[d]
                    final_rules[-1][-1].append((d, (l, r)))
        dnf = DNFClassifier(final_rules)
        return dnf

    def _init_itemset_support_idxs(self, itemset):
        support_idxs = [idx for idx, val in enumerate(self.explanations_itemsets) if val == itemset]
        self.__update_d_k_v(self.support_idxs, itemset, support_idxs)

    def init(self, n_classes, n_dims):
        self.n_init_samples = 0
        self.boxsystems = {}
        self.dnfclassifiers = {}
        self.labels = []
        self.samples = []
        self.n_classes = n_classes
        self.binarized_explanations = []
        self.explanations = []

    @staticmethod
    def box_is_singleton(box) -> bool:
        return all(abs(lb - ub) <= wconv._tau for lb, ub in zip(box.lower_bounds, box.upper_bounds))

    def init_fit(self, data, labels=None, explanations=None, use_closed_itemsets=False):
        if self.boxsystems is not None or self.explanations is not None:
            raise ValueError

        self.boxsystems = {}
        self.dnfclassifiers = {}

        if labels is None:
            labels = self.model_fn(data)
        self.labels = list(deepcopy(labels))
        self.n_classes = len(np.unique(self.labels))
        self.samples = list(deepcopy(data))
        self.n_init_samples = len(self.samples)

        if explanations is None:
            explanations = self.local_explainer(self.samples)
        self.explanations = explanations
        self.binarized_explanations = self.binarization_fn(self.explanations)
        # item_order = np.arange(self._data.shape[1])  # np.argsort(np.sum(expls_target, axis=1))[::-1]
        # itemset_results = __compute_rules_discr(expls_preproc, None,
        #                                         None, gely_threshold,
        #                                         k_means_max_bins, setcover_reduction,
        #                                         gely_sort_items, verbose, None, compute_rules=False)
        # itemsets, support_idxs = None, None
        if use_closed_itemsets:
            raise NotImplementedError
            # support_idxs_frequent_nodes:list[np.ndarray], nodes: list[ItemsetNode]
            # (support_idxs, nodes) = __compute_rules_discr(deepcopy(self._binarized_explanations),
            #                                         data_target=None, data_other=None,
            #                                         gely_threshold=self._frequency_threshold,
            #                                         gely_sort_items=False, verbose=False,
            #                                         model_callable=None, compute_rules=False)
            # itemsets = nodes[0].get_frequent_children()
            # itemsets = [OnlineBvFExplainer._OnlineBvFExplainer__itemset_to_key(i.itemset) for i in itemsets]
        else: # use identity, ie all dimensions marked in explanation
            # get all unique rows and to what unique element each index belongs
            unique_e_bin, inv_map = np.unique(self.binarized_explanations, axis=0, return_inverse=True)
            support_idxs = []
            _itemset_expls = unique_e_bin[inv_map]
            self.explanations_itemsets = [CoBoT._itemset_to_key(np.argwhere(_i).reshape(-1)) for _i in _itemset_expls]
            for _idx_unique, b in enumerate(unique_e_bin):
                support_idxs.append(
                    np.argwhere(_idx_unique == inv_map).reshape(-1)
                )
            itemsets = [CoBoT._itemset_to_key(np.argwhere(b).reshape(-1)) for b in unique_e_bin]

        self.itemsets = itemsets
        for (idxs, items) in zip(support_idxs, itemsets):
            try:
                _di = np.array([self.samples[i][list(items)] for i in idxs])
            except IndexError:
                print(idxs, items)
                print()
                pass
            if len(_di.shape)==1: _di = np.expand_dims(_di,0)
            _di = [Vec(d) for d in _di]
            _yi = [self.labels[i] for i in idxs]
            boxes = wconv.classification(_di, _yi)
            # save box system to dict
            self.__update_d_k_v(self.boxsystems, items, boxes)
            dnf = self.convert_boxsystem_to_dnf(boxes, items)
            # save dnf to dict
            self.__update_d_k_v(self.dnfclassifiers, items, dnf)
            self._init_itemset_support_idxs(items)
        self._calc_num_boxes()

    @staticmethod
    def _itemset_to_key(i):
        return tuple(sorted(i))

    def _calc_num_boxes(self):
        n_boxes = 0
        for itemset in self.itemsets:
            b = self.boxsystems[CoBoT._itemset_to_key(itemset)]
            n_boxes += len(b)
        self._n_boxes += n_boxes
        return n_boxes


    def __update_d_k_v(self, d, k, v):
        k = self._itemset_to_key(k)
        d.update({k: v})

    def _update(self, boxsystem, new_sample, new_label, mode:CoBoTExitCode, itemset):
        # SUPPORT FOR ITEMSET IS FULLY TRACKED IN _explain_sample
        k_itemset = CoBoT._itemset_to_key(itemset)
        l_itemset = list(itemset)
        # self.samples = np.vstack([self.samples, new_sample])
        # self.labels = np.array(list(self.labels)+ [new_label])
        new_sample = self.samples[-1]
        new_label = self.labels[-1]
        if mode == CoBoTExitCode.ITEMSET_MISSING:
            self.itemsets.append(CoBoT._itemset_to_key(itemset))
            self.history_n_itemsets.append((len(self.samples), len(self.itemsets)))
            self.__update_d_k_v(self.support_idxs, k_itemset, [])
            new_singleton_box = Rect.singleton(Vec(new_sample[l_itemset]))
            _l_boxsystem = [(new_singleton_box, wconv._tau, int(new_label))]
        elif mode == CoBoTExitCode.NO_BOXES:
            new_singleton_box = Rect.singleton(Vec(new_sample[l_itemset]))
            _l_boxsystem = list(boxsystem) + [(new_singleton_box, wconv._tau, int(new_label))]
        elif mode == CoBoTExitCode.WRONG_LABEL:
            wrong_box = None
            _l_boxsystem = list(self.boxsystems[k_itemset])
            for i, box  in enumerate(_l_boxsystem):
                if box[0].membership(Vec(new_sample[l_itemset])):
                    wrong_box = box
                    del _l_boxsystem[i]
                    break
            assert wrong_box is not None
            support_samples_itemset = [self.samples[s] for s in self.support_idxs[k_itemset]]
            support_samples_box = np.argwhere([wrong_box[0].membership(Vec(sample[l_itemset]))
                                        for sample in support_samples_itemset
                                       ]).reshape(-1)
            _l_boxsystem.extend([
                (Rect.singleton(Vec(self.samples[idx][l_itemset])), wconv._tau,
                                    self.labels[idx]
                 )
                 for idx in np.array(self.support_idxs[k_itemset])[support_samples_box]
            ])
        else:
            raise ValueError
        new_boxes = wconv.classification_from_boxes(_l_boxsystem)
        new_dnf = self.convert_boxsystem_to_dnf(new_boxes, k_itemset)
        self.__update_d_k_v(self.boxsystems, k_itemset, new_boxes)  # update boxsystems dict
        self.__update_d_k_v(self.dnfclassifiers, k_itemset, new_dnf)  # update DNFClassifier dict
        # update counts and history
        self._n_updates += 1
        if boxsystem is not None:
            self._n_boxes -= len(boxsystem)
            self._n_singletons -= sum([int(CoBoT.box_is_singleton(b[0])) for b in boxsystem])
        self._n_boxes += len(new_boxes)
        assert self._n_boxes >= 0
        self.history_n_boxes.append(self._n_boxes)
        self.history_compression.append(self.current_compression_rate())
        self._n_singletons += sum([int(CoBoT.box_is_singleton(b[0])) for b in new_boxes])
        self.history_n_singletons.append(self._n_singletons)
        self.history.append(
            (len(self.samples), mode)
        )
        if self._track_itemset_history:
            try:
                self.history_itemsets[k_itemset].append(
                    (self.get_n_samples(), self.get_n_updates(), deepcopy(self.boxsystems[k_itemset]))
                     )
            except KeyError:
                self.history_itemsets[k_itemset] = []
                self.history_itemsets[k_itemset].append(
                    (self.get_n_samples(), self.get_n_updates(), deepcopy(self.boxsystems[k_itemset]))
                     )


    @staticmethod
    def _binarized_expl_itemset_id(binary_expl):
        return np.argwhere(binary_expl).reshape(-1)

    def _expl_to_itemset(self, attr):
        # expexts a single attr -> shape == (d,)
        _binarized = self.binarization_fn(attr).reshape(-1)
        items = self.binarized_expl_to_itemset_fn(_binarized)
        return self._itemset_to_key(items)

    def _explain_sample(self, sample, target=None, attribution=None,
                        _internal_call_after_update: CoBoTExitCode=None, no_update=False):
        if attribution is None:
            attribution = self.local_explainer(sample)
        if target is None:
            target = self.model_fn(sample)

        itemset = self._expl_to_itemset(attribution)
        if len(itemset) == 0:
            if no_update:
                return (CoBoTExitCode.ATTRIBUTION_ERROR, f"explanation was mapped to empty itemset.")
            self.history.append((len(self.samples), CoBoTExitCode.ATTRIBUTION_ERROR))
            return (CoBoTExitCode.ATTRIBUTION_ERROR, f"explanation was mapped to empty itemset.")
            # raise ValueError(f"explanation was mapped to empty itemset.")

        if _internal_call_after_update is None and not no_update: # ie this is the first call of _explain_sample where we are not coming from an _update
            self.samples = self.samples + [sample]
            self.labels = np.array(list(self.labels) + [target])
            self.explanations_itemsets.append(itemset)

        bool_itemset_was_missing = False
        if not _internal_call_after_update:
            try:
                if no_update: #
                    if not itemset in self.support_idxs.keys():
                        raise KeyError
                else:
                    self.support_idxs[itemset].append(len(self.samples)-1)
            except KeyError:
                bool_itemset_was_missing = True
                if no_update:
                    return (CoBoTExitCode.ITEMSET_MISSING, f"ITEMSET MISSING.")
                self._update(None, sample, target, CoBoTExitCode.ITEMSET_MISSING, itemset)
                self.support_idxs[itemset].append(len(self.samples) - 1)

        dnf = self.dnfclassifiers[itemset]
        expl = dnf.describe([sample])[0] # since we only describe a single sample we [0]

        labels = [c for c in range(len(expl))  if len(expl[c]) > 0 ]

        if len(labels) == 0: # no boxes
            assert not bool_itemset_was_missing
            if no_update:
                return (CoBoTExitCode.NO_BOXES, f"NO FITTING BOX.")
            self._update(self.boxsystems[itemset], sample, target, CoBoTExitCode.NO_BOXES, itemset)
            return self._explain_sample(sample, target, attribution,
                                        _internal_call_after_update=CoBoTExitCode.NO_BOXES)

        if len(labels) > 1: # sanity check
            raise ValueError(f"{CoBoTExitCode.AMBIGUOUS}, ambiguous prediction, "
                             f"target {target} but got predicted classes {labels}, {expl}")

        # labels != 0 and labels <= 2 -> len(labels) == 1
        # print(f"target={target} v {labels}=labels")
        label = labels[0]


        if label != target: # update boxsystem
            assert not bool_itemset_was_missing
            if no_update:
                return (CoBoTExitCode.WRONG_LABEL, f"WRONG LABEL.")
            self._update(self.boxsystems[itemset], sample, target, CoBoTExitCode.WRONG_LABEL, itemset)
            return self._explain_sample(sample, target, attribution,
                                        _internal_call_after_update=CoBoTExitCode.WRONG_LABEL)

        if label == target: # yay
            if (not _internal_call_after_update and # we called _updated() from WRONG_LABEL/NO_BOX before
                    not bool_itemset_was_missing and # we _updated() because itemset was missing
                    not no_update): # we're only running inference
                self.history.append((len(self.samples), CoBoTExitCode.SUCCESS))
            return (CoBoTExitCode.SUCCESS,
                    f"rules match predicted class {target}", expl[label])


    def explain_samples(self, samples, labels=None, attributions=None):
        if labels is None:
            labels = [None]*len(samples)
        if attributions is None:
            attributions = [None]*len(samples)
        outs = []
        for s, l, a in zip(samples, labels, attributions):
            o = self._explain_sample(s, l, a)
            outs.append(o)
        return outs


class SimpleTorchWrapper():
    '''
    wrapper class that maps basic functions of an sklearn api based model to torch.

    '''

    def __init__(self, sklearn_model):
        self.model = sklearn_model
        self.inference_fn = self.model.predict_proba
        self.training = False
        self.device = 'cpu'


    # helper
    def __cast_input(self, X):
        try:
            X_np = X.detach().cpu().numpy()
        except AttributeError:
            X_np = np.array(X)
        return X_np

    # prediction wrapper
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # take torch tensor/ numpy array and return
        X_np = self.__cast_input(X)
        y = self.inference_fn(X_np)
        y_tensor = torch.tensor(y)
        return y_tensor

    def predict_batch(self, x):
        _X = self.__cast_input(x)
        pred = self.model.predict(x)
        pred = torch.tensor(pred)
        return pred

    # dummy functions
    def eval(self):
        return

    def to(self, device):
        return self

    def parameters(self):
        return iter([self.model])

    def train(self, b):
        return None


def merge_rules(a, b):
    # if not mergeable return (a, b)
    # if mergeabl then return (merged_rule, None)

    def get_dims(r):
        return set([t[0] for t in r])

    def overlap(i1, i2):
        i1, i2 = sorted([i1, i2], key=lambda x: (x[0], x[1]))
        if i1[1] >= i2[0]:
            return True
        return False

    def overlap_not_contain(i1, i2):
        i1, i2 = sorted([i1, i2], key=lambda x: (x[0], x[1]))
        if i1[1] < i2[0]: return False
        if i1[0] <= i2[0] and i1[1] >= i2[0] and i1[1] < i2[1]: return True
        return False

    def contains(i1, i2):
        return i1[0] <= i2[0] and i2[1] <= i1[1]

    a = sorted(a, key=lambda t: t[0])
    b = sorted(b, key=lambda t: t[0])
    da = get_dims(a)
    db = get_dims(b)

    if not (da <= db or db <= da): # if rules use different dims, return
        return a, b

    if da == db:
        # if a, b look at the same dims: if all dims overlap, the rules can be merged
        new_rule = []
        _n_overlap = 0
        _containment = None

        if np.all([ta == tb for ta, tb in zip(a, b)]):  # rules are identical
            return a, None

        _a_contains_b = np.array([contains(ta[1], tb[1]) for ta, tb in zip(a, b)])
        _b_contains_a = np.array([contains(tb[1], ta[1]) for ta, tb in zip(a, b)])

        if np.all(_a_contains_b):
            return a, None
        if np.all(_b_contains_a):
            return b, None

        _eq = np.logical_and(_a_contains_b, _b_contains_a)

        if sum(_eq) == len(_eq) - 1:
            for i, (ta, tb) in enumerate(zip(a, b)):
                if _a_contains_b[i]:
                    new_rule.append(ta)
                elif _b_contains_a[i]:
                    new_rule.append(tb)
                elif overlap_not_contain(ta[1], tb[1]):
                    _start = min(ta[1][0], tb[1][0])
                    _end = max(ta[1][1], tb[1][1])
                    new_rule.append((ta[0], (_start, _end)))
                else:  # if they don't overlap, don't merge
                    return a, b
            # all contained or overlapped
            return new_rule, None

        #
        return a, b

    # else: da != db

    if len(db) < len(da):
        a, b = b, a
        da, db = db, da

    # a is the shorter, potentially more general, rule
    # check that all intervals of terms in a fully contain the terms in b

    def __get_term_for_dim(terms, dim):
        for t in terms:
            if t[0] == dim: return t
        return None
    # assert that all terms in shorter rule (->a) contain all respective terms in longer rule (->b)
    _a_contains_b = np.array([contains(ta[1], __get_term_for_dim(b, ta[0])[1]) for ta, tb in zip(a, b)])
    if not all(_a_contains_b):
        return a, b
    # b partially marks other intervals in shared dimensions, hence keep both rules
    return a, None



class DNFClassifier:
    def __init__(self, rules: list[list[list[tuple]]], tie_break="first"):
        # rule format: dimension, interval -> tuple(dimension, tuple(lower_limit, upper_limit))
        self.n_classes = len(rules)
        self.rules = rules
        self.purge_dummy_rules()
        self.rule_performances = {c: {} for c in range(self.n_classes)}  # dict to collect performance statistics of rules
        self.rule_usage_stats = {c: {} for c in range(self.n_classes)}
        self._reset_usage_stats()
        self.tie_break = tie_break
        assert (self.tie_break in
                ["first", "shortest", "longest", "random", "accuracy", "f1"])
        self.n_literals = self.__comp_n_literals()
        self.n_rules = self.__comp_n_rules()
        self._meta_information = None
        self.score = None


    def __call__(self, samples):
        return self.predict(samples)

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.n_classes:
            return self.rules[self.current]
        raise StopIteration

    def __getitem__(self, idx):
        # make class subscriptable
        return self.rules[idx]

    def _reset_usage_stats(self):
        for c, r in zip(range(self.n_classes), self.rules):
            for term in r:
                self.rule_usage_stats[c][tuple(term)] = 0


    def purge_dummy_rules(self):
        purged_rules = []
        for class_rules in self.rules:
            purged_class = []
            for term in class_rules:
                term = list(filter(lambda x: x != (-1, (np.nan, np.nan)), term))
                purged_class.append(term)
            purged_rules.append(purged_class)
        self.rules = purged_rules

    def assert_no_infty(self):
        for class_rules in self.rules:
            for terms in class_rules:
                for literal in terms:
                    assert np.abs(literal[1][0]) != np.inf and np.abs(literal[1][1]) != np.inf

    def __set_rules(self, new_rules):
        self.rules = deepcopy(new_rules)
        self.n_classes = len(self.rules)
        self.n_literals = self.__comp_n_literals()
        self.n_rules = self.__comp_n_rules()
        return

    def __comp_n_literals(self) -> int:
        n_literals = 0
        for c in range(self.n_classes):
            c_dnf = self.rules[c]
            n_literals += sum([len(term) for term in c_dnf])

        return n_literals

    def get_n_terms(self):
        return self.n_literals

    def __comp_n_rules(self):
        n_rules = 0
        for class_rules in self.rules:
            n_rules += len(class_rules)
        return n_rules

    def get_n_rules(self):
        return self.n_rules

    def get_relevant_dims(self):
        d = []
        for c in range(self.n_classes):
            c_dnf = self.rules[c]
            for term in c_dnf:
                for literal in term:
                    if literal[0] not in d:
                        d.append(literal[0])
        d = sorted(d)
        return d

    def compute_rule_performance(self, X, Y):
        # compute performance statistics [accuracy, prec, recall, f1] for every single term of each class
        # if then our DNF predicts multiple classes, predict class of rules with "best" performance
        for c in range(self.n_classes):
            # Xc = X[Y == c]
            # Xother = X[Y != c]
            # Yother = Y != c
            _class_rules = self.rules[c]
            Yc = Y == c

            _c_stats = {}
            for clause in _class_rules:
                applicability = []
                for sample in X:
                    applicability.append(self.__clause_match_sample(sample=sample, clause=clause))
                applicability = np.array(applicability)
                precision = sklearn.metrics.precision_score(Yc, applicability, zero_division=0)
                recall = sklearn.metrics.recall_score(Yc, applicability, zero_division=0)
                f1 = sklearn.metrics.f1_score(Yc, applicability, zero_division=0)

                _c_stats[tuple(clause)] = {'accuracy':precision, 'f1':f1, 'recall': recall}
            self.rule_performances[c] = _c_stats
        return

    def simplify_merge_rules(self):
        new_rules = []
        for cdnf in self.rules:
            merged_rules = True
            new_cdnf = cdnf
            while merged_rules:
                merged_rules = False
                for i in range(len(new_cdnf)):
                    r1 = new_cdnf[i]
                    for j in range(i+1, len(new_cdnf)):
                        r2 = new_cdnf[j]
                        m1, m2 = merge_rules(a=r1, b=r2)
                        if m2 is None:
                            new_cdnf.remove(r1)
                            new_cdnf.remove(r2)
                            new_cdnf.append(m1)
                            merged_rules = True
                            break
                    if merged_rules:
                        break
            new_rules.append(new_cdnf)

        self.__set_rules(new_rules)

    def remove_empirically_redundant_rules(self, X, min_complexity=True):
        assert self.tie_break != "random"
        _tie_break = self.tie_break
        self.tie_break = "first"

        _pred_before = self(X)
        rules_before = deepcopy(self.rules)

        reduced_rules = []

        _list_removed_clauses = []
        for _, class_dnf in enumerate(self.rules):
            removed_supports = []
            supports = []
            for clause in class_dnf:
                # class dnf consists of multiple conjuctive clauses
                applicability = [self.__clause_match_sample(x, clause) for x in X]
                supports.append(set(np.argwhere(applicability).reshape(-1)))

            reduced_class_rules = deepcopy(class_dnf)
            # if support SA is subset of support SB, remove rule corresponding to SA
            # if support SA == SB, remove longer rule if min_complexity else remove less specific rule/ keep both
            removed_clause = True
            while removed_clause:
                removed_clause = False
                for i in range(len(reduced_class_rules)-1):
                    clause1, s1 = reduced_class_rules[i], supports[i]

                    if len(s1) == 0:  # remove rule that doesn't have support
                        removed_supports.append(deepcopy(s1))
                        _list_removed_clauses.append(deepcopy(clause1))
                        del reduced_class_rules[i]
                        del supports[i]
                        removed_clause = True; break

                    for j in range(i+1, len(reduced_class_rules)):
                        clause2, s2 = reduced_class_rules[j], supports[j]
                        if s1 == s2:  # if support sets are equal

                            if min_complexity:  # remove longer rule if we prioritize complexity > completeness
                                if len(clause2) < len(clause1):
                                    removed_supports.append(deepcopy(s1))
                                    _list_removed_clauses.append(deepcopy(clause1))
                                    del reduced_class_rules[i]
                                    del supports[i]
                                else:
                                    removed_supports.append(deepcopy(s2))
                                    _list_removed_clauses.append(deepcopy(clause2))
                                    applicable = []
                                    applicable_by_clause = None
                                    if not b_need_applicability_by_clause:
                                        # use faster version
                                        for _class_rules in self.rules:
                                            #
                                            applicable.append(
                                                self.__rule_match_sample(sample, _class_rules)
                                            )
                                    else:
                                        applicable_by_clause = []
                                        for _class_rules in self.rules:
                                            applicable_by_clause.append([])
                                            for clause in _class_rules:
                                                applicable_by_clause[-1].append(
                                                    self.__clause_match_sample(sample=sample, clause=clause))
                                            if np.any(applicable_by_clause[-1]):
                                                applicable.append(1)
                                            else:
                                                applicable.append(0)
                                    del supports[j]
                                    del reduced_class_rules[j]
                                removed_clause = True; break
                            else:  # keep more specific rule (copmleteness) or keep both if they use different dims
                                _clause_dims = lambda _clause: set(np.unique([t[0] for t in _clause]))
                                cd1, cd2 = _clause_dims(clause1), _clause_dims(clause2)
                                if cd1 <= cd2:  # clause2 more specific than clause 1
                                    removed_supports.append(deepcopy(s1))
                                    _list_removed_clauses.append(deepcopy(clause1))
                                    del reduced_class_rules[i]
                                    del supports[i]
                                    removed_clause = True; break
                                elif cd2 <= cd1:  # clause1 more specific than clause2
                                    removed_supports.append(deepcopy(s2))
                                    _list_removed_clauses.append(deepcopy(clause2))
                                    del supports[j]
                                    del reduced_class_rules[j]
                                    removed_clause = True; break
                                else:
                                    pass  # keep both because they map to different dims

                        elif s1 < s2:  # if s1 is contained in s2, remove clause1
                            removed_supports.append(deepcopy(s1))
                            _list_removed_clauses.append(deepcopy(clause1))
                            del reduced_class_rules[i]
                            del supports[i]
                            removed_clause = True; break
                        elif s2 < s1:  # if s2 is contained in s1, remove s2
                            removed_supports.append(deepcopy(s2))
                            _list_removed_clauses.append(deepcopy(clause2))
                            del supports[j]
                            del reduced_class_rules[j]
                            removed_clause = True; break

                        if removed_clause:
                            break
                    #
                    if removed_clause:
                        # start iteration over reduced_class_rules again after list changed
                        break

            _supports_left = set().union(*supports)
            _supports_removed = set().union(*removed_supports)
            assert len(_supports_removed - _supports_left) == 0  # assert we don't "lose" a sample
            reduced_rules.append(reduced_class_rules)

        n_literals_before = self.n_literals
        n_rules_before = self.n_rules
        self.__set_rules(deepcopy(reduced_rules))
        # print(f"removed redundant rules")
        # print(f"n_literals {n_literals_before} -> {self.n_literals}")
        # print(f"n_rules {n_rules_before} -> {self.n_rules}")
        _pred_after = self(X)
        _after_eq_before = np.all(_pred_before == _pred_after)
        # if not np.all(_pred_before == _pred_after):
        #     print(f"what happened")
        assert np.all(_pred_before == _pred_after)
        n_literals_before_simplification = self.n_literals
        n_rules_before_simplification = self.n_rules
        self.simplify_merge_rules()
        _pred_after_simplified = self(X)
        _simplified_eq_before = np.all(_pred_after_simplified == _pred_before)
        _simplified_eq_after = np.all(_pred_after_simplified == _pred_after)
        # if n_literals_before_simplification != self.n_literals:
        #     print(f"simplified rules")
        #     print(f"n_literals {n_literals_before_simplification} -> {self.n_literals}")
        #     print(f"n_rules {n_rules_before_simplification} -> {self.n_rules}")
        # print()
        # if not np.all(_pred_before == _pred_after_simplified):
        #     print(f"simplifying changed behavior")
        assert np.all(_pred_before == _pred_after_simplified)

        self.tie_break = _tie_break 
        return None

    def __rule_match_sample(self, sample, rule):
        # DNF
        for clause in rule:
            matches = True
            for literal in clause:
                dim, (start, end) = literal
                # TODO, checking <= on both sides may 'connect' intervals from different terms but hasn't happened so
                #  far. what _has_ happened is that the data is such that min == max ..
                if not start <= sample[dim] <= end:
                    # clause cannot be fulfilled anymore
                    matches = False
                    break
            # if all literals are fulfilled, return True
            if matches:
                return True
        # no clause returned True, hence return False
        return False

    def __clause_match_sample(self, sample, clause):
        matches = True
        for literal in clause:
            try:
                dim, (start, end) = literal
            except ValueError:
                print(literal)
            # TODO, checking <= on both sides may 'connect' intervals from different terms but hasn't happened so
            #  far. what _has_ happened is that the data is such that min == max ..
            if not start <= sample[dim] <= end:
                # clause cannot be fulfilled anymore
                matches = False
                break
        # if all literals are fulfilled, return True
        return matches

    def __predict_sample(self, sample, explain=False):

        # if explain also return all applicable rules from predicted class

        if self.tie_break not in ["first", "random"] or explain:
            b_need_applicability_by_clause = True
        else:
            b_need_applicability_by_clause = False

        applicable = []
        applicable_by_clause = None
        if not b_need_applicability_by_clause:
            # use faster version
            for _class_rules in self.rules:
                #
                applicable.append(
                    self.__rule_match_sample(sample, _class_rules)
                )
        else:
            applicable_by_clause = []
            for _class_rules in self.rules:
                applicable_by_clause.append([])
                for clause in _class_rules:
                    applicable_by_clause[-1].append(self.__clause_match_sample(sample=sample, clause=clause))
                if np.any(applicable_by_clause[-1]):
                    applicable.append(1)
                else:
                    applicable.append(0)
            pass

        if b_need_applicability_by_clause:
            assert applicable_by_clause is not None

        if self.n_classes == 1:
            if not explain:
                return applicable[0]
            elif applicable[0]:
                c = self.rules[0]
                return (applicable[0], [c[i] for i, a in enumerate(applicable_by_clause[0]) if a])
            else:
                return (False, None)

        if not np.any(applicable):  # reject
            prediction = -1
        else:
            # if multiple, choose first class that matches
            prediction = np.argwhere(applicable).squeeze()
            if prediction.size == 1:  # only one class predicted
                prediction = prediction.item()

            elif self.tie_break == "first":
                prediction = prediction[0]

            elif self.tie_break == "accuracy":  # or f1
                _best_class, _best_acc = prediction[0], -1
                for p in prediction:
                    p_applicable = applicable_by_clause[p]
                    for i, a in enumerate(p_applicable):
                        if not a:
                            continue
                        r = self.rules[p][i]
                        acc = self.rule_performances[p][tuple(r)]['accuracy']
                        if acc > _best_acc:
                            _best_class, _best_acc = p, acc
                prediction = _best_class

            elif self.tie_break == "f1":  # or f1
                _best_class, _best_acc = prediction[0], -1
                for p in prediction:
                    p_applicable = applicable_by_clause[p]
                    for i, a in enumerate(p_applicable):
                        if not a:
                            continue
                        r = self.rules[p][i]
                        acc = self.rule_performances[p][set(r)]['f1']
                        if acc > _best_acc:
                            _best_class, _best_acc = p, acc
                prediction = _best_class

            elif self.tie_break == "shortest":
                # TODO: write __rule_match_sample function that returns
                #  indicating for each term in class DNF if it matches
                # shortest: choose class with fewest rules matching "shortest explanation"
                raise NotImplementedError
            elif self.tie_break == "longest":
                # longest: choose class with the most number of rules matching "most specific/ certain(?) expl"
                raise NotImplementedError
            elif self.tie_break == "random": # self.tie_break = "random":
                prediction = np.random.choice(prediction, 1).item()#prediction[0]
            else:
                raise ValueError

        if explain:
            if isinstance(prediction, int) and prediction == -1:
                return (-1, None)

            _applicable_terms = [self.rules[prediction][i] for i, a in enumerate(applicable_by_clause[prediction]) if a]
            prediction = (prediction, _applicable_terms)

        return prediction

    def predict(self, samples, explain=False):
        predictions = []
        for sample in samples:
            predictions.append(self.__predict_sample(sample, explain))
        if not explain:
            return np.array(predictions)
        else:
            return predictions

    def get_num_rules(self):
        n_rules = []
        for c in range(len(self.rules)):
            n_rules.append(len(self.rules[c]))
        return n_rules

    def score_recall(self, X, Y):
        raise NotImplementedError

    def __describe_sample(self, sample, track_usage=True):
        # returns a list of lists of rules
        # index corresponds to class
        # each list holds all rules that were applicable from that class

        applicable_by_clause = []
        for c, _class_rules in enumerate(self.rules):
            applicable_by_clause.append([])
            for clause in _class_rules:
                if self.__clause_match_sample(sample=sample, clause=clause):
                    applicable_by_clause[-1].append(clause)
                    if track_usage:
                        self.rule_usage_stats[c][tuple(clause)] += 1
        return applicable_by_clause

    def describe(self, samples):
        dscriptions = []
        for sample in samples:
            dscriptions.append(self.__describe_sample(sample))
        return dscriptions


    def __literal_in_clause(self, literal, clause):
        for lc in clause:
            if literal[0] == lc[0]:
                if literal[1][0] >= lc[1][0] and literal[1][1] <= lc[1][1]:
                    return True
        return False
    def __clause_contained(self, clause1, clause2):
        for literal in clause1:
            if not self.__literal_in_clause(literal, clause2):
                return False
        return True

    def __dnf_contained(self, dnf1, dnf2):
        contained, diff = [], []
        for clause in dnf1:
            if np.any([self.__clause_contained(clause, clause2) for clause2 in dnf2]):
                contained.append(clause)
            else:
                diff.append(clause)
        return (contained, diff)

    def __contrast_classes(self, class1, class2):
        dnf1 = self.rules[class1]
        dnf2 = self.rules[class2]
        raise NotImplementedError

    def covers(self, sample):
        for cdnf in self.rules:
            for clause in cdnf:
                if self.__rule_match_sample(sample=sample, clause=clause):
                    return True
        return False


    @staticmethod
    def from_DT(dt: sklearn.tree.DecisionTreeClassifier, verbose=False):

        def __intervals_plausible(i1: tuple[int, tuple[float, float]],
                                  i2: tuple[int, [float, float]]) -> bool:
            if i1[0] != i2[0]: return True # if they apply to different dimensions
            interval1, interval2 = sorted([i1[1], i2[1]], key=lambda i: i[0])
            return interval1[1] > interval2[0]


        def __term_is_plausible(term: list[tuple[int, tuple[float, float]]]):
            '''
            checks if dnf (for one class) is plausible,
            ie that every term comprises of non-contradictory literals
            '''
            for literal in term:  # assert all literals have valid intervals where min < max
                if not literal[1][0] <= literal[1][1]: return False
            if len(term) == 1: return True
            # # all terms apply to the same dimension
            # assert len(np.unique([term[0] for term in terms])) == 1
            # assert that all terms intersect/ have partial overlap
            for ta, tb in combinations(term, 2):
                if not __intervals_plausible(ta, tb):
                    return False
            return True

        def get_path(leaf, l, r, remove_leaf=False):
            # given leaf node, walk up the binary tree until root, return path (optional: remove the leaf)
            def get_parent(child, _l, _r):
                if child in _l:
                    return np.argwhere(_l == child).squeeze().item()
                else:
                    return np.argwhere(_r == child).squeeze().item()

            current_node = leaf
            pth = [current_node]
            while current_node != 0:  # 0 is root_id
                current_node = get_parent(current_node, l, r)
                pth.append(current_node)
            pth = pth[::-1]
            if remove_leaf:
                return pth[:-1]
            else:
                return pth

        def path_to_dnf(path, dims, thresh, _lr):
            # given nodes on a path, their thresholds and information about direction left (lt) or right (gr)
            # convert path to dnf with (dim, (interval start, interval end))
            # where start==-np.inf if node is left from parent and end==np.inf if node is right of parent
            _dnf = []
            for idx, node in enumerate(path[:-1]):
                lr = _lr[path[idx + 1]]  # look up if child is left or right -> leq or gr
                term = None
                if lr == 'l':
                    term = (dims[node], (-np.inf, thresh[node]))
                else:
                    term = (dims[node], (thresh[node], np.inf))
                _dnf.append(term)
            _dnf = sorted(_dnf, key=lambda x: x[0])
            assert __term_is_plausible(_dnf)
            return _dnf

        def prune_dnf(term, _lr):
            # given a single dnf as [rule_1, rule_2, rule_3] w/ rule = (dim, (inverval start, interval end))
            # merge two rules if they operate on the same dim.
            # take the larger number if interval is open to the left (-np.inf),
            # take smaller number if interval is open to the right (np.inf)
            # merge rules [(d, (-np.inf, a)), (d, (b, np.inf))] -> (d (b, a))
            # because dnf stems from one path, the case where b > a should not occur, we assert nonetheless
            pruned_dnf = []
            dims = np.unique([t[0] for t in term])
            for dim in dims:
                literals = [t for t in term if t[0] == dim]
                if len(literals) <= 1:
                    pruned_dnf.extend(literals)
                    continue
                else:
                    lmax = max([l[1][0] for l in literals])
                    rmin = min([l[1][1] for l in literals])
                    # lmin = min([t[1][0] if t[1][0] != -np.inf else np.inf for t in literals])  # gr
                    # if lmin == np.inf:  # np.isnan(lmin): # lmin == np.nan is always False
                    #     lmin = -np.inf
                    # # rmax = max([t[1][1] if t[1][1] != np.inf else np.nan for t in _terms])
                    # rmax = max([t[1][1] if t[1][1] != np.inf else -np.inf for t in literals])
                    # if rmax == -np.inf:
                    #     rmax = np.inf
                    assert rmin >= lmax
                    assert rmin != -np.inf or lmax != np.inf
                    lit = (dim, (lmax, rmin))
                    pruned_dnf.append(lit)

            return pruned_dnf


        t = dt.tree_
        left = t.children_left
        right = t.children_right
        n_nodes = t.capacity
        _lr = ['l' if i in left else 'r' for i in range(n_nodes)]
        assert n_nodes == len(left)
        dims = [np.nan if left[i] == -1 or right[i] == -1 else t.feature[i] for i in range(n_nodes)]
        thresh = [np.nan if left[i] == -1 or right[i] == -1 else t.threshold[i] for i in range(n_nodes)]
        is_leaf = [True if left[i] == right[i] == -1 else False for i in range(n_nodes)]

        # https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/tree/_classes.py#L507
        _leaf_class = np.array([np.argmax(t.value[i]) if is_leaf[i] else -1 for i in range(n_nodes)])

        # assert len(np.unique(_leaf_class)) == len(dt.classes_) + 1  # at least one leaf per class + 1? make no sense

        paths_by_class = [[get_path(leaf, left, right) for leaf in np.argwhere(_leaf_class == j).reshape(-1)]
                          for j in dt.classes_]
        if verbose:
            print("TREE INFO")
            print(f"leafs:{[i for i, j in enumerate(is_leaf) if j]}")
            print("node, dim, thesh")
            for n, th, d, il in zip(range(n_nodes), thresh, dims, is_leaf):
                if not il:
                    print(f"{n}, {d}, {th:.4f}")

            print("\n\nPATHS")
            for i, p in enumerate(paths_by_class):
                print(f"class {i}")
                for pp in p:
                    print(pp)
                print()
        paths_by_class = [[p for p in pc] for pc in paths_by_class] # [:-1] to remove leaf node because we know the class already
        dnf = [[path_to_dnf(p, dims, thresh, _lr) for p in path_class] for path_class in paths_by_class]

        # pruned_dnf = [[prune_dnf(__dnf, _lr) for __dnf in cdnf if __dnf != []] for cdnf in dnf]
        # if verbose:
        #     print("DNFs")
        #     print(f"before \n ->  pruned")
        #     for i in range(len(dnf)):
        #         d = dnf[i]
        #         pd = pruned_dnf[i]
        #         print(f"class {i}")
        #         for dd, pd in zip(d, pd):
        #             print(f"{dd} \n -> {pd}")
        #         print()

        # _DNF = DNFClassifier(pruned_dnf)
        _DNF = DNFClassifier(dnf)
        # _DNF.simplify()
        return _DNF

def make_ff(shapes_layers, actfun_out=None):
    # shapes layers: output size layer is input size of next ..
    layers = [nn.Linear(s_in, s_out) for (s_in, s_out) in zip(shapes_layers[:-1], shapes_layers[1:])]
    actfun = nn.ReLU
    architecture = []
    for layer in layers:
        architecture.append(layer)
        architecture.append(actfun())
    architecture = architecture[:-1] # delete last actfun
    if actfun_out is not None:
        architecture.append(actfun())
    sequential = nn.Sequential(*architecture)
    return SimpleNet(sequential=sequential)

class SimpleNet(nn.Module):
    def __init__(self, sequential):
        super(SimpleNet, self).__init__()
        self.net = sequential
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return x

    def predict_batch(self, x):
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

    def predict_batch_softmax(self, x):
        pred = self.forward(x)
        sm_pred = self.softmax(pred)
        return sm_pred
