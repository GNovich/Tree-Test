import tqdm
import numpy as np
import ete3
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rv_discrete
from scipy.spatial.distance import squareform
import pickle
import os

class Tree_pickler():
    tree_name = 'tree'
    attr_name = 'dat'

    def saveTree(self, tree, save_dir, attr_list, sp_list):
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        # save newick
        tree.write(outfile=os.path.join(save_dir, self.tree_name), format=1)
        # save node data
        for sp_attr in sp_list:
            sp_dat = sp.vstack([getattr(n, sp_attr) for n in tree.traverse()])
            sp.save_npz(os.path.join(save_dir, sp_attr), sp_dat)
        # save other attr
        to_dump = {attr_: getattr(tree, attr_) for attr_ in attr_list}
        pickle.dump(to_dump, open(os.path.join(save_dir, self.attr_name), 'wb'))

    def loadTree(self, save_dir):
        # load newick
        t = ete3.Tree(os.path.join(save_dir, self.tree_name))

        # load node data
        sp_list = [base for base in os.listdir(save_dir) if '.npz' in base]
        for sp_attr in sp_list:
            attr_name = sp_attr.strip('.npz')
            sparse_dat = sp.load_npz(open(os.path.join(save_dir, sp_attr), 'rb'))
            for ind, n in enumerate(t.traverse()):
                setattr(n, attr_name, sparse_dat[ind])

        # load other attr
        attr_dict = pickle.load(open(os.path.join(save_dir, self.attr_name), 'rb'))
        for key in attr_dict:
            setattr(t, key, attr_dict[key])
        return t


def MaxParsimony(Tree, naming=False):
    """
    Reconstruct ancestral characters using MaxParsimony / occam's razor logic...
    Note this implementation relays on binary characters

    The first pass extracts info from the leaves upwards with the following function:
    Intersect when an intersection is available, otherwise yeild union.
    e.g given dijointed sets A,B

         AUB   |      A     |      B
         /^\   |     /^\    |     /^\
        A  B   |    A  AUB  |    B  B

    The second pass brings stable information from root to leaves.
    The root might need randomization to deside on a character.
    We consider events influenced by randomization as risky, and they are documented in 'random'.
    A parent will always be a single character, and a son might get updated by intersection with parent node:

         A     >      A     |     A     >      A     |
        /^\    >     /^\    |    /^\    >     /^\    |
       A  AUB  >    A  A    |   A  B    >    A   B    |

    :param X: sparse csr_mat data to reconstruct
    :param Tree: ete3 tree
    :param tip_row_dict: dict mapping leaf names to row number in X
    :return: A tree where each node has a
            .dat containing the reconstruction by X
            .random documenting non-determinant values
    """
    # 2 represents {0,1} set
    sp_to_arr = lambda sp_arr: np.array(sp_arr.todense().astype(np.int8))[0]
    tree_len = 0
    for _ in Tree.traverse(): tree_len += 1
    for i, node in tqdm.tqdm(enumerate(Tree.traverse('postorder')), total=tree_len,
                             desc='Ancestral Reconstruction: 1st pass'):
        if node.is_leaf():
            continue
        if naming: node.name = i
        children = [sp_to_arr(c.dat) for c in node.children]
        res = children[0].copy()
        eq = np.equal(*children)
        res[children[0] == 2] = children[1][children[0] == 2]  # 2 is the union {0,1}
        res[children[1] == 2] = children[0][children[1] == 2]
        res[(children[0] != 2) & (children[1] != 2) & ~eq] = 2
        node.dat = sp.csr_matrix(res)

    post = Tree.traverse('preorder')
    root = next(post)
    root.random = (sp_to_arr(root.dat) == 2)
    root.dat[root.dat == 2] = np.random.choice([1, 0], size=(root.dat == 2).sum())
    for node in tqdm.tqdm(post, total=tree_len - 1, desc='Ancestral Reconstruction: 2nd pass'):
        if node.is_leaf(): continue
        parent_ = sp_to_arr(node.up.dat)
        node_ = sp_to_arr(node.dat)
        res = node_.copy()
        res[node_ == 2] = parent_[node_ == 2]
        node.random = (node.up.random) & (node_ == 2)  # these are unstable positions
        node.dat = sp.csr_matrix(res)

    return Tree


def GetChangeRates(tree):
    """
    Get distribution of homoplasy
    """
    population = (len(tree) * 2) - 1
    order = tree.traverse()
    root = next(order)
    homoplasy_hist = sp.csr_matrix(np.zeros(root.dat.shape[1]))
    for cur_node in tqdm.tqdm(order, total=population + 1):
        homoplasy_hist += (cur_node.dat[0, :] != cur_node.up.dat[0, :])
    homoplasy_dist = pd.Series(homoplasy_hist.data).value_counts(normalize=True)
    fit_dist = rv_discrete(values=(homoplasy_dist.index.astype(int), homoplasy_dist.values.astype(float)))
    root.homoplasy = fit_dist
    root.homoplasy_hist = homoplasy_hist.toarray()[0].astype(np.int)
    return tree


def GetPhyloParallelScoreMat(tree):
    """
    This code utilizes the sparse matrix multiplication to calculate pairwise scores between all traits

    :param tree:
    :return:
    """
    population = tree.size
    score = None

    for i, cur_node in tqdm.tqdm(enumerate(tree.traverse()), total=population, desc='Score calc'):
        if not cur_node.is_root():

            state = (cur_node.dat > 0) * 1
            prev_state = (cur_node.up.dat > 0) * 1
            event = (prev_state - state)
            event = event.multiply(event.T)
            if score is None:
                score = event
            else:
                score += event

    score = score.toarray()
    # the diagonal is a "variance score"... might usefull in the future. needs to be zero right now for squareform.
    np.fill_diagonal(score, 0)
    return squareform(score).astype(np.int)


def my_direct_para_pval(p_rates, scores, population, branch_prop=None):
    p_rates, inver = np.unique(p_rates, return_inverse=True)

    res = []
    for curr_rate in tqdm.tqdm(p_rates, total=len(p_rates), desc='individual pvals conv'):
        x_ = np.array([.5 * curr_rate, 1 - curr_rate, .5 * curr_rate])
        x = x_
        if branch_prop:
            for prop in branch_prop:
                branch_rate = curr_rate * prop
                x_ = np.array([.5 * branch_rate, 1 - branch_rate, .5 * branch_rate])
                x = np.convolve(x, x_)
        else:
            i = 1
            while i ** 2 < population:
                x = np.convolve(x, x)
                i *= 2
            while i < population:
                x = np.convolve(x, x_)
                i += 1
        z_ = np.argmax(x)
        x = x[z_:]
        x[1:] *= 2
        x = np.cumsum(x[::-1])[::-1]
        res.append(x)
    res = np.array(res)
    return res.take((inver * res.shape[1] + scores))


def get_tables(tree):
    """

    :param tree: make sure 'dat', 'homoplasy_hist', and 'size' are fileds for each node
    :return:
    """
    # get pvals
    ind_rates = (tree.homoplasy_hist / tree.size)
    rate_mat = np.outer(ind_rates, ind_rates)
    np.fill_diagonal(rate_mat, 0)
    rate_mat = squareform(rate_mat)

    table_summary = pd.DataFrame()
    table_summary['Covariance_Score'] = GetPhyloParallelScoreMat(tree)
    table_summary['pval'] = my_direct_para_pval(rate_mat, abs(table_summary['Covariance_Score']), tree.size)

    ind_a = []
    ind_b = []
    trait_num = tree.dat.shape[-1]
    for i in range(trait_num-1):
        ind_a.extend([i]*(trait_num-i-1))
        ind_b.extend(list(range(i+1, trait_num)))
    table_summary['ind_a'] = ind_a
    table_summary['ind_b'] = ind_b

    return table_summary


def GetPhyloParallelScore(tree, with_rand=True):
    """
    Collects co-occuring events along the phylogeny

                                 /--(1,1)
                       /--(1,1)-|
             /--(0,0)-|          \--(0,0)
    --(0,1)-|          \--(1,0)
             \--(1,0)

    Would translate to the following score:

    sum((x_par - x_des)(y_par - y_des) for [(x_par,y_par), (x_des,y_des)] in branches)

    In this case: -1 + 1 + 1 = 1

    :param tree: ete3 tree
    :param phen_ind: index of target T
    :param skip: number of qualities to skip
    :return: A contingency table for each available quality in a nodes 'dat' value
    """

    population = tree.size if 'size' in vars(tree) else (len(tree) * 2) - 1
    score = np.zeros(tree.dat.shape[1])
    node_to_arr = lambda n: np.array(n.dat.todense().astype(np.int))[0]

    for i, cur_node in tqdm.tqdm(enumerate(tree.traverse()), total=population, desc='Iterating tree'):
        if not cur_node.is_root():
            if not cur_node.is_leaf() and with_rand and cur_node.random[0]: continue
            gene_state = node_to_arr(cur_node)[1:]
            prev_gene_state = node_to_arr(cur_node.up)[1:]

            phen_state = cur_node.dat[0, 0]
            prev_phen_state = cur_node.up.dat[0, 0]

            phen_event = (prev_phen_state - phen_state)
            gene_event = (prev_gene_state - gene_state)

            if with_rand: gene_event[cur_node.up.random] = 0
            score += phen_event * gene_event

    return score.astype(np.int)


def my_direct_ind_pval(rates_a, rate_b, scores, population, branch_prop=None):
    rates_a, inver = np.unique(rates_a, return_inverse=True)
    p_rates = rates_a * rate_b

    res = []
    for curr_rate in tqdm.tqdm(p_rates, total=len(p_rates), desc='individual pvals conv'):
        x_ = np.array([.5 * curr_rate, 1 - curr_rate, .5 * curr_rate])
        x = x_
        if branch_prop:
            for prop in branch_prop:
                branch_rate = curr_rate * prop
                x_ = np.array([.5 * branch_rate, 1 - branch_rate, .5 * branch_rate])
                x = np.convolve(x, x_)
        else:
            i = 1
            while i ** 2 < population:
                x = np.convolve(x, x)
                i *= 2
            while i < population:
                x = np.convolve(x, x_)
                i += 1
        z_ = np.argmax(x)
        x = x[z_:]
        x[1:] *= 2
        x = np.cumsum(x[::-1])[::-1]
        res.append(x)
    res = np.array(res)
    return res.take((inver * res.shape[1] + scores))


def get_target_first(tree):
    ind_rates = (tree.homoplasy_hist[1:] / tree.size)
    target_rate = tree.homoplasy_hist[0] / tree.size

    table_summary = pd.DataFrame()
    table_summary['Covariance_Score'] = GetPhyloParallelScore(tree)
    table_summary['pval'] = my_direct_ind_pval(ind_rates, target_rate, abs(table_summary['Covariance_Score']), tree.size)
    return table_summary


def Tree_Test(newick, SparseTable, just_first=False, Tree_save_path=None, Tree_pickle_path=None, Reconstruct=False):
    """
    :param newick: newick
    :param SparseTable: csr sparse binary table of with M traits: NxM
                        each row represents the matching node under *inorder* traversal of the tree
                        if Reconstruct is True, asuming M is the number of leaves
    :param just_first: is the first trait the target? if False - all traits are targets
    :param Tree_pickle_path: load path for an existing tree with data
    :param Tree_save_path: save path for computed tree, will create a directory storing tree files
    :param Reconstruct: use maximum parsimony to estimate internal nodes

    :return: pandas summery table of score and p-value
    """

    # load tree
    if Tree_pickle_path is not None:
        pickler = Tree_pickler()
        t = pickler.loadTree(Tree_pickle_path)
    else:
        # load tree
        t = ete3.Tree(newick)
        for ind in enumerate(t.traverse()): pass
        t.size = ind

        # load data
        for ind, n in enumerate(t if Reconstruct else t.traverse()):
            n.dat = SparseTable[ind]
        if Reconstruct:
            t = MaxParsimony(t)
        # estimate rates
        GetChangeRates(t)

    # Tree-Test
    res = get_target_first(t) if just_first is None else get_tables(t)

    # save tree
    if Tree_save_path is not None:
        pickler = Tree_pickler()
        attr_list = ['size', 'homoplasy', 'homoplasy_hist']
        sp_list = ['dat', 'random']
        pickler.saveTree(t, Tree_save_path, attr_list, sp_list)

    return res
