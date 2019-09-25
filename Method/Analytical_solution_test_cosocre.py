# coding: utf-8

# In[1]:

from Alice_Macros import *
import matplotlib.pylab as plt
from rpy2 import robjects
import numpy as np
from numpy.random import binomial
from scipy import sparse as sp
from scipy.stats import binom, poisson, rv_discrete, pearsonr, spearmanr
from scipy.stats.distributions import hypergeom, norm
import os
import bisect
import pickle
import sys
import ete3
import tqdm
import itertools
from scipy.stats import rv_histogram
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
statspackage = importr('stats', robject_translations={'format_perc': '_format_perc'})


def my_norm(rate_a, rate_b, population):
    expected_var = (rate_a * rate_b) * (1 - (rate_a * rate_b))
    expected_std = np.sqrt(expected_var) * np.sqrt(population)
    return norm(0, expected_std)


def my_direct(rate_a, rate_b, population, branch_prop=None):
    # TODO could optimize space here, cut tails of x when pval == 0
    # TODO [0, ..., 0, 1e-300, ..., 1, ..., 1e-300, 0, ..., 0] - > [0, 1e-300, ..., 1, ..., 1e-300, 0]
    # TODO is there a cheaper way to calculate a convolution between 2 symetric veriables?
    tot_rate = rate_a * rate_b
    x_ = [1 - tot_rate, tot_rate]
    x = x_
    if branch_prop:
        for prop in branch_prop:
            branch_rate = tot_rate * prop
            x_ = np.array([1 - branch_rate, branch_rate])
            x = np.convolve(x, x_)
    else:
        i = 1
        while i ** 2 < population:
            x = np.convolve(x, x)
            i *= 2
        while i < population:
            x = np.convolve(x, x_)
            i += 1

    pvals_sf = dict(zip(range(len(x)), np.concatenate([[0], np.cumsum(x[::-1])])[::-1]))
    tail = np.vectorize(lambda y: pvals_sf[bisect.bisect_left(range(len(pvals_sf)), y)])
    return tail


def my_direct_ind_pval(rates_a, rate_b, scores, population, branch_prop=None):
    rates_a, inver = np.unique(rates_a, return_inverse=True)
    p_rates = rates_a * rate_b

    res = []
    for curr_rate in tqdm.tqdm(p_rates, total=len(p_rates), desc='individual pvals conv'):
        x_ = np.array([1 - curr_rate, curr_rate])
        x = x_
        if branch_prop:
            for prop in branch_prop:
                branch_rate = curr_rate * prop
                x_ = np.array([1 - branch_rate, branch_rate])
                x = np.convolve(x, x_)
        else:
            i = 1
            while i ** 2 < population:
                x = np.convolve(x, x)
                i *= 2
            while i < population:
                x = np.convolve(x, x_)
                i += 1
        x = np.cumsum(x[::-1])[::-1]
        res.append(x)
    res = np.array(res)
    return res.take((inver * res.shape[1] + scores))


def left_tail_(samples):
    val, count = np.unique(samples, return_counts=True)
    count = count / len(samples)
    pval_sf_list = np.concatenate([[0], np.cumsum(count)])
    pval = np.vectorize(lambda x: pval_sf_list[bisect.bisect_right(val, x)])
    return pval


def right_tail_(samples):
    val, count = np.unique(samples, return_counts=True)
    count = count / len(samples)
    pval_sf_list = np.concatenate([[0], np.cumsum(count[::-1])])[::-1]
    pval = np.vectorize(lambda x: pval_sf_list[bisect.bisect_left(val, x)])
    return pval


def GenerateTree(ntax, birth_rate, death_rate, SaveDir=None):
    t = ete3.Tree(
        robjects.r("ape::write.tree(ape::rphylo({},{},{},fossils=FALSE))".format(ntax, birth_rate, death_rate))[0])
    for i, node in enumerate(t.traverse()): node.name = str(i)
    if SaveDir:
        if not os.path.exists(SaveDir): os.mkdir(SaveDir)
        pickle.dump(t, open(os.path.join(SaveDir, "tree"), 'wb'))
    return t


def RandomAtributes(tree, population, n_genes, mut_func):
    # set binary atributes across phylogeny
    order = tree.traverse('preorder')
    root = next(order)
    root.genotype = sp.csr_matrix(np.random.randint(2, size=n_genes))
    for node in tqdm.tqdm(order, total=population, desc='propageting events on tree'):
        mut_here = mut_func() if not tree.rand_branches else mut_func(node.dist)
        node.genotype = np.abs(node.up.genotype - sp.csr_matrix(mut_here))
    return tree


def RandomAtributesExact(tree, n_genes, homoplasy=1):
    # set binary atributes across phylogeny
    population = (len(tree) * 2) - 1
    chunks = int(1e5)
    parts = int(n_genes // chunks)
    events_per_mut = []
    lims = None if type(homoplasy) == int else homoplasy.rvs(size=n_genes)
    max_lim = lims.max() if lims is not None else homoplasy
    if tree.rand_branches:
        prob_ = np.array([x.dist for x in tree.traverse()][1:])  # node id are inorder
        prob_ = prob_ / prob_.sum()
        node_ind = np.array([x.id for x in tree.traverse()][1:])
        rand_func = lambda: node_ind[np.random.choice(len(node_ind), size=max_lim, p=prob_, replace=False)]
    for _ in tqdm.tqdm(range(parts), total=parts, desc='simulating events in chucks'):
        if tree.rand_branches:
            # draw proportional to dist
            events_per_mut.append(np.array([rand_func() for _ in range(chunks)]))
        else:
            events_per_mut.append((np.argsort(np.random.rand(chunks, population - 1))[:, :max_lim] + 1))
    events_per_mut = np.concatenate(events_per_mut)
    order = tree.traverse('preorder')
    root = next(order)
    root.genotype = sp.csr_matrix(np.random.randint(2, size=n_genes))
    node_to_arr = lambda n: np.array(n.todense().astype(np.int8))[0]
    for node in tqdm.tqdm(order, total=population, desc='propageting events on tree'):
        if lims is None:
            mut_here = np.any(node.id == events_per_mut, axis=1)
        else:
            mut_here = np.zeros(len(events_per_mut))
            idx_mut = np.argwhere(node.id == events_per_mut)
            idx_mut = idx_mut[idx_mut.T[1] < lims[idx_mut.T[0]]].T[0]
            mut_here[idx_mut] = 1
        node.genotype = sp.csr_matrix(np.abs(node_to_arr(node.up.genotype) - mut_here))
    return tree


def MarginalMaxLiklihood(X, Tree, tip_row_dict, do_tips=False, naming=False, m_rate=0.5):
    # 2 represents {0,1} set
    sp_to_arr = lambda sp_arr: np.array(sp_arr.todense().astype(np.int8))[0]
    wrap = lambda x: sp_to_arr(X[tip_row_dict[x.name]]) if x.is_leaf() and not do_tips else sp_to_arr(x.genotype)
    tree_len = 0
    for _ in Tree.traverse(): tree_len += 1
    for i, node in tqdm.tqdm(enumerate(Tree.traverse('postorder')), total=tree_len,
                             desc='Ancestral Reconstruction: 1st pass'):
        if node.is_leaf():
            if not do_tips:
                node.genotype = X[tip_row_dict[node.name]]
            node.zero_prob = sp.csr_matrix(node.genotype == 0)
            node.one_prob = sp.csr_matrix(node.genotype == 1)
            continue
        if naming: node.name = i
        node.one_prob = ((node.children[0].zero_prob * m_rate) + (node.children[0].one_prob * (1 - m_rate))) + (
                (node.children[1].zero_prob * m_rate) + (node.children[1].one_prob * (1 - m_rate)))
        node.zero_prob = ((node.children[0].zero_prob * (1 - m_rate)) + (node.children[0].one_prob * m_rate)) + (
                (node.children[1].zero_prob * (1 - m_rate)) + (node.children[1].one_prob * m_rate))
        node.genotype = (node.zero_prob < node.one_prob).astype(int)


def JointMaxLiklihood(X, Tree, tip_row_dict, do_tips=False, naming=False, m_rate=0.5):
    # As explained in FastML
    # 2 represents {0,1} set
    sp_to_arr = lambda sp_arr: np.array(sp_arr.todense().astype(np.int8))[0]
    wrap = lambda x: sp_to_arr(X[tip_row_dict[x.name]]) if x.is_leaf() and not do_tips else sp_to_arr(x.genotype)
    tree_len = 0
    for _ in Tree.traverse(): tree_len += 1
    for node in tqdm.tqdm(Tree.traverse('postorder'), total=tree_len, desc='Ancestral Reconstruction: 1st pass'):
        if node.is_leaf():
            """
            For each OTU y perform the following:
            1a. Let j be the amino acid at y. Set, for each amino acid i: Cy(i)= j. 
                This implies that no matter what is the amino acid in the father of y, j is assigned to node y.
            1b. Set for each amino acid i: Ly(i) = Pij(ty), where ty is the branch length between y and its father.
            """
            if not do_tips:
                node.genotype = X[tip_row_dict[node.name]]
            node.C_0 = node.genotype
            node.C_1 = node.genotype
            node.L_0 = np.ones((node.genotype).shape[1]) * (1 - m_rate)
            node.L_0[node.genotype.indices] = m_rate
            node.L_1 = np.ones((node.genotype).shape[1]) * m_rate
            node.L_1[node.genotype.indices] = (1 - m_rate)

            # log
            node.L_1 = np.log(node.L_1)
            node.L_0 = np.log(node.L_0)
            continue

        """
        Visit a nonroot internal node, z, which has not been visited yet, 
        but both of whose sons, nodes x and y, have already been visited, 
        i.e., Lx(j), Cx(j), Ly(j), and Cy(j) have already been defined for each j. 
        Let tz be the length of the branch connecting node z and its father. 
        For each amino acid i, compute Lz(i) and Cz(i) according to the following formulae:

        2a. Lz(i) = maxj Pij(tz)× Lx(j) × Ly(j).
        2b. Cz(i) = the value of j attaining the above maximum.
        """
        s0 = node.children[0].L_0 + node.children[1].L_0
        s1 = node.children[0].L_1 + node.children[1].L_1

        L0_0 = np.log(1 - m_rate) + s0
        L0_1 = np.log(m_rate) + s1
        node.L_0 = np.maximum(L0_0, L0_1)
        node.C_0 = L0_0 < L0_1

        L1_0 = np.log(m_rate) + s0
        L1_1 = np.log(1 - m_rate) + s1
        node.L_1 = np.maximum(L1_0, L1_1)
        node.C_1 = L1_0 < L1_1

        del node.children[0].L_0
        del node.children[1].L_0
        del node.children[0].L_1
        del node.children[1].L_1

    post = Tree.traverse('preorder')
    root = next(post)
    """
    Denote the sons of the root by x, y. 
    For each amino acid k, compute the expression Pk × Lx(k) × Ly(k). 
    Reconstruct r by choosing the amino acid k maximizing this expression. 
    The maximum value found is the likelihood of the best reconstruction.
    """
    root.genotype = sp.csr_matrix(root.L_0 < root.L_1)
    if naming: root.name = 0
    for i, node in tqdm.tqdm(enumerate(post), total=tree_len, desc='Ancestral Reconstruction: 2nd pass'):
        """
        Traverse the tree from the root in the direction of the OTUs, 
        assigning to each node its most likely ancestral character as follows:
        
        5a. Visit an unreconstructed internal node x whose father y has already been reconstructed. 
            Denote by i the reconstructed amino acid at node y.
        5b. Reconstruct node x by choosing Cx(i).
        5c. Return to step 5a until all internal nodes have already been reconstructed.
        """
        if not node.is_leaf():
            if naming: node.name = i + 1
            node.genotype = sp.csr_matrix(node.C_0)
            node.genotype[0, node.up.genotype.indices] = node.C_1[node.up.genotype.indices]
            # np.where(node.up.genotype, node.C_1, node.C_0)
        del node.C_0, node.C_1

    return Tree


def MaxParsimonyNoTable(X, Tree, tip_row_dict, do_tips=False, naming=False):
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
            .genotype containing the reconstruction by X
            .random documenting non-determinant values
    """
    # 2 represents {0,1} set
    sp_to_arr = lambda sp_arr: np.array(sp_arr.todense().astype(np.int8))[0]
    wrap = lambda x: sp_to_arr(X[tip_row_dict[x.name]]) if x.is_leaf() and not do_tips else sp_to_arr(x.genotype)
    tree_len = 0
    for _ in Tree.traverse(): tree_len += 1
    for i, node in tqdm.tqdm(enumerate(Tree.traverse('postorder')), total=tree_len,
                             desc='Ancestral Reconstruction: 1st pass'):
        if node.is_leaf():
            if not do_tips:
                node.genotype = X[tip_row_dict[node.name]]
            continue
        if naming: node.name = i
        children = [wrap(c) for c in node.children]
        res = children[0].copy()
        eq = np.equal(*children)
        res[children[0] == 2] = children[1][children[0] == 2]  # 2 is the union {0,1}
        res[children[1] == 2] = children[0][children[1] == 2]
        res[(children[0] != 2) & (children[1] != 2) & ~eq] = 2
        node.genotype = sp.csr_matrix(res)

    post = Tree.traverse('preorder')
    root = next(post)
    root.random = (wrap(root) == 2)
    root.genotype[root.genotype == 2] = np.random.choice([1, 0], size=(root.genotype == 2).sum())
    for node in tqdm.tqdm(post, total=tree_len - 1, desc='Ancestral Reconstruction: 2nd pass'):
        if node.is_leaf(): continue
        parent_ = wrap(node.up)
        node_ = wrap(node)
        res = node_.copy()
        res[node_ == 2] = parent_[node_ == 2]
        node.random = (node.up.random) & (node_ == 2)  # these are unstable positions - will not be counted
        node.genotype = sp.csr_matrix(res)

    return Tree


def GetPhyloSebsequentScore(tree, phenotree, phen_ind, skip=0, with_rand=False, dist_only=False, dist=None):
    """
    Collects co-occuring events along the phylogeny
    
                                 /--(1,1)
                       /--(0,1)-|
             /--(0,1)-|          \--(0,1)
    --(0,1)-|          \--(1,1)
             \--(0,1)

    Would translate to the following contingency table:

    T\Q| 0  1
    ---------
     0 | 4  1
     1 | 1  0

    :param tree: ete3 tree
    :param phen_ind: index of target T
    :param skip: number of qualities to skip
    :return: A contingency table for each available quality in a nodes 'genotype' value
    """
    population = (len(tree) * 2) - 1
    subscore = np.zeros(tree.genotype.shape[1] - skip)
    node_to_arr = lambda n: np.array(n.genotype.todense().astype(np.int))[0]
    for i, (cur_node, phen_node) in tqdm.tqdm(enumerate(zip(tree.traverse(), phenotree.traverse())),
                                              total=population, desc='Iterating tree'):
        if not cur_node.is_root():
            if not cur_node.is_leaf() and with_rand and cur_node.random[phen_ind]: continue
            node = node_to_arr(cur_node)
            prev_node = node_to_arr(cur_node.up)

            gene_state = node[skip:]
            prev_gene_state = prev_node[skip:]

            phen_state = phen_node.genotype[0, phen_ind]
            prev_phen_state = phen_node.up.genotype[0, phen_ind]

            subscore += np.abs((1.333 * prev_phen_state * prev_gene_state) +
                               (.666 * prev_phen_state * gene_state) +
                               (.666 * phen_state * prev_gene_state) +
                               (1.333 * phen_state * gene_state) -
                               phen_state -
                               prev_phen_state -
                               gene_state -
                               prev_gene_state +
                               1)

    if dist_only:
        hist_ = np.histogram(subscore, bins=int(1e7))
        fit_dist = rv_histogram(hist_)
        fit_dist.bin = np.diff(hist_[1]).max()
        return fit_dist
    if dist is not None:
        return dist.sf(subscore)
    else:
        return subscore


def GetPhyloCoEventScore(tree, phenotree, phen_ind, skip=0, with_rand=False):
    """
    Collects co-occuring events along the phylogeny

                                 /--(1,1)
                       /--(0,1)-|
             /--(0,1)-|          \--(0,1)
    --(0,1)-|          \--(1,1)
             \--(0,1)

    Would translate to the following contingency table:

    T\Q| 0  1
    ---------
     0 | 4  1
     1 | 1  0

    :param tree: ete3 tree
    :param phen_ind: index of target T
    :param skip: number of qualities to skip
    :return: A contingency table for each available quality in a nodes 'genotype' value
    """

    population = (len(tree) * 2) - 1
    score = np.zeros(tree.genotype.shape[1] - skip)
    node_to_arr = lambda n: np.array(n.genotype.todense().astype(np.int))[0]
    contingency_sum = 100
    for i, (cur_node, phen_node) in tqdm.tqdm(enumerate(zip(tree.traverse(), phenotree.traverse())),
                                              total=population, desc='Iterating tree'):
        if not cur_node.is_root():
            if not cur_node.is_leaf() and with_rand and cur_node.random[phen_ind]: continue
            node = node_to_arr(cur_node)
            prev_node = node_to_arr(cur_node.up)

            gene_state = node[skip:]
            prev_gene_state = prev_node[skip:]

            phen_state = phen_node.genotype[0, phen_ind]
            prev_phen_state = phen_node.up.genotype[0, phen_ind]

            phen_event = np.abs((prev_phen_state - phen_state))  # all that differs from paralel is an abs
            gene_event = np.abs((prev_gene_state - gene_state))

            if with_rand: gene_event[cur_node.up.random[skip:]] = 0
            score += phen_event * gene_event

    return score.astype(np.int)


def GetPhyloParallelScore(tree, phenotree, phen_ind, skip=0, with_rand=False):
    """
    Collects co-occuring events along the phylogeny
    
                                 /--(1,1)
                       /--(0,1)-|
             /--(0,1)-|          \--(0,1)
    --(0,1)-|          \--(1,1)
             \--(0,1)

    Would translate to the following contingency table:

    T\Q| 0  1
    ---------
     0 | 4  1
     1 | 1  0

    :param tree: ete3 tree
    :param phen_ind: index of target T
    :param skip: number of qualities to skip
    :return: A contingency table for each available quality in a nodes 'genotype' value
    """

    population = (len(tree) * 2) - 1
    score = np.zeros(tree.genotype.shape[1] - skip)
    node_to_arr = lambda n: np.array(n.genotype.todense().astype(np.int))[0]
    contingency_sum = 100
    for i, (cur_node, phen_node) in tqdm.tqdm(enumerate(zip(tree.traverse(), phenotree.traverse())),
                                              total=population, desc='Iterating tree'):
        if not cur_node.is_root():
            if not cur_node.is_leaf() and with_rand and cur_node.random[phen_ind]: continue
            node = node_to_arr(cur_node)
            prev_node = node_to_arr(cur_node.up)

            gene_state = node[skip:]
            prev_gene_state = prev_node[skip:]

            phen_state = phen_node.genotype[0, phen_ind]
            prev_phen_state = phen_node.up.genotype[0, phen_ind]

            phen_event = (prev_phen_state - phen_state)
            gene_event = (prev_gene_state - gene_state)

            if with_rand: gene_event[cur_node.up.random[skip:]] = 0
            score += phen_event * gene_event

    return score.astype(np.int)


def SetHomoplasy(tree, skip=0):
    """
    Get distribution of homoplasy
    """
    population = tree.size if 'size' in vars(tree) else (len(tree) * 2) - 1
    order = tree.traverse()
    root = next(order)
    homoplasy_hist = sp.csr_matrix(np.zeros(root.genotype.shape[1] - skip))
    for cur_node in tqdm.tqdm(order, total=population + 1):
        homoplasy_hist += (cur_node.genotype[0, skip:] != cur_node.up.genotype[0, skip:])
    homoplasy_dist = pd.Series(homoplasy_hist.data).value_counts(normalize=True)
    fit_dist = rv_discrete(values=(homoplasy_dist.index.astype(int), homoplasy_dist.values.astype(float)))
    root.homoplasy = fit_dist
    root.homoplasy_hist = homoplasy_hist.toarray()[0].astype(np.int)
    return tree


def score_to_pval(scores, pval_type='hist', right=True):
    colnames = ['pval', 'score']
    if type(pval_type) is rv_histogram:
        func = pval_type
        res = func(scores)
    elif 'hist' in pval_type:
        func = right_tail_(scores) if right else left_tail_(scores)
        if 'hist_only' in pval_type: return func
        res = func(scores)
    res = pd.DataFrame(np.vstack([res, scores]).T, columns=colnames)
    return (res, func)


# -------------------------------------------- #
""""            figure functions             """


# -------------------------------------------- #

def ods_fig(tot_tables, ods_columns, n_mutations):
    plt.close('all')
    with plt.style.context('default'):
        for col in ods_columns:
            plt.plot(sorted(tot_tables[col]), np.arange(len(tot_tables)) / len(tot_tables), label=col)
        plt.xlim(-6, 6)
        # plt.xlabel(str(spearmanr(tot_tables['ods'], tot_tables['ods_recon'])))
        plt.legend()
        plt.title(str(n_mutations) + ' mutations' + (' JointLike' if max_like else ' MaxPars') + ': ods ratio cdf')
        if save_fig: plt.savefig(str(n_mutations) + ('_max_like' if max_like else '') + '_ods.svg')
        # plt.show()


def ods_ratio_sf(tot_tables, ods_cols, n_mutations, outpath='', stat='pearson'):
    plt.close('all')
    with plt.style.context('default'):
        for col in ods_cols:
            plt.plot(sorted(tot_tables[col], reverse=True), np.arange(len(tot_tables)) / len(tot_tables), label=col)
        plt.yscale('log')
        stat_func = spearmanr if stat != 'pearson' else pearsonr
        plt.xlabel(str(stat_func(tot_tables[ods_cols[0]], tot_tables[ods_cols[1]])))
        plt.legend()
        plt.title(str(n_mutations) + ' mutations' + (' JointLike' if max_like else ' MaxPars') + ': ods ratio sf')

        if save_fig: plt.savefig(
            os.path.join(outpath, str(n_mutations) + ('_max_like' if max_like else '') + '_ods.svg'))
        # plt.show()


# In[22]:

def pval_hists(tables, pval_cols, n_mutations, cmap='rainbow', outpath='', stat='pearson'):
    plt.close('all')
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap(cmap)(range(0, 255, 256 // len(pval_cols)))
    with plt.style.context('default'):
        ax = plt.subplots(2)[1].ravel()
        ax[0].set_visible(False)
        ax = ax[-1]
        for col, color in zip(pval_cols, colors):
            ax.loglog(sorted(tables[col]), np.arange(len(tables)) / len(tables), label=col, color=color)
        ax.loglog(np.arange(len(tables)) / len(tables), np.arange(len(tables)) / len(tables), label='diag')
        ax.set_xlabel(str(n_mutations) + ' mutations' + (' JointLike' if max_like else ' MaxPars') + ': pval cdf')
        ax.set_ylabel('population fraction')
        stat_func = spearmanr if stat != 'pearson' else pearsonr
        cell_text = stat_func(tables[pval_cols])[0]
        cell_text = np.vectorize(lambda x: "{:.2e}".format(x))(cell_text)
        labels = pval_cols
        the_table = ax.table(cellText=cell_text,
                             colColours=colors,
                             colLabels=labels,
                             loc='top')

        if save_fig: plt.savefig(
            os.path.join(outpath, str(n_mutations) + ('_max_like' if max_like else '') + '_pval.svg'))
        # plt.show()


# In[23]:

def pval_corr(tot_tables, pval_cols, n_mutations, outpath='', stat='pearson'):
    plt.close('all')
    stat_func = spearmanr if stat != 'pearson' else pearsonr
    cell_text = stat_func(tot_tables[pval_cols])[0]
    cell_text = np.vectorize(lambda x: "{:.2e}".format(x))(cell_text)
    with plt.style.context('default'):
        tmp_pval = tot_tables[pval_cols][~(tot_tables[pval_cols] == 0).any(axis=1)]
        for i, col in enumerate(pval_cols[1:]):
            plt.scatter(np.log10(tmp_pval[pval_cols[0]].astype(float)), np.log10(tmp_pval[col].astype(float)),
                        alpha=.1, label=col + ' pval: ' + cell_text[0][i + 1])
        plt.xlabel('log pval ' + pval_cols[0])
        plt.legend()
        plt.title('Fisher pval correlation' + (' JointLike' if max_like else ' MaxPars'))
        if save_fig: plt.savefig(
            os.path.join(outpath, str(n_mutations) + ('_max_like' if max_like else '') + '_scatter.png'))
        # plt.show()
        del tmp_pval


# In[24]:

def hist_to_spearman(table, pval_main, cols, hist, n_mutations, outpath='', stat='pearson'):
    with plt.style.context('default'):
        plt.close('all')
        marker = itertools.cycle((',', '+', '.', 'o', '*'))
        for col in cols:
            pval_hist = []
            for val in range(1, int(hist.max() + 2)):
                tmp_pval = table.loc[(hist < val).indices]
                stat_func = spearmanr if stat != 'pearson' else pearsonr
                spear = stat_func(tmp_pval[[pval_main, col]])[0]
                pval_hist.append(spear)
            plt.scatter(range(int(hist.max() + 1)), pval_hist, label=col, alpha=.5, marker=next(marker))
        plt.xlabel('tree recon distance')
        plt.ylabel('spearman correlation for d or lower distance')
        plt.title('spearman for recon distance')
        plt.legend()
        plt.savefig(
            os.path.join(outpath, str(n_mutations) + ('_max_like' if max_like else '') + '_spearman_distance.svg'))
        # plt.show()

        plt.close('all')
        hist = sorted(hist.toarray()[0], reverse=True)
        plt.plot(hist, np.arange(len(hist)) / len(hist), label='source')
        plt.yscale('log')
        plt.title('recon distance over population sf')
        plt.xlabel('recon distance')
        plt.ylabel('population with <= recon distance')
        plt.savefig(os.path.join(outpath, str(n_mutations) + ('_max_like' if max_like else '') + '_distance_sur.svg'))
        # plt.show()


# In[25]:

def FitNormPrintExp(score_per_pos, norm_funcs):
    plt.close('all')
    plt.figure(figsize=(7, 7))
    for i, arr in enumerate(score_per_pos):
        plt.plot(sorted(arr), np.arange(len(arr)) / len(arr), label='source_' + str(i))
        dist = norm(*norm.fit(arr))
        text_ = np.vectorize(lambda x: "{:.2e}".format(x))(norm.fit(arr))
        x = np.arange(arr.min(), arr.max(), 0.001)
        plt.plot(x, dist.cdf(x), label='fit_' + str(i) + ' ' + str(text_))
    for i, norm_func in enumerate(norm_funcs):
        plt.plot(x, norm_func.cdf(x), label='expected_' + str(i) + ' ' + ("{:.2e}".format(norm_func.std())))
        plt.yscale('log')
    plt.legend()
    # plt.show()


"""
Plan
GroundTruth
1) build a tree (currently constant branch length)
2) simulate genotypes with a (fixed or from poiss dist) number of events by subsampling branches of tree
3) select a first genotype as phenotype
4) calculate contingencies of co-occuring events with phenotype of all traits
5) set pval as sf of score histogram

MyCalc
6) reconstruct events of the phylogeny, and estimate position homoplasy
7) calculate scores reconstructed tree to GroundTruth phenotype
8) set custom pvals

TreeWas
9) simulate 10x times the number of events
10) calculate TreeWas scores from simulated set to GroundTruth phenotype (assumed to be perfectly reconstructed)
11) calculate TreeWas scores from reconstructed tree to GroundTruth phenotype
"""


def pipe(n_mutations, population, outpath, random_prop=False, model_prop=False):
    print('mut: ' + str(n_mutations))
    print('population: ' + str(population))

    # 1) build a tree
    print('1) build a tree', flush=True)
    GroundTruthTree = ete3.Tree()
    GroundTruthTree.populate(population, random_branches=random_prop)
    GroundTruthTree.rand_branches = random_prop
    for i, n in enumerate(tqdm.tqdm(GroundTruthTree.traverse())): n.id = i
    population = i - 1  # now it is branch population
    if type(n_mutations) is list:
        n_mutations = np.array(n_mutations)

    m_rate = (n_mutations / population)

    def get_rand_func(x):
        if barnch_binom:
            def rand_func_const(branch=1):
                return binomial(1, branch * m_rate, x)

            def rand_func_mix(branch=1):
                res = binomial(1, branch * m_rate, size=(x, len(m_rate)))
                res = pd.DataFrame(res).sample(n=1, axis=1).values.T[0]
                return res

            return rand_func_const if np.ndim(m_rate) == 0 else rand_func_mix
        else:
            m_rate_per_position = poisson(n_mutations).rvs(x) / population

            def rand_func():
                return binomial(1, m_rate_per_position)

            return rand_func

    # 2) simulate a set number of events by subsampling branches of tree
    print('2) simulate a set number of events by subsampling branches of tree', flush=True)
    # TreeWas has a transition matrix Q for (g,p) to get number of interesting genes. We do things a bit differant
    GroundTruthTree = RandomAtributes(GroundTruthTree, population, n_genes, get_rand_func(n_genes))

    # 3) select a phenotype
    # 4) calculate contingencies of co-occuring events with phenotype of all traits
    print('3) select a phenotype', flush=True)
    print('4) calculate contingencies of co-occuring events with phenotype of all traits', flush=True)
    GroundTruthScore_para = GetPhyloCoEventScore(GroundTruthTree, GroundTruthTree, phen_ind=0, skip=1, with_rand=False)

    # 5) set pval as sf function of the calculated histogram of ods
    print('5) set pval as sf function of the calculated histogram of ods', flush=True)
    GroundTruthTables, GroundTruthDist = score_to_pval(np.abs(GroundTruthScore_para), 'hist', right=True)

    # 6) reconstruct event of the phylogeny
    # 6.1) choose ancestral reconstruction method
    print('6.1) reconstruct event of the phylogeny')
    if max_like:
        ReconTree = JointMaxLiklihood(None, GroundTruthTree.copy(), {}, do_tips=True, naming=False, m_rate=m_rate)
    else:
        ReconTree = MaxParsimonyNoTable(None, GroundTruthTree.copy(), {}, do_tips=True)

    # 6.2) estimate homoplay from ancestral reconstruction (optional - can be constant)
    print('6.2) estimate homoplay from ancestral reconstruction (optional - can be constant)', flush=True)
    ReconTree = SetHomoplasy(ReconTree, skip=0)
    GroundTruthTree = SetHomoplasy(GroundTruthTree, skip=0)

    ReconHist = sp.csr_matrix(np.zeros(shape=(1, n_genes)))
    for og_node, recon_node in zip(GroundTruthTree.traverse(), ReconTree.traverse()):
        ReconHist += (og_node.genotype != recon_node.genotype)
    ReconHist = ReconHist[0, 1:]  # no pheno
    ReconHist = ReconHist.toarray()[0].astype(int)
    # In[66]:

    # 7) calculate contingencies and ods from reconstructed tree to GroundTruth phenotype
    print('7) calculate contingencies and ods from reconstructed tree to GroundTruth phenotype', flush=True)
    tables_recon_para_score = GetPhyloCoEventScore(ReconTree, GroundTruthTree, phen_ind=0, skip=1, with_rand=False)

    # 8) set custom pval
    print('8) set custom pval', flush=True)
    phen_rate = GroundTruthTree.homoplasy_hist[0] / population

    mean_norm = my_norm((ReconTree.homoplasy_hist.mean() / population), phen_rate, population)
    direct_norm = my_norm((ReconTree.homoplasy_hist[1:] / population), phen_rate, population)
    mean_dir = my_direct((ReconTree.homoplasy_hist.mean() / population), phen_rate, population)

    tables_recon = pd.DataFrame()
    tables_recon['score'] = np.abs(tables_recon_para_score)

    if np.ndim(m_rate) == 0:
        expected_dir = my_direct(m_rate, phen_rate, population)
        expected_norm = my_norm(m_rate, phen_rate, population)
        tables_recon['exp_norm_pval'] = expected_norm.sf(tables_recon['score']) * 2
        tables_recon['exp_conv_pval'] = expected_dir(tables_recon['score'])

    tables_recon['mean_norm_pval'] = mean_norm.sf(tables_recon['score']) * 2
    tables_recon['direct_norm_pval'] = direct_norm.sf(tables_recon['score']) * 2

    tables_recon['mean_conv_pval'] = mean_dir(tables_recon['score'])
    tables_recon['direct_conv_pval'] = my_direct_ind_pval((ReconTree.homoplasy_hist[1:] / population), phen_rate,
                                                          tables_recon['score'].values, population)
    if model_prop:
        branch_prop = np.array([x.dist for x in ReconTree.traverse()][1:])
        branch_prop = branch_prop / sum(branch_prop)
        mean_dir_prop = my_direct((ReconTree.homoplasy_hist.mean() / population), phen_rate, population,
                                  branch_prop=branch_prop)
        tables_recon['mean_conv_prop_pval'] = mean_dir_prop(tables_recon['score'])
        tables_recon['direct_conv_prop_pval'] = my_direct_ind_pval((ReconTree.homoplasy_hist[1:] / population),
                                                                   phen_rate,
                                                                   tables_recon['score'].values, population,
                                                                   branch_prop=branch_prop)

    tables_recon['recon_dist'] = ReconHist

    # 9) simulate 10x times the number of events
    print('9) simulate 10x times the number of events', flush=True)
    # TODO could save space by calculating score while building tree, then erasing prev data
    n_sim_genes = min(n_genes * 10, int(5e5))
    SimTree = RandomAtributesExact(GroundTruthTree.copy(), n_sim_genes, ReconTree.homoplasy)

    # 10) calculate TreeWas scores distribution from simulated set to GroundTruth phenotype
    print('10) calculate TreeWas scores distribution from simulated set to GroundTruth phenotype', flush=True)
    tables_sim_para_score = GetPhyloCoEventScore(SimTree, GroundTruthTree, phen_ind=0, skip=1, with_rand=False)

    # 11) calculate TreeWas scores from reconstructed tree to GroundTruth phenotype
    print('11) calculate TreeWas scores from reconstructed tree to GroundTruth phenotype', flush=True)
    sim_dist_sf = score_to_pval(np.abs(tables_sim_para_score), 'hist_only', right=True)
    tables_sim = pd.Series(sim_dist_sf(np.abs(tables_recon['score'])), name='treewas_pval')

    tot_tables = GroundTruthTables.join(tables_recon, rsuffix='_recon').join(tables_sim, rsuffix='_treewas').dropna()
    if save_table:
        tot_tables.to_pickle(outpath)


# ----------------------------------------------------------- #
"""                  main program script                    """
# ----------------------------------------------------------- #

# Global (program) params
n_genes = int(5e4)
leaf_population_list = [100, 250, 500, 1000, 2000]
n_mutations_list = [10, 25, 50, 75, 100]
n_times = 10
barnch_binom = True
max_like = False  # Maxpasimony is used over max like in TreeWas: appendix 2
save_fig = False
save_table = True
outpath = 'figs_data_coevent'
if not os.path.exists(outpath): os.mkdir(outpath)
leaf_population = 500

# fig 1) data same tree size, many mutation rates
print("fig 1) data same tree size, many mutation rates")
for mut in n_mutations_list:
    for i in range(n_times):
        print(f"fig 1) {i}")
        table_path = os.path.join(outpath, f'mut_{mut}_pop_{leaf_population}_{i}')
        pipe(mut, leaf_population, table_path)


# fig 2) many data same tree sizes, constant rate, constant mut
leaf_population_list.remove(leaf_population)  # no need for 500
# 2 a - constant mut
print("fig 2) 2 a - constant mut")
mut = 50
for pop in leaf_population_list:
    for i in range(n_times):
        print(f"fig 2a) {i}")
        table_path = os.path.join(outpath, f'mut_{mut}_pop_{pop}_{i}')
        pipe(mut, pop, table_path)
# 2 b  - constant rate
print("fig 2) 2 b - constant rate")
rate = mut / ((leaf_population * 2) - 1)
for pop in leaf_population_list:
    mut = int(((pop * 2) - 1) * rate)
    for i in range(n_times):
        print(f"fig 2b) {i}")
        table_path = os.path.join(outpath, f'const_rate_pop_{pop}_{i}')
        pipe(mut, pop, table_path)

# fig 3) data same tree size, 2 mutation rates of increasing distance
print("fig 3) data same tree size, 2 mutation rates of increasing distance")
lower_rate = n_mutations_list[0]
for mut in n_mutations_list[1:]:
    for i in range(n_times):
        print(f"fig 1) {i}")
        table_path = os.path.join(outpath, f'mutmix_{lower_rate}_{mut}_pop_{leaf_population}_{i}')
        pipe([lower_rate, mut], leaf_population, table_path)

# fig 4) rerun with branch dist
print("fig 4) rerun with branch dist")
for mut in n_mutations_list:
    for i in range(n_times):
        print(f"fig 1) {i}")
        table_path = os.path.join(outpath, f'prop_mut_{mut}_pop_{leaf_population}_{i}')
        pipe(mut, leaf_population, table_path, random_prop=True, model_prop=True)

# asistent supplementary figures
"""
FitNormPrintExp([GroundTruthScore_para, tables_recon_para_score, tables_sim_para_score], [expected_norm, mean_norm])
pval_cols = [col for col in tot_tables if 'pval' in col]
pval_hists(tot_tables, pval_cols, outpath=outpath)
pval_corr(tot_tables, ['pval', 'mean_pois_pval', 'treewas_pval'], outpath=outpath)
hist_to_spearman(tot_tables, pval_cols[0], pval_cols[1:], ReconHist, outpath=outpath)
#ods_ratio_sf(tot_tables, ['ods', 'ods_para'], outpath=outpath)
"""
