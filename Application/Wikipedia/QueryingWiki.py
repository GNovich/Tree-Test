import os
import ete3
import tqdm
import re
import numpy as np
import random
import pickle
import requests
import scipy.sparse as sp
from collections import deque
import wikipediaapi as wiki_session
from process_wikipedia import remove_special_chars, remove_html_tags, clean_string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from gensim.corpora import Dictionary

def WikiAPITree(topic, max_depth=2, max_child=10, max_nodes_tot=100, branch_factor=5, keep_cat=False):
    """
        tree format is as follows:
        A category node is created (at root or as internal)
        Every node is aimed to be a MAIN page
        So if the category has a main page, we replace the node name with that page
        
        chechking for the existance of a main page in the tree is easy as (page.title in tree)
        
        To keep trak on used category pages, we save them at 
        
        chechking for the existance of a category page in the tree is easy as (page.title in tree.categories)
        a dict tree.categories marks the relationship between categoy and node name in tree
    """

    filter_pages = ['template', 'portal']
    wiki = wiki_session.Wikipedia('en')

    def cat_to_page(node):
        fp = requests.get(node.Catpage.fullurl)
        for l in fp.iter_lines():
            main_art = re.findall(b'mainarticle.*is.*title=\"(.*)\"', l)
            if main_art:
                main_page = wiki.page(str(main_art[0].decode('utf-8')))
                if main_page.exists():
                    node.page = main_page
                    node.pageless = False
                    node.name = node.page.title
                    break
                break
        return node

    t = ete3.Tree()
    t.name = topic
    t.Catpage = wiki.page('Category:' + topic if 'Category:' not in topic else topic)
    t.categories = dict()
    t.depth = 0
    t.pageless = True
    t = cat_to_page(t)
    t.categories[t.Catpage.title] = (t.page if not t.pageless else t.Catpage).title

    node_q = deque([t])
    bar = tqdm.tqdm(total=max_nodes_tot, desc='Constructing CatTree')
    node_counter = 0
    while node_q and node_counter < max_nodes_tot:  # wiki category BFS
        # next category node
        node = node_q.popleft()

        # get potential children
        # keep it real and not alpha-betic
        subitems = node.Catpage.categorymembers
        sub_keys = list(subitems.keys())
        random.shuffle(sub_keys)

        child_count = 0
        cat_count = 0
        # adding children
        for key in sub_keys:
            # avoid circles - main and cat check
            if key in t or key in t.categories: continue
            # filter hubs
            if any(filt in key for filt in filter_pages): continue

            child = ete3.TreeNode(name=key)
            child.depth = node.depth + 1
            if subitems[key].ns == wiki_session.Namespace.CATEGORY:
                if node.depth < max_depth and cat_count < branch_factor:
                    child.name = re.findall('Category:(.*)', key)[0]
                    child.Catpage = subitems[key]
                    child.pageless = True

                    # try to replace with main_artical 'pageless' might change inside
                    child = cat_to_page(child)
                    # we know the category is not in tree, but the new main article might
                    if not child.pageless and child.page.title in t: continue

                    # seems like a new node
                    # when adding a new category, trak cat and main page if exists
                    t.categories[child.Catpage.title] = (child.Catpage if child.pageless else child.page).title
                    node.add_child(child)  # main page name is in name already
                    node_q.append(child)
                    cat_count += 1
            elif child_count < max_child and subitems[key].ns == wiki_session.Namespace.MAIN:
                child.pageless = False
                child.page = subitems[key]
                child.name = child.page.title
                node.add_child(child)
                child_count += 1

        # category leaf should be erased - they are useless
        if len(node.children) < 1 and node.pageless:
            node.detach()
        else:
            node_counter += (1 + child_count)
            bar.update(1 + child_count)
    bar.close()

    # category leaf might still be in the queue
    bar = tqdm.tqdm(total=len(node_q), desc='Category Leaf Cleanup')
    while node_q:
        node = node_q.popleft()
        if node.pageless:
            # either you got a main page or you don't... and you don't
            parent = node.up
            node.detach()
            if (parent not in node_q and parent.pageless):
                if parent.is_leaf() and not parent.is_root():
                    # no children, the parent became a category leaf
                    node_q.append(parent)
        bar.update(1)

    pop = 0
    for _ in t.traverse(): pop += 1
    t.size = pop

    return t


def text_cleanup(text):
    text = remove_html_tags(text)
    text = remove_special_chars(text, ['\n', '–'] + list(punctuation))
    text = clean_string(text, set(stopwords.words('english')))
    return text


stemm = PorterStemmer()
lemma = WordNetLemmatizer()


def tokanizer(text, ret_trans=True, unique=False):
    words = nltk.word_tokenize(text)
    words, inv_words = np.unique(words, return_inverse=True)
    word_trans_ = dict.fromkeys(words)
    for i in range(len(words)):
        word_trans_[words[i]] = stemm.stem(lemma.lemmatize(words[i]))
        words[i] = word_trans_[words[i]]

    tokens, inv_tokens = np.unique(words, return_inverse=True)
    if not unique: tokens = (tokens[inv_tokens])[inv_words]
    tokens = tokens.tolist()
    if ret_trans:
        return (tokens, word_trans_)
    else:
        return tokens


def SetWordDataDict(tree, no_below=3):
    # collect tokens and translations
    token_corp = Dictionary()
    word_to_token = dict()
    for node in tqdm.tqdm(tree.traverse(), total=tree.size, desc='Collecting Page Tokens'):
        if node.pageless:
            node.text = text_cleanup(node.Catpage.text)
        else:
            node.text = text_cleanup(node.page.text)

        node.tokens, trans = tokanizer(text_cleanup(node.text), unique=True, ret_trans=True)
        word_to_token.update(trans)
        token_corp.add_documents([node.tokens])

    # filter token space to relevant tokens
    token_corp.filter_extremes(no_below=no_below)

    # token_to_word inv map
    dat_size = len(token_corp)
    tree.token_corp = token_corp
    tree.word_to_token = word_to_token
    tree.token_to_word = dict()
    for k, v in word_to_token.items():
        tree.token_to_word[v] = tree.token_to_word.get(v, [])
        tree.token_to_word[v].append(k)

    # word2vec mapping
    for node in tqdm.tqdm(tree.traverse('postorder'), total=tree.size, desc='Token2Vec_pass1'):
        if node.pageless:
            # ancestral reconstruction
            children_vecs_1 = sum([n.dat for n in node.children])
            children_vecs_0 = sum([1 * (n.dat != 1) for n in node.children])
            node.dat = children_vecs_1 > children_vecs_0
            node.dat[0, (children_vecs_0 == children_vecs_1).indices] = 2  # marks indicicive
        else:
            cols = np.array(token_corp.doc2idx(node.tokens))
            cols = cols[cols > -1]
            rows = [0] * len(cols)
            dat = [1] * len(cols)
            node.dat = sp.csr_matrix((dat, (rows, cols)), shape=(1, dat_size))

    for node in tqdm.tqdm(tree.traverse('preorder'), total=tree.size, desc='Token2Vec_pass2'):
        if node.pageless:
            indicicive = ((node.dat == 2).indices.tolist())
            if node.is_root() and indicicive:
                node.dat[0, indicicive] = np.random.randint(2, len(indicicive))
            else:
                node.dat[0, indicicive] = node.up.dat[0, indicicive]


def GenerateTree(topic, outpath):
    t = WikiAPITree(topic, max_depth=50, max_child=5, max_nodes_tot=5000, branch_factor=5, keep_cat=True)
    SetWordDataDict(t, no_below=int(t.size // 1000))
    pickle.dump(t, open(outpath, 'wb'))


# script
dir_path = 'data'
topics = ['Physics', 'Finance', 'Biology', 'Mathematics']
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

for topic in topics:
    outpath = os.path.join(dir_path, 'tree_' + topic + '.tree')
    GenerateTree(topic, outpath)
    break
