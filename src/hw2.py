import itertools
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple
from dict_vec import DictVector
import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as wn
import re
from tqdm import tqdm

### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])


stopwords = nltk_stopwords.words("english")
#stopwords = read_stopwords('../data/common_words')

stemmer = SnowballStemmer('english')


def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]


class DocFreqs(Counter):
    def __init__(self):
        super(DocFreqs, self).__init__()
        self.num_docs = 0

    def set_num_docs(self, n):
        self.num_docs = n

    def get_num_docs(self):
        return self.num_docs


# Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float


def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = DocFreqs()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    freq.set_num_docs(len(docs))
    return freq


def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict


def compute_tfidf(doc, doc_freqs, weights):
    tf = compute_tf(doc, doc_freqs, weights)
    tf_idf = {}
    N = doc_freqs.get_num_docs()
    for word in tf.keys():
        tf_idf[word] = tf[word] * np.log(N / (1 + doc_freqs[word]))
    return tf_idf


def compute_boolean(doc, doc_freqs, weights):
    vec = defaultdict(float)
    for word in doc.author:
        if word not in vec.keys():
            vec[word] = 1
    for word in doc.keyword:
        if word not in vec.keys():
            vec[word] = 1
    for word in doc.title:
        if word not in vec.keys():
            vec[word] = 1
    for word in doc.abstract:
        if word not in vec.keys():
            vec[word] = 1

    return dict(vec)



### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y):
    num = 2 * dictdot(x, y)
    if num == 0:
        return 0
    denom = sum(x.values()) + sum(y.values())
    return num / denom

def jaccard_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    denom = sum(x.values()) + sum(y.values()) - num
    if denom == 0:
        return 1
    return num / denom

def overlap_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    denom = min(sum(x.values()), sum(y.values()))
    return num / denom


# Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b


def precision(tp: int, total: int) -> float:
    if total == 0:
        return 1
    return tp / total


def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''
    relevant_set = set(relevant)
    total = len(relevant)
    counted = 0
    relevant_counted = 0
    cur_recall = 0
    for doc in results:
        if cur_recall == recall:
            return precision(relevant_counted, counted)
        elif cur_recall > recall:
            last_relevant_counted = relevant_counted - 1
            last_recall = last_relevant_counted / total
            return interpolate(last_recall, precision_at(last_recall, results, relevant),
                               cur_recall, precision_at(cur_recall, results, relevant), recall)
        else:
            counted += 1
            if doc in relevant_set:
                relevant_counted += 1
                cur_recall = relevant_counted / total
    return 1


def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3


def mean_precision2(results, relevant):
    precision_sum = 0
    for i in range(1, 11):
        precision_sum += (1 / 10) * precision_at(i / 10, results, relevant)
    return precision_sum


def norm_recall(results, relevant):
    N = len(results)
    relevant_set = set(relevant)
    # Sum from 1 to len(relevant)
    sum_rel = len(relevant) * (len(relevant) + 1) / 2
    sum_rank = 0
    for i in range(len(results)):
        if results[i] in relevant_set:
            sum_rank += i + 1
    num = sum_rank - sum_rel
    denom = len(relevant) * (N - len(relevant))
    return 1 - num / denom


def factorial(n):
    if n == 0:
        return 1
    if n > 15:
        return n * np.log(n)
    prod = 1
    for i in range(1, n + 1):
        prod *= i
    return prod


def norm_precision(results, relevant):
    N = len(results)
    Rel = len(relevant)
    relevant_set = set(relevant)
    # Can group all i's together inside one log
    sum_rel = sum([np.log(i) for i in range(1, Rel+1)])
    sum_rank = 0
    for i in range(len(results)):
        if results[i] in relevant_set:
            sum_rank += np.log(i + 1)
    num = sum_rank - sum_rel
    denom = N * np.log(N) - (N - Rel) * np.log(N - Rel) - Rel * np.log(Rel)
    return 1 - num / denom


# Extensions

# Constant length document embedder using BERT. Implemented this, might use it, probably not
class Embedder:
    def __init__(self, gpu=False):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                               output_hidden_states=True)
        self.hidden_dim = 768
        self.model.eval()
        if gpu:
            self.model = self.model.cuda()
        self.gpu = gpu

    def doc2tokens(self, doc: Document) -> List[List[str]]:
        """
        Converts a document to a list of tokenized sentences
        """
        tok_list = []
        s = " "
        if len(doc.keyword) > 0:
            keywords = "[CLS] " + s.join(doc.keyword) + " [SEP]"
            tok_list.append(self.tokenizer.tokenize(keywords))
        if len(doc.author) > 0:
            authors = "[CLS] " + s.join(doc.author) + " [SEP]"
            tok_list.append(self.tokenizer.tokenize(authors))
        if len(doc.title) > 0:
            title = "[CLS] " + s.join(doc.title) + " [SEP]"
            tok_list.append(self.tokenizer.tokenize(title))
        if len(doc.abstract) > 0:
            abstract = s.join(doc.abstract)
            abs_sent = nltk.sent_tokenize(abstract)
            for i, sentence in enumerate(abs_sent):
                abs_sent[i] = "[CLS] " + sentence + " [SEP]"
            for sentence in abs_sent:
                tok_list.append(self.tokenizer.tokenize(sentence))
        return tok_list

    def embed_tokens(self, tokens: List[str]) -> torch.Tensor:
        """
        Embeds a tokenized sentence into a vector using BERT
        """
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [1] * len(tokens)
        tokens_tensor = torch.Tensor([indexed_tokens]).long()
        segments_tensor = torch.Tensor([segment_ids]).long()
        if self.gpu:
            tokens_tensor, segments_tensor = tokens_tensor.cuda(), segments_tensor.cuda()
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding.cpu()

    def doc2vec(self, doc: Document) -> torch.Tensor:
        """
        Converts a Document to a vector representation
        """
        tok_list = self.doc2tokens(doc)
        vecs = torch.zeros([self.hidden_dim])
        for tokens in tok_list:
            vecs.add_(self.embed_tokens(tokens))
        vecs.div_(len(tok_list))
        return vecs


# Implementing query expansion, use of wordnet learned from
# https://medium.com/@swaroopshyam0/a-simple-query-expansion-49aef3442416
def pos_tagger(tokens):
    return nltk.pos_tag(tokens)


def remove_stopwords_nltk(tokens):
    stop = nltk_stopwords.words("english")
    out = []
    for token in tokens:
        if token[0].lower() not in stop:
            out.append(token)
    return out


def download_nltk_packages():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


def get_tokens_from_synsets(synsets):
    tokens = {}
    for synset in synsets:
        for s in synset:
            if s.name() in tokens:
                tokens[s.name().split('.')[0]] += 1
            else:
                tokens[s.name().split('.')[0]] = 1
    return tokens


def get_hypernyms(synsets):
    hypernyms = []
    for synset in synsets:
        for s in synset:
            hypernyms.append(s.hypernyms())

    return hypernyms


def get_tokens_from_hypernyms(synsets):
    tokens = {}
    for synset in synsets:
        for s in synsets:
            for ss in s:
                if ss.name().split('.')[0] in tokens:
                    tokens[(ss.name().split('.')[0])] += 1
                else:
                    tokens[(ss.name().split('.')[0])] = 1
    return tokens


def underscore_replacer(tokens):
    new_tokens = {}
    for key in tokens.keys():
        mod_key = re.sub(r'_', ' ', key)
        new_tokens[mod_key] = tokens[key]
    return new_tokens


def word_count(words: list):
    return Counter(words)


def compute_doc_freqs_from_dict(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = DocFreqs()
    for doc in docs:
        words = set()
        for word in doc.keys():
            words.add(word)
        for word in words:
            freq[word] += 1
    freq.set_num_docs(len(docs))
    return freq


def tokenizer(sentence):
    return word_tokenize(sentence)


class QueryExpander:
    """
    This class can compute the term frequencies for expanded queries
    using synonyms and hypernyms
    """
    def __init__(self):
        download_nltk_packages()
        self.pos_tags = {
        'NN': [wn.NOUN],
        'JJ': [wn.ADJ, wn.ADJ_SAT],
        'RB': [wn.ADV],
        'VB': [wn.VERB]
        }

    def pos_tag_converter(self, nltk_pos_tag):
        root_tag = nltk_pos_tag[0:2]
        try:
            self.pos_tags[root_tag]
            return self.pos_tags[root_tag]
        except KeyError:
            return ''

    def get_synsets(self, tokens):
        synsets = []
        for token in tokens:
            wn_pos_tag = self.pos_tag_converter(token[1])
            if wn_pos_tag == '':
                continue
            else:
                synsets.append(wn.synsets(token[0], wn_pos_tag))
        return synsets

    def generate_tokens(self, sentence):
        tokens = tokenizer(sentence)
        tokens = pos_tagger(tokens)
        tokens = remove_stopwords_nltk(tokens)
        synsets = self.get_synsets(tokens)
        synonyms = get_tokens_from_synsets(synsets)
        synonyms = underscore_replacer(synonyms)
        hypernyms = get_hypernyms(synsets)
        hypernyms = get_tokens_from_hypernyms(hypernyms)
        hypernyms = underscore_replacer(hypernyms)
        tokens = {**synonyms, **hypernyms}
        return tokens


    def wordlist2tokens(self, words):
        s = " "
        sentence = s.join(words)
        return self.generate_tokens(sentence)


    def compute_tf(self, doc: Document, weights: list):
        vec = defaultdict(float)
        for word in doc.author:
            vec[word] += weights.author
        keyword_toks = self.wordlist2tokens(doc.keyword)
        for word, count in keyword_toks.items():
            vec[word] += count * weights.keyword
        title_toks = self.wordlist2tokens(doc.title)
        for word, count in title_toks.items():
            vec[word] += count * weights.title
        abstract_toks = self.wordlist2tokens(doc.abstract)
        for word, count in abstract_toks.items():
            vec[word] += count * weights.abstract
        return dict(vec)


# Converts a query string to document
def query2doc(query: str) -> Document:
    doc = Document(-1, [], [], [], word_tokenize(query))
    return doc


class QueryProcessor:
    """
    This class can transform a string or document into a sparse tfidf vector
    """
    def __init__(self, term_weights, sim_func=cosine_sim, query_expand=True):
        self.sim_func = sim_func
        self.query_expander = None
        self.term_weights = term_weights
        if query_expand:
            self.query_expander = QueryExpander()

    def tfidf_from_doc(self, doc: Document, doc_freqs: DocFreqs) -> dict:
        if self.query_expander is not None:
            tf = self.query_expander.compute_tf(doc, self.term_weights)
        else:
            tf = compute_tf(doc, doc_freqs, self.term_weights)
        tf_idf = {}
        N = doc_freqs.get_num_docs()
        for word in tf.keys():
            tf_idf[word] = tf[word] * np.log(N / (1 + doc_freqs[word]))
        return DictVector(tf_idf)

    def tfidf_from_query(self, query, doc_freqs):
        doc = query2doc(query)
        return self.tfidf_from_doc(doc, doc_freqs)

    def tfidf_from_tf(self, tf, doc_freqs):
        tf_idf = {}
        N = doc_freqs.get_num_docs()
        for word in tf.keys():
            tf_idf[word] = tf[word] * np.log(N / (1 + doc_freqs[word]))
        return DictVector(tf_idf)



# TODO: put any extensions here


# Search

def experiment():
    docs = read_docs('../data/cacm.raw')
    queries = read_docs('../data/query.raw')
    rels = read_rels('../data/query.rels')
    stopwords = read_stopwords('../data/common_words')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

        #return  # TODO: just for testing; remove this when printing the full table


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim, verbose=True):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    if verbose:
        print('Query:', query)
        print('Relevant docs: ', relevant)
        print()
        for doc_id, score in results_with_score[:10]:
            print('Score:', score)
            print(docs[doc_id - 1])
            print()
    return results


def experiment_with_query_expansion():
    # Instantiate our query expander, load data
    weights = TermWeights(author=1, title=1, keyword=1, abstract=1)
    processer = QueryProcessor(weights)
    docs = read_docs('../data/cacm.raw')
    queries = read_docs('../data/query.raw')
    rels = read_rels('../data/query.rels')

    # Compute term freqs for all docs
    docs_tf = [(doc.doc_id, processer.query_expander.compute_tf(doc, weights)) for doc in tqdm(docs)]
    # Compute doc_freqs using term freqs as vocabulary
    doc_freqs = compute_doc_freqs_from_dict([j for i, j in docs_tf])

    # Compute tfidf of all docs
    doc_vectors = [(pair[0], processer.tfidf_from_tf(pair[1], doc_freqs))
                  for pair in tqdm(docs_tf)]

    # Compute tfidf of all queries
    query_vectors = [processer.tfidf_from_doc(query, doc_freqs) for query in queries]

    metrics = []
    ids = [query.doc_id for query in queries]
    queries = zip(ids, query_vectors)
    for id, query_vec in queries:
        results = search(doc_vectors, query_vec, cosine_sim)

        rel = rels[id]
        metrics.append([
            precision_at(0.25, results, rel),
            precision_at(0.5, results, rel),
            precision_at(0.75, results, rel),
            precision_at(1.0, results, rel),
            mean_precision1(results, rel),
            mean_precision2(results, rel),
            norm_recall(results, rel),
            norm_precision(results, rel)
        ])
    averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
                for i in range(len(metrics[0]))]
    print(*averages, sep='\t')

if __name__ == '__main__':
    #experiment()
    experiment_with_query_expansion()