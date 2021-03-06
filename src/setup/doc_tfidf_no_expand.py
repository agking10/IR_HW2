import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hw2
from tqdm import tqdm
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords

stemmer = SnowballStemmer('english')
stopwords = nltk_stopwords.words('english')
if not os.path.exists("../obj"):
    os.makedirs("../obj")

if __name__ == "__main__":
    docs = docs = hw2.read_docs('../../data/cacm.raw')
    processed_docs, _ = hw2.process_docs_and_queries(docs, [], stem=True, removestop=True, stopwords=stopwords)
    weights = hw2.TermWeights(author=1, title=1, keyword=1, abstract=1)
    processer = hw2.QueryProcessor(weights, query_expand=False)
    doc_freqs = hw2.compute_doc_freqs(docs)
    print("processing docs...")
    docs_tfidf = [(doc.doc_id, processer.tfidf_from_doc(doc, doc_freqs))
                  for doc in tqdm(docs)]
    print("saving docs")
    with open("../obj/docs_tfidf_no_expand.pkl", "wb") as fp:
        pickle.dump(docs_tfidf, fp)