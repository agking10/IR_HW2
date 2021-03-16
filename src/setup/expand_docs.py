import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hw2
from tqdm import tqdm
import pickle

if not os.path.exists("../obj"):
    os.makedirs("../obj")

if __name__ == "__main__":
    docs = docs = hw2.read_docs('../../data/cacm.raw')
    weights = hw2.TermWeights(author=1, title=1, keyword=1, abstract=1)
    processer = hw2.QueryProcessor(weights)
    print("processing docs...")
    docs_tf = [(doc.doc_id, processer.query_expander.compute_tf(doc, weights)) for doc in tqdm(docs)]
    print("docs processed.")
    print("saving docs")
    with open("../obj/docs_tf.pkl", "wb") as fp:
        pickle.dump(docs_tf, fp)