import hw2
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    with open("../obj/docs_tf.pkl", "rb") as fp:
        docs_tf = pickle.load(fp)
    weights = hw2.TermWeights(author=1, title=1, keyword=1, abstract=1)
    processer = hw2.QueryProcessor(weights)
    doc_freqs = hw2.compute_doc_freqs_from_dict([j for i, j in docs_tf])
    print("processing docs...")
    docs_tfidf = [(pair[0], processer.tfidf_from_tf(pair[1], doc_freqs))
                  for pair in tqdm(docs_tf)]
    print("saving docs")
    with open("../obj/docs_tfidf.pkl", "wb") as fp:
        pickle.dump(docs_tfidf, fp)