import hw2
import pickle

if __name__ == "__main__":
    weights = hw2.TermWeights(author=1, title=1, keyword=1, abstract=1)
    processer = hw2.QueryProcessor(weights)
    with open("../src/obj/docs_tfidf.pkl", "rb") as fp:
        docs_tfidf = pickle.load(fp)
    doc_freqs = hw2.compute_doc_freqs_from_dict([j for i, j in docs_tfidf])
    query = "This is a test. Dog cat ran."
    vec = processer.tfidf_from_query(query, doc_freqs)
    print(vec)