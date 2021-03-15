import hw2
import torch
from sqlitedict import SqliteDict
from flask import Flask, request, jsonify, Response
import json
import os
import pickle
from dict_vec import DictVector

embedder = hw2.Embedder()

"""
This will be our database API. Right now we are going to simply upload 
everything to memory when the server starts but maybe in the future it'll
be upgraded to a persistent DB.

Run this server on port 6000
"""

# Load precomputed tfidf kernel for all documents
weights = hw2.TermWeights(author=1, title=1, keyword=1, abstract=1)
processer = hw2.QueryProcessor(weights)

app = Flask(__name__)
with open(os.path.join(app.root_path, "obj/docs_tfidf.pkl"), "rb") as fp:
    docs_tfidf = pickle.load(fp)
doc_freqs = hw2.compute_doc_freqs_from_dict([j for i, j in docs_tfidf])
db_path = os.path.join(app.root_path, "db")
# query db stores the results of queries for later use
query_db = SqliteDict(os.path.join(db_path, "queries.db"))
query_db_path = os.path.join(db_path, "queries.db")
# query map is a map of queries to vectors
query_map = SqliteDict(os.path.join(db_path, "query_map.db"))
query_map_path = os.path.join(db_path, "query_map.db")
doc_vecs_db_path = os.path.join(db_path, "doc_vecs.db")


@app.route("/docs", methods=["POST"])
def upload_doc():
    doc_db = SqliteDict(os.path.join(db_path, "docs.db"))
    data = json.loads(request.data)
    doc = {"doc_id": data['doc_id'],
           "keyword": data['keyword'],
           "author": data['author'],
           "abstract": data['abstract'],
           "title": data['title']}
    doc_db[data["doc_id"]] = doc
    doc_db.commit()
    doc_db.close()
    return "Success", 201


@app.route("/docs/<int:doc_id>")
def get_doc(doc_id):
    doc_db = SqliteDict(os.path.join(db_path, "docs.db"))
    try:
        document = doc_db[doc_id]
    except KeyError:
        doc_db.close()
        return Response(status=404)
    doc_db.close()
    return jsonify(document)


@app.route("/docs/<int:doc_id>", methods=["DELETE"])
def remove_doc(doc_id):
    doc_db = SqliteDict(os.path.join(db_path, "docs.db"))
    del doc_db[doc_id]
    doc_db.close()
    return "Success", 200


# Returns k nearest neighbors to a query vector
@app.route("/docs/k_nearest")
def get_neighbors():
    data = json.loads(request.data)
    query = data['query']
    k = data['k']
    results = get_nearest(query, k=k)
    return jsonify(results)


def search(doc_pairs, query_vec, sim):
    """
    Linear search through documents. doc_pairs must be a tuple list of
    (<doc_id>, <sparse_doc_vector>) tuples
    """
    results_with_score = [(doc_id, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in doc_pairs]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


# Return sparse vector from query
def query2vec(query):
    processed_query = processer.tfidf_from_query(query, doc_freqs)
    processed_query = DictVector(processed_query)
    return processed_query


def get_nearest(query: str, k=20) -> dict:
    # TODO implement nearest neighbor search by query
    processed_query = query2vec(query)
    results = search(docs_tfidf, processed_query, hw2.cosine_sim)
    if k < len(results):
        results = results[:k]
    out = {}
    for result in results:
        out[result] = json.loads(get_doc(result).data)
    return out


# similarity metric for query strings
def query_sim(q1: str, q2: str):
    return 1


# Returns similar query if it exists, empty string if there is none
def search_for_query(sim_thresh=0.8) -> str:
    query_db = SqliteDict(os.path.join(db_path, "queries.db"))
    data = json.loads(request.data)
    query = data['query']
    max_sim_query = ""
    max_sim = 0
    for key in query_db.keys():
        sim = query_sim(query, key)
        if sim > max_sim:
            max_sim = sim
            max_sim_query = key
    query_db.close()
    if max_sim > sim_thresh:
        return max_sim_query
    else:
        return ""


# Puts a new query in database with results list.
# Returns True if successful, False otherwise
def upload_query(query, results):
    query_db = SqliteDict(query_db_path)
    query_map = SqliteDict(query_map_path)
    flag = False
    try:
        query_db[query] = results
        query_vec = query2vec(query)
        query_map[query] = query_vec
        query_db.commit()
        query_map.commit()
        flag = True
    except:
        pass
    query_db.close()
    query_map.close()
    return flag


def update_query(query, relevant=None, irrelevant=None):
    if relevant is None:
        relevant = []
    if irrelevant is None:
        irrelevant = []
    assert (query != "")




if __name__ == "__main__":
    app.run(host='127.0.0.1', port=6000)
