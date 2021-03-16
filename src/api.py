import hw2
from sqlitedict import SqliteDict
from flask import Flask, request, jsonify, Response
import json
import os
import pickle
from dict_vec import DictVector
import argparse


"""
This will be our database API. Right now we still have somefiles that have to be loaded into memory
when the server starts but maybe in the future everything will be persistent.

Run this server on port 6000
"""

parser = argparse.ArgumentParser()
parser.add_argument("--no_query_expand", dest="query_expand", action="store_false")
parser.set_defaults(query_expand=True)
args = parser.parse_args()

# Load precomputed tfidf kernel for all documents
weights = hw2.TermWeights(author=1, title=1, keyword=1, abstract=1)
processer = hw2.QueryProcessor(weights, query_expand=args.query_expand)

app = Flask(__name__)
if args.query_expand:
    with open(os.path.join(app.root_path, "obj/docs_tfidf.pkl"), "rb") as fp:
        docs_tfidf = pickle.load(fp)
else:
    with open(os.path.join(app.root_path, "obj/docs_tfidf_no_expand.pkl"), "rb") as fp:
        docs_tfidf = pickle.load(fp)

doc_freqs = hw2.compute_doc_freqs_from_dict([j for i, j in docs_tfidf])
db_path = os.path.join(app.root_path, "db")
if not os.path.exists(db_path):
    os.makedirs(db_path)
# query db stores the results of queries for later use
query_db_path = os.path.join(db_path, "queries.db")
# query map is a map of queries to vectors
query_map_path = os.path.join(db_path, "query_map.db")
doc_vecs_db_path = os.path.join(db_path, "doc_vecs.db")

# Upload all doc vectors to db
doc_vec_db = SqliteDict(doc_vecs_db_path)
for doc_id, doc_vec in docs_tfidf:
    try:
        vec = doc_vec_db[doc_id]
    except KeyError:
        doc_vec_db[doc_id] = DictVector(doc_vec)
doc_vec_db.commit()
doc_vec_db.close()


# Convert document to vector, then upload
def upload_doc_vec(doc):
    #TODO Maybe implement this if time permits
    return


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
    doc_vec_db = SqliteDict(doc_vecs_db_path)
    try:
        doc_vec_db[data["doc_id"]]
    except KeyError:
        upload_doc_vec(doc)
    doc_vec_db.close()
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
    closest = search_for_query(query)
    if closest != "":
        query_db = SqliteDict(query_db_path)
        results = query_db[closest]
        if len(results) > k:
            results = results[:k]
        query_db.close()
        results = get_docs_from_list(results)
        new_query = closest
    else:
        results = get_nearest(query, k=k)
        upload_query(query, results)
        new_query = query
    return jsonify({"results": results, "query": new_query})


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


def get_docs_from_list(doc_list):
    out = {}
    for doc in doc_list:
        out[doc] = json.loads(get_doc(doc).data)
    return out


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


def sim_query(q1: str, q2: str):
    return hw2.cosine_sim(query2vec(q1), query2vec(q2))


# Returns similar query if it exists, empty string if there is none
def search_for_query(query, sim_thresh=0.8) -> str:
    query_map = SqliteDict(query_map_path)
    max_sim_query = ""
    max_sim = 0
    for key in query_map.keys():
        sim = sim_query(key, query)
        if sim > max_sim:
            max_sim = sim
            max_sim_query = key
    query_map.close()
    if max_sim > sim_thresh:
        return max_sim_query
    else:
        return ""


def set_query_results(query, k=20):
    query_map = SqliteDict(query_map_path)
    query_vec = query_map[query]
    query_map.close()
    query_db = SqliteDict(query_db_path)
    results = search(docs_tfidf, query_vec, hw2.cosine_sim)
    query_db[query] = results[:k]
    query_db.commit()
    query_db.close()


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


def update_query(query, relevant=None, irrelevant=None, alpha=0.9, beta=0.5, gamma=0.1):
    """
    Update query in our db using rocchio algorithm. Note, the query string is not updated but
    the results in the results db are updated.
    :param query: query string
    :param relevant: list of doc_ids
    :param irrelevant: list of doc_ids
    :param alpha: weight of original query
    :param beta: weight of relevant docs
    :param gamma: weight or irrelevant docs
    :return: True if successful, False if unsuccessful
    """
    if relevant is None:
        relevant = []
    if irrelevant is None:
        irrelevant = []
    assert (query != "")
    query_map = SqliteDict(query_map_path)
    try:
        q0 = query_map[query]
    except KeyError:
        # Can't update queries we've never seen
        query_map.close()
        return False
    if not isinstance(q0, DictVector):
        q0 = DictVector(q0)
    doc_vec_db = SqliteDict(doc_vecs_db_path)
    Nr = len(relevant)
    for doc_id in relevant:
        try:
            doc_vec = doc_vec_db[doc_id]
            if not isinstance(doc_vec, DictVector):
                doc_vec = DictVector(doc_vec)
        except KeyError:
            continue
        q0 = q0 + (beta / Nr) * doc_vec
    Ni = len(irrelevant)
    for doc_id in irrelevant:
        try:
            doc_vec = doc_vec_db[doc_id]
            if not isinstance(doc_vec, DictVector):
                doc_vec = DictVector(doc_vec)
        except KeyError:
            continue
        q0 = q0 - (gamma / Ni) * doc_vec
    query_map[query] = q0
    query_map.commit()
    set_query_results(query)


def undo_update(query, relevant=None, irrelevant=None, alpha=0.9, beta=0.5, gamma=0.1):
    """
    Method for undoing an update if a user decides a post that was relevant isn't actually relevant.
    :param query: query string
    :param relevant: list of doc_ids
    :param irrelevant: list of doc_ids
    :param alpha: weight of original query
    :param beta: weight of relevant docs
    :param gamma: weight or irrelevant docs
    :return: True if successful, False if unsuccessful
    """
    if relevant is None:
        relevant = []
    if irrelevant is None:
        irrelevant = []
    assert (query != "")
    query_map = SqliteDict(query_map_path)
    try:
        q0 = query_map[query]
    except KeyError:
        # Can't update queries we've never seen
        query_map.close()
        return False
    if not isinstance(q0, DictVector):
        q0 = DictVector(q0)
    doc_vec_db = SqliteDict(doc_vecs_db_path)
    Nr = len(relevant)
    for doc_id in relevant:
        try:
            doc_vec = doc_vec_db[doc_id]
            if not isinstance(doc_vec, DictVector):
                doc_vec = DictVector(doc_vec)
        except KeyError:
            continue
        q0 = q0 - (beta / Nr) * doc_vec
    Ni = len(irrelevant)
    for doc_id in irrelevant:
        try:
            doc_vec = doc_vec_db[doc_id]
            if not isinstance(doc_vec, DictVector):
                doc_vec = DictVector(doc_vec)
        except KeyError:
            continue
        q0 = q0 + (gamma / Ni) * doc_vec
    query_map[query] = q0
    query_map.commit()
    set_query_results(query)


@app.route('/query/update', methods=["POST"])
def relevance_feedback():
    data = json.loads(request.data)
    relevant = data["relevant"]
    irrelevant = data["irrelevant"]
    undo = data["undo"]
    query = data["query"]
    if undo == "TRUE":
        undo_update(query, relevant=relevant, irrelevant=irrelevant)
    else:
        update_query(query, relevant, irrelevant)
    return Response(status=201)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=6000)
