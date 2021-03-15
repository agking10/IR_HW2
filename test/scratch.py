import json
import hw2

if __name__ == "__main__":
    docs = hw2.read_docs('../data/cacm.raw')
    doc = docs[0]
    doc_dict = {"id": doc.doc_id, "keyword": doc.keyword, "abstract":doc.abstract, "author": doc.author}
    json_doc = json.dumps(doc_dict)
    doc_dict_loaded = json.loads(json_doc)
    print(json.doc_id)
