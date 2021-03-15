import hw2
import json
import requests

url = "http://127.0.0.1:6000/docs"

if __name__ == "__main__":
    docs = hw2.read_docs('../../data/cacm.raw')
    for doc in docs:
        doc_dict = {"doc_id": doc.doc_id,
                    "keyword": doc.keyword,
                    "abstract": doc.abstract,
                    "author": doc.author,
                    "title": doc.title}
        json_doc = json.dumps(doc_dict)
        r = requests.post(url, data=json_doc)
        if r.status_code != 201:
            print("{} failed to upload".format(doc.doc_id))
