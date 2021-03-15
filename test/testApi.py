import unittest
import requests
import json
from hw2 import Document

url = "http://127.0.0.1:6000/docs"

class testApi(unittest.TestCase):

    def setUp(self) -> None:
        self.test_doc = Document(1,["This"], ["is", "a"], [], ["Test"])

    def testDocUploadReturnCode(self):
        doc = self.test_doc
        doc_dict = {"doc_id": doc.doc_id, "keyword": doc.keyword, "abstract": doc.abstract, "author": doc.author, "title": doc.title}
        json_doc = json.dumps(doc_dict)
        r = requests.post(url, data=json_doc)
        self.assertEqual(r.status_code, 201)

    def testFindDocDoesntExist(self):
        r = requests.get(url + "/2")
        self.assertEqual(r.status_code, 404)

    def testDocUploadDownload(self):
        doc = self.test_doc
        doc_dict = {"doc_id": doc.doc_id, "keyword": doc.keyword, "abstract": doc.abstract, "author": doc.author,"title": doc.title}
        json_doc = json.dumps(doc_dict)
        requests.post(url, data=json_doc)
        r = requests.get(url + "/{}".format(doc.doc_id))
        response = r.json()
        self.assertEqual(response, doc_dict)

    def testDeleteDoc(self):
        doc = self.test_doc
        doc_dict = {"doc_id": doc.doc_id, "keyword": doc.keyword, "abstract": doc.abstract, "author": doc.author, "title": doc.title}
        json_doc = json.dumps(doc_dict)
        requests.post(url, data=json_doc)
        r = requests.delete(url+ "/{}".format(doc.doc_id))
        self.assertEqual(r.status_code, 200)
        r = requests.get(url+ "/{}".format(doc.doc_id))
        self.assertEqual(r.status_code, 404)

    def testUpdateDoc(self):
        doc = self.test_doc
        doc_dict = {"doc_id": doc.doc_id, "keyword": doc.keyword, "abstract": doc.abstract, "author": doc.author,
                    "title": doc.title}
        json_doc = json.dumps(doc_dict)
        r = requests.post(url, data=json_doc)
        new_doc = Document(doc.doc_id, ["This"], ["is"], ["a"], ["new", "doc"])
        doc_dict = {"doc_id": new_doc.doc_id, "keyword": new_doc.keyword, "abstract": new_doc.abstract, "author": new_doc.author,
                    "title": new_doc.title}
        json_doc = json.dumps(doc_dict)
        r = requests.post(url, data=json_doc)
        r = requests.get(url + "/{}".format(doc.doc_id))
        response = r.json()
        self.assertEqual(response, doc_dict)


    def tearDown(self) -> None:
        doc = self.test_doc
        r = requests.delete(url+ "/{}".format(doc.doc_id))


