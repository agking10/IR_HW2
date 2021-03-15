import unittest
import hw2


class EmbedderTest(unittest.TestCase):

    def setUp(self) -> None:
        self.embedder = hw2.Embedder()
        self.docs = hw2.read_docs("../data/cacm.raw")

    def testEmbedderWorks(self):
        doc = self.docs[0]
        vec = self.embedder.doc2vec(doc)
        print(vec)