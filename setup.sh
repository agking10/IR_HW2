#!/usr/bin/bash

cd src/setup
python doc_tfidf_no_expand.py
python expand_docs.py
python calculate_doc_tfidf.py
