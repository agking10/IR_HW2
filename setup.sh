#!/usr/bin/bash

cd src/setup
python expand_docs.py
python calculate_doc_tfidf.py