
# Search Engine Implementation with Relevance Feedback
### HW2 Extenstion
### Andrew King (aking65)

For this assignment, I chose to implement extensions 1 and 12 (with a little bit)
of extension 6. That is, I chose to implement a full web server search engine
and the SMART Rocchio algorithm for relevance feedback. I say I did a little
bit of extension 6 because I did implement query expansion but a significant 
portion of it comes directly from a Medium article.

There are a lot of pieces to this server so here is my attempt to describe
the project at a high level. If you just downloaded this folder, there are a few
scripts to run to set things up. First, install the required packages using `pip install -r requirements.txt`. 
Next, enter the command `sh setup.sh`

This will create files that serialize the vector representations of all the docs we were given.
Next run `sh run_db.sh`

This will start the database server. Inside the file there is a flag `--no_query_expand` 
that is set by default. While this flag is set, the search engine will not use
query expansion. I tried to build a RESTful API for the
database so development would be easier. Next, in a new terminal window
run `sh upload_docs.sh`

This will upload all the documents to the document database. There are two 
<key, value> stores for the documents, both indexed by document id. 
The first stores the document as a dictionary and the second stores the 
tfidf vector representation of the document.

Finally, run `sh start_webserver.sh`

The search engine should be available at `http://127.0.0.1:5000/`

Once you pull the server up, type in any query and search to see the results.
The results are displayed with the document ID and title in bold with the abstract
below, if it is present. The queries are computed using the tfidf vector
representation and cosine similarity with equal weighting for all sections of
the document. I used query expansion with NLTK/Wordnet, so all synonyms and
hypernyms of words in the queries and documents are included as well. The expansion
is done using the class `QueryExpander` class. Processing into tfidf vectors
is done using the `QueryProcessor` class.

To the right of each result there is a pair
of thumbs up/down buttons. Selecting them is the interface for the relevance
feedback mechanism. The algorithm I implemented is the SMART Rocchio algorithm.
When you select the thumbs up button, a call is made to the server to 
add that document vector (multiplied by a hyperparameter) to the query vector.
Similarly, when you select the thumbs down button, a call is made to subtract
a fraction of that document vector from the query.

How is the query manipulation actually implemented? If you are watching the source
files while using the search engine, you'll see two files, queries.db and query_map.db
appear in the /db/ folder. When a query is sent to the database server, it is
converted to a query vector by the `QueryProcessor` class, and the database checks
query_map.db to see if it has seen any similar queries. Query_map.db stores 
a map between a query string and a query vector. Note that the query vector in this
database is not necessarily the same as the output of the processor when given 
the query key. The value stored in the database is the vector used for search, so it has
potentially been manipulated by relevance feedback.

Because of this, when the database compares the similarity of an incoming query with queries that
have been seen before, it uses the "vanilla" query vector for all strings, i.e.
the result of processing each query string from scratch. Similarity calculations
are done using cosine similarity.

If a suitable match is found, the results of the query are taken from queries.db, where the
results have already been precomputed and stored. If no suitable match is found, 
the query is stored along with its query vector in query_map.db. The server then calculates
the results of the query and stores these in queries.db. The results and the query (or query
match) are sent back to the webserver to be stored as variables for the user's session.

When a user clicks the like button on a search result, the webserver makes a request to 
the database server that includes the query that generated the result (original or matched) and
the document id of the result. The query vector stored in query_map.db is then 
modified according to the Rocchio algorithm and new search results are computed and stored
in queries.db. The function that actually carries out the algorithm is called `update_query`
in api.py.

An important fact about the implementation of this is that the webserver keeps track
of the state of each button so that the user can change her selection 
of relevance for the document. As an example, if a user selects the like button
and the query is modified in accordance with that document being relevant but then she
switches to the dislike button, the server will tell the database to "undo" the like 
as if it never happened. Similarly, if all buttons are unselected, the end result will
be as if no buttons had been pressed in the first place.

A class I implemented that made this easier is the `DictVector` class from dict_vec.py. It
extends the dictionary class to support vector addition and scalar multiplication.

To test this feature out, you can click a few like/dislike buttons then reenter
the same search and see how the results are affected. In the file hw2, I added 
a benchmarking experiment for query expansion. Interestingly, it made the performance
of the search worse. I measured an r_norm of 0.85 with expansion vs. 0.93 without and a p_norm
of 0.49 compared with 0.72. I believe this is due to the fact that the synonyms
were taken from Wordnet which is an extremely general purpose english language 
database, but these documents are from a highly specific field, so query expansion would 
just have the effect of making everything more similar to everything else.