from flask import Flask, request, render_template, session, Response
import json
import requests

"""
Client facing web application

Run this on port 5000 (default for Flask)
"""
db_url = "http://127.0.0.1:6000/"
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='dev'
)

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def results():
    query = request.form["query"]
    payload = json.dumps({'query': query, 'k': 20})
    resp = requests.get(db_url + "docs/k_nearest", data=payload)
    resp_json = json.loads(resp.content)
    docs = resp_json["results"]
    query = resp_json["query"]
    session["query"] = query
    session["docs"] = {}
    return render_template('results.html', docs=docs.items(), query=query)


possible_states = {"RELEVANT", "IRRELEVANT", "NULL"}


@app.route('/relevant', methods=["POST"])
def update_relevant_docs():
    data = json.loads(request.data)
    doc_id = data["doc_id"]
    state = data["action"].upper()
    assert (state in possible_states)
    doc_state = session.pop("docs", None)
    prev_state = doc_state.get(doc_id, None)
    query = session["query"]
    if prev_state is None:
        # If we are choosing fresh document just mark irrelevant or relevant
        status = send_relevace_to_db(query, doc_id, state)
        doc_state[doc_id] = state
    elif state == "NULL":
        # If unmarking document undo changes
        status = send_relevace_to_db(query, doc_id, action=prev_state, undo=True)
        del doc_state[doc_id]
    else:
        # If changing decision, undo previous choice, do new choice
        send_relevace_to_db(query, doc_id, action=prev_state, undo=True)
        status = send_relevace_to_db(query, doc_id, action=state)
        doc_state[doc_id] = state
    session["docs"] = doc_state
    return render_template("results.html")


def send_relevace_to_db(query, doc_id, action="RELEVANT", undo=False):
    relevant = []
    irrelevant = []
    if action == "RELEVANT":
        relevant.append(doc_id)
    elif action == "IRRELEVANT":
        irrelevant.append(doc_id)
    else:
        return
    data = {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "query": query,
        "undo": "FALSE" if not undo else "TRUE",
    }
    r = requests.post(db_url + "query/update", data=json.dumps(data))
    return r.status_code

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
