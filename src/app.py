from flask import Flask, request, render_template, url_for
import json
import requests

"""
Client facing web application

Run this on port 5000 (default for Flask)
"""
db_url = "http://127.0.0.1:6000/"
app = Flask(__name__)


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def results():
    query = request.form["query"]
    payload = json.dumps({'query': query, 'k': 20})
    docs = requests.get(db_url + "docs/k_nearest", data=payload)
    docs = json.loads(docs.content)
    return render_template('results.html', docs=docs.items())


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
