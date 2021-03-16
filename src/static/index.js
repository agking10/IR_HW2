const hostServer = "http://127.0.0.1:5000/"

function deactivate(caller) {
    if (caller.classList.contains("active")) {
        caller.classList.remove("active");
    }
}

function markRelevant(caller) {
    var doc_id = caller.parentElement.getAttribute('doc_id');
    if (caller.classList.contains("active")) {
    caller.classList.remove("active");
    // If we unmark on our own, deactivate marking
    fetch(hostServer + "relevant", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "doc_id": doc_id,
                "action": "null"
            })
        }).then(result => result);
    } else {
        // Mark as relevant and change button colors
        caller.classList.add("active");
        var dislike_btn = document.getElementById("dislike_"+doc_id);
        deactivate(dislike_btn);
        fetch(hostServer + "relevant", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "doc_id": doc_id,
                "action": "relevant"
            })
        }).then(result => result);
    }
}

function markIrrelevant(caller) {
    var doc_id = caller.parentElement.getAttribute('doc_id');
    if (caller.classList.contains("active")) {
    caller.classList.remove("active");
    // If we unmark on our own, deactivate marking
    fetch(hostServer + "relevant", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "doc_id": doc_id,
                "action": "null"
            })
        }).then(result => result);
    } else {
        // Mark as irrelevant and change button colors
        caller.classList.add("active");
        var like_btn = document.getElementById("like_"+doc_id);
        deactivate(like_btn);
        fetch(hostServer + "relevant", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "doc_id": doc_id,
                "action": "irrelevant"
            })
        }).then(result => result);
    }
}


