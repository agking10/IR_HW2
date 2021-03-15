const hostServer = "http://127.0.0.1:5000/"

function api(path, params, method) {
  let options;
  options = {
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    method: method,
    ...(params && { body: JSON.stringify(params) }),
  };

  return fetch(hostServer + path, options)
      .then(resp => resp.json())
      .then(json => json)
      .catch(error => error);
}

function search() {
    const query = document.getElementById("searchbar")
    let params = {query: query}
    console.log(hostServer)
    api("", 'GET', params).then(resp => resp)
}

function printHello() {
    console.log(hostServer)
    fetch(hostServer, {method:"GET"}).then(resp => resp)
}

document.getElementById("submit-search").onclick = search



