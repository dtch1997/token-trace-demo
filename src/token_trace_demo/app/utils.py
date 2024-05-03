import json
import urllib.parse
import webbrowser


def get_neuronpedia_url(
    layer: int, features: list[int], name: str = "temporary_list"
) -> str:
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": "gpt2-small", "layer": f"{layer}-res-jb", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    return url


def open_neuronpedia(layer: int, features: list[int], name: str = "temporary_list"):
    url = get_neuronpedia_url(layer, features, name)
    webbrowser.open(url)
