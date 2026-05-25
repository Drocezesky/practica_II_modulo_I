import requests
import httpx

def test_api():
    url = "http://localhost:8000/questions/433"
    payload = {
        "question": "Let's hang out"
    }
    response = requests.get(url)
    new_question = response.json()
    assert (new_question["question"] == payload["question"]) and (response.status_code == 200)

def test_api_2():
    url = "http://localhost:8000/questions/463"
    payload = {
        "question": "Who knows"
    }
    response = httpx.get(url)
    new_question = response.json()
    assert (new_question["question"] == payload["question"]) and (response.status_code == 200)