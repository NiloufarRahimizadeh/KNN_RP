import requests


download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

target_file = "abalone.data"

response = requests.get(download_url)
response.raise_for_status()
with open(target_file, "bw") as f:
    f.write(response.content)