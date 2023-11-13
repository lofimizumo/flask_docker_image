import requests


url = "https://df62c75409a1648289a3f219a687490c9.clg07azjl.paperspacegradient.com/start_shawsbay"
# url = "http://127.0.0.1:5234/"
data = {}
response = requests.post(url, json=data)
print(response.json())