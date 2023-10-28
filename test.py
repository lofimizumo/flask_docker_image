import requests


url = "https://df62c75409a1648289a3f219a687490c9.clg07azjl.paperspacegradient.com/stop"
data = {"deviceSn":"RX2505ACA10JOA160037"}
response = requests.post(url, json=data)
print(response.json())