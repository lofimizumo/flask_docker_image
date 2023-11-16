#%%
import requests


url = "https://d231f65111596470ebbc257f8269abcda.clg07azjl.paperspacegradient.com/start_shawsbay"
# url = "http://127.0.0.1:5234/"
data = {}
response = requests.post(url, json=data)
print(response.json())

#%%
import requests

url = "https://d231f65111596470ebbc257f8269abcda.clg07azjl.paperspacegradient.com/start"
# url = "http://127.0.0.1:5000/start"
data = {'deviceSn': 'RX2505ACA10J0A160016'}
response = requests.post(url, json=data)
print(response.json())
# %%
