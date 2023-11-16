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
import requests
amber_key = 'psk_2d5030fe84a68769b6f48ab73bd48ebf'
sn = 'RX2505ACA10J0A160016' 
start_date = '2023-11-14T00:00'
end_date = '2023-11-14T23:55'
url = "https://d231f65111596470ebbc257f8269abcda.clg07azjl.paperspacegradient.com/cost_savings"
data = {'start_date': start_date, 'end_date': end_date, 'amber_key': amber_key, 'deviceSn': sn}
response = requests.post(url, json=data)
print(response.json())
# %%
