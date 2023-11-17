import requests
import pytz
from datetime import datetime, timedelta
import numpy as np


class AmberFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = f"https://api.amber.com.au/v1"
        self.site_id = None

    def get_site(self):
        header = {'Authorization': f'Bearer {self.api_key}',
                  'accept': 'application/json'}
        response = requests.get(f"{self.base_url}/sites", headers=header)
        return response.json()[0]['id']

    def get_prices(self, start_date, end_date, resolution=5):
        if not self.site_id:
            self.site_id = self.get_site()
        header = {'Authorization': f'Bearer {self.api_key}',
                  'accept': 'application/json'}
        # make end_date 1 day before
        end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M") - timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%dT%H:%M")
        url = f"{self.base_url}/sites/{self.site_id}/prices?startDate={start_date}&endDate={end_date}&resolution={resolution}"
        response = requests.get(url, headers=header)
        if response.status_code != 200:
            raise Exception("Error: " + str(response.status_code))
        data = response.json()
        general_data = list(filter(lambda x: x['channelType'] == 'general', data))
        prices = [(x['nemTime'], x['perKwh']) for x in general_data]
        feed_in_data = list(filter(lambda x: x['channelType'] == 'feedIn', data))
        feed_in_prices = [(x['nemTime'], x['perKwh']) for x in feed_in_data]
        return prices, feed_in_prices




class BatteryMonitor:
    def __init__(self, api_version='dev3'):
        self.api = None
        if api_version == 'dev3':
            self.api = ApiCommunicator(
                base_url="https://dev3.redxvpp.com/restapi")
        else:
            self.api = ApiCommunicator(
                base_url="https://redxpower.com/restapi")
        self.token = None
        self.token_last_updated = datetime.now()

    def get_token(self, api_version='redx'):
        if self.token and (datetime.now(tz=pytz.timezone('Australia/Sydney')) - self.token_last_updated) < timedelta(hours=1):
            return self.token
        if api_version == 'redx':
            response = self.api.send_request("user/token", method='POST', data={
                'user_account': 'yetao_admin', 'secret': 'tpass%#%'}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        else:
            response = self.api.send_request("user/token", method='POST', data={
                'user_account': 'yetao_admin', 'secret': 'a~L$o8dJ246c'}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        self.token_last_updated = datetime.now(tz=pytz.timezone('Australia/Sydney'))
        return response['data']['token']

    def get_history_load(self,start_date, end_date,sn):
        def _get_history_usage(start_date, end_date, sn):
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            for i in range((end_date - start_date).days + 1):
                date = start_date + timedelta(days=i)
                daily_data = self._get_daily_usage(sn, date.strftime("%Y_%m_%d"))
                if daily_data is not None:
                    yield [x['loadP'] for x in daily_data]
        history_data = _get_history_usage(start_date, end_date, sn)
        load = list(history_data)

        return [item for sublist in load for item in sublist] 
    def get_history_grid_power(self,start_date, end_date,sn):
        def _get_history_usage(start_date, end_date, sn):
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            for i in range((end_date - start_date).days + 1):
                date = start_date + timedelta(days=i)
                daily_data = self._get_daily_usage(sn, date.strftime("%Y_%m_%d"))
                if daily_data is not None:
                    yield [x['gridP'] for x in daily_data]
        history_data = _get_history_usage(start_date, end_date, sn)
        load = list(history_data)
        return [item for sublist in load for item in sublist]

        
    
    def _get_daily_usage(self, sn, date):
        data = {'deviceSn': sn, 'logDate': date, 'samplePeriod': 5}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "device/get_daily_data", method='POST', json=data, headers=headers)
        return response['data']
    


class ApiCommunicator:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def send_request(self, command, method="GET", data=None, json=None, headers=None, retries=3):
        url = f"{self.base_url}/{command}"
        for _ in range(retries):
            try:
                if method == "GET":
                    response = self.session.get(url, headers=headers)
                elif method == "POST":
                    response = self.session.post(
                        url, data=data, json=json, headers=headers, timeout=None)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                print(f"Error occurred: {e}. Retrying...")

        raise ConnectionError(
            f"Failed to connect to {url} after {retries} attempts.")


def get_prices(start_date: str, end_date: str, amber_key: str):
    fetcher = AmberFetcher(amber_key)
    try:
        prices, feed_in_prices = fetcher.get_prices(start_date, end_date)
    except Exception:
        raise Exception('Failed to get prices')

    json_prices = [{'date': x[0], 'price': x[1]} for x in prices]
    json_feed_in_prices = [{'date': x[0], 'price': x[1]} for x in feed_in_prices]

    return {'general':json_prices, 'feedIn':json_feed_in_prices}


def cost_savings(start_date, end_date, amber_key, sn): 
    fetcher = AmberFetcher(amber_key)
    prices, feed_in_prices = fetcher.get_prices(start_date, end_date)
    
    monitor = BatteryMonitor(api_version='redx')
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M")
    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M")
    history_usage = monitor.get_history_load(start_date, end_date, sn)
    history_grid_power = monitor.get_history_grid_power(start_date, end_date, sn)
    feed_in_prices = [x[1] for x in feed_in_prices]
    prices = [x[1] for x in prices]
    cost_without_battery = calculate_cost(history_usage, prices, feed_in_prices)
    cost_with_battery = calculate_cost(history_grid_power, prices, feed_in_prices)
    if cost_without_battery == 0:
        return 0
    else:
        return {'cost_savings': (cost_without_battery - cost_with_battery) / cost_without_battery, 'cost_without_battery': cost_without_battery, 'cost_with_battery': cost_with_battery} 

def calculate_cost(usage, usage_price, feed_in_price):
    """
    Calculate the total cost based on usage and prices.

    Args:
    usage (list of float): Array of power usage in kilowatts, where negative values represent power fed into the grid.
    usage_price (list of float): Price per kilowatt for power usage.
    feed_in_price (list of float): Price per kilowatt for power fed into the grid.

    Returns:
    float: Total cost
    """
    total_cost = 0
    for i, power in enumerate(usage):
        if power > 0:
            # Power usage
            price = usage_price[min(i, len(usage_price) - 1)]  # Use the last price if index out of range
            total_cost += power * price
        else:
            # Power feed-in
            price = feed_in_price[min(i, len(feed_in_price) - 1)]  
            total_cost += -power * price  

    return total_cost


    # amber_key = 'psk_2d5030fe84a68769b6f48ab73bd48ebf'
    # sn = 'RX2505ACA10J0A160016' 
    # start_date = '2023-11-14T00:00'
    # end_date = '2023-11-14T23:55'
    # savings = cost_savings(start_date, end_date, amber_key, sn)
    # print(savings)

if __name__ == "__main__":
    prices = get_prices('2023-11-14T00:00', '2023-11-14T23:55', 'psk_2d5030fe84a68769b6f48ab73bd48ebf')
    print(prices)