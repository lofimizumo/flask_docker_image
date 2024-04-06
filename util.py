from functools import wraps
import requests
import time
from datetime import datetime, timedelta
import pytz
from itertools import cycle
import logging
import tomli
import os

def load_config(file_path='config.toml'):
    with open(file_path, 'rb') as file:
        config = tomli.load(file)
    return config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def api_status_check(max_retries=10, delay=10):
    """
    A decorator function that checks the status of a device after executing a command.

    Args:
        confirm_delay (int): The delay (in seconds) before checking the status.
        sn (str): The serial number of the device.

    Returns:
        function: The decorator function.
    """
    def _is_charging(status):
        charging_status = ['On-Grid Charging']
        return status in charging_status

    def _is_discharging(status):
        discharging_status = ['Off-Grid Discharging', 'On-Grid Discharging']
        return status in discharging_status

    def _is_idle(status):
        idle_status = ['PassBy', 'Idle', 'StandBy']
        return status in idle_status

    def _is_command_expected(status, command):
        """
        Checks if the current status matches the expected status based on the command.

        Args:
            status (str): The current status of the device.
            command (str): The command executed on the device.

        Returns:
            bool: True if the current status matches the expected status, False otherwise.
        """
        if command == 'Charge':
            return _is_charging(status)
        elif command == 'Discharge':
            return _is_discharging(status)
        elif command == 'Idle':
            return _is_idle(status)
        elif command == 'Clear Fault':
            pass
        elif command == 'Power Off':
            pass

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                response = func(*args, **kwargs)
                if not check_response(response):
                    logging.error(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(delay)  # Sleep for some time before retrying
                else:
                    # logging.info("Status successfully changed!")
                    return response
            logging.error(
                "Max retry limit reached! Stopping further attempts.")
            return response

        def check_response(response):
            if response is None or not isinstance(response, dict):
                logging.error(f"Invalid response: {response}")
                return False
                # raise ValueError("Invalid response: None or not a dictionary.")
            return True

        def check_status(args, kwargs):
            is_TestMode = args[0].test_mode
            if is_TestMode:
                return True

            sn = kwargs['sn']
            expected_status = kwargs
            api = ApiCommunicator(base_url="https://dev3.redxvpp.com/restapi")
            token = args[0].token

            # Send an API request to check the status
            status_response = api.send_request('device/get_latest_data', method='POST', json={
                'deviceSn': sn}, headers={'token': token})
            logging.info(f"Status: {status_response['data']['showStatus']}")

            if _is_command_expected(status_response['data']['showStatus'], expected_status):
                return True
            else:
                return False

        return wrapper

    return decorator


class PriceAndLoadMonitor:
    def __init__(self,  test_mode=False, api_version='dev3'):
        self.config = load_config()
        self.sim_load_iter = self.get_sim_load_iter()
        self.sim_time_iter = self.get_sim_time_iter()
        self.api = None
        if api_version == 'dev3':
            self.api = ApiCommunicator(
                base_url="https://dev3.redxvpp.com/restapi")
        else:
            self.api = ApiCommunicator(
                base_url="https://redxpower.com/restapi")
        self.token = None
        self.token_last_updated = datetime.now(
            tz=pytz.timezone('Australia/Sydney'))
        # Test Mode is for testing the battery control without sending command to the actual battery
        self.test_mode = test_mode
        self.get_project_stats_call_count = 0
        self.get_meter_reading_stats_call_count = 0
        self.amber_api_url_qld = self.config.get('apiurls', {}).get('apiurl_qld', None)
        self.amber_api_url_nsw = self.config.get('apiurls', {}).get('apiurl_nsw', None)
        self.amber_api_key_qld = os.getenv(self.config.get('apikey_varnames', {}).get('apikey_varname_qld', None))
        self.amber_api_key_nsw = os.getenv(self.config.get('apikey_varnames', {}).get('apikey_varname_nsw', None))
        logging.info(f"API KEY QLD: {self.amber_api_key_qld}")
        logging.info(f"API KEY NSW: {self.amber_api_key_nsw}")

    def get_realtime_price(self, location='qld'):
        if location == 'qld':
            url = self.amber_api_url_qld
            api_key = self.amber_api_key_qld
        elif location == 'nsw':
            url = self.amber_api_url_nsw
            api_key = self.amber_api_key_nsw
        header = {'accept': 'application/json',
                  'Authorization': f'Bearer {api_key}'}
        r = requests.get(url, headers=header, timeout=5)
        prices = [x['perKwh'] for x in r.json()]
        return (prices[0], -prices[1]) # prices[1] is the feed in price, so we return the negative value

    def get_price_history(self, location='qld'):
        if location == 'qld':
            api_key = self.amber_api_key_qld
        elif location == 'nsw':
            api_key = self.amber_api_key_nsw
        fetcher = AmberFetcher(api_key)
        yesterday_date = (datetime.now(tz=pytz.timezone(
            'Australia/Sydney')) - timedelta(days=1)).strftime("%Y-%m-%d")
        day_before_yesterday_date = (datetime.now(tz=pytz.timezone(
            'Australia/Sydney')) - timedelta(days=2)).strftime("%Y-%m-%d")
        response = fetcher.get_prices(
            day_before_yesterday_date, yesterday_date, resolution=30)
        prices = [x[1] for x in response]
        return prices

    def get_sim_price(self, current_time):
        _price_test = [17.12,
                       18.48,
                       18,
                       18,
                       15.7,
                       14.96,
                       14.94,
                       14.67,
                       14.98,
                       15.83,
                       15.9,
                       17.57,
                       16.19,
                       15.62,
                       16.65,
                       13.83,
                       17.06,
                       17.56,
                       17.48,
                       15.25,
                       15.39,
                       15.39,
                       14.84,
                       15.87,
                       16.54,
                       17.95,
                       17.8,
                       18.55,
                       18.08,
                       19.02,
                       17.95,
                       17.89,
                       18.79,
                       15.57,
                       16.39,
                       17.61,
                       17.39,
                       20.61,
                       20.42,
                       18.14,
                       18.43,
                       18.42,
                       17.91,
                       18.39,
                       17.16,
                       21.37,
                       18.89,
                       19.39,
                       ]
        start_time = datetime.strptime('00:00', '%H:%M')
        time_intervals = [(start_time + timedelta(minutes=30 * i)).time()
                          for i in range(48)]
        price_time_map = dict(zip(time_intervals, _price_test))

        # Find the correct time range for current_time
        current_time = datetime.strptime(current_time, '%H:%M').time()
        for t in time_intervals:
            if current_time <= t:
                return price_time_map[t]

        # If current_time is later than the last time interval, return the price for the first interval
        return price_time_map[time_intervals[0]]

    def get_sim_load_iter(self):
        _usage = [0.17,
                  0.18,
                  0.11,
                  0.1,
                  0.11,
                  0.17,
                  0.22,
                  0.15,
                  0.13,
                  0.13,
                  0.12,
                  0.13,
                  0.4,
                  0.42,
                  0.22,
                  0.48,
                  0.2,
                  0.23,
                  0.55,
                  0.7,
                  0.5,
                  0.85,
                  0.53,
                  0.48,
                  0.22,
                  0.23,
                  0.54,
                  0.51,
                  0.51,
                  0.57,
                  0.34,
                  0.21,
                  0.17,
                  0.24,
                  0.2,
                  0.35,
                  0.44,
                  0.51,
                  0.43,
                  0.49,
                  0.52,
                  0.57,
                  0.47,
                  0.43,
                  0.19,
                  0.22,
                  0.19,
                  ]
        return cycle(_usage)

    def get_sim_load(self):
        return next(self.sim_load_iter)

    def get_token(self, api_version='redx'):
        if self.token and (datetime.now(tz=pytz.timezone('Australia/Sydney')) - self.token_last_updated) < timedelta(hours=1):
            return self.token
        if api_version == 'redx':
            response = self.api.send_request("user/token", method='POST', data={
                'user_account': 'yetao_admin', 'secret': 'tpass%#%'}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        else:
            response = self.api.send_request("user/token", method='POST', data={
                'user_account': 'yetao_admin', 'secret': 'a~L$o8dJ246c'}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        self.token_last_updated = datetime.now(
            tz=pytz.timezone('Australia/Sydney'))
        if response is None:
            raise Exception('API failed: get_token')
        return response['data']['token']

    def get_realtime_battery_stats(self, sn):
        # Test Get latest summary data
        data = {'deviceSn': sn}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "device/get_latest_data", method='POST', json=data, headers=headers)
        if response is None:
            raise Exception('API failed: get_latest_data')
        return response['data']

    def is_VPP_on(self, sn):
        data = {'deviceSn': sn, 'sync': 0}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "device/get_params", method='POST', json=data, headers=headers)
        vpp = response.get('data', {}).get('operatingMode', 0)
        return True if vpp == '1' else False

    def get_project_stats(self, grid_ID=1, phase=2):
        '''
        Currently we have only one project, shawsbay, so we hard code the gridID as 1.
        '''
        data = {'gridId': grid_ID}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "grid/get_meter_reading", method='POST', json=data, headers=headers)
        self.get_meter_reading_stats_call_count += 1
        # logging.info(f'get_prediction_v2_api called: {self.get_meter_reading_stats_call_count}')
        if response is None:
            raise Exception('Get meter reading API failed')
        return response['data'][f'phase{phase}']

    def get_project_demand(self, grid_ID=1, phase=2):
        '''
        Currently we have only one project, shawsbay, so we hard code the gridID as 1.
        '''
        date_today = datetime.now(tz=pytz.timezone(
            'Australia/Sydney')).strftime("%Y_%m_%d")
        data = {'date': date_today, 'gridID': grid_ID, 'phase': phase}
        headers = {'token': self.get_token()}
        max_retries = 3
        response = {}

        for retry_count in range(max_retries):
            if response.get('data') is not None:
                break

            response = self.api.send_request(
                "grid/get_prediction", method='POST', json=data, headers=headers)

            if response.get('data') is None:
                response = self.api.send_request(
                    "grid/get_prediction_v2", method='POST', json=data, headers=headers)

            if response.get('data') is None:
                logging.error(
                    f"Failed to get prediction data, retrying... Attempt {retry_count + 1}")
                time.sleep(180)
        prediction_average = [
            (int(x['predictionLower']) + int(x['predictionUpper']))/2 for x in response['data']]
        self.get_project_stats_call_count += 1
        return prediction_average

    def get_sim_time_iter(self):
        start_time = datetime.strptime('00:00', '%H:%M')
        time_intervals = [(start_time + timedelta(minutes=30 * i)).time()
                          for i in range(48)]
        return cycle(time_intervals)

    def get_current_time(self, time_zone='Australia/Sydney'):
        '''
        Return the current time in the simulation in the format like 2021-10-01 21:00
        '''
        if self.test_mode:
            return next(self.sim_time_iter).strftime("%H:%M")
        local_time = datetime.now(
            tz=pytz.timezone(time_zone)).strftime("%H:%M")
        return local_time

    def get_params(self, sn):
        try:
            headers = {'token': self.get_token()}
            data = {'deviceSn': sn, 'sync': 0}
            response = self.api.send_request(
                "device/get_params", method='POST', json=data, headers=headers)
            return response
        except ConnectionError as e:
            logging.error(f"Connection error occurred: {e}")
            return None

    def set_antibackflow_register(self, data):
        try:
            headers = {'token': self.get_token()}
            sn = data.get('deviceSn', None)
            response = self.api.send_request(
                "device/set_register", method='POST', json={
                    "deviceSn": sn,
                    "addr": 58,
                    "value": 0
                },
                headers=headers
            )
        except Exception as e:
            logging.error(f"Set Anti-backflow: unexpected error occurred: {e}")
            response = None
        return response

    def set_min_register(self, data):
        key_register_map = {
            'chargeStart1': 9,
            'chargeEnd1': 11,
            'dischargeStart1': 13,
            'dischargeEnd1': 15,
        }

        try:
            headers = {'token': self.get_token()}
            sn = data.get('deviceSn', None)
            for key, value in data.items():
                if key in key_register_map:
                    minute = int(value.split(':')[1])
                    response = self.api.send_request(
                        "device/set_register", method='POST', json={
                            "deviceSn": sn,
                            "addr": key_register_map[key],
                            "value": minute
                        },
                        headers=headers
                    )
        except ConnectionError as e:
            logging.error(f"Connection error occurred: {e}")
            response = None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            response = None
        return response

    def set_hour_register(self, data):
        key_register_map = {
            'chargeStart1': 8,
            'chargeEnd1': 10,
            'dischargeStart1': 12,
            'dischargeEnd1': 14,
        }

        try:
            headers = {'token': self.get_token()}
            sn = data.get('deviceSn', None)
            for key, value in data.items():
                if key in key_register_map:
                    hour = int(value.split(':')[0])
                    response = self.api.send_request(
                        "device/set_register", method='POST', json={
                            "deviceSn": sn,
                            "addr": key_register_map[key],
                            "value": hour
                        },
                        headers=headers
                    )
        except ConnectionError as e:
            logging.error(f"Connection error occurred: {e}")
            response = None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            response = None
        return response

    @api_status_check(max_retries=3, delay=60)
    def send_battery_command(self, peak_valley_command=None, json=None, sn=None):
        if self.test_mode:
            return

        command_map = {
            'Idle': 3,
            'Charge': 2,
            'Discharge': 1,
            'Clear Fault': 4,
            'Power Off': 5,
        }

        mode_map = {
            'Auto': 0,
            'Vpp': 1,
            'Time': 2,
        }

        anti_backflow_map = {
            True: 1,
            False: 0,
        }

        grid_charge_map = {
            True: 1,
            False: 0,
        }

        def _convert_floats_to_ints(d):
            for key, value in d.items():
                if isinstance(value, float):
                    d[key] = int(value)
                elif isinstance(value, dict):
                    d[key] = _convert_floats_to_ints(value)
            return d

        def _get_amber_command(command, peak_valley_command, sn):
            from datetime import datetime, timedelta
            data = {}
            current_time_str = self.get_current_time(
                time_zone='Australia/Brisbane')
            current_time = datetime.strptime(current_time_str, '%H:%M')
            empty_time = '00:00'

            if command == 'Charge':
                grid_charge = peak_valley_command.get('grid_charge', False)
                grid_charge = grid_charge_map[grid_charge]
                start_time = self.get_current_time(
                    time_zone='Australia/Brisbane')
                end_time = (datetime.strptime(start_time, '%H:%M') +
                            timedelta(minutes=40)).strftime("%H:%M")
                data = {
                    'deviceSn': sn,
                    'controlCommand': command_map[command],
                    'operatingMode': mode_map['Time'],
                    'chargeStart1': start_time,
                    'chargeEnd1': end_time,
                    'dischargeStart1': empty_time,
                    'dischargeEnd1': empty_time,
                    'chargePower1': peak_valley_command.get('power', 800),
                    'enableGridCharge1': grid_charge
                }

            elif command == 'Discharge':
                start_time = self.get_current_time(
                    time_zone='Australia/Brisbane')
                end_time = (datetime.strptime(start_time, '%H:%M') +
                            timedelta(minutes=40)).strftime("%H:%M")
                data = {
                    'deviceSn': sn,
                    'controlCommand': command_map[command],
                    'operatingMode': mode_map['Time'],
                    'dischargeStart1': start_time,
                    'dischargeEnd1': end_time,
                    'chargeStart1': empty_time,
                    'chargeEnd1': empty_time,
                    'antiBackflowSW': anti_backflow_map[peak_valley_command.get('anti_backflow', True)],
                    'dischargePower1': peak_valley_command.get('power', 2500)
                }

            elif command == 'Idle':
                data = {
                    'deviceSn': sn,
                    'controlCommand': command_map[command],
                    'operatingMode': mode_map['Time'],
                    'dischargeStart1': empty_time,
                    'dischargeEnd1': empty_time,
                    'chargeStart1': empty_time,
                    'chargeEnd1': empty_time,
                }

            return data

        # Check if the device is on VPP mode
        if self.is_VPP_on(sn):
            logging.info(f"Device {sn} is on VPP mode, skipping...")
            return None

        data = {}

        if peak_valley_command:
            cmd = peak_valley_command.copy()
            command = cmd.get('command', None)
            data = _get_amber_command(command, cmd, sn)

        if json:
            data = json.copy()

        data = _convert_floats_to_ints(data)

        try:
            headers = {'token': self.get_token()}
            self.set_hour_register(data)
            self.set_min_register(data)
            self.set_antibackflow_register(data)
            data.pop('chargeStart1', None)
            data.pop('chargeEnd1', None)
            data.pop('dischargeStart1', None)
            data.pop('dischargeEnd1', None)
            response = self.api.send_request(
                "device/set_params", method='POST', json=data, headers=headers)
        except Exception as e:
            logging.error(f"An unexpected error occurred at sending command: {e}")
            response = None

        return response


class ApiCommunicator:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def send_request(self, command, method="GET", data=None, json=None, headers=None, retries=3):
        """
        Send a request to the API.

        :param command: API command/endpoint to be accessed.
        :param method: HTTP method like GET, POST, PUT, DELETE.
        :param data: Payload to send (if any).
        :param headers: Additional headers to be sent.
        :param retries: Number of retries in case of a failure.
        :return: Response from the server.
        """
        url = f"{self.base_url}/{command}"

        for _ in range(retries):
            try:
                if method == "GET":
                    response = self.session.get(url, headers=headers)
                elif method == "POST":
                    response = self.session.post(
                        url, data=data, json=json, headers=headers, timeout=None)

                # Raises an exception for HTTP errors.
                response.raise_for_status()
                # Assuming JSON response. Modify as needed.
                return response.json()
            except requests.RequestException as e:
                logging.error(f"Error occurred: {e}. Retrying...")

        raise ConnectionError(
            f"Failed to connect to {url} after {retries} attempts.")

    def is_cmd_succ(self, api, expected_output, command, method="GET", json=None, headers=None, retries=3):
        raise NotImplementedError
        if expected_output != self.send_request(command, method, data, headers, retries):
            raise ValueError(f"Command {command} failed to execute.")
        return True


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

    def get_prices(self, start_date, end_date, resolution=30):
        if not self.site_id:
            self.site_id = self.get_site()
        header = {'Authorization': f'Bearer {self.api_key}',
                  'accept': 'application/json'}
        url = f"{self.base_url}/sites/{self.site_id}/prices?startDate={start_date}&endDate={end_date}&resolution={resolution}"
        response = requests.get(url, headers=header)
        data = response.json()
        data = list(filter(lambda x: x['channelType'] == 'general', data))
        prices = [(x['nemTime'], x['perKwh']) for x in data]
        return prices
