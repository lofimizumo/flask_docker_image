import json
import base64
from functools import wraps
import requests
import time
from datetime import datetime, timedelta
import pytz
from itertools import cycle
from email.mime.text import MIMEText
import logging
import tomli
import os
import numpy as np
from batteryexceptions import *
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
# from algorithms import DemandPredictorFactory


def load_config(file_path='config.toml'):
    with open(file_path, 'rb') as file:
        config = tomli.load(file)
    return config


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)


setup_logger('logger', 'logs.txt', logging.INFO)
setup_logger('shawsbay_logger', 'sb_logs.txt', logging.INFO)
logger = logging.getLogger('logger')


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
                    logger.error(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(delay)  # Sleep for some time before retrying
                else:
                    # logging.info("Status successfully changed!")
                    return response
            logger.error(
                "Max retry limit reached! Stopping further attempts.")
            return response

        def check_response(response):
            if response is None or not isinstance(response, dict):
                logger.error(f"Invalid response: {response}")
                return False
            return True

        def check_status(args, kwargs):
            is_TestMode = args[0].test_mode
            if is_TestMode:
                return True

            sn = kwargs['sn']
            expected_status = kwargs
            api = RedXServerClient(base_url="https://dev3.redxvpp.com/restapi")
            token = args[0].token

            # Send an API request to check the status
            status_response = api.send_request('device/get_latest_data', method='POST', json={
                'deviceSn': sn}, headers={'token': token})
            logger.info(f"Status: {status_response['data']['showStatus']}")

            if _is_command_expected(status_response['data']['showStatus'], expected_status):
                return True
            else:
                return False

        return wrapper

    return decorator


def send_email(api_key, domain, sender, recipients, subject, text):
    api_url = f"https://api.mailgun.net/v3/{domain}/messages"
    auth = ("api", api_key)

    data = {
        "from": sender,
        "to": recipients,
        "subject": subject,
        "text": text
    }

    response = requests.post(api_url, auth=auth, data=data)

    if response.status_code == 200:
        print("Email sent successfully!")
    else:
        print(f"Error sending email. Status code: {response.status_code}")
        print(response.text)


def decode_model_data(encoded_data):
    try:
        # Decode from base64 to JSON string
        model_data_json = base64.b64decode(encoded_data).decode('utf-8')
        # Parse JSON string to list of floats
        model_data = json.loads(model_data_json)
        return model_data
    except Exception as e:
        print(f"Error decoding model_data: {e}")
        return None


class UserManager:
    def __init__(self, config_file='user_config.toml'):
        self.plant_to_devices: Dict[int, List[str]] = {}
        self.device_to_plant: Dict[str, int] = {}
        self.device_to_type: Dict[str, int] = {}
        self.plant_to_user: Dict[int, str] = {}
        self.toml = load_config(config_file)
        self.parse_plant_device()

    def add_plant(self, plant_id: int):
        if plant_id not in self.plant_to_devices:
            self.plant_to_devices[plant_id] = []

    def add_device(self, plant_id: int, device_id: str):
        if plant_id not in self.plant_to_devices:
            self.add_plant(plant_id)
        self.plant_to_devices[plant_id].append(device_id)
        self.device_to_plant[device_id] = plant_id

    def get_devices_for_plant(self, plant_id: int) -> List[str]:
        return self.plant_to_devices.get(plant_id, [])

    def get_plant_for_device(self, device_id: str) -> int:
        return self.device_to_plant.get(device_id)
    
    def get_plants(self) -> List[int]:
        return list(self.plant_to_devices.keys())
    
    def get_users(self) -> List[str]:
        users = self.toml.get('users', {})
        return list(users.keys())
    
    def get_algo_type(self, plant_id: int) -> str:
        user = self.get_user_for_plant(plant_id)
        return self.get_user_profile(user).get('algo_type_old', 'sell_to_grid')
    
    def get_device_type(self, sn: str) -> str:
        return self.device_to_type.get(sn, 2505)

    def parse_plant_device(self):
        user_profiles = self.toml.get('users', {})
        for username, user_data in user_profiles.items():
            plant_id = user_data.get('plant_id')
            if plant_id is not None:
                self.add_plant(plant_id)
                self.plant_to_user[plant_id] = username
                for device in user_data.get('devices', []):
                    device_id = device.get('id')
                    device_type = device.get('type')
                    if device_id:
                        self.add_device(plant_id, device_id)
                    if device_type:
                        self.device_to_type[device_id] = device_type

    def get_user_profile(self, username: str) -> Dict[str, Any]:
        user_profiles = self.toml.get('users', {})
        return user_profiles.get(username, {})

    def update_user_profile(self, username: str, profile_data: Dict[str, Any]):
        user_profiles = self.toml.get('users', {})
        if username not in user_profiles:
            user_profiles[username] = {}
        
        user_profiles[username].update(profile_data)
        
        plant_id = profile_data.get('plant_id')
        if plant_id is not None:
            self.add_plant(plant_id)
            for device in profile_data.get('devices', []):
                device_id = device.get('id')
                if device_id:
                    self.add_device(plant_id, device_id)
    def get_user_for_plant(self, plant_id: int) -> str:
        return self.plant_to_user.get(plant_id, None)

    def get_api_url(self, plant_id: str) -> str:
        user = self.get_user_for_plant(plant_id)
        return self.get_user_profile(user).get('api_url', None)
    
    def get_api_key(self, plant_id: str) -> str:
        user = self.get_user_for_plant(plant_id)
        return self.get_user_profile(user).get('api_key', None)
    
    def get_retailer_type(self, plant_id: str) -> str:
        user = self.get_user_for_plant(plant_id)
        return self.get_user_profile(user).get('retailer', 'amber')
    
    def get_partner_id(self, plant_id: str) -> str:
        user = self.get_user_for_plant(plant_id)
        return self.get_user_profile(user).get('partner_id', None)

class PriceAndLoadMonitor:
    def __init__(self,  test_mode=False, api_version='dev3'):
        self.config = load_config()
        self.user_manager = UserManager("user_config.toml")
        self.sim_load_iter = self.get_sim_load_iter()
        self.sim_time_iter = self.get_sim_time_iter()
        self.api = None
        if api_version == 'dev3':
            self.api = RedXServerClient(
                base_url="https://dev3.redxvpp.com/restapi")
        else:
            self.api = RedXServerClient(
                base_url="https://redxpower.com/restapi")
        self.token = None
        self.token_last_updated = datetime.now(
            tz=pytz.timezone('Australia/Sydney'))
        # Test Mode is for testing the battery control without sending command to the actual battery
        self.test_mode = test_mode

        self.default_prices = {}

    def get_realtime_price(self, retailer='amber', plant_id=None):
        if retailer == 'amber':
            return self._get_realtime_amber_price(plant_id)
        elif retailer == 'lv':
            return self._get_realtime_lv_price(plant_id)

    def get_realtime_price_from_server(self, sn):
        # TODO: implement the function to get the realtime price from the server
        # Send a request to the RedX Prod server to get the realtime price
        raise NotImplementedError

    def _get_realtime_lv_price(self, plant_id):
        # TODO: replace the API key with the device's API key and partner ID
        url = self.user_manager.get_api_url(plant_id)
        api_key = f'apikey {self.user_manager.get_api_key(plant_id)}'
        partner_id = self.user_manager.get_partner_id(plant_id) 
        header = {
            'Authorization': api_key, 'partner': partner_id}
        try:
            r = requests.get(url, headers=header, timeout=5)
            prices = [(x['costsFlexUp'], x['earningsFlexUp'])
                      for x in r.json()]
            cast_prices = [float(x) for x in prices[0]]
            is_expected_data = r.json()[0]['quality'] == 'Exp'
            return cast_prices, is_expected_data
        except Exception as e:
            raise ValueError(f"Failed to get localvolts price data: {e}")

    def _get_realtime_amber_price(self, plant_id):
        url = self.user_manager.get_api_url(plant_id)
        api_key = self.user_manager.get_api_key(plant_id)
        header = {'accept': 'application/json',
                  'Authorization': f'Bearer {api_key}'}
        try:
            r = requests.get(url, headers=header, timeout=5)
            prices = [x['perKwh'] for x in r.json()]
        # prices[1] is the feed in price, so we return the negative value
        # True means it's expected data, Amber doesn't have forecast data, so we always return True
            return (prices[0], -prices[1]), True
        except Exception as e:
            raise ValueError(f"Failed to get price data: {e}")

    def _init_default_prices(self, retailer):
        # LocalVolts doesn't have historical price data, so we only need to get the historical price data from Amber instead
        default_amber_plant_id = 317 # Plant ID of Dion's Amber account, We use this account to request the initial price data
        if retailer == 'amber' or retailer == 'lv':
            try:
                api_key = self.user_manager.get_api_key(default_amber_plant_id)
            except Exception as e:
                raise ValueError(f"Failed to get API key at _init_default_prices: {e}")
            fetcher = AmberFetcher(api_key)
            yesterday_date = (datetime.now(tz=pytz.timezone(
                'Australia/Sydney')) - timedelta(days=1)).strftime("%Y-%m-%d")
            day_before_yesterday_date = (datetime.now(tz=pytz.timezone(
                'Australia/Sydney')) - timedelta(days=2)).strftime("%Y-%m-%d")
            response = fetcher.get_prices(
                day_before_yesterday_date, yesterday_date, resolution=30)
            prices = [x[1] for x in response]
            return prices
        else:
            raise ValueError(f"Retailer {retailer} is not supported")

    def _get_device_info(self, sn):
        location = self.config.get('battery_locations', {}).get(sn, 'qld')
        retailer = self.config.get('retailers', {}).get(sn, 'amber')
        return location, retailer

    def get_price_history(self, sn, length):
        plant_id = self.user_manager.get_plant_for_device(sn)
        retailer = self.user_manager.get_retailer_type(plant_id)
        ret = None
        if retailer == 'amber':
            if self.default_prices.get('amber', None) is None:
                self.default_prices['amber'] = self._init_default_prices(retailer)
            ret = self.default_prices.get('amber')
        elif retailer == 'lv':
            if self.default_prices.get('lv', None) is None:
                self.default_prices['lv'] = self._init_default_prices(retailer)
            ret = self.default_prices.get('lv')
        ret = np.interp(
            np.linspace(0, 1, int(length)),
            np.linspace(0, 1, len(ret)),
            ret
        ).tolist()
        return ret

    def get_sim_price(self, current_time):
        _price_test = [20] * 48
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
        _usage = [0.2] * 48
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
        # raise BatteryStatsUpdateFailure(
        #     f"Failed to get latest summary data for device {sn}")
        data = {'deviceSn': sn}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "device/get_latest_data", method='POST', json=data, headers=headers)
        if response.get('data', None) is None:
            # raise custom exception
            raise BatteryStatsUpdateFailure(
                f"Failed to get latest data for device {sn}")
        return response['data']

    def is_VPP_on(self, sn):
        data = {'deviceSn': sn, 'sync': 0}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "device/get_params", method='POST', json=data, headers=headers)
        try:
            vpp = response.get('data', {}).get('operatingMode', 0)
        except Exception as e:
            return False
        return True if vpp == '1' else False

    def get_project_stats(self, grid_ID=1, phase=2):
        '''
        Currently we have only one project, shawsbay, so we hard code the gridID as 1.
        '''
        data = {'gridId': grid_ID}
        headers = {'token': self.get_token()}
        response = self.api.send_request(
            "grid/get_meter_reading", method='POST', json=data, headers=headers)
        # logger.info(f'get_prediction_v2_api called: {self.get_meter_reading_stats_call_count}')
        if response is None:
            raise Exception('Get meter reading API failed')
        return response['data'][f'phase{phase}']

    # def get_device_demand_pred(self, sn):
    #     factory = DemandPredictorFactory('config.toml')
    #     model = factory.get_demand_predictor(sn)
    #     return model.predict()

    # def get_device_price_pred(self, sn):
    #     raise NotImplementedError
    #     factory = PricePredictorFactory('config.toml')
    #     model = factory.get_price_predictor(sn)
    #     return model.predict()

    def get_project_demand_pred(self, grid_ID=1, phase=2):
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
                logger.error(
                    f"Failed to get prediction data, retrying... Attempt {retry_count + 1}")
                time.sleep(180)
        prediction_average = [
            (int(x['predictionLower']) + int(x['predictionUpper']))/2 for x in response['data']]
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
            logger.error(f"Connection error occurred: {e}")
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
            logger.error(f"Set Anti-backflow: unexpected error occurred: {e}")
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
            logger.error(f"Connection error occurred: {e}")
            response = None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
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
            logger.error(f"Connection error occurred: {e}")
            response = None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            response = None
        return response

    # @api_status_check(max_retries=1, delay=60)
    # Device status check is not enabled for now, consider enabling it in the future
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

            # Prepare the data for the command to be sent
            if command == 'Charge':
                grid_charge = peak_valley_command.get('grid_charge', False)
                grid_charge = grid_charge_map[grid_charge]
                start_time = self.get_current_time(
                    time_zone='Australia/Brisbane')
                end_time = (datetime.strptime(start_time, '%H:%M') +
                            timedelta(minutes=40)).strftime("%H:%M")
                data = {
                    'deviceSn': sn,
                    # 'controlCommand': command_map[command],
                    # 'operatingMode': mode_map['Time'],
                    'chargeStart1': start_time,
                    'chargeEnd1': end_time,
                    # 'dischargeStart1': empty_time,
                    # 'dischargeEnd1': empty_time,
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
                    # 'controlCommand': command_map[command],
                    # 'operatingMode': mode_map['Time'],
                    'dischargeStart1': start_time,
                    'dischargeEnd1': end_time,
                    # 'chargeStart1': empty_time,
                    # 'chargeEnd1': empty_time,
                    'antiBackflowSW': anti_backflow_map[peak_valley_command.get('anti_backflow', True)],
                    'dischargePower1': peak_valley_command.get('power', 2500)
                }

            elif command == 'Idle':
                data = {
                    'deviceSn': sn,
                    # 'controlCommand': command_map[command],
                    # 'operatingMode': mode_map['Time'],
                    'dischargeStart1': empty_time,
                    'dischargeEnd1': empty_time,
                    'chargeStart1': empty_time,
                    'chargeEnd1': empty_time,
                }

            # Sometimes the device will lose all the settings, so we need to set the settings every 30 minutes in case of the settings lost
            if not hasattr(self, 'last_command_time'):
                self.last_command_time = {}

            if sn not in self.last_command_time or self.last_command_time[sn] is None or (current_time - self.last_command_time[sn]).total_seconds() >= 5400:
                data['controlCommand'] = command_map['Idle']
                data['operatingMode'] = mode_map['Time']
                data['antiBackflowSW'] = anti_backflow_map[True]
                data['globalMaxSOC'] = "100"
                data['globalMinSOC'] = "10"
                self.last_command_time[sn] = current_time

            return data

        # Check if the device is on VPP mode
        if self.is_VPP_on(sn):
            logger.info(f"Device {sn} is on VPP mode, skipping...")
            return {'status': 'VPP mode', 'message': 'Device is on VPP mode, skipping...'}

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
            # self.set_hour_register(data)
            # self.set_min_register(data)
            self.set_antibackflow_register(data)
            # data.pop('chargeStart1', None)
            # data.pop('chargeEnd1', None)
            # data.pop('dischargeStart1', None)
            # data.pop('dischargeEnd1', None)
            response = self.api.send_request(
                "device/set_params", method='POST', json=data, headers=headers)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred at sending command: {e}")
            response = None

        return response


class RedXServerClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def send_request(self, command, method="GET", data=None, json=None, headers=None, retries=2):
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
                if e.response is not None and e.response.status_code == 504:
                    return None
                else:
                    logger.error(
                        f"Failed to connect to {url}. Retrying...")

        raise ConnectionError(
            f"Failed to connect to {url} after {retries} attempts.")


class AIServerClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIServerClient, cls).__new__(cls)
            cls._instance.base_url = "https://ai.redxvpp.com"
            cls._instance.data_url = f"{cls._instance.base_url}/api/v1.0/data/"
            cls._instance.session = None
            cls._instance.id = None
            cls._instance.email = None
            cls._instance.token = None
            cls._instance.token_expiry = None
            cls._instance.credentials = None
        return cls._instance

    async def ensure_login(self, email: str, password: str):
        self.credentials = (email, password)
        if not self.is_logged_in:
            await self.login(email, password)

    async def login(self, email: str, password: str):
        if not self.credentials:
            raise ValueError("Credentials not set. Call ensure_login first.")
        email, password = self.credentials
        body_data = {"email": email, "password": password}
        response_data = await self._request("/api/login", body_data)
        if response_data and response_data.get('errorCode') == 0:
            self.id = response_data['data']['id']
            self.email = response_data['data']['email']
            self.token = response_data['data']['token']
            self.token_expiry = datetime.now(tz=pytz.timezone(
                'Australia/Brisbane')) + timedelta(hours=24)
        return response_data

    @property
    def is_logged_in(self):
        return self.token is not None and datetime.now(tz=pytz.timezone('Australia/Brisbane')) < self.token_expiry

    async def _request(self, url_slug: str, body_data: Dict[str, Any], is_data_api: bool = False) -> Optional[Dict[str, Any]]:
        if not self.session:
            self.session = aiohttp.ClientSession()

        full_url = self.data_url + url_slug if is_data_api else self.base_url + url_slug

        headers = {"Content-Type": "application/json"}
        if is_data_api and self.id and self.email and self.token:
            headers.update({
                "id": str(self.id),
                "email": self.email,
                "token": self.token,
            })

        try:
            async with self.session.post(full_url, headers=headers, json=body_data, timeout=10) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if response_data.get('errorCode') == 0:
                        pass
                    else:
                        print(
                            f"Error: Response status code: {response.status}, response error code: {response_data.get('errorCode')}, response info text: {response_data.get('infoText')}.")
                    return response_data
                else:
                    print(f"Error: Response status code: {response.status}, whole response text: {await response.text()}")
        except asyncio.TimeoutError:
            print("Request timed out")
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")

        return None

    async def data_api_request(self, url_slug: str, body_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return await self._request(url_slug, body_data, is_data_api=True)

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


class AmberFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = f"https://api.amber.com.au/v1"
        self.site_id = None

    def get_site(self):
        header = {'Authorization': f'Bearer {self.api_key}',
                  'accept': 'application/json'}
        response = requests.get(f"{self.base_url}/sites", headers=header)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]['id']
        else:
            raise ValueError("No site found in the response")

    def get_prices(self, start_date, end_date, resolution=30):
        if not self.site_id:
            self.site_id = self.get_site()
        header = {'Authorization': f'Bearer {self.api_key}',
                  'accept': 'application/json'}
        url = f"{self.base_url}/sites/{self.site_id}/prices?startDate={start_date}&endDate={end_date}&resolution={resolution}"
        response = requests.get(url, headers=header)
        response.raise_for_status()
        data = response.json()
        data = list(filter(lambda x: x['channelType'] == 'general', data))
        prices = [(x['nemTime'], x['perKwh']) for x in data]
        return prices
