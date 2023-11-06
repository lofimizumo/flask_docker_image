from functools import wraps
import requests
import time
from datetime import datetime, timedelta
import pytz
from itertools import cycle


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
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(delay)  # Sleep for some time before retrying
                else:
                    print("Status successfully changed!")
                    return response
            print("Max retry limit reached! Stopping further attempts.")
            return response

        def check_response(response):
            if response['errorCode'] != 0:
                raise ValueError(
                    f"Command failed to execute. Response: {response}")
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
            print(f"Status: {status_response['data']['showStatus']}")

            if _is_command_expected(status_response['data']['showStatus'], expected_status):
                return True
            else:
                return False

        return wrapper

    return decorator


class PriceAndLoadMonitor:
    def __init__(self,  test_mode=False, api_version='dev3'):
        self.sim_load_iter = self.get_sim_load_iter()
        self.sim_time_iter = self.get_sim_time_iter()
        self.api = None
        if api_version == 'dev3':
            self.api = ApiCommunicator(
                base_url="https://dev3.redxvpp.com/restapi")
        else:
            self.api = ApiCommunicator(
                base_url="https://redxpower.com/restapi")
        self.token = self.get_token(api_version=api_version)
        # Test Mode is for testing the battery control without sending command to the actual battery
        self.test_mode = test_mode

    def get_realtime_price(self):
        url = 'https://api.amber.com.au/v1/sites/01HDN4PXKQ1MR29SWJPHBQE8M8/prices/current?next=0&previous=0&resolution=30'
        header = {'accept': 'application/json','Authorization': 'Bearer psk_2d5030fe84a68769b6f48ab73bd48ebf'}
        r = requests.get(url, headers=header)
        prices = [x['perKwh'] for x in r.json()]
        return prices[0]

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

    def get_token(self, api_version='dev3'):
        if api_version == 'redx':
            response = self.api.send_request("user/token", method='POST', data={
                'user_account': 'yetao_admin', 'secret': 'tpass%#%'}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        else:
            response = self.api.send_request("user/token", method='POST', data={
                'user_account': 'yetao_admin', 'secret': 'a~L$o8dJ246c'}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        return response['data']['token']

    def get_realtime_battery_stats(self, sn):
        # Test Get latest summary data
        data = {'deviceSn': sn}
        headers = {'token': self.token}
        response = self.api.send_request(
            "device/get_latest_data", method='POST', json=data, headers=headers)
        return response['data']

    def get_project_demand(self, grid_ID=1, phase=2):
        '''
        Currently we have only one project, shawsbay, so we hard code the gridID as 1.
        '''
        date_today = datetime.now(tz=pytz.timezone(
            'Australia/Brisbane')).strftime("%Y_%m_%d")
        data = {'date': date_today, 'gridID': grid_ID, 'phase': phase}
        headers = {'token': self.token}
        response = self.api.send_request(
            "grid/get_prediction_v2", method='POST', json=data, headers=headers)
        prediction_average = [
            (int(x['predictionLower']) + int(x['predictionUpper']))/2 for x in response['data']]
        return prediction_average

    def get_sim_time_iter(self):
        start_time = datetime.strptime('00:00', '%H:%M')
        time_intervals = [(start_time + timedelta(minutes=30 * i)).time()
                          for i in range(48)]
        return cycle(time_intervals)

    def get_current_time(self):
        '''
        Return the current time in the simulation in the format like 2021-10-01 21:00
        '''
        if self.test_mode:
            return next(self.sim_time_iter).strftime("%H:%M")
        gold_coast_time = time.gmtime(time.time() + 3600 * 10)
        return time.strftime("%H:%M", gold_coast_time)

    @api_status_check(max_retries=3, delay=60)
    def send_battery_command(self, command=None, json=None, sn=None):
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

        if command:
            from datetime import datetime, timedelta
            formatted_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            end_time = datetime.now() + timedelta(minutes=5)
            formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
            data = {}
            command = command
            start_time = self.get_current_time()
            end_time = (datetime.strptime(start_time, '%H:%M') +
                        timedelta(minutes=30)).strftime("%H:%M")
            empty_time = '00:00'
            if command == 'Charge':
                data = {'deviceSn': sn,
                        'controlCommand': command_map[command],
                        'operatingMode': mode_map['Time'],
                        'chargeStart1': start_time,
                        'chargeEnd1': end_time,
                        'dischargeStart1': empty_time,
                        'dischargeEnd1': empty_time,
                        'chargePower1': 800
                        }

            elif command == 'Discharge':
                data = {'deviceSn': sn,
                        'controlCommand': command_map[command],
                        'operatingMode': mode_map['Time'],
                        'dischargeStart1': start_time,
                        'dischargeEnd1': end_time,
                        'chargeStart1': empty_time,
                        'chargeEnd1': empty_time,
                        'antiBackflowSW': 1,
                        'dischargePower1': 2500,
                        }
            elif command == 'Idle':
                data = {'deviceSn': sn,
                        'controlCommand': command_map[command],
                        'operatingMode': mode_map['Time'],
                        'dischargeStart1': empty_time,
                        'dischargeEnd1': empty_time,
                        'chargeStart1': empty_time,
                        'chargeEnd1': empty_time, }
        if json:
            data = json

        headers = {'token': self.token}
        response = self.api.send_request(
            "device/set_params", method='POST', json=data, headers=headers)
        print(f'Send command {command} to battery {sn}, response: {response}')
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
                print(f"Error occurred: {e}. Retrying...")

        raise ConnectionError(
            f"Failed to connect to {url} after {retries} attempts.")

    def is_cmd_succ(self, api, expected_output, command, method="GET", json=None, headers=None, retries=3):
        raise NotImplementedError
        if expected_output != self.send_request(command, method, data, headers, retries):
            raise ValueError(f"Command {command} failed to execute.")
        return True
