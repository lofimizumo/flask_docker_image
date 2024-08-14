from datetime import datetime, timedelta
import time
import pytz
import tomli
from enum import Enum
from util import RedXServerClient
import logging
import optuna
from battery_automation import DeviceType
from scipy.ndimage import gaussian_filter1d
import numpy as np
from solar_prediction import WeatherInfoFetcher


class RedXDemandPredictor:
    """
    DemandPredictor with simple statistical method.
    """

    def __init__(self, sn, model_path, config=None):
        """
        Initialize the predictor algorithm with the model path and configuration.

        Args:
            model_path (str): Path to the trained model file.
            config (dict): Configuration parameters for the predictor algorithm.
        """
        self.model_path = model_path
        self.config = config
        self.logger = logging.getLogger('logger')
        self.sn = sn
        try:
            api_version = self.config.get('api_version').get(self.sn, 'redx')
        except AttributeError:
            raise ValueError("api_version missing in config")
        if api_version == 'dev3':
            self.api = RedXServerClient(
                base_url="https://dev3.redxvpp.com/restapi")
        else:
            self.api = RedXServerClient(
                base_url="https://redxpower.com/restapi")
        self.token = None
        self.token_last_updated = datetime.now(
            tz=pytz.timezone('Australia/Sydney'))

    def _get_token(self, api_version='redx'):
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
        return response['data']['token']

    def predict(self, **kwargs):
        """
        Predict the demand given the input data.

        Args:
            data (dict): Input data for prediction.

        Returns:
            Predicted demand value(s).
        """
        date_today = datetime.now(tz=pytz.timezone(
            'Australia/Sydney')).strftime("%Y_%m_%d")
        if 'grid_ID' not in kwargs:
            raise ValueError("grid_ID is required for prediction")
        if 'phase' not in kwargs:
            raise ValueError("phase is required for prediction")
        grid_ID = kwargs['grid_ID']
        phase = kwargs['phase']
        data = {'date': date_today, 'gridID': grid_ID, 'phase': phase}
        headers = {'token': self._get_token()}
        max_retries = 3
        response = {}

        for retry_count in range(max_retries):
            if response.get('data') is not None:
                break

            response = self.api.send_request(
                "grid/get_prediction", method='POST', json=data, headers=headers)

            if response.get('data') is None:
                self.logger.error(
                    f"Failed to get prediction data, retrying... Attempt {retry_count + 1}")
                time.sleep(180)
        prediction_average = [
            (int(x['predictionLower']) + int(x['predictionUpper']))/2 for x in response['data']]
        return prediction_average


class DemandPredictorType(Enum):
    """
    Enum class for the different types of demand predictors.
    """
    REDX = RedXDemandPredictor


class DemandPredictorFactory:
    """
    Factory class for creating demand predictor objects based on device serial numbers.
    """

    def __init__(self, config_file):
        """
        Initialize the factory with a mapping of serial numbers to predictor types.

        Args:
            config_file (str): Path to the TOML configuration file.
        """
        self.predictor_classes = {}
        with open(config_file, 'rb') as f:
            config = tomli.load(f)
            predictor_classes_config = config.get('predictor_classes', {})

            for sn, predictor_type in predictor_classes_config.items():
                try:
                    predictor_class = DemandPredictorType[predictor_type].value
                    self.predictor_classes[sn] = predictor_class
                except KeyError:
                    raise ValueError(
                        f"Invalid predictor type: {predictor_type}")

    def get_demand_predictor(self, sn):
        """
        Get the demand predictor object for the given serial number.

        Args:
            sn (str): The serial number of the device.

        Returns:
            A demand predictor object for the given serial number.

        Raises:
            ValueError: If the serial number is not found in the mapping.
        """
        try:
            predictor_class = self.predictor_classes[sn]
        except KeyError:
            raise ValueError(f"No predictor found for serial number: {sn}")

        # Create and return an instance of the predictor class
        return predictor_class()


class BatteryProactiveScheduler:
    def __init__(self, sn, config):
        self.config = config

        device_type = DeviceType(
            self.config.get('battery_types').get(sn, 5000))
        match device_type:
            case DeviceType.FIVETHOUSAND:
                self.battery_max_capacity_kwh = 10
                self.min_discharge_rate_kw = 0.5
                self.max_discharge_rate_kw = 5
            case DeviceType.TWOFIVEZEROFIVE:
                self.battery_max_capacity_kwh = 5
                self.min_discharge_rate_kw = 0.5
                self.max_discharge_rate_kw = 2.5

    def generate_schedule(self, consumption, price, batterie_capacity_kwh, num_batteries):
        # 1. Discharging
        consumption = self._smooth_demand(consumption, sigma=6)
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective_discharging, n_trials=100)
        best_durations = [
            study.best_params[f"duration_{i}"] for i in range(num_batteries)]
        discharging_capacity = [[batterie_capacity_kwh, duration]
                                for duration in best_durations]
        net_consumption, battery_discharges = self._greedy_battery_discharge(
            consumption, price, discharging_capacity)

        # 2. Charging
        # charge power set to 670W, the interval is 5 minutes, so the charge duration is 90
        charge_duration = 70
        charging_needs = [[batterie_capacity_kwh, charge_duration]
                          for x in range(num_batteries)]
        masks = [all(mask[i] == 0 for mask in battery_discharges)
                 for i in range(len(battery_discharges[0]))]

        _, battery_charges = self._greedy_battery_charge_with_mask(
            consumption, price, self._get_solar(), charging_needs, masks)

        discharge_schedules = [self._get_charging_window(
            i) for i in battery_discharges]
        charge_schedules = [self._get_charging_window(
            i) for i in battery_charges]
        flat_discharge_schedules = [
            item for sublist in discharge_schedules for item in sublist]
        flat_charge_schedules = [
            item for sublist in charge_schedules for item in sublist]

        discharge_sched_split = self._split_schedules(
            flat_discharge_schedules, 1, 'Discharge')
        charge_sched_split = self._split_schedules(
            flat_charge_schedules, 1, 'Charge')

        # Use hardcoded schedules for charging windows.
        all_schedules = sorted(discharge_sched_split +
                               charge_sched_split, key=lambda x: x[1])
        return all_schedules

    def _get_solar(self, interval=0.5, test_mode=False, max_solar_power=5000):
        max_solar = max_solar_power

        def gaussian_mixture(interval=0.5):
            gaus_x = np.arange(0, 24, interval)
            mu1 = 10.35
            mu2 = 13.65
            sigma = 1.67
            gaus_y1 = np.exp(-0.5 * ((gaus_x - mu1) / sigma)
                             ** 2) / (sigma * np.sqrt(2 * np.pi))
            gaus_y2 = np.exp(-0.5 * ((gaus_x - mu2) / sigma)
                             ** 2) / (sigma * np.sqrt(2 * np.pi))
            gaus_y = (gaus_y1 + gaus_y2) / np.max(gaus_y1 + gaus_y2)
            return gaus_x, gaus_y
        if test_mode:
            max_solar = 5000
        else:
            weather_fetcher = WeatherInfoFetcher('Shaws Bay')
            rain_info = weather_fetcher.get_rain_cloud_forecast_24h(
                weather_fetcher.get_response())
            # here we give clouds more weight than rain based on the assumption that clouds have a bigger impact on solar generation
            max_solar = (
                1-(1.4*rain_info['clouds']+0.6*rain_info['rain'])/(2*100))*max_solar

        _, gaus_y = gaussian_mixture(interval)
        return gaus_y*max_solar

    def _greedy_battery_discharge(self, consumption, price, batteries, price_weight=1):
        num_hours = len(consumption) - 24
        net_consumption = consumption.copy()
        battery_discharges = [[0] * num_hours for _ in batteries]

        for _, (capacity, duration) in enumerate(batteries):
            best_avg = float('-inf')
            best_start = 0

            # Find the window with the highest rolling average
            # Discharging start from 16:30 PM
            for start in range(198, num_hours - duration + 1):
                weight_adjusted_array = [i*j*price_weight for i, j in zip(
                    net_consumption[start:start+duration], price[start:start+duration])]
                avg = sum(weight_adjusted_array) / duration
                if avg > best_avg:
                    best_avg = avg
                    best_start = start

            # Place the battery's discharge
            capacity = 12*1000 * capacity
            discharge_rate = capacity / duration
            for h in range(best_start, best_start + duration):
                # discharged = min(discharge_rate, 1+net_consumption[h])
                battery_discharges[_][h] = discharge_rate
                net_consumption[h] -= discharge_rate

        return net_consumption, battery_discharges

    def _greedy_battery_charge_with_mask(self, consumption, price, solar, batteries, engaged_slots, price_weight=1):
        charge_rate = 1000
        num_hours = len(consumption)
        net_consumption = consumption.copy()
        battery_charges = [[0] * num_hours for _ in batteries]

        np_price = np.array(price)
        solar = np.array(solar)
        consumption = np.array(consumption)

        for battery_idx, (capacity, duration) in enumerate(batteries):
            surplus_solar = np.clip(solar - net_consumption, 0, None)
            power_from_grid = np.clip(charge_rate - surplus_solar, 0, None)
            charging_cost = power_from_grid * np_price
            best_avg = float('inf')
            best_start = 0

            for start in range(12, num_hours - duration + 1):  # start from 1:00 AM
                # check if all slots in this window are available
                if all(engaged_slots[start:start+duration]):
                    period_cost = charging_cost[start:start+duration]
                    avg = sum(period_cost) / duration
                    if avg < best_avg:
                        best_avg = avg
                        best_start = start

            total_capacity_kWh = 12 * 1000 * capacity
            charge_per_hour = total_capacity_kWh / duration

            for h in range(best_start, best_start + duration):
                battery_charges[battery_idx][h] = charge_per_hour
                net_consumption[h] += charge_per_hour

        return net_consumption, battery_charges

    def _objective_discharging(self, trial, consumption, price, num_batteries, batterie_capacity_kwh, min_discharge_rate_kw, max_discharge_rate_kw, price_weight):
        max_discharge_length = self.battery_max_capacity_kwh / \
            self.min_discharge_rate_kw * len(consumption) / 24
        min_discharge_length = self.battery_max_capacity_kwh / \
            self.max_discharge_rate_kw * len(consumption) / 24
        durations = [trial.suggest_int(
            f'duration_{i}', min_discharge_length, max_discharge_length, step=int((max_discharge_length-min_discharge_length)/10)) for i in range(num_batteries)]
        batteries = [(batterie_capacity_kwh, duration)
                     for duration in durations]
        net_consumption, _ = self._greedy_battery_discharge(
            consumption, price, batteries, price_weight)
        avg_consumption = sum(net_consumption) / len(net_consumption)
        variance = sum((x - avg_consumption) **
                       2 for x in net_consumption) / len(net_consumption)
        return variance

    def _smooth_demand(self, data, sigma):
        """Expand peaks using Gaussian blur."""
        return gaussian_filter1d(data, sigma)

    def _get_charging_window(self, battery_charge_schedule):
        p_left = 0
        p_right = 0
        durations = []
        while p_right < len(battery_charge_schedule)-1:
            p_right += 1
            if battery_charge_schedule[p_right] == 0 or p_right == len(battery_charge_schedule)-1:
                if p_right - p_left > 1:
                    durations.append((p_left+1, p_right-p_left-1))
                p_left = p_right
        return durations

    def _split_single_schedule(self, schedule, num_windows):
        start = schedule[0]
        duration = schedule[1]
        segments = [(round(i*duration/num_windows+start), round((i+1)*duration/num_windows+start) -
                    round(i*duration/num_windows+start)) for i in range(num_windows)]
        return segments

    def _split_schedules(self, schedules, num_windows, task_type='Discharge'):
        ret = []
        for schedule in schedules:
            ret.extend(self._split_single_schedule(schedule, num_windows))
        ret = [(task_type, i[0], i[1]) for i in ret]
        return ret
