import numpy as np
import pytz
import aiohttp
from datetime import datetime, timedelta, time as datetime_time
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass
import time
import copy
import util
from util import async_retry
import math
import logging
import tomli
from threading import Thread
import pickle
from solar_prediction import WeatherInfoFetcher
from optimizer.battery_optimizer import BatteryScheduler
import concurrent.futures
import traceback
from batteryexceptions import *
import warnings
import asyncio


class DeviceType(Enum):
    FIVETHOUSAND = 5000
    TWOFIVEZEROFIVE = 2505
    SEVENTHOUSAND = 7000


class ErrorCode(Enum):
    SUCCESS = 0
    ALREADY_EXIST = 3005


def load_config(file_path):
    with open(file_path, 'rb') as file:
        config = tomli.load(file)
    return config


class BatterySchedulerManager:
    """
    BatterySchedulerManager class manages the scheduling of battery operations.
    This is for the new model V2.0 on non-Shawsbay devices.

    This class is responsible for initializing and managing battery schedulers,
    handling different scheduler types, and coordinating battery operations
    across multiple devices.

    Attributes:
        config_old_model (dict): Configuration loaded from the old model config file.
        logger (logging.Logger): Logger for recording events and errors.
        scheduler: The active scheduler instance.
        monitor (util.PriceAndLoadMonitor): Monitor for price and load data.
        ai_client (util.AIServerClient): Client for AI server interactions.
        user_manager (util.UserManager): Manager for user-related operations.
        test_mode (bool): Flag indicating if the scheduler is in test mode.
        is_running (bool): Flag indicating if the scheduler is currently running.
        pv_sn (str): Serial number of the PV device.
        plant_list (list): List of plant IDs.
        sn_list (list): List of battery serial numbers.
        user_names (list): List of user names.
        last_bat_sched_time: Timestamp of the last battery schedule.
        schedule (dict): Current battery schedule.
        schedule_for_compare (dict): Schedule used for comparison.
        prepare_battery_sched_loop (asyncio.AbstractEventLoop): Event loop for battery scheduling.
        last_schedule_peakvalley (dict): Last schedule for peak-valley algorithm.
        should_update_schedule (bool): Flag indicating if the schedule should be updated.
        project_phase (int): Phase of the project.
        project_mode (str): Mode of the project (e.g., 'normal', 'Peak Shaving').
        sn_locations (dict): Locations of batteries keyed by serial number.
        last_command_time (dict): Last command times for each device.
        current_prices (dict): Current prices for each plant.

    Args:
        scheduler_type (str): Type of scheduler to use (default: 'PeakValley').
        battery_sn (list): List of battery serial numbers.
        test_mode (bool): Whether to run in test mode (default: False).
        api_version (str): Version of the API to use (default: 'dev3').
        pv_sn (str): Serial number of the PV device (default: None).
        phase (int): Project phase (default: 2).
        config_old_model (str): Path to the old model config file (default: 'config.toml').
        config_new_model (str): Path to the new model config file (default: 'user_config.toml').
        project_mode (str): Mode of the project (default: 'normal').
    """

    def __init__(self, scheduler_type='PeakValley',
                 battery_sn=['011LOKL140104B',
                             '011LOKL140058B',
                             'RX2505ACA10J0A180003',
                             'RX2505ACA10J0A160016',
                             ],
                 test_mode=False,
                 api_version='dev3',
                 pv_sn=None,
                 phase=2,
                 config_old_model='config.toml',
                 config_new_model='user_config.toml',
                 project_mode='normal'
                 ):
        '''
        Args:
        pv_sn: str, the serial number of the PV device in a Project
        battery_sn: list, a list of serial numbers of the battery devices
        test_mode: bool, if True, the scheduler will use the test mode for debugging
        api_version: str, the version of the API, e.g., 'dev3', 'redx'
        phase: int, the phase of the project
        config: str, the path to the config file, config file should be in TOML format, see config.toml for an example
        project_mode: str, the mode of the project of AI scheduler, e.g., 'Peak Shaving', 'Money Saving', 'Normal'
        '''
        self.config_old_model = load_config(config_old_model)
        self.logger = logging.getLogger('logger')
        self.scheduler = None
        self.monitor = util.PriceAndLoadMonitor(api_version=api_version)
        self.ai_client = util.AIServerClient()
        self.user_manager = util.UserManager(config_file=config_new_model)
        self.test_mode = test_mode
        self.is_runing = False
        self.pv_sn = pv_sn
        self.plant_list = self.user_manager.get_plants()
        self.sn_list = battery_sn if type(battery_sn) == list else [
            battery_sn]
        self.user_names = self.user_manager.get_users()
        self.last_bat_sched_time = None
        self.schedule = {}
        self.schedule_for_compare = {}
        self.prepare_battery_sched_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.prepare_battery_sched_loop)
        self.last_schedule_peakvalley = {}
        self.plant_data = {}
        self.should_update_schedule = False
        self.project_phase = phase
        self.project_mode = project_mode
        self.sn_locations = self.config_old_model.get('battery_locations', {})
        self.last_command_time = {}
        self.current_prices = {plant: {'buy': 0.0, 'feedin': 0.0}
                               for plant in self.plant_list}

        self.init_params = (scheduler_type, api_version, pv_sn)

    def _set_scheduler(self, scheduler_type, api_version, pv_sn=None):
        if scheduler_type == 'PeakValley':
            self.scheduler = HybridAlgo(monitor=self.monitor)
            for sn in self.sn_list:
                self.scheduler.init_price_history(sn)
            self.sample_interval = self.config_old_model.get(
                'peakvalley', {}).get('SampleInterval', 120)
        elif scheduler_type == 'AIScheduler':
            self.sample_interval = self.config_old_model.get(
                'shawsbay', {}).get('SampleInterval', 900)
            self.scheduler = ShawsbayAlgo(
                sn_list=self.sn_list, api_version=api_version, pv_sn=pv_sn,
                phase=self.project_phase,
                mode=self.project_mode)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _get_battery_command(self, **kwargs):
        if not self.scheduler:
            raise ValueError("Scheduler not set. Use set_scheduler() first.")

        return self.scheduler.step(**kwargs)

    async def _make_battery_decision(self):
        await asyncio.sleep(5)
        while self.is_running:
            try:
                await self._process_peak_valley_scheduler()
                await asyncio.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error("Scheduling error: %s", e)
                self.logger.error("Traceback: %s", traceback.format_exc())
                await asyncio.sleep(self.sample_interval)

    def _seconds_to_next_n_minutes(self, current_time, n=5):
        next_time = current_time + timedelta(minutes=n)
        next_time = next_time.replace(
            minute=(next_time.minute // n) * n, second=0, microsecond=0)
        time_diff = next_time - current_time
        seconds_to_wait = time_diff.total_seconds()
        return int(seconds_to_wait)

    async def _collect_amber_prices(self):
        while self.is_running:
            try:
                self.logger.info("Updating Amber Prices...")
                self._update_prices('amber')
                # Amber updates prices every 2 minutes
                current_time = datetime.now(pytz.timezone('Australia/Brisbane'))
                delay = self._seconds_to_next_n_minutes(
                    current_time=current_time, n=2)
                await asyncio.sleep(delay)
            except Exception as e:
                self.logger.error(
                    "An exception occurred while updating Amber Prices: %s", str(e))
                await asyncio.sleep(5)

    async def _collect_localvolts_prices(self):
        while self.is_running:
            try:
                self.logger.info("Updating LocalVolts Prices...")
                quality = self._update_prices('lv')
                if quality:
                    # Local Volts updates prices every 5 minutes
                    current_time = datetime.now(
                        pytz.timezone('Australia/Brisbane'))
                    delay = self._seconds_to_next_n_minutes(current_time, n=5)
                    # add 30 seconds to make sure the price is updated on Local Volts
                    delay = delay + 30
                    await asyncio.sleep(delay)
                else:
                    self.logger.info(
                        "Quality is not good or Error happened, waiting for 10 seconds...")
                    await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(
                    "An exception occurred while updating LocalVolts Prices: %s", str(e))
                await asyncio.sleep(5)

    def _update_prices(self, target_retailer):
        """
        Update prices for only the devices that are using the target retailer.
        e.g., if target_retailer is 'amber', only update prices for devices that are using Amber.
        """
        def _update_prices_per_plant_id(retailer, plant_id):
            data, quality = self.get_current_price(
                retailer=retailer, plant_id=plant_id)
            if data:
                self.current_prices[plant_id]['buy'], self.current_prices[plant_id]['feedin'] = data
                return True
            else:
                return False

        plants = self.user_manager.get_plants()
        for plant_id in plants:
            retailer_type = self.user_manager.get_retailer_type(plant_id)
            if target_retailer != retailer_type:
                continue
            return _update_prices_per_plant_id(retailer=retailer_type, plant_id=plant_id)

    async def _process_peak_valley_scheduler(self):
        while True:
            tasks = [self._process_send_cmd_each_sn(sn) for sn in self.sn_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sn, result in zip(self.sn_list, results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Processing failed for sn %s: %s", sn, result)
                    self.logger.info("Restarting processing for sn %s", sn)
                    asyncio.create_task(self._process_send_cmd_each_sn(sn))

            await asyncio.sleep(self.sample_interval)

    def make_battery_schedule(self, actions: List[float], buy_prices: List[float], sell_prices: List[float], loads: List[float], solar_outputs: List[float]):
        actions_copy = copy.deepcopy(actions)
        # Create new x values for interpolation
        x = np.arange(len(actions_copy))
        x_new = np.linspace(0, len(actions_copy)-1, 288)

        # Interpolate all lists to 288 points and round to 2 decimal places
        actions_288 = np.round(np.interp(x_new, x, actions_copy), 2)

        battery_actions = [
            BatteryAction(
                action_plant=actions_288[i],
                action_device=None,
                env=ActionEnvObservation(
                    buy_price=buy_prices[i],
                    sell_price=sell_prices[i],
                    load=loads[i],
                    solar=solar_outputs[i]
                )
            )
            for i in range(288)
        ]
        return BatterySchedule(actions=battery_actions)

    async def _process_send_cmd_each_sn(self, sn):
        try:
            self.logger.info("Processing sn: %s", sn)
            plant_id = self.user_manager.get_plant_for_device(sn)
            time_zone = self.user_manager.get_time_zone(plant_id)
            plant_stats = await self.get_current_plant_stats(sn, is_sn=True)
            bat_stats = await self.get_current_battery_stats(sn)
            current_batP = bat_stats.get('batP', 0) if bat_stats else 0
            current_usage_kw = bat_stats.get('loadP', 0) if bat_stats else 0
            current_soc = bat_stats.get(
                'soc', 0) / 100.0 if bat_stats else 0
            current_pv_kw = bat_stats.get('ppv', 0) if bat_stats else 0
            plant_pv_kw = sum(plant_stats.get('ppv', 0)
                              for plant_stats in plant_stats.values())
            plant_load_kw = sum(plant_stats.get('loadP', 0)
                                for plant_stats in plant_stats.values())
            devices_need_charge = [
                device for device, stats in plant_stats.items() if stats.get('soc', 0) < 97]
            device_type = DeviceType(self.user_manager.get_device_type(sn))
            algo_type = self.user_manager.get_algo_type(plant_id)
            buy_price_c = self.current_prices[plant_id]['buy'] if plant_id else 0.0
            feedin_price_c = self.current_prices[plant_id]['feedin'] if plant_id else 0.0
            current_time = self.get_current_time(
                state=self.sn_locations.get(sn, 'qld'))

            # Prepare for V2.0 Model
            plant_id = self.user_manager.get_plant_for_device(sn)
            schedule = self.schedule_for_compare.get(
                self.user_manager.get_user_for_plant(plant_id), None)
            schedule_adjusted = self.adjust_power_for_plant(schedule, sn)
            device_percentage = self.get_device_power_percentage(
                sn, devices_need_charge)

            command = self._get_battery_command(
                current_buy_price=buy_price_c,
                current_feedin_price=feedin_price_c,
                current_usage=current_usage_kw,
                current_time=current_time,
                current_soc=current_soc,
                current_pv=current_pv_kw,
                current_batP=current_batP,
                device_type=device_type,
                device_sn=sn,
                algo_type=algo_type,
                schedule=schedule_adjusted,
                device_percentage=device_percentage,
                plant_pvKW=plant_pv_kw,
                plant_loadKW=plant_load_kw
            )

            last_command = self.last_schedule_peakvalley.get(sn, {})
            c_datetime = datetime.strptime(
                current_time, '%H:%M')
            last_command_time = self.last_command_time.get(sn, c_datetime)
            minute_passed = abs(c_datetime.minute -
                                last_command_time.minute)

            if command != last_command or minute_passed >= 5:
                if not self.test_mode:
                    await self.send_battery_command(command=command, sn=sn, time_zone=time_zone)
                self.last_command_time[sn] = c_datetime
                self.last_schedule_peakvalley[sn] = command
                self.logger.info(
                    "Successfully sent command for %s: %s", sn, command)
            else:
                self.logger.info(
                    "Command Skipped: Command: %s, Last Command: %s, Time: %s, Last Time: %s", command, last_command, c_datetime, last_command_time)
        except BatteryStatsUpdateFailure:
            self.logger.error(
                "Failed to update battery stats for %s.", sn)
        except Exception as e:
            error_message = "Error processing sn:%s: %s \n Traceback: %s" % (
                sn, e, traceback.format_exc())
            self.logger.error(error_message)
            raise e

    def _setup(self):
        self._set_scheduler(*self.init_params)
        tasks = [
            self._collect_amber_prices(),
            self._collect_localvolts_prices(),
            self.prepare_bat_sched_loop(),
            self._make_battery_decision(),
        ]
        return tasks

    def start(self):
        '''
        Main loop for the battery scheduler.
        Steps:

        1. Collect Amber prices Per 2 minutes
        2. Collect LocalVolts prices Per 5 minutes
        3. Make battery decision Periodically (Check SampleInterval in the config.toml file)
        '''
        self.is_running = True

        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async setup and keep the tasks running
            background_tasks = loop.run_until_complete(
                asyncio.gather(*self._setup()))
            loop.run_forever()
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.is_running = False
        self.logger.info("Stopped.")
        for task in asyncio.all_tasks(asyncio.get_event_loop()):
            task.cancel()

    def get_device_power_percentage(self, sn, devices_need_charge):
        plant_id = self.user_manager.get_plant_for_device(sn)
        device_type = self.user_manager.get_device_type(sn)
        user_name = self.user_manager.get_user_for_plant(plant_id)
        devices_meta_data = self.user_manager.get_user_profile(
            user_name).get('devices', [])
        bat_powers = [dev.get('discharge_power', 2.5)
                      for dev in devices_meta_data if dev.get('id') in devices_need_charge]
        total_discharge_power = sum(bat_powers)
        device_discharge_power = 5 if DeviceType(
            device_type) == DeviceType.FIVETHOUSAND else 2.5

        adjustment_factor = device_discharge_power / \
            total_discharge_power if total_discharge_power > 0 else 0  # avoid division by zero
        return adjustment_factor

    def adjust_power_for_plant(self, schedule, sn: str):
        if not schedule:
            return None
        plant_id = self.user_manager.get_plant_for_device(sn)
        device_type = self.user_manager.get_device_type(sn)
        total_discharge_power = self.user_manager.get_user_profile(
            self.user_manager.get_user_for_plant(plant_id)).get('total_bat_discharge_power', 0)
        device_discharge_power = 5 if DeviceType(
            device_type) == DeviceType.FIVETHOUSAND else 2.5

        adjustment_factor = device_discharge_power / total_discharge_power

        adjusted_actions = [
            BatteryAction(
                action_plant=action.action_plant,
                action_device=round(action.action_plant *
                                    adjustment_factor, 2),
                env=ActionEnvObservation(
                    buy_price=action.env.buy_price,
                    sell_price=action.env.sell_price,
                    load=action.env.load,
                    solar=action.env.solar
                )
            )
            for action in schedule.actions
        ]

        return BatterySchedule(actions=adjusted_actions)

    @async_retry(max_retries=3, delay=5)
    async def get_prices(self, plant_id) -> List[float]:
        # Elmar's API call to get prices
        # Ensure login before making the request
        await self.ai_client.ensure_login("ye.tao@redx.com.au", "1111")

        now = datetime.now(pytz.timezone('Australia/Brisbane'))
        day = now.strftime('%Y-%m-%d')
        model_data = {
            "model_date": day,
            "plant_id": plant_id,
        }
        response = await self.ai_client.data_api_request("price_prediction/get", model_data)
        if response and ErrorCode(response.get('errorCode')) == ErrorCode.SUCCESS:
            data = response.get('data')
            price_buy_decoded = util.decode_model_data(
                data[0].get('model_price_buy'))
            price_sell_decoded = util.decode_model_data(
                data[0].get('model_price_sell'))
            self.logger.info(
                "Price data accessed successfully for %s", plant_id)
            return {'buy': price_buy_decoded, 'sell': price_sell_decoded}

    @async_retry(max_retries=3, delay=5)
    async def get_solar(self, plant_id) -> List[float]:
        # Elmar's API call to get solar prediction
        await self.ai_client.ensure_login("ye.tao@redx.com.au", "1111")

        now = datetime.now(pytz.timezone('Australia/Brisbane'))
        day = now.strftime('%Y-%m-%d')
        model_data = {
            "model_date": day,
            "plant_id": plant_id,
        }
        response = await self.ai_client.data_api_request("pv_prediction/get", model_data)
        if response and ErrorCode(response.get('errorCode')) == ErrorCode.SUCCESS:
            data = response.get('data')
            pv = util.decode_model_data(data[0].get('model_adjusted_pv'))
            self.logger.info("PV data accessed successfully for %s", plant_id)
            return pv

    @async_retry(max_retries=3, delay=5)
    async def get_load(self, plant_id) -> List[float]:
        # Elmar's API call to get load prediction
        await self.ai_client.ensure_login("ye.tao@redx.com.au", "1111")

        now = datetime.now(pytz.timezone('Australia/Brisbane'))
        day = now.strftime('%Y-%m-%d')
        model_data = {
            "model_date": day,
            "plant_id": plant_id,
        }
        response = await self.ai_client.data_api_request("load_prediction/get", model_data)
        if response and ErrorCode(response.get('errorCode')) == ErrorCode.SUCCESS:
            data = response.get('data')
            load = util.decode_model_data(data[0].get('model_prediction_load'))
            self.logger.info(
                "Load data accessed successfully for %s", plant_id)
            return load

    def optimize(self, prices: List[float], sell_prices: List[float], solars: List[float], loads: List[float], max_charge_power: float, max_discharge_power: float, capacity: float, init_kwh: float, bat_kwh_now: float, current_time_index: int) -> List[float]:
        """
        Optimizes the battery charging and discharging schedule based on various input parameters.

        This method uses a battery scheduling algorithm to determine the optimal charging and discharging
        actions for a battery system over a 24-hour period. It takes into account electricity prices,
        solar generation, load demand, and battery characteristics to maximize the economic benefit
        or minimize the cost of electricity usage.

        Args:
            prices (List[float]): List of electricity buy prices for each time interval.
            sell_prices (List[float]): List of electricity sell prices for each time interval.
            solars (List[float]): List of solar generation values for each time interval.
            loads (List[float]): List of load demand values for each time interval.
            max_charge_power (float): Maximum charging power of the battery in kW.
            max_discharge_power (float): Maximum discharging power of the battery in kW.
            capacity (float): Total capacity of the battery in kWh.
            init_kwh (float): Initial energy stored in the battery in kWh.
            bat_kwh_now (float): Current energy stored in the battery in kWh.
            current_time_index (int): Index of the current time in the input lists.

        Returns:
            List[float]: A list of battery actions (positive for charging, negative for discharging)
                        for each time interval over the next 24 hours.

        Note:
            - The method uses a charge mask to constrain charging to specific time periods.
            - It employs a battery scheduling algorithm to optimize the charge/discharge schedule.
            - The returned list represents the optimal battery actions for the next 24 hours,
            starting from the current time index.
        """
        # Charge_mast is a list of 0s and 1s, 1 means the battery can be charged at that time
        # This is essential for a non-linear optimization problem, otherwise, the optimizer will not be able to solve the problem
        charge_mask = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        def adjust_middle_value(arr, window_size=3, threshold=0.5):
            if len(arr) < window_size:
                return arr

            result = arr.copy()

            for i in range(1, len(arr) - 1):
                left = arr[i - 1]
                middle = arr[i]
                right = arr[i + 1]

                if middle > left * (1 - threshold) and middle > right * (1 - threshold):
                    new_value = 0.80 * ((left + right) / 2)
                    result[i] = new_value

            return result

        def resample_data(data, target_length=48):
            data = [float(x) for x in data]
            current_length = len(data)
            if current_length == target_length:
                return data
            elif current_length > target_length:
                # Downsample to target_length
                indices = np.linspace(
                    0, current_length - 1, target_length, dtype=int)
                return [data[i] for i in indices]
            else:
                # Upsample to target_length
                return np.interp(np.linspace(0, current_length - 1, target_length),
                                 np.arange(current_length), data).tolist()

        # Resample the data to 48 points
        prices = resample_data(prices)
        sell_prices = resample_data(sell_prices)
        loads = resample_data(loads)
        solars = resample_data(solars)

        config = {
            'b': prices,
            's': sell_prices,
            'l': loads,
            'p': solars,
            'R_c': max_charge_power,
            'R_d': max_discharge_power,
            'capacity': capacity,
            'init_kwh': init_kwh,
            'bat_kwh_now': bat_kwh_now,
            'current_time_index': current_time_index,
            'charge_mask': charge_mask,
        }
        scheduler = BatteryScheduler(config)
        x_vals, _ = scheduler.solve()
        # x_vals = adjust_middle_value(x_vals)
        # import random
        # rand_file_names = ['battery_sched.pkl', 'battery_sched_2.pkl', 'battery_sched_3.pkl', 'battery_sched_4.pkl', 'battery_sched_5.pkl']
        # pickle.dump((config, x_vals, _, charge_mask), open(random.choice(rand_file_names), 'wb'))
        return x_vals

    def _is_update_schedule_time(self, current_time, update_time):
        return current_time >= update_time

    def _is_update_time(self, current_time, update_time):
        return (current_time >= update_time and
                (self.last_bat_sched_time is None or
                 self.last_bat_sched_time < update_time or
                 self.last_bat_sched_time >= current_time))

    async def prepare_bat_sched_loop(self):
        while True:
            current_time = datetime.now(
                tz=pytz.timezone('Australia/Brisbane')).time()
            morning_update_time = datetime_time(0, 30)  # 0:30 AM
            afternoon_update_time = datetime_time(16, 45)  # 4:45 PM

            # We will force a re-schedule update at 4:45 PM
            if self._is_update_time(current_time, afternoon_update_time) or self._is_update_time(current_time, morning_update_time):
                # Push all customers' schedule to the AI server
                await self.prepare_bat_sched_all_users()
                self.last_bat_sched_time = current_time

            # Wait for 1 minute before checking again
            await asyncio.sleep(60)

    async def prepare_bat_sched_all_users(self):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for user_name in self.user_names:
                tasks.append(self.prep_bat_sched_each_user(user_name))
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_current_time_index(self, time_interval):
        now = datetime.now(pytz.timezone('Australia/Brisbane'))
        minutes_since_midnight = now.hour * 60 + now.minute
        index = (minutes_since_midnight * time_interval) // (24 * 60)
        return index

    async def prep_bat_sched_each_user(self, user_name):
        try:
            self.logger.info("Preparing schedule for %s", user_name)
            plant_id = self.user_manager.get_plant_for_user(user_name)

            prices = await self.get_prices(plant_id)
            pvs = await self.get_solar(plant_id)
            loads = await self.get_load(plant_id)
            plant_stats = await self.get_current_plant_stats(plant_id)
            soc_percentage = sum(device.get(
                'soc', 0) for device in plant_stats.values()) / (len(plant_stats)*100)
            bat_capacity = self.user_manager.get_user_profile(
                user_name).get('capacity', 0)
            bat_kwh_now = max(bat_capacity * soc_percentage, 0.1*bat_capacity)
            buy_prices = prices['buy']
            sell_prices = prices['sell']
            plant_charge_power = self.user_manager.get_user_profile(
                user_name).get('total_bat_charge_power', 0)
            plant_discharge_power = self.user_manager.get_user_profile(
                user_name).get('total_bat_discharge_power', 0)
            capacity = self.user_manager.get_user_profile(
                user_name).get('capacity', 0)
            current_time_index = self.get_current_time_index(48)
            existing_schedule = await self.get_schedule(plant_id)
            if existing_schedule is None:
                init_kwh = bat_kwh_now
            else:
                init_kwh = -existing_schedule[0]
            schedule = self.optimize(
                buy_prices, sell_prices, pvs, loads, plant_charge_power, plant_discharge_power, capacity, init_kwh, bat_kwh_now, current_time_index)

            await self.push_schedule_to_AI(plant_id, schedule, existing_schedule)
            self.schedule_for_compare[user_name] = self.make_battery_schedule(
                schedule, buy_prices, sell_prices, loads, pvs)
            self.schedule[user_name] = schedule
        except Exception as e:
            error_message = "Error preparing schedule for %s: %s \n Traceback: %s" % (
                user_name, e, traceback.format_exc())
            self.logger.error(error_message)

    async def push_schedule_to_AI(self, plant_id, schedule, existing_schedule) -> List[float]:
        schedule_copy = copy.deepcopy(schedule)
        await self.ai_client.ensure_login("ye.tao@redx.com.au", "1111")
        now = datetime.now(pytz.timezone('Australia/Brisbane'))
        date = now.strftime('%Y-%m-%d')
        # Upsample schedule to 288 points
        schedule_copy = np.interp(np.linspace(0, len(schedule_copy)-1, 288),
                                  np.arange(len(schedule_copy)), schedule_copy).tolist()
        # Round the float to 2 decimal places
        schedule_copy = [round(x, 2) for x in schedule_copy]

        # Merge the schedule with the existing schedule when pushing the schedule
        if existing_schedule is not None:
            current_time_index = self.get_current_time_index(288)
            schedule_copy = existing_schedule[:current_time_index] + \
                schedule_copy[current_time_index:]

        # Flip the schedule to match the AI server's format
        schedule_copy = [-x for x in schedule_copy]
        model_data = {
            "plant_id": plant_id,
            "date": date,
            "action_power": schedule_copy,
        }
        self.logger.info("Pushing schedule for %s: %s", plant_id, model_data)
        response = await self.ai_client.data_api_request("battery_actions/set", model_data)
        if response and ErrorCode(response.get('errorCode')) == ErrorCode.SUCCESS:
            self.logger.info(
                "Schedule data pushed successfully for %s", plant_id)
        elif response and ErrorCode(response.get('errorCode')) == ErrorCode.ALREADY_EXIST:
            self.logger.info(
                "Schedule data already exists for %s. Overwritting the data..", plant_id)
            response = await self.ai_client.data_api_request("battery_actions/update", model_data)
            if response and ErrorCode(response.get('errorCode')) == ErrorCode.SUCCESS:
                self.logger.info(
                    "Schedule data overwritten successfully for %s", plant_id)
            else:
                raise RuntimeError(
                    "Error overwriting schedule for %s: %s" % (plant_id, response.get('infoText')))
        else:
            raise RuntimeError(
                "Error pushing schedule for %s: %s" % (plant_id, response.get('infoText')))

    async def get_schedule(self, plant_id) -> List[float]:
        try:
            await self.ai_client.ensure_login("ye.tao@redx.com.au", "1111")

            now = datetime.now(pytz.timezone('Australia/Brisbane'))
            date = now.strftime('%Y-%m-%d')
            model_data = {
                "plant_id": plant_id,
                "date": date,
            }
            response = await self.ai_client.data_api_request("battery_actions/get", model_data)
            if response and ErrorCode(response.get('errorCode')) == ErrorCode.SUCCESS:
                battery_actions = util.decode_model_data(
                    response.get('data')[0].get('action_power'))
                # Flip the schedule to match the AI server's format
                battery_actions = [-x for x in battery_actions]
                self.logger.info(
                    "Schedule data obtained successfully for %s", plant_id)
                return battery_actions
            else:
                self.logger.info("Error request schedule for %s: %s",
                                plant_id, response.get('infoText'))
                return None
        except Exception as e:
            self.logger.error("Error getting schedule for %s: %s",
                             plant_id, e)
            return None

    def get_logs(self):
        with open('logs.txt', 'r') as file:
            logs = file.read()
        return logs

    def _init_device_profiles(self, sn):
        plant_id = self.user_manager.get_plant_for_device(sn)
        self.current_prices[plant_id] = {'buy': 0.0, 'feedin': 0.0}
        self.last_schedule_peakvalley[sn] = {'command': 'Idle'}

    def add_amber_device(self, sn):
        if sn not in self.sn_list:
            self.sn_list.append(sn)
            self._init_device_profiles(sn)
            self.logger.info("Added device %s to the scheduler.", sn)

    def remove_amber_device(self, sn):
        if sn in self.sn_list:
            self.sn_list.remove(sn)
            self.logger.info("Removed device %s from the scheduler.", sn)

    def get_current_price(self, retailer='amber', plant_id=None):
        return self.monitor.get_realtime_price(retailer=retailer, plant_id=plant_id)

    def get_current_time(self, state='qld') -> str:
        state_timezone_map = {
            'qld': 'Australia/Brisbane',
            'nsw': 'Australia/Sydney'
        }
        return self.monitor.get_current_time(state_timezone_map[state])

    async def get_current_plant_stats(self, identifier, is_sn=False):
        if is_sn:
            sn = identifier
            plant_id = self.user_manager.get_plant_for_device(sn)
        else:
            plant_id = identifier
        devices = self.user_manager.get_devices_for_plant(plant_id)
        stats = await self._get_plant_stats(devices)
        stats_dict = {device: stats for device, stats in zip(devices, stats)}
        return stats_dict

    async def _get_plant_stats(self, devices):
        stats = await asyncio.gather(
            *[self.monitor.get_realtime_battery_stats(device) for device in devices]
        )
        return stats

    async def get_current_battery_stats(self, sn):
        return await self.monitor.get_realtime_battery_stats(sn)

    async def send_battery_command(self, command=None, json=None, sn=None, time_zone='Australia/Brisbane'):
        await self.monitor.send_battery_command(
            peak_valley_command=command, json=json, sn=sn, time_zone=time_zone)


class ShawsbaySchedulerManager:
    """
    ShawsbaySchedulerManager class manages the scheduling of battery operations only for Shawsbay devices.

    This class is responsible for initializing and managing battery schedulers,
    handling different scheduler types, and coordinating battery operations
    across multiple devices specifically for Shawsbay projects.

    Attributes:
        config (dict): Configuration loaded from the config file.
        logger (logging.Logger): Logger for recording events and errors.
        scheduler: The active scheduler instance.
        monitor (util.PriceAndLoadMonitor): Monitor for price and load data.
        test_mode (bool): Flag indicating if the scheduler is in test mode.
        is_running (bool): Flag indicating if the scheduler is currently running.
        pv_sn (str): Serial number of the PV device.
        sn_list (list): List of battery serial numbers.
        last_schedule_ai (dict): Last schedule from AI algorithm.
        last_schedule_peakvalley (dict): Last schedule from peak-valley algorithm.
        last_five_metre_readings (list): Last five meter readings.
        battery_original_discharging_powers (dict): Original discharging powers for batteries.
        battery_original_charging_powers (dict): Original charging powers for batteries.
        project_phase (int): Phase of the project.
        project_mode (str): Mode of the project (e.g., 'normal', 'Peak Shaving').
        sn_types (dict): Types of batteries keyed by serial number.
        sn_locations (dict): Locations of batteries keyed by serial number.
        sn_retailers (dict): Retailers of batteries keyed by serial number.
        algo_types (dict): Algorithm types for batteries keyed by serial number.
        last_command_time (dict): Last command times for each device.
        current_prices (dict): Current prices for each device.
        sample_interval (int): Interval for sampling data.

        Args:
        scheduler_type (str): Type of scheduler to use (default: 'AIScheduler').
        battery_sn (list): List of battery serial numbers.
        test_mode (bool): Whether to run in test mode (default: False).
        api_version (str): Version of the API to use (default: 'dev3').
        pv_sn (str): Serial number of the PV device (default: None).
        phase (int): Project phase (default: 2).
        config (str): Path to the config file (default: 'config.toml').
        project_mode (str): Mode of the project (default: 'normal').
    """

    def __init__(self, scheduler_type='AIScheduler',
                 battery_sn=[],
                 test_mode=False,
                 api_version='dev3',
                 pv_sn=None,
                 phase=2,
                 config='config.toml',
                 project_mode='normal'
                 ):
        '''
        Args:
        pv_sn: str, the serial number of the PV device in a Project
        battery_sn: list, a list of serial numbers of the battery devices
        test_mode: bool, if True, the scheduler will use the test mode for debugging
        api_version: str, the version of the API, e.g., 'dev3', 'redx'
        phase: int, the phase of the project
        config: str, the path to the config file, config file should be in TOML format, see config.toml for an example
        project_mode: str, the mode of the project of AI scheduler, e.g., 'Peak Shaving', 'Money Saving', 'Normal'
        '''
        self.config = load_config(config)
        self.logger = logging.getLogger('shawsbay_logger')
        self.scheduler = None
        self.monitor = util.PriceAndLoadMonitor(api_version=api_version)
        self.test_mode = test_mode
        self.is_runing = False
        self.pv_sn = pv_sn
        self.sn_list = battery_sn if type(battery_sn) == list else [
            battery_sn]
        self.last_schedule_ai = {}
        self.last_schedule_peakvalley = {}
        self.last_five_metre_readings = []
        self.battery_original_discharging_powers = {}
        self.battery_original_charging_powers = {}
        self.project_phase = phase
        self.project_mode = project_mode
        self.sn_types = self.config.get('battery_types', {})
        self.sn_locations = self.config.get('battery_locations', {})
        self.sn_retailers = self.config.get('battery_retailers', {})
        self.algo_types = self.config.get('battery_algo_types', {})
        self.last_command_time = {}
        self.current_prices = {sn: {'buy': 0.0, 'feedin': 0.0}
                               for sn in self.sn_list}
        scheduler_type == 'AIScheduler'
        self.sample_interval = self.config.get(
            'shawsbay', {}).get('SampleInterval', 900)
        self.scheduler = ShawsbayAlgo(
            sn_list=self.sn_list, api_version=api_version, pv_sn=pv_sn,
            phase=self.project_phase,
            mode=self.project_mode)

    def _get_battery_command(self, **kwargs):
        if not self.scheduler:
            raise ValueError("Scheduler not set. Use set_scheduler() first.")

        return self.scheduler.step(**kwargs)

    def _make_battery_decision(self):
        while True:
            if not self.is_running:
                self.logger.info("Exiting the battery scheduler...")
                return

            try:
                self._process_ai_scheduler()
                time.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error("Scheduling error: %s", e)
                self.logger.error("Traceback: %s", traceback.format_exc())
                time.sleep(self.sample_interval)
                # self._make_battery_decision()

    def _process_ai_scheduler(self):
        schedule = self._get_battery_command()
        load = self.get_project_status(phase=self.project_phase)
        current_time = self.get_current_time()
        schedule = self.scheduler.adjust_discharge_power(
            schedule, load, current_time)
        schedule = self.scheduler.adjust_charge_power(
            schedule, load, current_time)
        self.logger.info("Schedule: %s", schedule)
        time_zone = 'Australia/Sydney'

        for sn in self.sn_list:
            battery_schedule = schedule.get(sn, {})
            last_battery_schedule = self.last_schedule_ai.get(sn, {})
            if all(battery_schedule.get(k) == last_battery_schedule.get(k) for k in battery_schedule) and all(battery_schedule.get(k) == last_battery_schedule.get(k) for k in last_battery_schedule):
                # self.logger.info(f"Schedule for {sn} is the same as the last one, skip sending command.")
                continue
            try:
                thread = Thread(target=self.send_battery_command,
                                kwargs={'json': battery_schedule, 'sn': sn, 'time_zone': time_zone})
                thread.start()
            except Exception as e:
                self.logger.error("Error sending battery command: %s", e)
                continue

        self.last_schedule_ai = copy.deepcopy(schedule)

    def start(self):
        '''
        Main loop for the battery scheduler.
        Steps:

        1. Collect Amber prices Per 2 minutes
        2. Collect LocalVolts prices Per 5 minutes
        3. Make battery decision Periondically (Check SampleInterval in the config.toml file)
        '''
        self.is_running = True
        with concurrent.futures.ThreadPoolExecutor() as executor:
            time.sleep(5)
            executor.submit(self._make_battery_decision)
            executor.submit(self._health_checker_devices)

    def stop(self):
        self.is_runing = False
        self.logger.info("Stopped.")

    def _health_checker_devices(self):
        # TODO: Implement health checker
        pass

    def get_logs(self):
        with open('sb_logs.txt', 'r') as file:
            logs = file.read()
        return logs

    def _init_device_profiles(self, sn):
        self.current_prices[sn] = {'buy': 0.0, 'feedin': 0.0}
        self.last_schedule_peakvalley[sn] = {'command': 'Idle'}

    def get_current_price(self, location='qld', retailer='amber'):
        return self.monitor.get_realtime_price(location=location, retailer=retailer)

    def get_current_time(self, state='qld') -> str:
        state_timezone_map = {
            'qld': 'Australia/Brisbane',
            'nsw': 'Australia/Sydney'
        }
        return self.monitor.get_current_time(state_timezone_map[state])

    def get_project_status(self, project_id: int = 1, phase: int = 2) -> float:
        if len(self.last_five_metre_readings) >= 2:
            self.last_five_metre_readings.pop(0)
        try:
            new_value = self.monitor.get_project_stats(project_id, phase)
            self.last_five_metre_readings.append(new_value)
            return sum(self.last_five_metre_readings) / len(self.last_five_metre_readings)
        except AttributeError:
            return 0

    async def get_current_battery_stats(self, sn):
        return await self.monitor.get_realtime_battery_stats(sn)

    async def send_battery_command(self, command=None, json=None, sn=None, time_zone='Australia/Brisbane'):
        await self.monitor.send_battery_command(
            peak_valley_command=command, json=json, sn=sn, time_zone=time_zone)


@dataclass
class ChargingPoint:
    '''
    Charging Points:
    ================
    Each charging point represents a specific point in time during the charging process.
    Charging Points is used to calculate the weighted price.
    '''
    charging_price: float
    grid_charge_power: float
    battery_charge_power: float


@dataclass
class DayChargingData:
    '''
    Day Charging Data:
    ==================
    Store the charging data of a device for a specific day.
    '''
    last_soc: float
    date: str
    charging_points: list[ChargingPoint]
    weighted_charging_cost: float


@dataclass
class ActionEnvObservation:
    buy_price: float
    sell_price: float
    load: float
    solar: float


@dataclass
class BatteryAction:
    action_plant: float
    action_device: float
    env: ActionEnvObservation

    @property
    def is_anti_backflow_on(self):
        threshold = 0.5
        return abs(-self.action_plant - self.env.load) < threshold

    @property
    def is_grid_charge_on(self):
        threshold = 1.0
        return self.action_plant - (self.env.solar - self.env.load) > threshold

    def is_confident(self, real_env: ActionEnvObservation):
        buy_price_diff = abs(self.env.buy_price - real_env.buy_price)
        sell_price_diff = abs(self.env.sell_price - real_env.sell_price)
        threshold_dollar = 1.0
        return ((buy_price_diff + sell_price_diff) / 2) < threshold_dollar


@dataclass
class BatterySchedule:
    actions: list[BatteryAction]


class HybridAlgo():
    def __init__(self, config_path='config.toml', monitor=None):
        peak_valley_config = load_config(config_path)['peakvalley']
        self.config = load_config(config_path)
        self.logger = logging.getLogger('logger')
        self.monitor = monitor
        self.BatNum = peak_valley_config['BatNum']
        self.BatMaxCapacity = peak_valley_config['BatMaxCapacity']
        self.BatCap = self.BatNum * self.BatMaxCapacity
        self.BatChgMax = self.BatNum * \
            peak_valley_config['BatChgMaxMultiplier']
        self.BatDisMax = self.BatNum * \
            peak_valley_config['BatDisMaxMultiplier']
        self.HrMin = peak_valley_config['HrMin']
        self.SellDiscount = peak_valley_config['SellDiscount']
        self.SpikeLevel = peak_valley_config['SpikeLevel']
        self.SolarCharge = peak_valley_config['SolarCharge']
        self.SellBack = peak_valley_config['SellBack']
        self.BuyPct = peak_valley_config['BuyPct']
        self.SellPct = peak_valley_config['SellPct']
        self.PeakPct = peak_valley_config['PeakPct']
        self.PeakPrice = peak_valley_config['PeakPrice']
        self.LookBackDays = peak_valley_config['LookBackDays']
        self.sample_interval = peak_valley_config['SampleInterval']
        self.LookBackBars = 24 * 60 / \
            (self.sample_interval / 60) * self.LookBackDays
        self.ChgStart1 = peak_valley_config['ChgStart1']
        self.ChgEnd1 = peak_valley_config['ChgEnd1']
        self.DisChgStart2 = peak_valley_config['DisChgStart2']
        self.DisChgEnd2 = peak_valley_config['DisChgEnd2']
        self.DisChgStartTest = peak_valley_config['DisChgStartTest']
        self.DisChgEndTest = peak_valley_config['DisChgEndTest']
        self.DisChgStart1 = peak_valley_config['DisChgStart1']
        self.DisChgEnd1 = peak_valley_config['DisChgEnd1']
        self.PeakStart = peak_valley_config['PeakStart']
        self.PeakEnd = peak_valley_config['PeakEnd']

        self.date = None
        self.last_updated_times = {}
        self.last_soc = None

        # Initial data containers and setup
        self.price_historys = {}
        self.charging_costs = {}
        self.solar = None

        # Convert start and end times to datetime.time
        self.t_chg_start1 = datetime.strptime(
            self.ChgStart1, '%H:%M').time()
        self.t_chg_end1 = datetime.strptime(
            self.ChgEnd1, '%H:%M').time()
        self.t_dis_start2 = datetime.strptime(
            self.DisChgStart2, '%H:%M').time()
        self.t_dis_end2 = datetime.strptime(
            self.DisChgEnd2, '%H:%M').time()
        self.t_dis_start1 = datetime.strptime(
            self.DisChgStart1, '%H:%M').time()
        self.t_dis_end1 = datetime.strptime(
            self.DisChgEnd1, '%H:%M').time()
        self.t_dis_start_test = datetime.strptime(
            self.DisChgStartTest, '%H:%M').time()
        self.t_dis_end_test = datetime.strptime(
            self.DisChgEndTest, '%H:%M').time()
        self.t_peak_start = datetime.strptime(
            self.PeakStart, '%H:%M').time()
        self.t_peak_end = datetime.strptime(
            self.PeakEnd, '%H:%M').time()

    def init_device_charge_cost(self, device_sn):
        self.charging_costs[device_sn] = DayChargingData(
            last_soc=0.0,
            date=datetime.now().strftime('%Y-%m-%d'),
            charging_points=[ChargingPoint(
                charging_price=0.0,
                grid_charge_power=0.0,
                battery_charge_power=0.0
            )],
            weighted_charging_cost=0.0
        )

    def init_price_history(self, sn, length=720):
        '''
        sn: str, the serial number of the device
        length: int, the length of the returned price history, it's interpolated from the original price history with a 30-minute interval
        '''
        price_history = self.monitor.get_price_history(sn, length)
        self.price_historys[sn] = price_history

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
            try:
                weather_fetcher = WeatherInfoFetcher('Shaws Bay')
                rain_info = weather_fetcher.get_rain_cloud_forecast_24h(
                    weather_fetcher.get_response())
                # here we give clouds more weight than rain based on the assumption that clouds have a bigger impact on solar generation
                max_solar = (
                    1-(1.4*rain_info['clouds']+0.6*rain_info['rain'])/(2*100))*max_solar
                self.logger.info(
                    "Weather forecast: rain: %s, clouds: %s, max_solar: %s", rain_info["rain"], rain_info["clouds"], max_solar)
            except Exception as e:
                self.logger.error("%s", e)

        _, gaus_y = gaussian_mixture(interval)
        return gaus_y*max_solar

    def _discharge_confidence(self, current_price):
        """
        Calculate the discharge confidence level based on the current price.

        Parameters:
        current_price (float): The current price value.

        Returns:
        float (0-1): The discharge confidence level.
        """
        conf_level = 0.97 - 0.8824 * math.exp(-0.033 * current_price)
        return conf_level

    def _algo_smart(self, current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pvkW, device_type: DeviceType, device_sn, schedule: BatterySchedule, device_percentage, plant_pvKW, plant_loadKW):
        # Compare with the pre-calculated schedule
        current_time = datetime.strptime(current_time, '%H:%M').time()
        if schedule:
            actual_env = ActionEnvObservation(
                buy_price=current_buy_price/100, sell_price=current_feedin_price/100, load=current_usage, solar=current_pvkW)
            current_total_min = current_time.hour * 60 + current_time.minute
            # 5 minutes interval, 288 points in total
            current_time_idx = current_total_min // 5
            battery_action = schedule.actions[current_time_idx]
            is_confident = battery_action.is_confident(actual_env)

            if is_confident:
                # Use the pre-calculated action from the schedule
                scheduled_action = battery_action.action_device
                scheduled_action = scheduled_action * 1000
                if scheduled_action > 0:
                    is_grid_charge_on = battery_action.is_grid_charge_on
                    # if the scheduled action is grid charge, use the scheduled action
                    # otherwise, absorb the excess solar with the adjusted power
                    if is_grid_charge_on:
                        command = {'command': 'Charge', 'power': abs(
                            scheduled_action), 'grid_charge': True}
                    else:
                        excess_solar = max(0, plant_pvKW - plant_loadKW)*1000
                        excess_solar = min(max(0, excess_solar), scheduled_action)
                        adjusted_power_w = excess_solar * device_percentage
                        command = {'command': 'Charge', 'power': adjusted_power_w, 'grid_charge': True} # enable grid_charge in case some devices have extra solar but some don't, force all to follow plan to achieve plant level charge alignment
                    # command = {'command': 'Charge', 'power': abs(
                    #     scheduled_action), 'grid_charge': True}
                elif scheduled_action < 0:
                    is_anti_backflow_on = battery_action.is_anti_backflow_on
                    command = {'command': 'Discharge', 'power': abs(
                        scheduled_action), 'anti_backflow': is_anti_backflow_on}
                else:
                    command = {"command": "Idle"}

                self.logger.info(
                    "Using scheduled action for %s: %s", device_sn, command)
                return command

        # If no schedule available or not confident, proceed with the original logic
        # Update price history
        price_history = self.price_historys.get(device_sn, None)
        if price_history is None:
            self.init_price_history(device_sn)
            price_history = self.price_historys[device_sn]
        last_updated_time = self.last_updated_times.get(device_sn, None)
        if last_updated_time is None or current_time.minute != last_updated_time.minute:
            self.last_updated_times[device_sn] = current_time
            price_history.append(current_buy_price)
            if len(price_history) > self.LookBackBars:
                price_history.pop(0)

        # Set current_price to the median of the last five minutes
        sample_points_per_minute = 60 / self.sample_interval
        if current_buy_price < self.PeakPrice:
            current_buy_price = np.mean(
                price_history[int(-5 * sample_points_per_minute):])

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            price_history, [self.BuyPct, self.SellPct])
        peak_price = self.PeakPrice

        command = {"command": "Idle"}

        # Charging logic
        if self._is_charging_period(current_time) and ((current_buy_price <= buy_price) or (current_pvkW > current_usage)):
            maxpower, minpower = self._get_charge_limit(device_type)
            power, grid_charge = self._calculate_charging_power(
                current_time, current_pvkW, current_usage, minpower, maxpower, current_buy_price <= buy_price)
            command = {'command': 'Charge',
                       'power': power, 'grid_charge': grid_charge}

        # Discharging logic
        if self._is_discharging_period(current_time) and (current_buy_price >= sell_price) and current_soc > 0.1:
            power = self._get_discharge_limit(device_type)
            anti_backflow_threshold = np.percentile(
                price_history, self.PeakPct)
            anti_backflow = current_buy_price <= anti_backflow_threshold
            conf_level = self._discharge_confidence(
                current_buy_price - anti_backflow_threshold)
            power = max(min(power * conf_level + 900, power), 1000)
            command = {'command': 'Discharge', 'power': power,
                       'anti_backflow': anti_backflow}

        self.logger.info("AmberModel (Sell to Grid): price: %s, sell price: %s, usage: %s, "
                         "time: %s, command: %s", current_buy_price, sell_price, current_usage,
                         current_time, command)

        return command

    def _algo_sell_to_grid(self, current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pvkW, device_type: DeviceType, device_sn):
        # Update price history
        current_time = datetime.strptime(current_time, '%H:%M').time()
        price_history = self.price_historys.get(device_sn, None)
        if price_history is None:
            self.init_price_history(device_sn)
            price_history = self.price_historys[device_sn]
        last_updated_time = self.last_updated_times.get(device_sn, None)
        if last_updated_time is None or current_time.minute != last_updated_time.minute:
            self.last_updated_times[device_sn] = current_time
            price_history.append(current_buy_price)
            if len(price_history) > self.LookBackBars:
                price_history.pop(0)

        # Set current_price to the median of the last five minutes
        sample_points_per_minute = 60 / self.sample_interval
        if current_buy_price < self.PeakPrice:
            current_buy_price = np.mean(
                price_history[int(-5 * sample_points_per_minute):])

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            price_history, [self.BuyPct, self.SellPct])
        peak_price = self.PeakPrice

        command = {"command": "Idle"}

        # Charging logic
        if self._is_charging_period(current_time) and ((current_buy_price <= buy_price) or (current_pvkW > current_usage)):
            maxpower, minpower = self._get_charge_limit(device_type)
            power, grid_charge = self._calculate_charging_power(
                current_time, current_pvkW, current_usage, minpower, maxpower, current_buy_price <= buy_price)
            command = {'command': 'Charge',
                       'power': power, 'grid_charge': grid_charge}

        # Discharging logic
        if self._is_discharging_period(current_time) and (current_buy_price >= sell_price) and current_soc > 0.1:
            power = self._get_discharge_limit(device_type)
            anti_backflow_threshold = np.percentile(
                price_history, self.PeakPct)
            anti_backflow = current_buy_price <= anti_backflow_threshold
            conf_level = self._discharge_confidence(
                current_buy_price - anti_backflow_threshold)
            power = max(min(power * conf_level + 900, power), 1000)
            command = {'command': 'Discharge', 'power': power,
                       'anti_backflow': anti_backflow}

        self.logger.info("AmberModel (Sell to Grid): price: %s, sell price: %s, usage: %s, "
                         "time: %s, command: %s", current_buy_price, sell_price, current_usage,
                         current_time, command)

        return command

    def _algo_auto_time(self, current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pv, current_batP, device_type: DeviceType, device_sn):
        # Use the Auto Mode (Now actually we are using charge with solar only, not using the auto mode) in the morning charging, and the Time Mode in the afternoon discharging
        # Update price history
        current_time = datetime.strptime(current_time, '%H:%M').time()
        price_history = self.price_historys[device_sn]
        if price_history is None:
            self.init_price_history(device_sn)
            price_history = self.price_historys[device_sn]
        last_updated_time = self.last_updated_times.get(device_sn, None)
        if last_updated_time is None or current_time.minute != last_updated_time.minute:
            self.last_updated_times[device_sn] = current_time
            price_history.append(current_buy_price)
            if len(price_history) > self.LookBackBars:
                price_history.pop(0)

        # Set current_price to the median of the last five minutes
        sample_points_per_minute = 60 / self.sample_interval
        if current_buy_price < self.PeakPrice:
            current_buy_price = np.mean(
                price_history[int(-5 * sample_points_per_minute):])

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            price_history, [self.BuyPct, self.SellPct])
        command = {"command": "Idle"}

        # Charging logic
        if self._is_charging_period(current_time) and ((current_buy_price <= buy_price) or (current_pv > current_usage)):
            maxpower, minpower = self._get_charge_limit(device_type)
            power, grid_charge = self._calculate_charging_power(
                current_time, current_pv, current_usage, minpower, maxpower, current_buy_price <= buy_price)
            command = {'command': 'Charge',
                       'power': power, 'grid_charge': grid_charge}

        # Discharging logic
        if self._is_discharging_period(current_time) and (current_buy_price >= sell_price) and current_soc > 0.1:
            power = self._get_discharge_limit(device_type)
            command = {'command': 'Discharge', 'power': power,
                       'anti_backflow': False}

        self.logger.info("AmberModel (Auto-Time): price: %s, usage: %s, "
                         "time: %s, command: %s", current_buy_price, current_usage,
                         current_time, command)

        return command

    def _algo_cover_usage(self, current_buy_price, current_feedin_price, current_time, current_usagekW, current_soc, current_pvkW, current_batpowerkW, device_type: DeviceType, device_sn):
        # Update price history
        current_time = datetime.strptime(current_time, '%H:%M').time()
        price_history = self.price_historys[device_sn]
        if price_history is None:
            self.init_price_history(device_sn)
            price_history = self.price_historys[device_sn]
        last_updated_time = self.last_updated_times.get(device_sn, None)
        if last_updated_time is None or current_time.minute != last_updated_time.minute:
            self.last_updated_times[device_sn] = current_time
            price_history.append(current_buy_price)
            if len(price_history) > self.LookBackBars:
                price_history.pop(0)

        # Set current_price to the median of the last five minutes
        sample_points_per_minute = 60 / self.sample_interval
        if current_buy_price < self.PeakPrice:
            current_buy_price = np.mean(
                price_history[int(-5 * sample_points_per_minute):])

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            price_history, [self.BuyPct, self.SellPct])
        peak_price = self.PeakPrice

        command = {"command": "Idle"}

        # Charging logic
        # Init the WeightedPrice (Charging Cost) for the device
        if device_sn not in self.charging_costs:
            self.init_device_charge_cost(device_sn)
        if self._is_charging_period(current_time) and ((current_buy_price <= buy_price) or (current_pvkW > current_usagekW)):
            maxpower, minpower = self._get_charge_limit(device_type)
            powerkW, grid_charge = self._calculate_charging_power(
                current_time, current_pvkW, current_usagekW, minpower, maxpower, current_buy_price <= buy_price)
            command = {'command': 'Charge',
                       'power': powerkW, 'grid_charge': grid_charge}
            # Update the weighted charging costs
            charge_powerkW = -current_batpowerkW
            self.update_weightedPrice(
                current_buy_price, current_usagekW, current_pvkW, device_sn, charge_powerkW)

        device_charge_cost = self.charging_costs.get(device_sn, None)
        weighted_price = device_charge_cost.weighted_charging_cost if device_charge_cost else None

        # Discharging logic
        # Turn off the debug flag to use the actual discharging period
        if self._is_discharging_period(current_time, debug=False) and (current_feedin_price >= weighted_price) and current_soc > 0.1 and current_pvkW < current_usagekW:
            anti_backflow = True
            powerkW = self._get_discharge_limit(device_type)
            device_charge_cost = self.charging_costs.get(device_sn, None)

            # Add dynamic discharging when price is very high
            anti_backflow_threshold = 150  # This is a provisional value, need to be adjusted
            if current_feedin_price > anti_backflow_threshold:
                conf_level = self._discharge_confidence(
                    current_feedin_price - anti_backflow_threshold)
                powerkW = max(min(powerkW * conf_level + 900, powerkW), 1000)
                anti_backflow = False
            command = {'command': 'Discharge', 'power': powerkW,
                       'anti_backflow': anti_backflow}

        self.logger.info("AmberModel(Cover Usage) : price: %s, WeightedPrice: %s, usage: %s, "
                         "time: %s, command: %s", current_buy_price, weighted_price, current_usagekW,
                         current_time, command)

        return command

    def update_weightedPrice(self, current_buy_price, current_usagekW, current_pvkW, device_sn, charge_powerkW):
        device_charge_cost = self.charging_costs.get(device_sn, None)

        # Ensure the stored charge_cost is only for the current day
        now_date_str = datetime.now().strftime('%Y-%m-%d')
        if now_date_str != device_charge_cost.date:
            self.init_device_charge_cost(device_sn)

        excess_energy = max(0, current_pvkW - current_usagekW)
        grid_charge_power = max(0, charge_powerkW - excess_energy)
        device_charge_cost.charging_points.append(ChargingPoint(
            charging_price=current_buy_price,
            grid_charge_power=grid_charge_power,
            battery_charge_power=charge_powerkW
        ))
        charging_prices = [
            point.charging_price for point in device_charge_cost.charging_points]
        grid_charge_powers = [
            point.grid_charge_power for point in device_charge_cost.charging_points]
        battery_charge_powers = [
            point.battery_charge_power for point in device_charge_cost.charging_points]

        device_charge_cost.weighted_charging_cost = np.multiply(np.array(charging_prices), np.array(
            grid_charge_powers)).sum()/(np.array(battery_charge_powers).sum() + 1e-6)

    def step(self, current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pv, current_batP, device_type, device_sn, algo_type='sell_to_grid', schedule=None, device_percentage=None, plant_pvKW=None, plant_loadKW=None):
        if algo_type == 'sell_to_grid':
            return self._algo_sell_to_grid(
                current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pv, device_type, device_sn)
        elif algo_type == 'auto_time':
            return self._algo_auto_time(
                current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pv, current_batP, device_type, device_sn)
        elif algo_type == 'smart':
            return self._algo_smart(
                current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pv, device_type, device_sn, schedule, device_percentage, plant_pvKW, plant_loadKW)

        else:
            return self._algo_cover_usage(
                current_buy_price, current_feedin_price, current_time, current_usage, current_soc, current_pv, current_batP, device_type, device_sn)

    def _get_charge_limit(self, device_type: DeviceType):
        match device_type:
            case DeviceType.FIVETHOUSAND:
                maxpower = 3000
                minpower = 1250
            case DeviceType.TWOFIVEZEROFIVE:
                maxpower = 1500
                minpower = 700
            case DeviceType.SEVENTHOUSAND:
                maxpower = 4600
                minpower = 1750
        return maxpower, minpower

    def _get_discharge_limit(self, device_type):
        match device_type:
            case DeviceType.FIVETHOUSAND:
                power = 5000
            case DeviceType.TWOFIVEZEROFIVE:
                power = 2500
            case DeviceType.SEVENTHOUSAND:
                power = 7000
        return power

    def _calculate_charging_power(self, current_time, current_pv, current_usage, minpower, maxpower, low_price=False):
        power = 0
        grid_charge = True
        excess_solar = 1000 * (current_pv - current_usage)

        if low_price:
            power = minpower
            grid_charge = True

        if excess_solar > 0:
            power = max(minpower, min(excess_solar, maxpower))
            grid_charge = True

        if datetime_time(6, 0) <= current_time <= datetime_time(9, 0):
            power = maxpower
            grid_charge = False

        return power, grid_charge

    def _is_charging_period(self, t):
        return t >= self.t_chg_start1 and t <= self.t_chg_end1

    def _is_discharging_period(self, t, debug=False):
        if debug:
            return t >= self.t_dis_start_test and t <= self.t_dis_end_test
        return (t >= self.t_dis_start2 and t <= self.t_dis_end2) or (t >= self.t_dis_start1 and t <= self.t_dis_end1)

    def _is_peak_period(self, t):
        return t >= self.t_peak_start and t <= self.t_peak_end


class ShawsbayAlgo():

    def __init__(self, sn_list, pv_sn, api_version='redx', phase=2, mode='normal'):
        self.logger = logging.getLogger('shawsbay_logger')
        self.battery_max_capacity_kwh = 5
        if sn_list is None:
            raise ValueError('sn_list is None')
        self.num_batteries = len(sn_list)
        self.price_weight = 1
        self.min_discharge_rate_kw = 1.25
        self.max_discharge_rate_kw = 2.5
        self.sn_list = sn_list
        self.pv_sn = pv_sn
        self.project_phase = phase
        self.battery_monitors = {sn: util.PriceAndLoadMonitor(
            test_mode=False, api_version=api_version) for sn in sn_list}
        self.schedule = None
        self.last_scheduled_date = None
        self.battery_original_discharging_powers = {}
        self.battery_original_charging_powers = {}
        self.mode = mode

    def _get_demand_and_price(self):
        # we don't need to get each battery's demand, just use the get_project_demand() method to get the total demand instead.
        # take the first battery monitor from the list
        try:
            demand = self.battery_monitors[self.sn_list[0]].get_project_demand_pred(phase=self.project_phase
                                                                                    )
        except Exception as e:
            demand, price = pickle.load(open('demand_price.pkl', 'rb'))
            self.logger.info(
                "Error fetching demand with get_prediction API, use the default demand instead. Error: %s", e)

        interval = int(24*60/len(demand))

        def time_to_minutes(t):
            hours, minutes = map(int, t.split(':'))
            return hours * 60 + minutes

        def get_price_array(shoulder_period, peak_period, shoulder_price, off_peak_price, peak_price, interval):
            shoulder_start1, shoulder_end1 = map(
                time_to_minutes, shoulder_period[0])
            shoulder_start2, shoulder_end2 = map(
                time_to_minutes, shoulder_period[1])
            peak_start1, peak_end1 = map(time_to_minutes, peak_period[0])
            peak_start2, peak_end2 = map(time_to_minutes, peak_period[1])
            price_array = []
            for i in range(0, 24 * 60, interval):
                mid_point = i + interval // 2
                if shoulder_start1 <= mid_point <= shoulder_end1 or shoulder_start2 <= mid_point <= shoulder_end2:
                    price_array.append(shoulder_price)
                elif peak_start1 <= mid_point <= peak_end1 or peak_start2 <= mid_point <= peak_end2:
                    price_array.append(peak_price)
                else:
                    price_array.append(off_peak_price)
            return price_array

        # Price for Shawsbay
        shoulder_period = [('9:00', '16:59'), ('20:00', '21:59')]
        peak_period = [('7:00', '8:59'), ('17:00', '19:59')]
        shoulder_price = 0.076586 + 0.059583 + 0.016196 + 0.007801
        off_peak_price = 0.058583 + 0.034259 + 0.016196 + 0.007801
        peak_price = 0.088695 + 0.079609 + 0.016196 + 0.007801

        price = get_price_array(shoulder_period, peak_period,
                                shoulder_price, off_peak_price, peak_price, interval)
        return np.array(demand, dtype=np.float64), np.array(price, dtype=np.float64)

    def _get_solar(self, interval=5/60, test_mode=False, max_solar_power=5000):
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
            try:
                weather_fetcher = WeatherInfoFetcher('Shaws Bay')
                rain_info = weather_fetcher.get_rain_cloud_forecast_24h(
                    weather_fetcher.get_response())
                # here we give clouds more weight than rain based on the assumption that clouds have a bigger impact on solar generation
                # sigma is used to adjust the impact of weather on solar generation
                sigma = 0.5
                max_solar = (
                    1-sigma*(1.4*rain_info['clouds']+0.6*rain_info['rain'])/(2*100))*max_solar
                self.logger.info(
                    "Weather forecast: rain: %s, clouds: %s, max_solar: %s", rain_info["rain"], rain_info["clouds"], max_solar)
            except Exception as e:
                self.logger.error("%s", e)

        _, gaus_y = gaussian_mixture(interval)
        return gaus_y*max_solar

    def _get_battery_status(self):
        battery_status = {}
        for sn in self.sn_list:
            battery_status[sn] = self.battery_monitors[sn].get_realtime_battery_stats(
                sn)

        return battery_status

    def adjust_charge_power(self, schedule, load, current_time):
        '''
        1. No adjustments are made during certain hours (16:00 to 06:00).
        2. Surplus power is calculated based on the load and mode.
        3. If surplus power is low (<=0W), reduce charging power for devices within their charging window.
        4. If surplus power is moderate (<=threshold+2000W), no adjustments are made.
        5. If surplus power is high (>3000W), increase charging power for devices not in their discharging window, up to a maximum limit.
        6. Update the charging start and end times in the schedule based on the current time and a 30-minute window.
        7. Return the updated schedule.

        '''
        if self.mode == 'normal':
            surplus_power = - load
        else:
            surplus_power = - load + 5000
        max_charging_power = 1500
        current_time_str = current_time
        current_time = datetime.strptime(current_time_str, '%H:%M')
        if current_time > datetime.strptime('16:00', '%H:%M') or current_time < datetime.strptime('06:00', '%H:%M'):
            return schedule

        # Return original schedule if surplus power is less than 0W
        # Reduce charging power, but not discharge to the grid, use current_load to track the remaining power
        # If the current_load reaches 0, stop reducing the charging power.
        threshold = 0
        if surplus_power <= threshold:
            current_load = surplus_power
            for sn in self.sn_list:
                if current_load >= threshold:
                    break
                start_time_str = schedule.get(
                    sn, {}).get('chargeStart1', '00:00')
                end_time_str = schedule.get(sn, {}).get('chargeEnd1', '00:00')
                if not (datetime.strptime(start_time_str, '%H:%M') <= current_time <= datetime.strptime(end_time_str, '%H:%M')):
                    continue
                original_charging_power = schedule.get(
                    sn, {}).get('chargePower1', 800)
                adjusted_charging_power = schedule.get(
                    sn, {}).get('chargePower1', 800)
                adjusted_charging_power = max(
                    adjusted_charging_power*0.6, self.battery_original_charging_powers.get(sn, 800))
                difference = original_charging_power - adjusted_charging_power
                # self.logger.info(
                # f'No surplus power, return original charging power for: {sn}')
                schedule[sn]['chargePower1'] = adjusted_charging_power
                current_load += difference
            return schedule

        # After we adjusted the power, the surplus power will change accordingly.
        # In order to avoid oscillation, we set a 2000 buffer to avoid frequent adjustments.
        if surplus_power <= threshold+2000:
            return schedule

        # Only when surplus power exceeded this 2000 buffer, then we can start to increase the charging power
        # We track the remaining power in the surplus_power variable, so that the surplus power will be used up.
        for sn in self.sn_list:
            if surplus_power <= 0:
                break
            discharge_start_str = schedule.get(
                sn, {}).get('dischargeStart1', '00:00')
            discharge_end_str = schedule.get(
                sn, {}).get('dischargeEnd1', '00:00')
            discharge_start = datetime.strptime(discharge_start_str, '%H:%M')
            discharge_end = datetime.strptime(discharge_end_str, '%H:%M')
            if current_time > discharge_start and current_time < discharge_end:
                continue

            current_charging_power = schedule.get(
                sn, {}).get('chargePower1', 800)
            adjusted_charging_power = min(
                max_charging_power, current_charging_power + surplus_power)
            surplus_power -= (adjusted_charging_power - current_charging_power)
            surplus_power = max(0, surplus_power)  # Avoid negative surplus
            end_30mins_later_str = (
                current_time + timedelta(minutes=30)).strftime('%H:%M')
            end_30mins_later = datetime.strptime(end_30mins_later_str, '%H:%M')
            if end_30mins_later > discharge_start:
                continue
            if sn in schedule:
                schedule[sn]['chargePower1'] = adjusted_charging_power
                schedule[sn]['chargeStart1'] = current_time_str
                if schedule[sn]['chargeEnd1'] < end_30mins_later_str:
                    schedule[sn]['chargeEnd1'] = end_30mins_later_str
                # self.logger.info(
                #     f'Increased charging power for Device: {sn} by {adjusted_charging_power - current_charging_power}W due to excess solar power.')
        return schedule

    def adjust_discharge_power(self, schedule: dict, load, current_time) -> dict:
        threshold_lower_bound = 4000
        threshold_upper_bound = threshold_lower_bound + 2000
        threshold_peak_bound = 12000

        current_time_str = current_time
        current_time = datetime.strptime(current_time, '%H:%M')
        load_now = load
        if current_time < datetime.strptime('15:00', '%H:%M'):
            return schedule

        if load >= threshold_peak_bound:
            # self.logger.info(
            #     f"Load is above peak threshold: {load}, Start increasing discharge power")
            for sn in self.sn_list:
                start_time = datetime.strptime(
                    schedule[sn]['dischargeStart1'], '%H:%M')
                end_time = datetime.strptime(
                    schedule[sn]['dischargeEnd1'], '%H:%M')
                end_time_str = schedule[sn]['dischargeEnd1']
                if start_time > current_time:
                    # self.logger.info(
                    #     f'--Device: {sn} is not in discharging period, skip.')
                    continue
                if end_time < current_time:
                    end_time = current_time + timedelta(minutes=30)
                    end_time_str = end_time.strftime('%H:%M')
                    schedule[sn]['dischargeEnd1'] = end_time_str
                # self.logger.info(
                #     f'Peak: Maximized discharging power for Device: {sn} by 30 minutes.')
                schedule[sn]['dischargePower1'] = 2000
            return schedule

        if load >= threshold_upper_bound:
            # self.logger.info(
            #     f"Load is above upper threshold: {load}, Start increasing discharge power")
            for sn in self.sn_list:
                start_time = datetime.strptime(
                    schedule[sn]['dischargeStart1'], '%H:%M')
                end_time = datetime.strptime(
                    schedule[sn]['dischargeEnd1'], '%H:%M')
                end_time_str = schedule[sn]['dischargeEnd1']
                if load_now <= threshold_upper_bound:
                    break
                if not start_time <= current_time <= end_time:
                    # self.logger.info(
                    #     f'--Device: {sn} is not in discharging period, skip.')
                    continue
                increased_power = schedule.get(
                    sn, {}).get('dischargePower1', 0)
                increased_power = min(
                    700+increased_power, self.battery_original_discharging_powers.get(sn, 1000))
                difference = increased_power - schedule[sn]['dischargePower1']
                # self.logger.info(
                #     f'--Increased discharge power for Device: {sn} by {difference}W. from: {schedule[sn]["dischargePower1"]}, to: {increased_power}')
                schedule[sn]['dischargePower1'] = increased_power
                load_now -= difference

        elif load <= threshold_lower_bound:
            # self.logger.info(
            #     f'Load is below lower threshold: {load}, Start lowering discharge power')
            for sn in self.sn_list:
                start_time = datetime.strptime(
                    schedule[sn]['dischargeStart1'], '%H:%M')
                end_time = datetime.strptime(
                    schedule[sn]['dischargeEnd1'], '%H:%M')
                end_time_str = schedule[sn]['dischargeEnd1']
                if load_now >= threshold_lower_bound:
                    break
                if not start_time <= current_time <= end_time:
                    # self.logger.info(
                    #     f'--Device: {sn} is not in discharging period, skip.')
                    continue
                decreased_power = schedule.get(
                    sn, {}).get('dischargePower1', 0)
                decreased_power = 0 if 0.5*decreased_power < 200 else 0.5*decreased_power
                difference = schedule[sn]['dischargePower1'] - decreased_power
                # self.logger.info(
                #     f'--Decreased discharge power for Device: {sn} by {difference}W. from: {schedule[sn]["dischargePower1"]}, to: {decreased_power}')
                schedule[sn]['dischargePower1'] = decreased_power
                load_now += difference

        return schedule

    def generate_schedule(self, consumption, price, batterie_capacity_kwh, num_batteries, stats, price_weight=1):
        import optuna
        from scipy.ndimage import gaussian_filter1d

        def greedy_battery_discharge(consumption, price, batteries, price_weight=1):
            # in order to let the discharging period end before 22:00 (1 hour has 12 * 5min intervals)
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

        def greedy_battery_charge_with_mask(consumption, price, solar, batteries, engaged_slots, price_weight=1):
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

        def objective_discharging(trial):
            max_discharge_length = self.battery_max_capacity_kwh / \
                self.min_discharge_rate_kw * len(consumption) / 24
            min_discharge_length = self.battery_max_capacity_kwh / \
                self.max_discharge_rate_kw * len(consumption) / 24
            durations = [trial.suggest_int(
                f'duration_{i}', min_discharge_length, max_discharge_length, step=int((max_discharge_length-min_discharge_length)/10)) for i in range(num_batteries)]
            batteries = [(batterie_capacity_kwh, duration)
                         for duration in durations]
            net_consumption, _ = greedy_battery_discharge(
                consumption, price, batteries, price_weight)
            avg_consumption = sum(net_consumption) / len(net_consumption)
            variance = sum((x - avg_consumption) **
                           2 for x in net_consumption) / len(net_consumption)
            return variance

        def smooth_demand(data, sigma):
            """Expand peaks using Gaussian blur."""
            return gaussian_filter1d(data, sigma)

        # 0. Preprocessing Consumption Data
        # Because the consumption is taking the battery charging energy into the total consumption
        # If we want to have the pure consumption data, we need to subtract the charging energy from the total consumption
        # The charging energy is estimated at 670W*4h = 2.68kWh
        # the subtraction is only applied on time from 8:00 to 15:00

        def preprocess_consumption(consumption: list) -> list:
            # 1. Remove the energy consumption by charging the battery
            samples_per_hour = len(consumption)/24
            charging_power_per_hour = 2100

            start_time = 8  # 8:00 AM
            end_time = 15   # 3:00 PM
            start_index = int(start_time * samples_per_hour)
            end_index = int(end_time * samples_per_hour)

            preprocessed_consumption = consumption[:]

            for i in range(start_index, end_index):
                preprocessed_consumption[i] -= charging_power_per_hour

            # 2. Clip the negative consumption prediction
            preprocessed_consumption = [max(0, i)
                                        for i in preprocessed_consumption]
            return preprocessed_consumption

        consumption = preprocess_consumption(consumption)

        # 1. Discharging
        consumption = smooth_demand(consumption, sigma=6)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_discharging, n_trials=100)
        best_durations = [
            study.best_params[f"duration_{i}"] for i in range(num_batteries)]
        discharging_capacity = [[batterie_capacity_kwh, duration]
                                for duration in best_durations]
        net_consumption, battery_discharges = greedy_battery_discharge(
            consumption, price, discharging_capacity)

        # 2. Charging
        # charge power set to 670W, the interval is 5 minutes, so the charge duration is 90
        charge_duration = 70
        charging_needs = [[batterie_capacity_kwh, charge_duration]
                          for x in range(num_batteries)]
        masks = [all(mask[i] == 0 for mask in battery_discharges)
                 for i in range(len(battery_discharges[0]))]
        num_pv_panels = len(self.pv_sn) if self.pv_sn is not None else 1

        _, battery_charges = greedy_battery_charge_with_mask(
            consumption, price, self._get_solar(max_solar_power=5000*num_pv_panels), charging_needs, masks)

        def _get_charging_window(battery_charge_schedule):
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

        def split_single_schedule(schedule, num_windows):
            start = schedule[0]
            duration = schedule[1]
            segments = [(round(i*duration/num_windows+start), round((i+1)*duration/num_windows+start) -
                        round(i*duration/num_windows+start)) for i in range(num_windows)]
            return segments

        def split_schedules(schedules, num_windows, task_type='Discharge'):
            ret = []
            for schedule in schedules:
                ret.extend(split_single_schedule(schedule, num_windows))
            ret = [(task_type, i[0], i[1]) for i in ret]
            return ret

        discharge_schedules = [_get_charging_window(
            i) for i in battery_discharges]
        charge_schedules = [_get_charging_window(i) for i in battery_charges]
        flat_discharge_schedules = [
            item for sublist in discharge_schedules for item in sublist]
        flat_charge_schedules = [
            item for sublist in charge_schedules for item in sublist]

        discharge_sched_split = split_schedules(
            flat_discharge_schedules, 1, 'Discharge')
        charge_sched_split = split_schedules(
            flat_charge_schedules, 1, 'Charge')

        # Use hardcoded schedules for charging windows.
        all_schedules = sorted(discharge_sched_split +
                               charge_sched_split, key=lambda x: x[1])

        class Battery:
            def __init__(self, soc, max_capacity, sn):
                self.sn = sn
                self.capacity = max_capacity*1000
                self.current_charge = soc  # Set initial charge to full capacity
                self.fully_charged_rounds = 0
                self.fully_discharged_rounds = 0
                self.available_at = 0  # The time when the battery is available again

            def is_depleted(self):
                return self.fully_discharged_rounds >= 2 and self.fully_charged_rounds >= 2

            def is_charged(self):
                return self.fully_charged_rounds >= 1

            def is_discharged(self):
                return self.fully_discharged_rounds >= 1

            def can_discharge(self, current_time):
                return self.current_charge > 0 and not self.is_discharged() and self.available_at <= current_time

            def can_charge(self, current_time):
                return not self.is_charged() and self.available_at <= current_time

            def discharge(self, duration, current_time):
                if self.can_discharge(current_time):
                    self.current_charge = 0
                    self.fully_discharged_rounds += 1
                    self.available_at = current_time + duration

            def charge(self, duration, current_time):
                if self.can_charge(current_time):
                    self.current_charge = self.capacity
                    self.fully_charged_rounds += 1
                    self.available_at = current_time + duration

        class ProjectBatteryManager:
            def __init__(self, tasks, battery_init_status, battery_max_capacity_kwh=5, sample_interval=5):
                self.tasks = sorted(tasks, key=lambda x: x[1])
                self.batteries = [
                    Battery(x[1]['soc']/100*battery_max_capacity_kwh*1000, battery_max_capacity_kwh, x[0]) for x in battery_init_status.items()]
                self.output = []
                self.battery_max_capacity_kwh = battery_max_capacity_kwh
                self.sample_interval = sample_interval

            def allocate_battery(self, task_time, task_type, task_duration):
                for battery in self.batteries:
                    if task_type == "Discharge" and battery.can_discharge(task_time):
                        battery.discharge(task_duration, task_time)
                        self.output.append(
                            (battery.sn, task_time, task_type, task_duration))
                        return True
                    elif task_type == "Charge" and battery.can_charge(task_time):
                        battery.charge(task_duration, task_time)
                        self.output.append(
                            (battery.sn, task_time, task_type, task_duration))
                        return True
                return False

            def generate_json_commands(self):
                for task_type, task_time, task_duration in self.tasks:
                    if not self.allocate_battery(task_time, task_type, task_duration):
                        self.logger.info(
                            "task %s %s %s failed to allocate battery", task_time, task_type, task_duration)

                def unit_to_time(unit, sample_interval):
                    total_minutes = int(unit * sample_interval)
                    hour = total_minutes // 60
                    minute = total_minutes % 60
                    return "{:02d}:{:02d}".format(hour, minute)

                schedules = {}
                for sn, task_time, task_type, task_duration in self.output:
                    power = int(batterie_capacity_kwh * 1000 *
                                60 / (self.sample_interval * task_duration))
                    start_time = unit_to_time(task_time, self.sample_interval)
                    end_time = unit_to_time(
                        task_time + task_duration, self.sample_interval)

                    if task_type == 'Discharge':
                        data = {
                            'deviceSn': sn,
                            'dischargeStart1': start_time if task_type == 'Discharge' else "00:00",
                            'dischargeEnd1': end_time if task_type == 'Discharge' else "00:00",
                            'dischargePower1': power,
                        }
                        schedules[sn] = data | schedules.get(sn, {})
                    # todo: Now using fixed time schedule, change back to start_time and end_time later.
                    elif task_type == 'Charge':
                        data = {
                            'deviceSn': sn,
                            'chargeStart1': start_time if task_type == 'Charge' else "00:00",
                            'chargeEnd1': end_time if task_type == 'Charge' else "00:00",
                            'chargePower1': power+200,  # Charging Power is increased by 200W to compensate in case
                        }
                        schedules[sn] = data | schedules.get(sn, {})
                return schedules

        tasks = all_schedules

        sample_interval = 24*60/len(consumption)
        battery_manager = ProjectBatteryManager(
            tasks, stats, self.battery_max_capacity_kwh, sample_interval)
        json_schedule = battery_manager.generate_json_commands()

        return json_schedule

    def _get_command_from_schedule(self, current_time):
        return 'Idle'

    def step(self):
        current_day = datetime.now(tz=pytz.timezone('Australia/Sydney')).day
        if self.last_scheduled_date != current_day:
            self.logger.info(
                "Updating schedule: day %s, time: %s, last_scheduled_date: %s", current_day, datetime.now(tz=pytz.timezone("Australia/Sydney")), self.last_scheduled_date)
            demand, price = self._get_demand_and_price()
            self.logger.info("demand: %s", demand)
            stats = self._get_battery_status()
            self.schedule = self.generate_schedule(
                demand, price, self.battery_max_capacity_kwh, self.num_batteries, stats, self.price_weight)
            self.last_scheduled_date = current_day
            for sn in self.sn_list:
                self.battery_original_discharging_powers[sn] = self.schedule.get(
                    sn, {}).get('dischargePower1', 0)
                self.battery_original_charging_powers[sn] = self.schedule.get(
                    sn, {}).get('chargePower1', 0)

        return self.schedule


def get_test_devices(count=100):
    import json
    import os
    file_path = os.path.join(os.path.dirname(
        __file__), 'ignoreMe/device_lists.json')
    with open(file_path) as f:
        data = json.load(f)
        sns = [device['deviceSn'] for device in data['data']]
    # choose random 100 devices
    import random
    random.shuffle(sns)
    sns = sns[:count]
    return sns


def test_shawsbay(test_mode=False):
    # For Phase 2
    scheduler = ShawsbaySchedulerManager(
        scheduler_type='AIScheduler',
        battery_sn=['RX2505ACA10J0A180011', 'RX2505ACA10J0A170035',
                    'RX2505ACA10J0A170033', 'RX2505ACA10J0A160007', 'RX2505ACA10J0A180010'],
        test_mode=False,
        api_version='redx',
        pv_sn=['RX2505ACA10J0A170033'],
        phase=2)
    scheduler.start()

    # For Phase 3
    # scheduler = BatteryScheduler(
    #     scheduler_type='AIScheduler',
    #     battery_sn=['RX2505ACA10J0A170013', 'RX2505ACA10J0A150006', 'RX2505ACA10J0A180002',
    #                 'RX2505ACA10J0A170025', 'RX2505ACA10J0A170019', 'RX2505ACA10J0A150008'],
    #     test_mode=False,
    #     api_version='redx',
    #     pv_sn=['RX2505ACA10J0A170033', 'RX2505ACA10J0A170019'],
    #     phase=3)


def test_scheduler(test_mode=False):
    # For Phase 2
    # scheduler = BatteryScheduler(
    #     scheduler_type='AIScheduler',
    #     battery_sn=['RX2505ACA10J0A180011', 'RX2505ACA10J0A170035', 'RX2505ACA10J0A170033', 'RX2505ACA10J0A160007', 'RX2505ACA10J0A180010'],
    #     test_mode=False,
    #     api_version='redx',
    #     pv_sn=['RX2505ACA10J0A170033'],
    #     phase=2)

    # For Phase 3
    # scheduler = BatteryScheduler(
    #     scheduler_type='AIScheduler',
    #     battery_sn=['RX2505ACA10J0A170013', 'RX2505ACA10J0A150006', 'RX2505ACA10J0A180002',
    #                 'RX2505ACA10J0A170025', 'RX2505ACA10J0A170019', 'RX2505ACA10J0A150008'],
    #     test_mode=False,
    #     api_version='redx',
    #     pv_sn=['RX2505ACA10J0A170033', 'RX2505ACA10J0A170019'],
    #     phase=3)

    # For Amber Johnathan (QLD)
    # scheduler = BatteryScheduler(scheduler_type='PeakValley', test_mode=False, api_version='redx')
    # For Amber Dion (NSW)
    scheduler = BatterySchedulerManager(
        scheduler_type='PeakValley',
        battery_sn=['011LOKL140104B',
                    '011LOKL140058B',
                    'RX2505ACA10J0A180003',
                    'RX2505ACA10J0A160016',
                    ],
        test_mode=test_mode, api_version='redx')
    scheduler.start()
    # time.sleep(300)
    # print('Scheduler started')
    # time.sleep(3)
    # scheduler.add_amber_device('011LOKL140104B')
    # scheduler.add_amber_device('RX2505ACA10J0A160016')


def multiple_random_devices_test(count=5):
    sns = get_test_devices(count=count)
    scheduler = BatterySchedulerManager(
        scheduler_type='PeakValley', battery_sn=sns, test_mode=True, api_version='redx')
    scheduler.start()


if __name__ == '__main__':
    # multiple_random_devices_test()
    test_scheduler()
    # test_shawsbay()
