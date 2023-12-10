# The class for auto schedule of charging/discharging the battery
import sched
from datetime import datetime, timedelta
import time
import numpy as np
import util
import pytz
import logging
import pickle
import pandas as pd
import copy
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)


class SimulationScheduler:

    def __init__(self, scheduler_type='PeakValley',
                 battery_sn=None,
                 test_mode=False,
                 api_version='dev3',
                 pv_sn=None,
                 phase=2,
                 discharging_starting_load=2000,
                 simulate_day='2023-04-09'):
        self.s = sched.scheduler(time.time, time.sleep)
        self.scheduler = None
        self.monitor = util.PriceAndLoadMonitor(
            test_mode=test_mode, api_version=api_version)
        self.test_mode = test_mode
        self.event = None
        self.is_runing = False
        self.pv_sn = pv_sn
        self.sn_list = battery_sn if type(battery_sn) == list else [
            battery_sn]
        self.last_schedule = {}
        self.schedule_before_hotfix = {}
        self.last_scheduled_date = None
        self.discharging_buffer = discharging_starting_load
        self.last_five_metre_readings = []
        self.battery_original_discharging_powers = {}
        self.battery_original_charging_powers = {}
        self.project_phase = phase
        self.generator = TimeIntervalGenerator(start_date_str=simulate_day)
        self.current_time = None
        self.load_df = None
        self.schedule_analyser = ScheduleAnalyser()
        self.schedule_list = []
        self._set_scheduler(scheduler_type, api_version, pv_sn=pv_sn)

    def _set_scheduler(self, scheduler_type, api_version, pv_sn=None):
        self.scheduler = AIScheduler(
            sn_list=self.sn_list, api_version=api_version, pv_sn=pv_sn)

    def _get_battery_command(self, **kwargs):
        if not self.scheduler:
            raise ValueError("Scheduler not set. Use set_scheduler() first.")

        required_data = self.scheduler.required_data()
        if not all(key in kwargs for key in required_data):
            raise ValueError(
                f"Data missing for {self.scheduler.__class__.__name__}. Required: {required_data}")

        return self.scheduler.step(**kwargs)

    def _start(self, interval=0.01):
        if not self.is_running:
            return
        self.current_time = self.generator.get_time()
        if self.current_time > self.current_time.replace(hour=23, minute=50):
            self.stop()
            self.plot_schedule()
            return
        logging.info(f'Current time: {self.current_time}')
        if isinstance(self.scheduler, AIScheduler):
            self._process_ai_scheduler()
        self.event = self.s.enter(interval, 1, self._start)

    def _process_ai_scheduler(self):
        schedule = self._get_battery_command()
        load = self.get_project_status(phase=self.project_phase)
        current_time = self.current_time
        current_time_str = datetime.strftime(current_time, '%H:%M')
        # schedule = self.scheduler.daytime_hotfix_discharging_smooth(schedule, load, current_time_str)
        # schedule = self.scheduler.daytime_hotfix_charging(schedule, load, current_time_str)
        # logging.info(f"Schedule: {schedule}")

        self.last_schedule = copy.deepcopy(schedule)
        for sn in self.sn_list:
            battery_schedule = schedule.get(sn, {})
            last_battery_schedule = self.last_schedule.get(sn, {})
            if all(battery_schedule.get(k) == last_battery_schedule.get(k) for k in battery_schedule) and all(battery_schedule.get(k) == last_battery_schedule.get(k) for k in last_battery_schedule):
                self.schedule_analyser.update_schedule(
                    sn, (battery_schedule, current_time_str))

    def start(self):
        self.is_running = True
        self._start()
        self.s.run()

    def stop(self):
        self.is_runing = False
        logging.info("Stopped.")

    def plot_schedule(self):
        self.schedule_analyser.plot_battery_power()

    def get_current_price(self):
        return self.monitor.get_realtime_price()

    def generate_5min_intervals(self, date_str='2022-11-25'):
        return self.generator.get_time()

    def get_project_status(self, project_id: int = 1, phase: int = 2) -> float:
        if self.load_df is None:
            path = 'data/shaws_bay_total.csv'
            self.load_df = self.load_sim_data(path)
        try:
            current_load = self.load_df.loc[self.load_df['time'] == self.current_time]['meter+battery'].values[0]
        except IndexError:
            logging.error(
                f'Error getting load for {self.current_time.strftime("%Y-%m-%d %H:%M")}')
            current_load = 2000
        return current_load

    def get_current_battery_stats(self, sn):
        raise NotImplementedError

    def load_sim_data(self, path):
        sim_df = pd.read_csv(path)
        sim_df['time'] = pd.to_datetime(
            sim_df['time'], format="%d/%m/%Y %H:%M")
        sim_df = sim_df.sort_values(by=['time'])
        return sim_df

    def load_sim_data_raw(self, path):
        self.sim_df = pd.read_csv(path)
        start_date = "2022-11-24"
        end_date = "2023-2-22"
        self.sim_df['logTime'] = pd.to_datetime(
            self.sim_df['logTime'], format="%Y-%m-%d %H:%M:%S.%f")
        self.sim_df = self.sim_df[self.sim_df['logTime'] < pd.to_datetime(
            end_date)]
        self.sim_df = self.sim_df[self.sim_df['logTime']
                                  > pd.to_datetime(start_date)]
        self.sim_df = self.sim_df.sort_values(by=['logTime'])
        voltages_phase1 = self.sim_df['VoltageA'].values
        currents_phase1 = self.sim_df['CurrentA'].values
        voltages_phase2 = self.sim_df['VoltageB'].values
        currents_phase2 = self.sim_df['CurrentB'].values
        voltages_phase3 = self.sim_df['VoltageC'].values
        currents_phase3 = self.sim_df['CurrentC'].values

        load_phase1 = np.multiply(voltages_phase1, currents_phase1)/10000
        load_phase2 = np.multiply(voltages_phase2, currents_phase2)/10000
        load_phase3 = np.multiply(voltages_phase3, currents_phase3)/10000
        new_df = pd.DataFrame({'logTime': self.sim_df['logTime'].values,
                               'load_phase1': load_phase1,
                               'load_phase2': load_phase2,
                               'load_phase3': load_phase3, }
                              )
        return new_df


class TimeIntervalGenerator:
    def __init__(self, start_date_str='2023-04-09'):
        self.start_time = datetime.fromisoformat(start_date_str)
        self.generator = self._time_generator()

    def _time_generator(self):
        current_time = self.start_time
        while True:
            yield current_time
            current_time += timedelta(minutes=5)

    def get_time(self):
        return next(self.generator)


class ScheduleAnalyser:
    def __init__(self, schedules=None):
        self.schedules = schedules if schedules else {}

    def update_schedule(self, sn, schedule):
        """Update the schedule dictionary with a new schedule."""
        if sn not in self.schedules:
            self.schedules[sn] = []
        new_schedule = copy.deepcopy(schedule)
        self.schedules[sn].append(new_schedule)

    def parse_time(self, time_str):
        """Parse a time string in the format HH:MM to a datetime object."""
        return datetime.strptime(time_str, '%H:%M')

    def battery_schedule(self, schedule_list, initial_power=0, max_capacity=5000):
        """Process a series of battery schedules and return power usage."""
        all_power_usage = []
        for i, (schedule, start_time_str) in enumerate(schedule_list):
            start_time = self.parse_time(start_time_str)
            end_time = self.parse_time('23:59') if i == len(
                schedule_list) - 1 else self.parse_time(schedule_list[i + 1][1])
            power_usage = self.calculate_power(
                schedule, start_time, end_time, initial_power, max_capacity)
            all_power_usage.extend(power_usage)
            # Update initial power for the next schedule
            initial_power = power_usage[-1][1]

        return all_power_usage

    def calculate_power(self, schedule, start_time, end_time, initial_power, max_capacity):
        """Calculate the power usage or charging for a given time period."""
        remaining_power = initial_power
        power_usage = []

        current_time = start_time
        while current_time < end_time:
            is_charging = False
            is_discharging = False
            charging_power = 0
            discharge_power = 0

            charge_start = self.parse_time(schedule['chargeStart1'])
            charge_end = self.parse_time(
                schedule['chargeStart1'.replace('Start', 'End')])
            charge_power = schedule['chargeStart1'.replace('Start', 'Power')]

            # Adjust the charge start and end times to the current date for comparison
            charge_start = current_time.replace(
                hour=charge_start.hour, minute=charge_start.minute)
            charge_end = current_time.replace(
                hour=charge_end.hour, minute=charge_end.minute)

            if charge_start <= current_time <= charge_end:
                is_charging = True
                charging_power = charge_power

            discharge_start = self.parse_time(schedule['dischargeStart1'])
            discharge_end = self.parse_time(
                schedule['dischargeStart1'.replace('Start', 'End')])
            discharge_power = schedule['dischargeStart1'.replace(
                'Start', 'Power')]

            # Adjust the discharge start and end times to the current date for comparison
            discharge_start = current_time.replace(
                hour=discharge_start.hour, minute=discharge_start.minute)
            discharge_end = current_time.replace(
                hour=discharge_end.hour, minute=discharge_end.minute)

            if discharge_start <= current_time <= discharge_end:
                is_discharging = True
                discharging_power = discharge_power
            if is_charging:
                # Calculate the charging for this hour
                if remaining_power + charging_power > max_capacity:
                    charging_power = max_capacity - remaining_power
                remaining_power = min(
                    remaining_power + charging_power/12, max_capacity)
            elif is_discharging:
                # Calculate the discharging for this hour
                if remaining_power - discharging_power < 0:
                    discharging_power = 0
                remaining_power = max(
                    remaining_power - discharging_power/12, 0)

            else:
                charging_power = 0
                discharging_power = 0

            power_usage.append((current_time.strftime(
                '%H:%M'), remaining_power, charging_power if is_charging else -discharging_power))
            current_time += timedelta(minutes=5)

        return power_usage

    def plot_battery_power(self, initial_power=0, max_capacity=5000):
        """Plot a series of battery schedules."""
        time_axis = []

        total_values = []
        for schedule in self.schedules.items():
            all_power_usage = self.battery_schedule(
                schedule[1], initial_power, max_capacity)
            power_values = []
            start_time = self.parse_time("00:00")
            number_of_5_min_per_day = 24 * 60 / 5

            for _5min in range(int(number_of_5_min_per_day)):
                current_time = (
                    start_time + timedelta(minutes=5 * _5min)).strftime('%H:%M')

                power_for_current_time = next(
                    (power for time, _, power in all_power_usage if time == current_time), None)
                if power_for_current_time is not None:
                    power_values.append(power_for_current_time)
                else:
                    power_values.append(
                        power_values[-1] if power_values else 0)

            power_values = np.array(power_values)
            total_values.append(power_values)

        total_values = np.stack(total_values).sum(axis=0)


        consumption, _ = pickle.load(open('demand_price.pkl', 'rb'))
        power_from_grid = total_values + consumption
        time_axis = np.arange(
            0, len(total_values)*5, 5)

        # Plotting the graph
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, total_values)
        plt.plot(time_axis, power_from_grid)
        plt.plot(time_axis, consumption)
        plt.xticks(rotation=45)
        plt.xlabel('Time')
        plt.ylabel('Power Usage (kWh)')
        plt.title('Battery Schedule')
        plt.grid(True)
        plt.show()


class AIScheduler():

    def __init__(self, sn_list, pv_sn, api_version='redx',phase=2, mode='normal'):
        self.battery_max_capacity_kwh = 5
        self.num_batteries = len(sn_list)
        self.price_weight = 1
        self.min_discharge_rate_kw = 1.25
        self.max_discharge_rate_kw = 2.5
        self.api = util.ApiCommunicator(
            'https://da2e586eae72a40e5bde4ead0fe77b2f0.clg07azjl.paperspacegradient.com/')
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
        demand = self.battery_monitors[self.sn_list[0]].get_project_demand(
        )
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

        _, gaus_y = gaussian_mixture(interval)
        return gaus_y*max_solar

    def _get_battery_status(self):
        battery_status = {}
        for sn in self.sn_list:
            battery_status[sn] = self.battery_monitors[sn].get_realtime_battery_stats(
                sn)

        return battery_status

    def daytime_hotfix_charging(self, schedule, load, current_time):
            if self.mode == 'normal':
                surplus_power =  - load
            else:
                surplus_power = - load + 5000
            max_charging_power = 1500
            current_time_str = current_time 
            current_time = datetime.strptime(current_time_str, '%H:%M')
            if current_time > datetime.strptime('16:00', '%H:%M') or current_time < datetime.strptime('06:00', '%H:%M'):
                return schedule

            # Return original schedule if surplus power is less than 0W
            threshold = 0
            if surplus_power <= threshold:
                power_now = surplus_power
                for sn in self.sn_list:
                    if power_now >= threshold:
                        break
                    start_time_str = schedule.get(sn, {}).get('chargeStart1', '00:00')
                    end_time_str = schedule.get(sn, {}).get('chargeEnd1', '00:00')
                    if not (datetime.strptime(start_time_str, '%H:%M') <= current_time <= datetime.strptime(end_time_str, '%H:%M')):
                        continue
                    original_charging_power = schedule.get(
                        sn, {}).get('chargePower1', 800)
                    adjusted_charging_power = schedule.get(
                        sn, {}).get('chargePower1', 800)
                    adjusted_charging_power = max(adjusted_charging_power*0.6, self.battery_original_charging_powers.get(sn, 800))
                    difference = original_charging_power - adjusted_charging_power
                    # logging.info(
                        # f'No surplus power, return original charging power for: {sn}')
                    schedule[sn]['chargePower1'] = adjusted_charging_power
                    power_now += difference
                return schedule

            if surplus_power <= threshold+2000:
                return schedule
            
            # Only when surplus power is greater than 3000W, we start to increase the charging power
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
                end_30mins_later_str = (current_time + timedelta(minutes=30)).strftime('%H:%M')
                end_30mins_later = datetime.strptime(end_30mins_later_str, '%H:%M')
                if end_30mins_later > discharge_start: 
                    continue
                if sn in schedule:
                    schedule[sn]['chargePower1'] = adjusted_charging_power
                    schedule[sn]['chargeStart1'] = current_time_str
                    if schedule[sn]['chargeEnd1'] < end_30mins_later_str:
                        schedule[sn]['chargeEnd1'] = end_30mins_later_str
                    logging.info(
                        f'Increased charging power for Device: {sn} by {adjusted_charging_power - current_charging_power}W due to excess solar power.')
            return schedule

    def daytime_hotfix_discharging_smooth(self, schedule: dict, load, current_time) -> dict:
        threshold_lower_bound = 5000
        threshold_upper_bound = threshold_lower_bound + 3000 
        threshold_peak_bound = 14000

        current_time_str = current_time
        current_time = datetime.strptime(current_time, '%H:%M')
        load_now = load 
        if current_time < datetime.strptime('15:00', '%H:%M'):
            return schedule
        

        if load >= threshold_peak_bound:
            logging.info(f"Load is above peak threshold: {load}, Start increasing discharge power")
            for sn in self.sn_list:
                start_time = datetime.strptime(schedule[sn]['dischargeStart1'], '%H:%M')
                end_time = datetime.strptime(schedule[sn]['dischargeEnd1'], '%H:%M')
                end_time_str = schedule[sn]['dischargeEnd1']
                if start_time > current_time:
                    logging.info(f'--Device: {sn} is not in discharging period, skip.')
                    continue
                if end_time < current_time:
                    end_time = current_time +timedelta(minutes=30)
                    end_time_str = end_time.strftime('%H:%M')
                    schedule[sn]['dischargeEnd1'] = end_time_str 
                logging.info(f'Peak: Maximized discharging power for Device: {sn} by 30 minutes.')
                schedule[sn]['dischargePower1'] = 2500
            return schedule

        if load >= threshold_upper_bound:
            logging.info(f"Load is above upper threshold: {load}, Start increasing discharge power")
            for sn in self.sn_list:
                start_time = datetime.strptime(schedule[sn]['dischargeStart1'], '%H:%M')
                end_time = datetime.strptime(schedule[sn]['dischargeEnd1'], '%H:%M')
                end_time_str = schedule[sn]['dischargeEnd1']
                if load_now <= threshold_upper_bound:
                    break
                if not start_time <= current_time <= end_time:
                    logging.info(f'--Device: {sn} is not in discharging period, skip.')
                    continue
                increased_power = schedule.get(
                    sn, {}).get('dischargePower1', 0)
                increased_power = min(700+increased_power, self.battery_original_discharging_powers.get(sn, 1000))
                difference = increased_power - schedule[sn]['dischargePower1']
                logging.info(f'--Increased discharge power for Device: {sn} by {difference}W. from: {schedule[sn]["dischargePower1"]}, to: {increased_power}')
                schedule[sn]['dischargePower1'] = increased_power
                load_now -= difference

        elif load <= threshold_lower_bound:
            logging.info(f'Load is below lower threshold: {load}, Start lowering discharge power')
            for sn in self.sn_list:
                start_time = datetime.strptime(schedule[sn]['dischargeStart1'], '%H:%M')
                end_time = datetime.strptime(schedule[sn]['dischargeEnd1'], '%H:%M')
                end_time_str = schedule[sn]['dischargeEnd1']
                if load_now >= threshold_lower_bound:
                    break
                if not start_time <= current_time <= end_time:
                    logging.info(f'--Device: {sn} is not in discharging period, skip.')
                    continue
                decreased_power = schedule.get(
                    sn, {}).get('dischargePower1', 0)
                decreased_power = 0 if 0.5*decreased_power < 200 else 0.5*decreased_power 
                difference = schedule[sn]['dischargePower1'] - decreased_power
                logging.info(f'--Decreased discharge power for Device: {sn} by {difference}W. from: {schedule[sn]["dischargePower1"]}, to: {decreased_power}')
                schedule[sn]['dischargePower1'] = decreased_power
                load_now += difference
        
        return schedule        

    def generate_schedule(self, consumption, price, batterie_capacity_kwh, num_batteries, stats, price_weight=1):
        import optuna
        from scipy.ndimage import gaussian_filter1d

        def greedy_battery_discharge(consumption, price, batteries, price_weight=1):
            num_hours = len(consumption)
            net_consumption = consumption.copy()
            battery_discharges = [[0] * num_hours for _ in batteries]

            for _, (capacity, duration) in enumerate(batteries):
                best_avg = float('-inf')
                best_start = 0

                # Find the window with the highest rolling average
                for start in range(0, num_hours - duration + 1):
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
                    discharged = min(discharge_rate, net_consumption[h])
                    battery_discharges[_][h] = discharged
                    net_consumption[h] -= discharged

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

                for start in range(0, num_hours - duration + 1):
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
        _, battery_charges = greedy_battery_charge_with_mask(
            consumption, price, self._get_solar(), charging_needs, masks)

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

            def manage_tasks(self):
                for task_type, task_time, task_duration in self.tasks:
                    # logging.info(
                    # f'try to allocate a battery for: (task_type: {task_type}, task_time: {task_time}, task_duration: {task_duration}))')
                    if not self.allocate_battery(task_time, task_type, task_duration):
                        logging.info(
                            f'task {task_time} {task_type} {task_duration} failed to allocate battery')

                def unit_to_time(unit, sample_interval):
                    total_minutes = int(unit * sample_interval)
                    hour = total_minutes // 60
                    minute = total_minutes % 60
                    return "{:02d}:{:02d}".format(hour, minute)

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
                            'operatingMode': mode_map['Time'],
                            'dischargeStart1': start_time if task_type == 'Discharge' else "00:00",
                            'dischargeEnd1': end_time if task_type == 'Discharge' else "00:00",
                            'dischargePower1': power,
                        }
                        schedules[sn] = data | schedules.get(sn, {})
                    # todo: Now using fixed time schedule, change back to start_time and end_time later.
                    elif task_type == 'Charge':
                        data = {
                            'deviceSn': sn,
                            'operatingMode': mode_map['Time'],
                            'chargeStart1': start_time if task_type == 'Charge' else "00:00",
                            'chargeEnd1': end_time if task_type == 'Charge' else "00:00",
                            'chargePower1': power,
                            # 'chargeStart1': "09:00",
                            # 'chargeEnd1':  "15:00",
                            # 'chargePower1': "800",
                        }
                        schedules[sn] = data | schedules.get(sn, {})
                return schedules

        tasks = all_schedules

        sample_interval = 24*60/len(consumption)
        battery_manager = ProjectBatteryManager(
            tasks, stats, self.battery_max_capacity_kwh, sample_interval)
        json_schedule = battery_manager.manage_tasks()

        # self.plot(consumption, price, battery_discharges, net_consumption)
        return json_schedule

    def _get_command_from_schedule(self, current_time):
        return 'Idle'

    def step(self):
        current_date = datetime.now(tz=pytz.timezone('Australia/Sydney')).day
        # Check if it's a new day or there's no existing schedule
        if self.last_scheduled_date != current_date:
            demand, price = pickle.load(open('demand_price.pkl', 'rb'))
            # demand, price = self._get_demand_and_price()
            stats = self._get_battery_status()
            self.schedule = self.generate_schedule(
                demand, price, self.battery_max_capacity_kwh, self.num_batteries, stats, self.price_weight)
            self.last_scheduled_date = current_date

        return self.schedule

    def required_data(self):
        return []


def test_func():
    schedule_list = {'sn1':
                     [({'chargeStart1': '00:05', 'chargeEnd1': '05:50', 'chargePower1': 869}, '08:00'),
                         ({'chargeStart2': '08:30', 'chargeEnd2': '10:00',
                          'chargePower2': 200}, '09:00'),
                      ({'chargeStart2': '10:30', 'chargeEnd2': '15:00',
                       'chargePower2': 700}, '10:00'),
                      ({'dischargeStart2': '15:30', 'dischargeEnd2': '19:00',
                          'dischargePower2': 700}, '10:00')],
                     'sn2':
                     [({'chargeStart1': '00:05', 'chargeEnd1': '05:50', 'chargePower1': 869}, '08:00'),
                         ({'chargeStart2': '08:30', 'chargeEnd2': '10:00',
                          'chargePower2': 200}, '09:00'),
                         ({'chargeStart2': '10:30', 'chargeEnd2': '15:00',
                          'chargePower2': 700}, '10:00')]}

    schedule_analyser = ScheduleAnalyser(schedule_list)
    print(schedule_analyser.plot_battery_power())


if __name__ == '__main__':
    scheduler = SimulationScheduler(
        scheduler_type='AIScheduler', battery_sn=['RX2505ACA10J0A180011', 'RX2505ACA10J0A170035', 'RX2505ACA10J0A170033', 'RX2505ACA10J0A160007', 'RX2505ACA10J0A180010'], test_mode=False, api_version='redx', pv_sn='RX2505ACA10J0A170033')

    scheduler.start()
    # test_func()
