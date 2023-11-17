# The class for auto schedule of charging/discharging the battery
import sched
from datetime import datetime, timedelta
import time
import numpy as np
import util
import pytz
import logging
from solar_prediction import WeatherInfoFetcher

logging.basicConfig(level=logging.INFO)


class BatteryScheduler:

    def __init__(self, scheduler_type='PeakValley', battery_sn=None, test_mode=False, api_version='dev3', pv_sn=None):
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
        self.last_five_metre_readings = []
        self._set_scheduler(scheduler_type, api_version, pv_sn=pv_sn)

    def _set_scheduler(self, scheduler_type, api_version, pv_sn=None):
        if scheduler_type == 'PeakValley':
            self.scheduler = PeakValleyScheduler()
            self.scheduler.init_price_history(self.monitor.get_price_history())
        elif scheduler_type == 'AIScheduler':
            self.scheduler = AIScheduler(
                sn_list=self.sn_list, api_version=api_version, pv_sn=pv_sn)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _get_battery_command(self, **kwargs):
        if not self.scheduler:
            raise ValueError("Scheduler not set. Use set_scheduler() first.")

        required_data = self.scheduler.required_data()
        if not all(key in kwargs for key in required_data):
            raise ValueError(
                f"Data missing for {self.scheduler.__class__.__name__}. Required: {required_data}")

        return self.scheduler.step(**kwargs)

    def _start(self, interval=1800):
        if not self.is_running:
            return

        try:
            if isinstance(self.scheduler, AIScheduler):
                interval = 360
                self._process_ai_scheduler()
            elif isinstance(self.scheduler, PeakValleyScheduler):
                self._process_peak_valley_scheduler()
            if self.test_mode:
                interval = 0.1
            self.event = self.s.enter(interval, 1, self._start)
        except Exception as e:
            logging.error(f"An error occurred in _start: {e}")
            self.event = self.s.enter(interval, 1, self._start)

    def _process_ai_scheduler(self):
        current_day = datetime.now(tz=pytz.timezone('Australia/Sydney')).day
        if self.last_scheduled_date != current_day:
            self.last_schedule = self._get_battery_command()
            self.last_scheduled_date = current_day

        schedule = self.daytime_hotfix_discharging(self.last_schedule)
        schedule = self.daytime_hotfix_charging(schedule)
        logging.info(f"Schedule: {schedule}")

        for sn in self.sn_list:
            battery_schedule = schedule.get(sn, None)
            if not battery_schedule:
                continue
            try:
                self.send_battery_command(json=battery_schedule, sn=sn)
                current_time = self.get_current_time()
                logging.info(
                    f'Schedule sent to battery: {sn} at {current_time}')
            except Exception as e:
                logging.error(f"Error sending battery command: {e}")
                continue

        self.last_schedule = schedule

    def _process_peak_valley_scheduler(self):
        for sn in self.sn_list:
            current_price = self.get_current_price()
            bat_stats = self.get_current_battery_stats(sn)
            current_usage = bat_stats['loadP']
            current_soc = bat_stats['soc'] / 100.0
            current_time = self.get_current_time(
                time_zone='Australia/Brisbane')
            current_pv = bat_stats['ppv']
            command = self._get_battery_command(
                current_price=current_price, current_usage=current_usage,
                current_time=current_time, current_soc=current_soc, current_pv=current_pv)
            self.send_battery_command(command=command, sn=sn)
        logging.info(f"Current price: {current_price}, current usage: {current_usage}, "
                     f"current time: {current_time}, current soc: {current_soc}, command: {command}")

    def start(self):
        self.is_running = True
        try:
            self._start()
            self.s.run()
        except KeyboardInterrupt:
            logging.info("Stopped.")
        except Exception as e:
            logging.error(f"Error in start: {e}")

    def stop(self):
        self.is_runing = False
        logging.info("Stopped.")

    def daytime_hotfix_charging(self, schedule):
        try:
            load = self.get_project_status()
        except Exception as e:
            logging.error(
                f"Error getting project status or battery stats: {e}")
            return schedule

        surplus_power = max(0, 0 - load-500)
        max_charging_power = 1500
        now = self.get_current_time()
        if datetime.strptime(now, '%H:%M') > datetime.strptime('15:00', '%H:%M'):
            return schedule
        if surplus_power <= 1000:
            return schedule

        self.schedule_before_hotfix = schedule.copy()
        for sn in self.sn_list:
            if surplus_power <= 0:
                break
            current_charging_power = schedule.get(
                sn, {}).get('chargePower1', 0)
            adjusted_charging_power = min(
                max_charging_power, current_charging_power + surplus_power)
            surplus_power -= (adjusted_charging_power - current_charging_power)
            surplus_power = max(0, surplus_power)  # Avoid negative surplus
            now_str = self.get_current_time()
            end_10mins_later_str = (datetime.strptime(
                now_str, '%H:%M') + timedelta(minutes=10)).strftime('%H:%M')
            discharge_start = schedule.get(
                sn, {}).get('dischargeStart1', '00:00')
            end_10mins_later = datetime.strptime(end_10mins_later_str, '%H:%M')
            if end_10mins_later > datetime.strptime(discharge_start, '%H:%M'):
                continue
            if sn in schedule:
                schedule[sn]['chargePower1'] = adjusted_charging_power
                schedule[sn]['chargeStart1'] = now_str
                if schedule[sn]['chargeEnd1'] < end_10mins_later_str:
                    schedule[sn]['chargeEnd1'] = end_10mins_later_str
                logging.info(
                    f'Increased charging power for Device: {sn} by {adjusted_charging_power - current_charging_power}W due to excess solar power.')
        return schedule

    def daytime_hotfix_discharging(self, schedule: dict) -> dict:
        threshold = 1000
        try:
            load = self.get_project_status()
        except Exception as e:
            logging.error(f"Error getting project status: {e}")
            load = 2000  # Fallback load value

        now = self.get_current_time()
        if datetime.strptime(now, '%H:%M') < datetime.strptime('15:00', '%H:%M'):
            return schedule
        if load >= threshold:
            return schedule

        load_now = load
        for sn in self.sn_list:
            if load_now >= threshold:
                break
            current_time = self.get_current_time()
            start_time = schedule[sn]['dischargeStart1']
            end_time = schedule[sn]['dischargeEnd1']
            discharge_power = schedule[sn]['dischargePower1']
            try:
                current_time = datetime.strptime(current_time, '%H:%M')
                start_time = datetime.strptime(start_time, '%H:%M')
                end_time = datetime.strptime(end_time, '%H:%M')
            except ValueError as e:
                logging.error(f"Time conversion error for Device: {sn}: {e}")
                continue
            if not (start_time <= current_time <= end_time):
                continue
            adjusted_start_time = current_time + timedelta(minutes=7)
            adjusted_end_time = adjusted_start_time + timedelta(minutes=7)
            if adjusted_end_time > datetime(hour=23, minute=55):
                continue
            schedule[sn]['dischargeStart1'] = adjusted_start_time.strftime(
                '%H:%M')
            load_now += discharge_power
            logging.info(
                f'Delayed dischargeStart for Device: {sn} by 3 mins due to low load. Load now: {load_now}')

        return schedule

    def get_current_price(self):
        return self.monitor.get_realtime_price()

    def get_current_time(self, time_zone='Australia/Sydney') -> str:
        return self.monitor.get_current_time(time_zone)

    def get_project_status(self, project_id: int = 1, phase: int = 2) -> float:
        if len(self.last_five_metre_readings) >= 2:
            self.last_five_metre_readings.pop(0)
        try:
            new_value = self.monitor.get_project_stats(project_id, phase)
            self.last_five_metre_readings.append(new_value)
            return sum(self.last_five_metre_readings) / len(self.last_five_metre_readings)
        except AttributeError:
            return 0

    def get_current_battery_stats(self, sn):
        return self.monitor.get_realtime_battery_stats(sn)

    def update_battery_state(self, **kwargs):
        # Placeholder for potential updates to the scheduler's internal state
        pass

    def send_battery_command(self, command=None, json=None, sn=None):
        self.monitor.send_battery_command(command=command, json=json, sn=sn)


class BaseScheduler:

    def step(self, **kwargs):
        """
        Execute one scheduling step and return a battery command.
        Override this method in the derived class.
        """
        raise NotImplementedError

    def required_data(self):
        """
        Return a list of data fields required by the scheduler.
        Override this method in the derived class.
        """
        raise NotImplementedError


class PeakValleyScheduler(BaseScheduler):
    def __init__(self, batnum=1):
        # Constants and Initializations
        self.BatNum = batnum
        self.BatMaxCapacity = 5
        self.BatCap = self.BatNum * self.BatMaxCapacity
        self.BatChgMax = self.BatNum * 1.5
        self.BatDisMax = self.BatNum * 2.5
        self.BatSocMin = 0.1
        self.HrMin = 30 / 60
        self.SellDiscount = 0.12
        self.SpikeLevel = 300
        self.SolarCharge = 0
        self.SellBack = 0
        self.BuyPct = 30
        self.SellPct = 80
        self.LookBackBars = 2 * 48
        self.ChgStart1 = '5:00'
        self.ChgEnd1 = '16:00'
        self.DisChgStart2 = '16:05'
        self.DisChgEnd2 = '23:55'
        self.DisChgStart1 = '0:00'
        self.DisChgEnd1 = '5:00'

        self.date = None

        # Initial data containers and setup
        self.price_history = None
        self.solar = None
        self.soc = self.BatSocMin
        self.bat_cap = self.soc * self.BatCap

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

    def init_price_history(self, price_history):
        self.price_history = price_history

    def _get_solar(self, interval=0.5, test_mode=False):
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
                    1-(1.4*rain_info['clouds']+0.6*rain_info['rain'])/(2*100))*5000
                logging.info(
                    f'Weather forecast: rain: {rain_info["rain"]}, clouds: {rain_info["clouds"]}, max_solar: {max_solar}')
            except Exception as e:
                logging.error(e)
                max_solar = 5000

        _, gaus_y = gaussian_mixture(interval)
        return gaus_y*max_solar

    def step(self, current_price, current_time, current_usage, current_soc, current_pv):
        # Update solar data each day
        if self.date != datetime.now(tz=pytz.timezone('Australia/Brisbane')).day or self.solar is None:
            self.solar = self._get_solar()
            self.date = datetime.now(
                tz=pytz.timezone('Australia/Brisbane')).day

        # Update battery state
        self.bat_cap = current_soc * self.BatCap

        self.price_history.append(current_price)
        if len(self.price_history) > self.LookBackBars:
            self.price_history.pop(0)

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            self.price_history, [self.BuyPct, self.SellPct])

        current_timenum = datetime.strptime(
            current_time, '%H:%M').time()

        command = ['Idle', 0]  # No action by default

        if self._is_charging_period(current_timenum) and (current_price <= buy_price or current_pv > current_usage):
            # Charging logic
            chg_delta = self.BatChgMax * self.HrMin
            temp_chg = chg_delta + self.bat_cap

            self.bat_cap = min(temp_chg, self.BatMaxCapacity)
            command = ['Charge', self.BatChgMax]

        elif self._is_discharging_period(current_timenum) and (current_price >= sell_price or current_price > self.SpikeLevel) and current_pv < current_usage:
            # Discharging logic
            dischg_delta = self.BatDisMax * \
                self.HrMin if self.SellBack else min(
                    current_usage, self.BatDisMax) * self.HrMin
            temp_dischg = self.bat_cap - dischg_delta

            if temp_dischg >= self.BatCap * self.BatSocMin:
                self.bat_cap = temp_dischg  # discharge battery
                _value = -self.BatDisMax if self.SellBack else - \
                    min(current_usage, self.BatDisMax)
                command = ['Discharge', _value]

        return command[0]

    def _is_charging_period(self, t):
        return t >= self.t_chg_start1 and t <= self.t_chg_end1

    def _is_discharging_period(self, t):
        return (t >= self.t_dis_start2 and t <= self.t_dis_end2) or (t >= self.t_dis_start1 and t <= self.t_dis_end1)

    def required_data(self):
        return ['current_price', 'current_time', 'current_usage', 'current_soc', 'current_pv']


class AIScheduler(BaseScheduler):

    def __init__(self, sn_list, pv_sn, api_version='redx'):
        self.battery_max_capacity_kwh = 5
        self.num_batteries = len(sn_list)
        self.price_weight = 1
        self.min_discharge_rate_kw = 0.5
        self.max_discharge_rate_kw = 2.5
        self.api = util.ApiCommunicator(
            'https://da2e586eae72a40e5bde4ead0fe77b2f0.clg07azjl.paperspacegradient.com/')
        self.battery_sn_list = sn_list
        self.pv_sn = pv_sn
        self.battery_monitors = {sn: util.PriceAndLoadMonitor(
            test_mode=False, api_version=api_version) for sn in sn_list}
        self.schedule = None
        self.last_scheduled_date = None

    def _get_demand_and_price(self):
        # we don't need to get each battery's demand, just use the get_project_demand() method to get the total demand instead.
        # take the first battery monitor from the list
        demand = self.battery_monitors[self.battery_sn_list[0]].get_project_demand(
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

    def _get_solar(self, interval=5/60, test_mode=False):
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
                    1-sigma*(1.4*rain_info['clouds']+0.6*rain_info['rain'])/(2*100))*5000
                logging.info(
                    f'Weather forecast: rain: {rain_info["rain"]}, clouds: {rain_info["clouds"]}, max_solar: {max_solar}')
            except Exception as e:
                logging.error(e)
                max_solar = 5000

        _, gaus_y = gaussian_mixture(interval)
        return gaus_y*max_solar

    def _get_battery_status(self):
        battery_status = {}
        for sn in self.battery_sn_list:
            battery_status[sn] = self.battery_monitors[sn].get_realtime_battery_stats(
                sn)

        return battery_status

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

    def plot(self, consumption, price, battery_discharges, net_consumption):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Seaborn settings
        sns.set(style="whitegrid")
        sns.set_palette("tab10")

        def time_string(minute_count):
            """ Convert total minutes to a formatted time string like 14:35 """
            hour = minute_count // 60
            minute = minute_count % 60
            return f"{hour:02d}:{minute:02d}"

        def time_to_minutes(t):
            hours, minutes = map(int, t.split(':'))
            return hours * 60 + minutes

        def plot_greedy_solution(hours, consumption, battery_discharges, net_consumption, weight_adjusted_array):
            plt.figure(figsize=(15, 6))

            # Generate the times list for the x-axis
            times = [time_string(i * 5) for i in range(288)]

            # Plot original consumption
            sns.lineplot(x=times, y=consumption, marker='o',
                         label='Original Consumption')

            # Plot net consumption after battery discharge
            sns.lineplot(x=times, y=net_consumption,
                         linestyle='--', label='Net Consumption')

            # Plot the weighted consumption
            sns.lineplot(x=times, y=weight_adjusted_array,
                         linestyle='dashdot', label='Weighted Consumption')

            # Plot the constant discharge rates for each battery as rectangles
            palette = sns.color_palette(n_colors=15)
            legend_added = set()

            # Maintain a height tracker for each hour to determine the starting height for each battery's rectangle
            height_tracker = [0] * len(times)

            for i, discharge in enumerate(battery_discharges):
                label = f'Battery {i+1} Discharge'
                for j, rate in enumerate(discharge):
                    if rate > 0:  # If the battery is discharging at this hour
                        # Base the start of discharge at net consumption + height of previous batteries
                        start = net_consumption[j] + height_tracker[j]
                        end = start + rate
                        plt.fill_between([times[j], times[j+1]], [start, start], [end, end], color=palette[i],
                                         alpha=0.5, label=label if i not in legend_added else "")
                        legend_added.add(i)

                        # Update the height tracker for the next battery
                        height_tracker[j] += rate

            # Define a soft color palette for the periods
            soft_yellow = "#FFFACD"  # lemon chiffon
            soft_red = "#FFC1C1"  # rosy brown

            shoulder_period = [('9:00', '16:59'), ('20:00', '21:59')]
            peak_period = [('7:00', '8:59'), ('17:00', '19:59')]

            # Add the shaded regions with dotted frame for shoulder period
            for i, period in enumerate(shoulder_period):
                start = times[time_to_minutes(period[0]) // 5]
                end = times[time_to_minutes(period[1]) // 5]
                plt.axvspan(start, end, color=soft_yellow, alpha=0.6, edgecolor='yellow',
                            linestyle='dotted', linewidth=1.5, label='Shoulder Period' if i == 0 else "")

            # Add the shaded regions with dotted frame for peak period
            for i, period in enumerate(peak_period):
                start = times[time_to_minutes(period[0]) // 5]
                end = times[time_to_minutes(period[1]) // 5]
                plt.axvspan(start, end, color=soft_red, alpha=0.6, edgecolor='red',
                            linestyle='dotted', linewidth=1.5, label='Peak Period' if i == 0 else "")

            plt.xlabel('Time')
            plt.ylabel('Consumption/Discharge Rate')
            plt.title('Energy Consumption and Battery Discharge')
            tick_spacing = 12
            plt.xticks([times[i]
                       for i in range(0, len(times), tick_spacing)], rotation=45)

            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()

        def plot_charge_windows(hours, consumption, battery_charges, net_consumption):
            net_consumption = np.array(net_consumption)
            net_consumption += np.sum(battery_charges, axis=0)
            plt.figure(figsize=(15, 6))

            sns.lineplot(x=hours, y=consumption, marker='o',
                         label='Original Consumption')
            sns.lineplot(x=hours, y=net_consumption, linestyle='--',
                         label='Net Consumption after Charge')
            palette = sns.color_palette(n_colors=15)
            legend_added = set()
            height_tracker = [0] * len(hours)

            for i, charge in enumerate(battery_charges):
                label = f'Battery {i+1} Charge'
                for hour, rate in enumerate(charge):
                    if rate > 0:
                        start = consumption[hour] + height_tracker[hour]
                        end = start + rate
                        plt.fill_between([hour, hour+1], [start, start], [end, end], color=palette[i],
                                         alpha=0.5, label=label if i not in legend_added else "")
                        legend_added.add(i)
                        height_tracker[hour] += rate

            plt.xlabel('Hour')
            plt.ylabel('Consumption/Charge Rate')
            plt.title('Energy Consumption and Battery Charge')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()
        weight_adjusted_array = [i*j*self.price_weight for i, j in zip(
            consumption, price)]
        hours = list(range(len(consumption)))

        plot_greedy_solution(hours, consumption,
                             battery_discharges, net_consumption, weight_adjusted_array)
        # plot_charge_windows(hours, consumption, battery_charges, net_consumption)

    def step(self):
        current_date = datetime.now(tz=pytz.timezone('Australia/Sydney'))
        # Check if it's a new day or there's no existing schedule
        if self.last_scheduled_date is None or current_date > self.last_scheduled_date:
            demand, price = self._get_demand_and_price()
            stats = self._get_battery_status()
            self.schedule = self.generate_schedule(
                demand, price, self.battery_max_capacity_kwh, self.num_batteries, stats, self.price_weight)
            self.last_scheduled_date = current_date

        return self.schedule

    def required_data(self):
        return []


if __name__ == '__main__':
    scheduler = BatteryScheduler(
        scheduler_type='AIScheduler', battery_sn=['RX2505ACA10J0A180011', 'RX2505ACA10J0A170035', 'RX2505ACA10J0A170033', 'RX2505ACA10J0A160007', 'RX2505ACA10J0A180010'], test_mode=False, api_version='redx', pv_sn='RX2505ACA10J0A170033')
    # scheduler = BatteryScheduler(scheduler_type='PeakValley', battery_sn=['RX2505ACA10JOA160037'], test_mode=False, api_version='dev3')
    scheduler.start()
