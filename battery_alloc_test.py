# The class for auto schedule of charging/discharging the battery
import sched
import datetime
import time
import numpy as np
import util


class BatteryScheduler:

    def __init__(self, scheduler_type='PeakValley', battery_sn=None, test_mode=False):
        self.s = sched.scheduler(time.time, time.sleep)
        self.scheduler = None
        self.monitor = util.PriceAndLoadMonitor(test_mode=test_mode)
        self.test_mode = test_mode
        self.event = None
        self.is_runing = False
        self.sn_list = battery_sn if type(battery_sn) == list else [
            battery_sn]
        self._set_scheduler(scheduler_type)

    def _set_scheduler(self, scheduler_type):
        if scheduler_type == 'PeakValley':
            self.scheduler = PeakValleyScheduler()
        elif scheduler_type == 'AIScheduler':
            self.scheduler = AIScheduler(sn_list=self.sn_list)
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
        if not self.is_runing:
            return

        if type(self.scheduler) == AIScheduler:
            schedule = self._get_battery_command()
            for sn in self.sn_list:
                json = schedule[sn]
                self.send_battery_command(json=json, sn=sn)
        elif type(self.scheduler) == PeakValleyScheduler:
            for sn in self.sn_list:
                _current_price = self.get_current_price()
                _bat_stats = self.get_current_battery_stats(sn)
                _current_usage = _bat_stats['loadP']
                _current_soc = _bat_stats['soc']/100.0
                _current_time = self.get_current_time()
                _command = self._get_battery_command(
                    current_price=_current_price, current_usage=_current_usage, current_time=_current_time, current_soc=_current_soc)
                self.send_battery_command(command=_command, sn=sn)
        print(
            f"Current price: {_current_price}, current usage: {_current_usage}, current time: {_current_time}, current soc: {_current_soc}, command: {_command}")
        if self.test_mode:
            interval = 0.1
        self.event = self.s.enter(interval, 1, self._start)

    def start(self):
        self.is_runing = True
        try:
            self._start()
            self.s.run()
        except KeyboardInterrupt:
            print("Stopped.")

    def stop(self):
        self.is_runing = False
        print("Stopped.")

    def get_current_price(self):
        return self.monitor.get_sim_price(self.get_current_time())

    def get_current_time(self):
        return self.monitor.get_current_time()

    def get_current_battery_stats(self, sn):
        return self.monitor.get_realtime_battery_stats(sn)

    def update_battery_state(self, **kwargs):
        # Placeholder for potential updates to the scheduler's internal state
        pass

    def send_battery_command(self, command=None, json=None, sn=None):
        self.monitor.send_battery_command(command, json, sn)


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
        self.BuyPct = 50
        self.SellPct = 75
        self.LookBackBars = 1 * 48
        self.ChgStart1 = '8:00'
        self.ChgEnd1 = '16:00'
        self.DisChgStart2 = '16:05'
        self.DisChgEnd2 = '23:55'

        # Initial data containers and setup
        self.price_history = [17.12,
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
        self.soc = self.BatSocMin
        self.bat_cap = self.soc * self.BatCap

        # Convert start and end times to datetime.time
        self.t_chg_start1 = datetime.datetime.strptime(
            self.ChgStart1, '%H:%M').time()
        self.t_chg_end1 = datetime.datetime.strptime(
            self.ChgEnd1, '%H:%M').time()
        self.t_dis_start2 = datetime.datetime.strptime(
            self.DisChgStart2, '%H:%M').time()
        self.t_dis_end2 = datetime.datetime.strptime(
            self.DisChgEnd2, '%H:%M').time()

    def step(self, current_price, current_time, current_usage, current_soc):
        # Update battery state
        self.bat_cap = current_soc * self.BatCap

        self.price_history.append(current_price)
        if len(self.price_history) > self.LookBackBars:
            self.price_history.pop(0)

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            self.price_history, [self.BuyPct, self.SellPct])

        current_timenum = datetime.datetime.strptime(
            current_time, '%H:%M').time()

        command = ['Idle', 0]  # No action by default

        if self._is_charging_period(current_timenum) and current_price <= buy_price:
            # Charging logic
            chg_delta = self.BatChgMax * self.HrMin
            temp_chg = chg_delta + self.bat_cap

            self.bat_cap = min(temp_chg, self.BatMaxCapacity)
            command = ['Charge', self.BatChgMax]

        elif self._is_discharging_period(current_timenum) and (current_price >= sell_price or current_price > self.SpikeLevel):
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
        return (t >= self.t_dis_start2 and t <= self.t_dis_end2)

    def required_data(self):
        return ['current_price', 'current_time', 'current_usage']


class AIScheduler(BaseScheduler):

    def __init__(self, sn_list):
        self.battery_capacity_kwh = 5
        self.num_batteries = 5
        self.price_weight = 1
        self.min_discharge_rate_kw = 0.5
        self.max_discharge_rate_kw = 2.5
        self.api = util.ApiCommunicator(
            'https://da2e586eae72a40e5bde4ead0fe77b2f0.clg07azjl.paperspacegradient.com/')
        self.battery_sn_list = sn_list
        self.battery_monitors = {sn: util.PriceAndLoadMonitor(
            test_mode=False) for sn in sn_list}
        self.schedule = None

    def _get_demand_and_price(self):
        # we don't need to get each battery's demand, just use the get_project_demand() method to get the total demand instead.
        # take the first battery monitor from the list
        demand = self.battery_monitors[self.battery_sn_list[0]].get_project_demand(
        )
        price = [1 for _ in range(len(demand))]
        return np.array(demand, dtype=np.float64), np.array(price, dtype=np.float64)

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

                # Step 1: Find the window with the highest rolling average
                for start in range(0, num_hours - duration + 1):
                    weight_adjusted_array = [i*j*price_weight for i, j in zip(
                        net_consumption[start:start+duration], price[start:start+duration])]
                    avg = sum(weight_adjusted_array) / duration
                    if avg > best_avg:
                        best_avg = avg
                        best_start = start

                # Step 3: Place the battery's discharge
                discharge_rate = capacity / duration
                for h in range(best_start, best_start + duration):
                    discharged = min(discharge_rate, net_consumption[h])
                    battery_discharges[_][h] = discharged
                    net_consumption[h] -= discharged

            return net_consumption, battery_discharges

        def greedy_battery_charge(consumption, price, batteries, price_weight=1):
            num_hours = len(consumption)
            net_consumption = consumption.copy()
            battery_charges = [[0] * num_hours for _ in batteries]

            for _, (capacity, duration) in enumerate(batteries):
                best_avg = float('inf')
                best_start = 0

                charge_rate = capacity / duration
                # Step 1: Find the window with the lowest rolling average
                for start in range(0, num_hours - duration + 1):
                    weight_adjusted_array = [
                        charge_rate*i for i in price[start:start+duration]]
                    avg = sum(weight_adjusted_array) / duration
                    if avg < best_avg:
                        best_avg = avg
                        best_start = start

                # Step 3: Place the battery's charge
                for h in range(best_start, best_start + duration):
                    charged = min(charge_rate, net_consumption[h])
                    battery_charges[_][h] = charged
                    net_consumption[h] += charged

            return net_consumption, battery_charges

        def objective(trial):
            max_discharge_length = self.battery_capacity_kwh / \
                self.min_discharge_rate_kw * len(consumption) / 24
            min_discharge_length = self.battery_capacity_kwh / \
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

        consumption = smooth_demand(consumption, sigma=6)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_durations = [
            study.best_params[f"duration_{i}"] for i in range(num_batteries)]
        discharging_capacity = [[batterie_capacity_kwh, duration]
                                for duration in best_durations]
        # charging_needs = [[30000, 24] for x in range(num_batteries)]

        net_consumption, battery_discharges = greedy_battery_discharge(
            consumption, price, discharging_capacity)
        # _, battery_charges = greedy_battery_charge(consumption,price, charging_needs)

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

        for i, b in enumerate(battery_discharges):
            print(f'Battery {i+1} discharging window:',
                  _get_charging_window(b))

        def split_single_schedule(schedule, num_windows):
            start = schedule[0]
            duration = schedule[1]
            segments = [(round(i*duration/num_windows+start), round((i+1)*duration/num_windows+start) -
                        round(i*duration/num_windows+start)) for i in range(num_windows)]
            return segments

        def split_schedules(schedules, num_windows, task_type='discharge'):
            ret = []
            for schedule in schedules:
                ret.extend(split_single_schedule(schedule, num_windows))
            ret = [(task_type, i[0], i[1]) for i in ret]
            return ret

        discharge_schedules = [_get_charging_window(
            i) for i in battery_discharges]
        flat_discharge_schedules = [
            item for sublist in discharge_schedules for item in sublist]
        # charge_schedules = [_get_charging_window(i) for i in battery_charges]

        discharge_sched_split = split_schedules(
            flat_discharge_schedules, 1, 'discharge')
        # charge_sched_split = split_schedules(flat_charge_schedules, 1, 'charge')

        # Use hardcoded schedules for charging windows.
        charge_sched_split = [('charge', 75, 24),
                              ('charge', 75, 24),
                              ('charge', 75, 24),
                              ('charge', 75, 24),
                              ('charge', 75, 24)]
        all_schedules = sorted(discharge_sched_split +
                               charge_sched_split, key=lambda x: x[1])

        class Battery:
            def __init__(self, capacity, battery_id):
                self.id = battery_id
                self.capacity = 30000
                self.current_charge = capacity  # Set initial charge to full capacity
                self.fully_charged_rounds = 0
                self.fully_discharged_rounds = 0
                self.available_at = 0  # The time when the battery is available again

            def is_depleted(self):
                return self.fully_discharged_rounds >= 2 and self.fully_charged_rounds >= 2

            def can_discharge(self, current_time):
                return self.current_charge > 0 and not self.is_depleted() and self.available_at <= current_time

            def can_charge(self, current_time):
                return self.current_charge < self.capacity and not self.is_depleted() and self.available_at <= current_time

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

        class BatteryManager:
            def __init__(self, tasks, battery_init_status, battery_capacity_kwh=5, time_unit=5):
                self.tasks = sorted(tasks, key=lambda x: x[1])
                self.batteries = [
                    Battery(power_level, i) for i, power_level in enumerate(battery_init_status)]
                self.output = []
                self.battery_capacity_kwh = battery_capacity_kwh

            def allocate_battery(self, task_time, task_type, task_duration):
                for battery in self.batteries:
                    if task_type == "discharge" and battery.can_discharge(task_time):
                        battery.discharge(task_duration, task_time)
                        self.output.append(
                            (battery.id, task_time, task_type, task_duration))
                        return True
                    elif task_type == "charge" and battery.can_charge(task_time):
                        battery.charge(task_duration, task_time)
                        self.output.append(
                            (battery.id, task_time, task_type, task_duration))
                        return True
                return False

            def manage_tasks(self):
                for task_type, task_time, task_duration in self.tasks:
                    print(
                        f'try to allocate a battery for: (task_type: {task_type}, task_time: {task_time}, task_duration: {task_duration}))')
                    if not self.allocate_battery(task_time, task_type, task_duration):
                        print(
                            f'task {task_time} {task_type} {task_duration} failed to allocate battery')
                def unit_to_time(unit):
                    total_minutes = unit * 5
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

                schedules = []
                for sn, task_time, task_type, task_duration in self.output:
                    power = batterie_capacity_kwh * 1000 * \
                        60 / (time_unit * task_duration)
                    start_time = unit_to_time(task_time)
                    end_time = unit_to_time(task_time + task_duration)

                    data = {
                        'deviceSn': sn,
                        'controlCommand': command_map[task_type],
                        'operatingMode': mode_map['Time'],
                        'dischargeStart1': start_time if task_type == 'discharge' else "00:00",
                        'dischargeEnd1': end_time if task_type == 'discharge' else "00:00",
                        'chargeStart1': start_time if task_type == 'charge' else "00:00",
                        'chargeEnd1': end_time if task_type == 'charge' else "00:00",
                        'antiBackflowSW': 1,
                        'dischargePower1': power,
                        'dischargeSOC1': 10,
                        'dischargePowerLimit1': 0,
                        'chargePower1': power,
                        'enableGridCharge1': 1
                    }
                    schedules.append(data)

                return schedules


        tasks = all_schedules

        time_unit = 24*60/len(consumption)
        battery_init_status = [np.random.randint(
            1000, 30000) for _ in range(15)]
        battery_manager = BatteryManager(tasks, battery_init_status)
        json_schedule = battery_manager.manage_tasks()
        return json_schedule 

    def _get_command_from_schedule(self, current_time):
        return 'Idle'

    def step(self):
        if self.schedule is None:
            demand, price = self._get_demand_and_price()
            stats = self._get_battery_status()
            self.schedule = self.generate_schedule(
                demand, price, self.battery_capacity_kwh, self.num_batteries, stats, self.price_weight)
        else:
            return self.schedule 

    def required_data(self):
        return []


if __name__ == '__main__':
    scheduler = BatteryScheduler(
        scheduler_type='AIScheduler', battery_sn='RX2505ACA10JOA160037', test_mode=True)
    scheduler.start()
