# The class for auto schedule of charging/discharging the battery
import requests
import pandas as pd
import datetime
import numpy as np
import json
import sched
import time
import util


class BatteryScheduler:

    def __init__(self, scheduler_type='PeakValley', battery_sn=None, test_mode=False):
        self.s = sched.scheduler(time.time, time.sleep)
        self.scheduler = None
        self._set_scheduler(scheduler_type)
        self.monitor = util.PriceAndLoadMonitor(
            sn=battery_sn, test_mode=test_mode)
        self.test_mode = test_mode
        self.event = None

    def _set_scheduler(self, scheduler_type):
        if scheduler_type == 'PeakValley':
            self.scheduler = PeakValleyScheduler()
        elif scheduler_type == 'AIScheduler':
            self.scheduler = AIScheduler()
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

    def start(self, interval=300):
        _current_price = self.get_current_price()
        _bat_stats = self.get_current_battery_stats()
        _current_usage = _bat_stats['loadP']
        _current_soc = _bat_stats['soc']/100.0
        _current_time = self.get_current_time()
        _command = self._get_battery_command(
            current_price=_current_price, current_usage=_current_usage, current_time=_current_time, current_soc=_current_soc)
        print(
            f"Current price: {_current_price}, current usage: {_current_usage}, current time: {_current_time}, current soc: {_current_soc}, command: {_command}")
        self.send_battery_command(_command)
        if self.test_mode:
            interval = 0.1
        self.event = self.s.enter(interval, 1, self.start)
        try:
            self.s.run(blocking=True)
        except KeyboardInterrupt:
            print("Stopped.")

    def stop(self):
        if self.event:
            self.s.cancel(self.event)
            self.event = None
        print("Stopped.")

    def get_current_price(self):
        return self.monitor.get_sim_price(self.get_current_time())

    def get_current_time(self):
        return self.monitor.get_current_time()

    def get_current_battery_stats(self):
        return self.monitor.get_realtime_battery_stats()

    def update_battery_state(self, **kwargs):
        # Placeholder for potential updates to the scheduler's internal state
        pass

    def send_battery_command(self, command):
        self.monitor.send_battery_command(command)


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
        self.BatCap = self.BatNum * 5
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
        self.soc = current_soc

        self.price_history.append(current_price)
        if len(self.price_history) > self.LookBackBars:
            self.price_history.pop(0)

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            self.price_history, [self.BuyPct, self.SellPct])

        # chec if current_time is pandas timestamp
        if isinstance(current_time, pd.Timestamp):
            current_timenum = current_time.time()
        else:
            current_timenum = datetime.datetime.strptime(
                current_time, '%H:%M').time()

        command = ['Idle', 0]  # No action by default

        if self._is_charging_period(current_timenum) and current_price <= buy_price:
            # Charging logic
            chg_delta = self.BatChgMax * self.HrMin
            temp_chg = chg_delta + self.bat_cap

            if temp_chg <= self.BatCap:
                self.bat_cap = temp_chg  # charge battery
                self.soc = self.bat_cap / self.BatCap
                command = ['Charge', self.BatChgMax]

        elif self._is_discharging_period(current_timenum) and (current_price >= sell_price or current_price > self.SpikeLevel):
            # Discharging logic
            dischg_delta = self.BatDisMax * \
                self.HrMin if self.SellBack else min(
                    current_usage, self.BatDisMax) * self.HrMin
            temp_dischg = self.bat_cap - dischg_delta

            if temp_dischg >= self.BatCap * self.BatSocMin:
                self.bat_cap = temp_dischg  # discharge battery
                self.soc = self.bat_cap / self.BatCap
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

    def step(self, price_prediction_curve):
        # Implementation specific to AIScheduler
        # ... (compute the command based on the provided data)
        raise NotImplementedError

    def required_data(self):
        return ['price_prediction_curve']


if __name__ == '__main__':
    scheduler = BatteryScheduler(
        scheduler_type='PeakValley', battery_sn='RX2505ACA10JOA160037')
    scheduler.start()
