# battery_scheduler/scheduler.py
import os
from pyomo.environ import *
from typing import List
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class BatterySchedulerConfig():
    b: List[float]
    s: List[float]
    l: List[float]
    p: List[float]
    R_c: float
    R_d: float
    capacity: float
    bat_kwh_now: float
    init_kwh: float
    current_time_index: int
    charge_mask: List[int]


class BatteryScheduler:
    def __init__(self, config):
        self.config = BatterySchedulerConfig(**config)
        self.interval_coef = 60 / (1440 / len(self.config.b))

    def create_model(self):
        WLSACCESSID = os.environ.get('WLSACCESSID')
        WLSSECRET = os.environ.get('WLSSECRET')
        LICENSEID = os.environ.get('LICENSEID')

        if WLSACCESSID and WLSSECRET and LICENSEID:
            options = {
                "WLSACCESSID": WLSACCESSID,
                "WLSSECRET": WLSSECRET,
                "LICENSEID": int(LICENSEID),
            }
        else:
            options = {}

        env = gp.Env(params=options)
        model = gp.Model("BatteryScheduler", env=env)

        T = range(len(self.config.b))

        # Variables
        x = model.addVars(T, lb=-self.config.R_d, ub=self.config.R_c, name="x")
        state_of_charge = model.addVars(T, lb=0, ub=self.config.capacity, name="state_of_charge")
        g_c = model.addVars(T, lb=-1e6, ub=1e6, name="g_c")  # Use large but finite bounds
        z = model.addVars(T, lb=-1e6, ub=1e6, name="z")  # Use large but finite bounds

        # Constraints
        model.addConstrs((g_c[t] == self.config.l[t] - self.config.p[t] + x[t] for t in T), "grid_consumption")
        
        # model.addConstr(state_of_charge[0] <= 0.01, "initial_soc")
        model.addConstr(state_of_charge[self.config.current_time_index] <= self.config.bat_kwh_now, "initial_soc")
        model.addConstr(state_of_charge[self.config.current_time_index] >= self.config.bat_kwh_now, "initial_soc")
        model.addConstrs((state_of_charge[t] == state_of_charge[t-1] + x[t-1]/self.interval_coef for t in range(1, len(T))), "soc_evolution")
        model.addConstrs((state_of_charge[t] >= 0.10 * self.config.capacity for t in T), "minimum_soc")
        model.addConstrs((x[t] >= -state_of_charge[t-1]*self.interval_coef for t in range(1, len(T))), "discharge_limit")
        model.addConstr(x[0] >= -self.config.capacity, "initial_discharge_limit")
        model.addConstr(state_of_charge[len(T)-1] <= self.config.capacity*0.5, "final_soc")
        
        model.addConstrs((x[t] <= 0 for t in T if self.config.charge_mask[t] == 0), "charge_mask_discharge")
        model.addConstrs((x[t] >= 0 for t in T if self.config.charge_mask[t] == 1), "charge_mask_charge")

        # Piecewise linear constraint for z
        max_g_c = max(max(self.config.l), max(self.config.p)) + max(self.config.R_c, self.config.R_d)
        for t in T:
            model.addGenConstrPWL(g_c[t], z[t], 
                                  [-max_g_c, 0, max_g_c], 
                                  [-max_g_c * self.config.s[t], 0, max_g_c * self.config.b[t]])

        # Objective
        model.setObjective(gp.quicksum(z[t] for t in T), GRB.MINIMIZE)

        return model, x, state_of_charge, g_c, z

    def solve(self):
        model, x, state_of_charge, g_c, z = self.create_model()
        
        # Set solver parameters
        model.Params.TimeLimit = 30
        model.Params.MIPFocus = 3
        model.Params.MIPGap = 0.001
        model.Params.Cuts = 2
        model.Params.Presolve = 2
        model.Params.Heuristics = 0.05
        model.Params.ImproveStartTime = 300
        model.Params.ImproveStartGap = 0.02
        model.Params.NodefileStart = 0.5
        model.Params.NodefileDir = "."
        model.Params.Threads = 0

        model.optimize()

        if model.status == GRB.OPTIMAL:
            x_vals = [x[t].X for t in range(len(self.config.b))]
            soc_vals = [state_of_charge[t].X for t in range(len(self.config.b))]
            g_c_vals = [g_c[t].X for t in range(len(self.config.b))]
            
            print(f"Optimization finished successfully")
            print(f"Best objective value: {model.ObjVal}")
            
            return x_vals, soc_vals
        else:
            print(f"Optimization failed with status {model.status}")
            return None, None

    def plot(self, charge_mask, socs, x_vals):
        config = self.config
        loads = config.l
        solars = config.p
        prices = config.b
        sell_prices = config.s
        battery_actions = x_vals

        df = pd.DataFrame({
            'Load': loads,
            'Solar': solars,
            'Battery': battery_actions,
            'Buy_Price': prices,
            'Sell_Price': sell_prices,
            'SOC': socs
        })

        # Calculate Net Grid
        df['Net_Grid'] = (df['Load'] - df['Solar'] + df['Battery'])

        # Set up Seaborn
        sns.set_theme(style="whitegrid")

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax2 = ax.twinx()
        ax3 = ax.twinx()

        # Define the color palette to match the image
        colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']

        # Line plots with fill
        ax.plot(df.index, df['Load'], color=colors[3],
                linewidth=2, label='Consumption')
        ax.fill_between(df.index, 0, df['Load'], alpha=0.5, color=colors[3])
        ax.plot(df.index, df['Solar'], color=colors[0],
                linewidth=2, label='Generation')
        ax.fill_between(df.index, 0, df['Solar'], alpha=0.5, color=colors[0])
        ax.plot(df.index, df['Net_Grid'], color=colors[1],
                linewidth=2, label='Grid')
        ax.fill_between(df.index, 0, df['Net_Grid'],
                        alpha=0.5, color=colors[1])
        ax.plot(df.index, df['Battery'], color=colors[2],
                linewidth=2, label='Storage')
        ax.fill_between(df.index, 0, df['Battery'], alpha=0.5, color=colors[2])

        ax2.plot(df.index, df['Buy_Price'], color='black',
                 linewidth=1, linestyle='--', label='Buy Price')
        ax2.plot(df.index, df['Sell_Price'], color='black',
                 linewidth=1, linestyle=':', label='Sell Price')
        ax.plot(df.index, df['SOC'], color='red',
                linewidth=1, linestyle='-', label='SOC')

        # Draw charge mask
        for i, mask in enumerate(charge_mask):
            if mask == 0:
                ax.axvspan(i, i+1, color='gray', alpha=0.1)

        # Set x-axis tick labels and positions
        # Tick positions every 2 data points (every hour)
        xticks = range(0, len(df), 2)
        # Tick labels in 30-minute format
        xticklabels = [
            f"{int(i/2):02d}:{'00' if i % 2 == 0 else 30}" for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=60, ha='right')

        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Battery Power(kWh/30min)')
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(
            0.5, -0.15), fancybox=True, shadow=True, ncol=4)
        ax.set_title('Energy Management')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def post_process(model):
        socs = [model.state_of_charge[t]() for t in model.T]
        x_vals = [model.x[t]() for t in model.T]
        return x_vals, socs

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

if __name__ == '__main__':
    config, x_vals, socs, charge_mask = pickle.load(open('battery_sched.pkl', 'rb'))
    config['b'] = resample_data(config['b'], 48)
    config['s'] = resample_data(config['s'], 48)
    config['l'] = resample_data(config['l'], 48)
    config['p'] = resample_data(config['p'], 48)
    config['charge_mask'] = resample_data(config['charge_mask'], 48)
    config['init_kwh'] = 0.0
    config['bat_kwh_now'] = 2.0
    config['current_time_index'] = 28
    scheduler = BatteryScheduler(config)
    x_vals, socs = scheduler.solve()
    # x_vals = adjust_middle_value(x_vals)
    scheduler.plot(config['charge_mask'], socs, x_vals)

