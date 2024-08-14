# battery_scheduler/scheduler.py

from pyomo.environ import *
from typing import List
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


@dataclass
class BatterySchedulerConfig():
    b: List[float]
    s: List[float]
    l: List[float]
    p: List[float]
    R_c: int
    R_d: int
    capacity: int
    charge_mask: List[int]


class BatteryScheduler:
    def __init__(self, config):
        self.config = BatterySchedulerConfig(**config)

    def create_model(self):
        model = ConcreteModel()
        model.T = Set(initialize=range(len(self.config.b)))
        model.x = Var(model.T, bounds=(-self.config.R_d, self.config.R_c))
        model.state_of_charge = Var(
            range(len(self.config.b)), bounds=(0, self.config.capacity))
        model.q = Var(model.T, within=NonNegativeIntegers)
        return model

    def define_objective(self, model):
        def obj_rule(model):
            return sum((self.config.l[t] - self.config.p[t] + model.x[t]) *
                       (self.config.b[t] * (1 - model.q[t]) +
                        self.config.s[t] * model.q[t])
                       for t in model.T)
        model.obj = Objective(rule=obj_rule, sense=minimize)

    def define_constraints(self, model):
        def soc_constraint_rule(model, t):
            if t == 0:
                # Initialize SOC to full
                return model.state_of_charge[t] <= 0.01
            else:
                return model.state_of_charge[t] == model.state_of_charge[t-1] + (model.x[t-1])/2
        model.soc_constraint = Constraint(model.T, rule=soc_constraint_rule)

        def soc_bounds_rule(model, t):
            return (0, model.state_of_charge[t], self.config.capacity)
        model.soc_bounds = Constraint(model.T, rule=soc_bounds_rule)

        def discharge_constraint_rule(model, t):
            if t == 0:
                return model.x[t] >= -self.config.capacity
            else:
                return model.x[t] >= -model.state_of_charge[t-1]
        model.discharge_constraint = Constraint(
            model.T, rule=discharge_constraint_rule)

        def q_constraint_rule1(model, t):
            return self.config.l[t] - self.config.p[t] + model.x[t] <= 10000000 * (1 - model.q[t])
        model.q_constraint1 = Constraint(model.T, rule=q_constraint_rule1)

        def q_constraint_rule2(model, t):
            return self.config.l[t] - self.config.p[t] + model.x[t] >= -10000000 * model.q[t]
        model.q_constraint2 = Constraint(model.T, rule=q_constraint_rule2)

        def final_soc_constraint_rule(model):
            return model.state_of_charge[len(self.config.b)-1] <= 0.01
        model.final_soc_constraint = Constraint(rule=final_soc_constraint_rule)

        # Cycle Count Constraint
        # def cycle_indicator_rule(model, t):
        #     if t > 0:
        #         return model.x[t] * model.x[t-1] >= -model.M * model.cycle_indicator[t]
        #     else:
        #         return Constraint.Skip
        # model.cycle_indicator_constraint = Constraint(model.T, rule=cycle_indicator_rule)

        # def cycle_count_rule(model):
        #     return model.cycle_count == sum(model.cycle_indicator[t] for t in model.T)
        # model.cycle_count_constraint = Constraint(rule=cycle_count_rule)

        # def total_cycle_constraint_rule(model):
        #     return model.cycle_count <= 2
        # model.total_cycle_constraint = Constraint(rule=total_cycle_constraint_rule)

        # Define the charge constraint based on the mask
        def charge_constraint_rule(model, t):
            # If charge mask is 0, then the battery should not charge
            if self.config.charge_mask[t] == 0:
                return model.x[t] <= 0
            elif self.config.charge_mask[t] == 1:
                return model.x[t] >= 0
            else:
                return Constraint.Skip
        model.charge_constraint = Constraint(
            model.T, rule=charge_constraint_rule)

    def solve(self):
        model = self.create_model()
        self.define_objective(model)
        self.define_constraints(model)
        solver = SolverFactory('mindtpy')
        _ = solver.solve(model, strategy='OA',
                               mip_solver='glpk', nlp_solver='ipopt')
        __ = [model.state_of_charge[t]() for t in model.T]
        x_vals = [model.x[t]() for t in model.T]
        return x_vals, __

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


if __name__ == '__main__':
    config, x_vals, socs, charge_mask = pickle.load(open('battery_sched.pkl', 'rb'))
    scheduler = BatteryScheduler(config)
    scheduler.plot(charge_mask, socs, x_vals)

