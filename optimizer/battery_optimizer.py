# battery_scheduler/scheduler.py

from pyomo.environ import *
from pydantic import BaseModel, validator
from typing import List

class BatterySchedulerConfig(BaseModel):
    b: List[float]
    s: List[float]
    l: List[float]
    p: List[float]
    R_c: float
    R_d: float
    capacity: float
    charge_mask: List[int]

    @validator('charge_mask')
    def validate_charge_mask(cls, v):
        if not all(x in [0, 1] for x in v):
            raise ValueError('charge_mask must contain only 0 or 1')
        return v
class BatteryScheduler:
    def __init__(self, config: BatterySchedulerConfig):
        self.config = config

    def create_model(self):
        model = ConcreteModel()
        model.T = Set(initialize=range(len(self.config.b)))
        model.x = Var(model.T, bounds=(-self.config.R_d, self.config.R_c))
        model.state_of_charge = Var(range(len(self.config.b)), bounds=(0, self.config.capacity))
        model.q = Var(model.T, within=NonNegativeIntegers)
        return model

    def define_objective(self, model):
        def obj_rule(model):
            return sum((self.config.l[t] - self.config.p[t] + model.x[t]) *
                       (self.config.b[t] * (1 - model.q[t]) + self.config.s[t] * model.q[t])
                       for t in model.T)
        model.obj = Objective(rule=obj_rule, sense=minimize)

    def define_constraints(self, model):
        # ... (include all the constraints from the original code)
        pass

    def solve(self):
        model = self.create_model()
        self.define_objective(model)
        self.define_constraints(model)
        solver = SolverFactory('mindtpy')
        results = solver.solve(model, strategy='OA', mip_solver='glpk', nlp_solver='ipopt')
        return model, results

    @staticmethod
    def post_process(model):
        socs = [model.state_of_charge[t]() for t in model.T]
        x_vals = [model.x[t]() for t in model.T]
        return x_vals, socs