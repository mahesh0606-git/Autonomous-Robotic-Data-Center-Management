"""Analytical toy scenario evaluator. NOT Bullet, CFD or physical validation."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    name: str
    task_count: int
    completed: int
    collision_count: int
    power_watts: float
    cooling_kwh: float

def evaluate(scenario,duration_s=3600.0):
    if scenario.task_count<0 or not 0<=scenario.completed<=scenario.task_count:
        raise ValueError('Invalid task counts')
    if duration_s<=0 or scenario.power_watts<0 or scenario.cooling_kwh<0 or scenario.collision_count<0:
        raise ValueError('Invalid scenario parameters')
    return {'scenario':scenario.name,
            'task_completion_rate':scenario.completed/scenario.task_count if scenario.task_count else 1.0,
            'collision_incidents':scenario.collision_count,
            'robot_energy_kwh':scenario.power_watts*duration_s/3_600_000,
            'cooling_energy_kwh':scenario.cooling_kwh}
