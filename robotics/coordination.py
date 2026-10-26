"""Small offline capability-constrained task assignment; no robot messaging."""
from dataclasses import dataclass
from itertools import permutations

@dataclass(frozen=True)
class Robot:
    name: str
    capabilities: frozenset

@dataclass(frozen=True)
class Task:
    name: str
    required: frozenset
    duration: float
    energy: float
    risk: float

def assign(robots,tasks,energy_weight=.1,risk_weight=10.0):
    """Brute force for small problems, one task per robot; not GA/Hungarian."""
    if len(tasks)>len(robots):return None
    best=None
    for choices in permutations(range(len(robots)),len(tasks)):
        if any(not tasks[j].required.issubset(robots[i].capabilities) for j,i in enumerate(choices)):
            continue
        cost=sum(t.duration+energy_weight*t.energy+risk_weight*t.risk for t in tasks)
        # Stable tie-break; no modeled robot-specific time or collision risk.
        candidate=(cost,choices)
        if best is None or candidate<best:best=candidate
    if best is None:return None
    return {tasks[j].name:robots[i].name for j,i in enumerate(best[1])}
