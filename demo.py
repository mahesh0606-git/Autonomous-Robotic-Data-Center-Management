"""Offline simulation-only examples; no robotics hardware or trained AI."""
import numpy as np
from robotics.fusion import PlanarEKF,bayesian_decision
from robotics.predictive import statistical_features,thermal_step
from robotics.navigation import astar,inflated_grid
from robotics.manipulation import InsertionEvidence,verify_insertion
from robotics.coordination import Robot,Task,assign
from robotics.digital_twin import Scenario,evaluate

def main():
    ekf=PlanarEKF();print('EKF predicted state:',ekf.predict(.1))
    print('Example features:',statistical_features(np.arange(40).reshape(10,4)))
    print('Next thermal state:',thermal_step(50,25,100,500,3,1))
    print('Bayes example:',bayesian_decision([.8,.2],[.1,.8],[[0,10],[1,0]]))
    grid=np.zeros((8,8),dtype=bool);grid[3,2:6]=True
    print('Offline grid path:',astar(inflated_grid(grid,0),(0,0),(7,7)))
    print('Insertion verified:',verify_insertion(InsertionEvidence(.02,2,True,True,True)))
    robots=[Robot('R1',frozenset({'thermal'})),Robot('R2',frozenset({'vision'}))]
    tasks=[Task('inspect',frozenset({'thermal'}),2,1,.01)]
    print('Offline assignment:',assign(robots,tasks))
    print('Toy twin metrics:',evaluate(Scenario('example',2,1,0,100,0.2)))

if __name__=='__main__':main()
