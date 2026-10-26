"""Smoke tests for simulation-only reference components."""
import unittest
import numpy as np
from robotics.fusion import PlanarEKF
from robotics.predictive import thermal_step
from robotics.navigation import astar
from robotics.manipulation import InsertionEvidence, verify_insertion
from robotics.digital_twin import Scenario, evaluate

class SmokeTests(unittest.TestCase):
    def test_filter(self):
        f=PlanarEKF()
        self.assertEqual(f.predict(0.1).shape, (5,))
    def test_thermal(self):
        self.assertAlmostEqual(thermal_step(50,25,100,500,3,1),50.05)
    def test_path(self):
        self.assertIsNotNone(astar(np.zeros((4,4),dtype=bool),(0,0),(3,3)))
    def test_offline_verification(self):
        evidence=InsertionEvidence(.01,1.0,True,True,True)
        self.assertTrue(verify_insertion(evidence))
    def test_scenario(self):
        result=evaluate(Scenario('demo',2,1,0,100,0.2))
        self.assertEqual(result['task_completion_rate'],.5)

if __name__=='__main__':
    unittest.main()
