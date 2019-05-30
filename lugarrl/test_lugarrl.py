#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from . import lugarrl, lugartypes, simulator


class MockLugarRL(object):
    def __getattr__(self, item):
        return lambda: None


class TestLugarRL(unittest.TestCase):
    def setUp(self):
        self.lugar = MockLugarRL()

    def test_simulation(self):
        self.lugar.simulate()

    def test_single_step(self):
        self.lugar.step()


class TestTimeBaseLugarRL(TestLugarRL):
    def setUp(self):
        self.lugar = lugarrl.LugarRL(lugartypes.SimulationType.TIME_BASED)

    def test_time_based_simulator_instance(self):
        self.assertTrue(isinstance(self.lugar.simulator, simulator.TimeBasedSimulator))

