#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from lugarrl import lugarrl

if __name__ == '__main__':
    numpy.random.seed(12345)
    jrl = lugarrl.LugarRL()
    for i in range(1000):
        jrl.step()