# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:07:44 2023

@author: sowmya
"""

import numpy as np
class colorUtils:
    @staticmethod
    def random_color():
        # Generate a random bright color
        return tuple(np.random.randint(0, 255, 3).tolist())