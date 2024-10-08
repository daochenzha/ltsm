""" This file contains the test cases for the csv_reader module. """
import os
import pandas as pd
import numpy as np

def DK_TestCases_input():
    df1 = pd.DataFrame([
    ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In'],
    ['6/30/2023 19:01:24', 0, 1, 98.34],
    ['6/30/2023 19:03:04', 0, 3, 98.93],
    ['6/30/2023 19:04:44', 0, 2, 97.90],
    ['6/30/2023 19:06:22', 2, 3, 98.37],
    ['6/30/2023 19:08:03', 3, 1, 98.37]
])
    
    df2 = pd.DataFrame([
    ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
    ['6/30/2023 19:09:43', 1, 2, 98.43, 109.31],
    ['6/30/2023 19:11:23', 1, 3, 97.64, 109.18],
    ['6/30/2023 19:13:02', 1, 4, np.nan, 109.18]
])

    df3 = pd.DataFrame([
        ['Updated Time', 'Suction Pressure', 'Suction temp'],
        ['6/30/2023 19:01:24', 61.71, 102.75],
        ['6/30/2023 19:03:04', 69.21, 103.19],
        ['6/30/2023 19:04:44', 71.35, 103.34],
        ['6/30/2023 19:06:22', 68.14, 102.60],
        ['6/30/2023 19:08:03', 83.47, 103.26],
        ['6/30/2023 19:09:43', 63.14, 103.85]
    ])

    df4 = pd.DataFrame([
        ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out'],
        ['6/30/2023 19:01:24', 61.71, 102.75, np.nan, 109.73],
        ['6/30/2023 19:03:04', np.nan, 103.19, 98.93, 109.73],
        ['6/30/2023 19:04:44', 71.35, 103.34, np.nan, 108.87],
        ['6/30/2023 19:06:22', np.nan, 102.60, 98.37, np.nan]
    ])

    df5 = pd.DataFrame([
        ['Updated Time', 'Suction Pressure', 'Suction temp', 'Condenser In', 'Condenser Out', 'Additional Column'],
        ['6/30/2023 19:01:24', 61.71, 102.75, 98.34, 109.73, 1],
        ['6/30/2023 19:03:04', 69.21, 103.19, 98.93, 109.73, 2],
        ['6/30/2023 19:04:44', 71.35, 103.34, 97.90, 108.87, 3],
        ['6/30/2023 19:06:22', 68.14, 102.60, 98.37, 109.48, 4],
        ['6/30/2023 19:08:03', 83.47, 103.26, 98.37, 109.44, 5]
    ])
    
    return [df1, df2, df3, df4, df5]

def DK_TestCases_Output():
    df1_expected = pd.DataFrame({
            0: [0, 1, 98.34],
            1: [0, 3, 98.93],
            2: [0, 2, 97.90],
            3: [2, 3, 98.37],
            4: [3, 1, 98.37]
        })
    
    df2_expected = pd.DataFrame({
        0: [1, 2, 98.43, 109.31],
        1: [1, 3, 97.64, 109.18],
        2: [1, 4, 0, 109.18]
    })

    df3_expected = pd.DataFrame({
        0: [61.71, 102.75],
        1: [69.21, 103.19],
        2: [71.35, 103.34],
        3: [68.14, 102.60],
        4: [83.47, 103.26],
        5: [63.14, 103.85]
    })

    df4_expected = pd.DataFrame({
        0: [61.71, 102.75,0, 109.73],
        1: [0,	103.19,	98.93,109.73],
        2: [71.35,	103.34,	0,	108.87],
        3: [0,102.6,98.37,0]
    })
    
    df5_expected = pd.DataFrame({
        0: [61.71, 102.75, 98.34, 109.73, 1],
        1: [69.21, 103.19, 98.93, 109.73, 2],
        2: [71.35, 103.34, 97.9, 108.87, 3],
        3: [68.14, 102.6, 98.37, 109.48, 4],
        4: [83.47, 103.26, 98.37, 109.44, 5]
    })
    return [df1_expected, df2_expected, df3_expected, df4_expected, df5_expected]



DK_TestCases_input()
DK_TestCases_Output()