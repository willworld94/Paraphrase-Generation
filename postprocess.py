import os
import re
import pandas as pd
import numpy as np
'''
    replace tree output to plain string
    the input is of string type
'''
def tree_postprocess(input_):
    result = input_.replace('(', '')
    result = result.replace(')', '')
    #     result = result.replace("'", '')
    list_ = result.split(' ')
    output = ' '.join([c for c in list_ if c.islower()]) + ' .'
    return output
