"""
function:The function is to read 2-D matrix from txt
input:filename:file's absolute or relative path
output:ndarry data
author:zhipeng.liu
"""
import numpy as np
import os
def readMatrixFromTxt(filename):
    if(os.path.exists(filename) == True):
        filestream = open(filename)
        data = np.loadtxt(filestream)
        return data
    else:
        print filename + " does not exist!"
        return None
