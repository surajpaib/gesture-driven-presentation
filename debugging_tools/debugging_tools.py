
import pandas as pd
import numpy as np

from datetime import datetime

# Print variable, type and shape:


def debug(variable, p=True):
    """
    variable: a variable
    p: If p is True the variable is printed.
    """
    if p:
            print(variable)

    print(type(variable))

    if isinstance(variable, (list, dict, tuple)):
        print(len(variable))

    elif isinstance(variable, (np.ndarray, pd.DataFrame,
                    pd.core.series.Series)):

        print(variable.shape)
    else:
        print('The type of the variable isn´t one of',
              '[list, dict, tuple, np.array, pd.DataFrame, pd.Series].')

    return None

##############################

# Pause a program:


def pause():
    return input("Press the <ENTER> key to continue...")

##############################

# Print everything:


def printall(variable):
    """
    Print everything of numpy or pandas variable.
    """
    np.set_printoptions(threshold=np.nan)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(variable)
    return None

##############################

# Get time:


def getTime():
    return datetime.now()

#############################

# Get execution time of a function:


def execTime(function, *args, **kwargs):
    """
    Get execution time of a function. Copy this function to the same file of
    the function to run to test:
    function: The function without ().
    Example: execTime(function,input,[100,100,100])
    """
    start = getTime()
    function(*args, **kwargs)
    end = getTime()
    return print(function, "\n", end-start)
