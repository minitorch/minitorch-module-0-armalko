"""Collection of the core mathematical operators used throughout the code base."""

import math
import numpy as np

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(a, b):
    return a * b

def id(x):
    return x

def add(a, b):
    return a + b

def neg(x):
    return -x

def lt(a, b):
    return a < b

def eq(a, b):
    return a == b

def max(a, b):
    return max(a, b)

def is_close(a, b):
    return abs(a - b) < 1e-2

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return 1 / (1 + math.exp(-x))

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def log(x):
    return math.log(x)

def exp(x):
    return math.exp(x)

def inv(x):
    return 1 / x


def log_back(x, d_out):
    """
    Computes the derivative of the natural logarithm with respect to x,
    and then multiplies it by d_out.
    """
    return (1 / x) * d_out


def inv_back(x, d_out):
    """
    Computes the derivative of the reciprocal function (1/x) with respect to x,
    and then multiplies it by d_out.
    """
    return (-1 / (x ** 2)) * d_out


def relu_back(x, d_out):
    """
    Computes the derivative of the ReLU function with respect to x,
    and then multiplies it by d_out.
    """
    relu_derivative = np.where(x > 0, 1, 0)  # Derivative of ReLU: 1 if x > 0, else 0
    return relu_derivative * d_out

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
