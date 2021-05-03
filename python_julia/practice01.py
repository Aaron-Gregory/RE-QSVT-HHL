#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:41:13 2021

@author: Aaron Gregory
"""

from julia.api import Julia
jl = Julia(compiled_modules=False)
x = jl.include("test.jl")