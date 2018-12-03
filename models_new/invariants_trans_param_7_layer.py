#!/usr/bin/python3

"""
   
    Necessary parameters for Invariance transformation net

"""
import math

with_X_transformation =True
sorting_method= None
with_global=True
data_dim = 4 # XYZI
x = 4

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(12, 1, -1, 16 * x, []),
                 (16, 1, 768, 32 * x, []),
                 (16, 2, 384, 64 * x, []),
                 (16, 2, 128, 96 * x, [])
]]

print(xconv_params)
xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 2, 3, 2),
                  (16, 1, 2, 1),
                  (12, 1, 1, 0)
]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(16 * x, 0.0),
              (16 * x, 0.5)]]


sampling = 'fps'
