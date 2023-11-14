#!/usr/bin/python

import matplotlib.pyplot as plt
import pyarma
import sys

fname_prefix = "svr_parameters_q_svrwave_nzdjpyaudjpy_avg_3600_audjpy_avg_bid_"
fname_actual_suffix = ".tsvactual_values"
fname_pred_suffix = ".tsvkernel_predict_values"
fname_linpred_suffix = ".tsvlin_predict_values"

actual_mat_zero = pyarma.mat()
actual_mat_zero.load(fname_prefix + "0" + fname_actual_suffix)
pred_diff = pyarma.mat(actual_mat_zero.n_rows, actual_mat_zero.n_cols)
pred_diff.fill(0)

lin_diff = pyarma.mat(actual_mat_zero.n_rows, actual_mat_zero.n_cols)
lin_diff.fill(0)


for l in range(0, 62, 2):
    actual_mat = pyarma.mat()
    actual_mat.load(fname_prefix + str(l) + fname_actual_suffix)

    pred_mat = pyarma.mat()
    pred_mat.load(fname_prefix + str(l) + fname_pred_suffix)

    lin_mat = pyarma.mat()
    lin_mat.load(fname_prefix + str(l) + fname_linpred_suffix)

    pred_diff += actual_mat - pred_mat
    lin_diff += actual_mat - lin_mat

