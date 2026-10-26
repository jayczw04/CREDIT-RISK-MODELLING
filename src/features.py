import numpy as np
import pandas as pd

## columns kept out of the model
BENCHMARK_COLS = ["grade", "sub_grade", "int_rate"]

EXCLUDE_COLS = [
    "issue_d", "earliest_cr_line",
    "fico_range_low", "fico_range_high",
    "funded_amnt", "funded_amnt_inv",
    "addr_state",
    "initial_list_status", "disbursement_method",
]

NEW_BUREAU_COLS = [
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "inq_fi", "total_cu_tl", "inq_last_12m",
]


## engineered features
def add_features(df):
    df = df.copy()
    monthly_inc = (df["annual_inc"] / 12).replace(0, np.nan)
    annual_inc = df["annual_inc"].replace(0, np.nan)

    df["credit_hist_months"] = (
        (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
        + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
    )
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["pti"] = df["installment"] / monthly_inc
    df["dti_incl_loan"] = df["dti"] + 100 * df["pti"]
    df["loan_to_income"] = df["loan_amnt"] / annual_inc
    df["revol_bal_to_income"] = df["revol_bal"] / annual_inc

    return df


## columns allowed into the model
def feature_columns(df, target="target"):
    drop = set(BENCHMARK_COLS + EXCLUDE_COLS + NEW_BUREAU_COLS + [target])
    return [c for c in df.columns if c not in drop]


## out-of-time split
def time_split(df, train_start="2013-01-01", test_start="2015-07-01"):
    train = df[(df["issue_d"] >= train_start) & (df["issue_d"] < test_start)].copy()
    test = df[df["issue_d"] >= test_start].copy()
    return train, test