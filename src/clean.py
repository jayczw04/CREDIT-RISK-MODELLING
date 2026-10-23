"""Load the LendingClub sample and return a model-ready dataframe."""
import pandas as pd

# ---- Settings (CAPITALS = constants that don't change) ----
LEAKAGE_COLS = [
    # payments made after the loan started
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    # credit info pulled after the loan started
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # flags that only exist once a borrower is struggling
    "pymnt_plan", "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
]
JUNK_COLS = ["id", "member_id", "url", "desc", "title", "emp_title", "zip_code"]
BAD_STATUSES = ["Charged Off", "Default"]
GOOD_STATUSES = ["Fully Paid"]
DATE_COLS = ["issue_d", "earliest_cr_line"]


def find_junk(df, max_missing=0.9, protect=()):
    """Named junk + columns with one value or more than max_missing empty."""
    rule_based = [
        c for c in df.columns
        if c not in protect
        and (df[c].nunique() <= 1 or df[c].isna().mean() > max_missing)
    ]
    return list(dict.fromkeys([c for c in JUNK_COLS if c in df.columns] + rule_based))


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)

    # 1. individual applications only
    df = df[df["application_type"] == "Individual"]

    # 2. drop leakage + junk
    df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    df = df.drop(columns=find_junk(df, protect=["loan_status"]))

    # 3. fix dates, term, emp_length, categories
    df = df.assign(
        **{c: pd.to_datetime(df[c], format="%b-%Y", errors="coerce") for c in DATE_COLS},
        term=df["term"].str.extract(r"(\d+)")[0].astype(float),            # " 60 months" -> 60
        emp_length=df["emp_length"]
            .replace({"< 1 year": "0", "10+ years": "10"})
            .str.extract(r"(\d+)")[0].astype(float),                       # "10+ years" -> 10
    )

    # 4. build target (finished loans only), then drop loan_status: it IS the answer
    df = df[df["loan_status"].isin(BAD_STATUSES + GOOD_STATUSES)]
    df = df.assign(target=df["loan_status"].isin(BAD_STATUSES).astype(int))
    df = df.drop(columns="loan_status")

    # 5. time window filter: only cohorts that have fully matured
    #    (every loan is now 36 months, so term carries no information)
    df_model = df[(df["issue_d"].dt.year <= 2015) & (df["term"] == 36)].drop(columns="term")

    text_cols = df_model.select_dtypes(include="str").columns
    df_model[text_cols] = df_model[text_cols].astype("category")
    return df_model


if __name__ == "__main__":
    df_model = load_and_clean("data/loans_sample.csv.gz")
    print(df_model.shape)
    print(df_model["target"].value_counts(), df_model["target"].mean().round(4))
