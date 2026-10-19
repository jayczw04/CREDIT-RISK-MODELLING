# Credit Risk Modelling: Probability of Default (PD) Scorecard

Predicting the probability that a consumer loan defaults, using ~2M LendingClub loans (2007–2018).
Built the way a bank risk team would: an interpretable **WoE logistic-regression scorecard** (champion)
benchmarked against a **LightGBM** model (challenger), validated **out-of-time**, and turned into a
**credit cut-off strategy**.

> 🚧 Work in progress. Sections marked *TBD* will be filled in as the project develops.

---

## 1. Business problem
A lender needs to decide **who to approve** and **how much risk** each loan carries.
This project estimates each applicant's **Probability of Default (PD)** at the point of application,
converts it into a credit score, and recommends an approval cut-off that balances
**approval rate** against **bad rate**.

Extension: **Expected Loss = PD × LGD × EAD** (Basel / IFRS 9 framing).

## 2. Data
- **Source:** [LendingClub loan data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club): `accepted_2007_to_2018Q4.csv.gz`
- **Size:** ~2.2M loans × ~150 features
- Data is **not included** in this repo (size + licensing). Download it and place it in `data/raw/`.

**Target definition**
| `loan_status` | Target |
|---|---|
| Charged Off, Default | 1 (bad) |
| Fully Paid | 0 (good) |
| Current, Late, In Grace Period | Excluded (outcome unknown) |

Only loans issued early enough to have reached an outcome are used, to avoid bias toward early defaults.

## 3. Approach
1. **Leakage control:** removed all post-origination variables (payments received, recoveries,
   last FICO pull, hardship/settlement fields). Only information available at application is used.
2. **EDA:** default rates by grade, term, purpose, FICO band, DTI.
3. **Feature engineering:** credit history length, payment-to-income, loan-to-income, etc.
4. **Out-of-time split:** train on earlier vintages, test on a later vintage (mirrors real deployment).
5. **Champion model:** WoE binning + Information Value feature selection → logistic regression scorecard.
6. **Challenger model:** LightGBM, explained with SHAP.
7. **Scorecard scaling:** PD → points (e.g. 600 points = 50:1 odds, PDO = 20).
8. **Strategy:** approval rate vs bad rate trade-off and recommended cut-off.

## 4. Results *(TBD)*
| Model | AUC | Gini | KS |
|---|---|---|---|
| LendingClub grade (benchmark) | TBD | TBD | TBD |
| WoE Logistic Scorecard | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |

*Charts to add: default rate by grade, ROC curves, calibration plot, score distribution, cut-off trade-off.*

## 5. Business impact *(TBD)*
*e.g. "At a cut-off of X points, the bad rate falls from A% to B% while keeping a C% approval rate."*

## 6. Project structure
```
credit-risk-modelling/
├── data/                 # not tracked: raw & processed data
├── notebooks/            # EDA and modelling notebooks
├── src/                  # reusable scripts (cleaning, features, modelling)
├── reports/figures/      # charts used in this README
├── requirements.txt
└── README.md
```

## 7. How to run
```bash
git clone https://github.com/<your-username>/credit-risk-modelling.git
cd credit-risk-modelling
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download accepted_2007_to_2018Q4.csv.gz into data/raw/
```

## 8. Limitations & next steps
- **Selection bias:** the model only sees approved applicants. Reject inference using
  `rejected_2007_to_2018Q4.csv.gz` is a possible extension.
- Add **LGD** and **EAD** models to estimate Expected Loss.
- Monitor stability over time with **PSI**.

## Author
**Jay**: aspiring Risk Analyst · [LinkedIn](https://www.linkedin.com/in/jaychong04)