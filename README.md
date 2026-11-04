# QuantInvent

ML + Quantum (VQE) + classical OR inventory planner on the M5 Walmart dataset (store CA_1, FOODS_3, 3 SKUs x 5 weeks = 15 qubits).

## Pipeline
1. LSTM + attention forecaster gives mean and std per SKU
2. Conformal safety stock gives guaranteed coverage instead of a Gaussian assumption
3. ML-parameterised QUBO solved by VQE, warm-started from a rule-based schedule
4. Classical constraint repair + local search
5. Bayesian-optimised penalty weights

## Setup
1. Download M5 from Kaggle into `backend/data/raw/` (sales_train_evaluation.csv, calendar.csv, sell_prices.csv)
2. `python -m venv venv` then `venv\Scripts\activate`
3. `pip install -r backend/requirements.txt`
4. Run in order:
   `python backend\ml\data.py`, `python backend\ml\train.py`, `python backend\ml\conformal.py`,
   `python backend\bo.py`, `python backend\run_eval.py`
5. API: `cd backend` then `uvicorn api:app --reload`
6. UI: `cd frontend` then `npm install` then `npm run dev`

## Findings
- Proposed pipeline cuts cost about 39% vs classical EOQ-style rule and meets the 95% service level
- At 15 qubits, ML + classical (no quantum) is competitive; the quantum layer is a proof of concept
- Warm-start halves the VQE optimality gap vs random initialisation (24% to 12%)

## Data and citation
M5 data is not included; download it from Kaggle. Makridakis et al. (2022), M5 competition.