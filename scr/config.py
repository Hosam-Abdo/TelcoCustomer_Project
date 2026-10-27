# =============================================================================
# 0. CONFIGURATION
# =============================================================================
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
Base_Dir  = Path(__file__).resolve().parent.parent
Output_Dir= Base_Dir/"output"
Data_Path = Base_Dir/"Data"/"Telco-Customer-Churn.csv"
