# =============================================================================
# Requirements:
#  pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
# =============================================================================
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
# =============================================================================
import config
import data_loading
import PlotRankedBars
from config import Output_Dir
import MyEncoddi
# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = data_loading.df
# =============================================================================
# 2. make a copy from df, drop NaN, columns="customerID", change the data type
#    of the feature "TotalCharges"
# ==============================================================================
df_neue = df.copy().drop_duplicates()
df_neue = df_neue.dropna()
df_neue = df_neue.drop(columns="customerID")

df_neue["TotalCharges"] = pd.to_numeric(df_neue["TotalCharges"],errors="coerce")

print([element for element in df_neue["TotalCharges"] if not isinstance(element,float)] )
print([element for element in df_neue["TotalCharges"] if pd.isna(element)] )
df_neue = df_neue.dropna(subset =["TotalCharges","MonthlyCharges"])

# =============================================================================
# 3 Wir untersuchen den Zusammenhang zwischen den Merkmalen und dem 
#   Zielmerkmal  mit zwei Methoden „Churn“ :
# =============================================================================
df_neue["Churn"] = df_neue["Churn"].map({"Yes":1, "No":0})

ausgeschlossen_features_screening = []
gesucht_features_scre  = [col for col in df_neue.columns if col != "Churn"]

schwelle = 10          # Prozentpunkte
entfernte = []         # merkt sich, welche Features neu ausgeschlossen wurden
diffs = {}
for feat in gesucht_features_scre :
    summary = df_neue.groupby(feat).agg(CustomerCount=("Churn", "size"),
                                        ChurnRate=("Churn", "mean"))
    summary["ChurnRate"] = (summary["ChurnRate"] * 100 ).round(2)
    print(summary)

    differenz = summary["ChurnRate"].max() - summary["ChurnRate"].min()
    print(f"-> {feat}: Differenz = {differenz:.2f}% \n")
    diffs[feat] = differenz

    if differenz < schwelle:
        ausgeschlossen_features_screening.append(feat)
        entfernte.append(feat)

gesucht_features_screening = [col for col in gesucht_features_scre  if 
                              col not in ausgeschlossen_features_screening]

print("ausgeschlossen features screening:", ausgeschlossen_features_screening )    
print("gesuchten features screening:", gesucht_features_screening ) 

# =============================================================================
# 4. Ein rangiertes horizontales Balkendiagramm der Unterschiede für 
#    alle Merkmale gleichzeitig:
# =============================================================================
diff_s = pd.Series(diffs).sort_values()

diff_classes = [
    (75, "#08306b", "≥ 75"),
    (50, "#2171b5", "50 to < 75"),
    (20, "#4292c6", "20 to < 50"),
    (10, "#9ecae1", "10 to < 20"),
    (0,  "#d9d9d9", "< 10"),
]

PlotRankedBars.plot_ranked_bars(
    values=pd.Series(diffs),
    classes=diff_classes,
    xlabel="Max. - Min. Abwanderungsrate (Prozentpunkte)",
    title="Auswirkung der einzelnen Merkmale auf die Abwanderung",
    filename=config.Output_Dir/"figures"/"effect_features_on_churn_group.png",
    legend_title="Differenz (Prozentpunkte)",
    fmt="%.1f",
    min_xmax=10,
)
# =============================================================================
#5. Wir untersuchen den Zusammenhang Chi-Quadrat-Test χ² und φ (Korrelation) 
# zwischen den Merkmalen und dem Zielmerkmal  „Churn“ 
# =============================================================================
gesucht_features_corr  = [col for col in df_neue.columns if col != "Churn"]

alpha = 0.01
ergebnisse = []

for feature in gesucht_features_corr :
    ct = pd.crosstab(df_neue[feature], df_neue["Churn"])
    chi2, p, dof, expected = chi2_contingency(ct)

    n = ct.values.sum()
    cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5

    ergebnisse.append({
        "feature": feature,
        "p_value": p,
        "cramers_v": round(cramers_v, 3),
        "min_expected": expected.min().round(1),
    })

ergebnisse = pd.DataFrame(ergebnisse).sort_values("p_value").reset_index(drop=True)
ausgeschlossen_feature_chi  = ergebnisse.loc[ergebnisse["p_value"] >= alpha, "feature"].tolist()
gesuchten_feature_chi  = ergebnisse.sort_values("cramers_v",ascending=False).loc[ergebnisse["p_value"] < alpha, "feature"].tolist()

print("Ausgeschlossene Feature nach Chi-squared:", ausgeschlossen_feature_chi)
print("Gesuchte Featuresnach Chi-squared:", gesuchten_feature_chi)
# =============================================================================
# Besondere Aufmerksamkeit bei der Interpretation des Chi-Quadrat-Tests χ²
# =============================================================================
hervorheben = ["MonthlyCharges", "TotalCharges"]

def highlight_rows(row):
    if row["feature"] in hervorheben:
        return ["background-color: yellow; color: red"] * len(row)
    return [""] * len(row)

ergebnisse_sorted = ergebnisse.sort_values("cramers_v", ascending=False)
print(ergebnisse_sorted.to_string())
# =============================================================================
# . Ein rangiertes horizontales Balkendiagramm der corellationen für 
#     alle Merkmale gleichzeitig:
# =============================================================================
# only features where the chi-square test is valid (expected counts >= 5)
v_classes = [
    (0.30, "#08306b", "≥ 0.30  strong"),
    (0.20, "#2171b5", "0.20 to < 0.30"),
    (0.10, "#6baed6", "0.10 to < 0.20  weak"),
    (0.00, "#d9d9d9", "< 0.10  negligible"),
]

valid = ergebnisse[ergebnisse["min_expected"] >= 5]
nicht_signifikant = set(valid.loc[valid["p_value"] >= alpha, "feature"])

PlotRankedBars.plot_ranked_bars(
    values=valid.set_index("feature")["cramers_v"],
    classes=v_classes,
    xlabel="Cramérs V (Stärke des Zusammenhangs mit der Abwanderung)",
    title="Chi-Quadrat-Test: Auswirkung der einzelnen Merkmale auf die Abwanderung",
    filename=config.Output_Dir/"figures"/"effect_features_on_churn_chi2.png",
    legend_title="Cramérs V",
    fmt="%.3f",
    not_significant=nicht_signifikant,
    alpha=alpha,
    min_xmax=0.1,
)
# =============================================================================
# Define a Funktion encoddi()
# Die Funktion encoddi() wandelt kategoriale Merkmale in numerische Werte um, 
# amit sie für die Datenanalyse und Machine Learning verwendet werden können.
#  Dafür erstellt sie zunächst eine Kopie des DataFrames und unterteilt die 
# Merkmale in drei Gruppen:
# =============================================================================
# Segmentanalyse: Kundengruppen mit erhöhtem Churn-Risiko identifizieren
# Ziel: Zwei Kundensegmente identifizieren, bei denen die Abwanderungsrate
# untersucht wird, um mögliche Risikogruppen zu erkennen.
# =============================================================================
df_risk, encoders = MyEncoddi.encoddi(df_neue, drop_first=False)
df_risk = df_risk.loc[:, ~df_risk.T.duplicated()]
features = df_risk.columns.drop(["Churn", "TotalCharges"], errors="ignore")


def hoch(col):
    """True for the 'high' customers of a feature:
    value 1 for binary columns, top 25 % (>= 75th percentile) otherwise."""
    if df_risk[col].nunique() == 2:
        return df_risk[col] == df_risk[col].max()
    return df_risk[col] >= df_risk[col].quantile(0.75)


segmente = []
for elem1, elem2 in combinations(features, 2):
    segment = df_risk[hoch(elem1) & hoch(elem2)]
    if len(segment) == 0:
        continue

    kundenanzahl = len(segment)

    segmente.append({
        "Segment": f"{elem1} mit {elem2}",
        "Kundenanzahl": kundenanzahl,
        "ChurnRate": round(segment["Churn"].mean() * 100, 2),
    })

# Sort: highest churn rate first; ties broken by more customers first
segmente = (
    pd.DataFrame(segmente)
      .sort_values(["ChurnRate", "Kundenanzahl"], ascending=[False, False])
      .reset_index(drop=True)
)

# Print every segment in descending order of the churn rate
for _, zeile in segmente.iterrows():
    print(f"\n{zeile['Segment']}")
    print(f"Kundenanzahl: {zeile['Kundenanzahl']}")
    print(f"Churn-Rate: {zeile['ChurnRate']:.2f}%")

# Write ALL segments to LaTeX (no head, no filter)
latex = segmente.to_latex(
    index=False,
    escape=True,
    float_format="%.2f",
    column_format="p{9cm}rr",
    longtable=True,
    caption="All risk segments, sorted by churn rate (descending)",
    label="tab:risk-segments-all",
)

dokument = (
    "\\documentclass[a4paper,11pt]{article}\n"
    "\\usepackage[T1]{fontenc}\n"
    "\\usepackage[utf8]{inputenc}\n"
    "\\usepackage{booktabs}\n"
    "\\usepackage{longtable}\n"
    "\\begin{document}\n"
    + latex +
    "\\end{document}\n"
)
REPORTS_FILES = config.Output_Dir/"reports"
with open(REPORTS_FILES/"segmente_komplett.tex", "w", encoding="utf-8") as f:
    f.write(dokument)

print("Rows written:", len(segmente), "->", REPORTS_FILES/ "segmente_komplett.tex")