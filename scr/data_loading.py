# =============================================================================
# 1. LOAD DATA
# =============================================================================
from pathlib import Path
import pandas as pd
import config

if config.Data_Path.exists():
    df = pd.read_csv(config.Data_Path)
else:
    df = pd.read_csv(config.URL)
    config.Data_Path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.Data_Path, index=False)


