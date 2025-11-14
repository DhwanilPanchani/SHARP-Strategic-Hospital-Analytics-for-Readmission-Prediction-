from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path("data")

def _detect_year(path: Path) -> int:
    m = re.search(r"fy(\d{4})", path.name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Year not found in filename: {path}")

def load_ipps_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        rename = {
            "DRG_Definition": "DRG_Definition",
            "Provider_Id": "Provider_Id",
            "Provider_Name": "Provider_Name",
            "Provider_City": "Provider_City",
            "Provider_State": "Provider_State",
            "Provider_Zip_Code": "Provider_Zip_Code",
            "Total_Discharges": "Total_Discharges",
            "Average_Covered_Charges": "Average_Covered_Charges",
            "Average_Total_Payments": "Average_Total_Payments",
            "Average_Medicare_Payments": "Average_Medicare_Payments",
        }
        df = df.rename(columns=rename)
        df["year"] = _detect_year(f)
        frames.append(df)
    cms = pd.concat(frames, ignore_index=True)
    return cms