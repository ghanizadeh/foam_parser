# utility.py
import re
import numpy as np
import pandas as pd
 
def extract_samples_complete_fixed(df):
    samples = []
    formulations = {}
    row = 0
    last_formulation = None
    last_dilution = None
    last_tube_volume = None
    column_map = {}
    current_dilution_rows = []

    dilution_has_8c = np.nan
    dilution_has_4c = np.nan

    def flush_current_dilution():
        for row_data in current_dilution_rows:
            row_data["Stable at 8C"] = dilution_has_8c
            row_data["Stable at 4C"] = dilution_has_4c
            row_data["Tube Volume (mL)"] = last_tube_volume
            samples.append(row_data)
        current_dilution_rows.clear()

    def parse_formulation(text):
        data = {
            "SampleID": re.search(r"\((.*?)\)", text).group(1).strip() if re.search(r"\((.*?)\)", text) else None
        }
        seen_keys_lower = set()
        for p in re.split(r'[,-]', text):
            p = p.strip()
            ppm_match = re.match(r"(\d+\.?\d*)\s*ppm\s*(.*)", p, re.IGNORECASE)
            if ppm_match:
                val, chem = ppm_match.groups()
                chem = re.sub(r"\(.*?\)", "", chem).strip()
                chem_key = f"{chem} (ppm)"
                if chem_key.lower() not in seen_keys_lower:
                    data[chem_key] = float(val)
                    seen_keys_lower.add(chem_key.lower())
                continue
            percent_match = re.match(r"(\d+\.?\d*)%\s*(.*)", p, re.IGNORECASE)
            if percent_match:
                val, chem = percent_match.groups()
                chem = re.sub(r"\(.*?\)", "", chem).strip()
                chem_key = f"{chem} (%)"
                if chem_key.lower() not in seen_keys_lower:
                    data[chem_key] = float(val)
                    seen_keys_lower.add(chem_key.lower())
        return data

    while row < df.shape[0]:
        cell = str(df.iat[row, 0]).strip()

        # --- Formulation row detection ---
        if any(sym in cell.lower() for sym in ["%", "ppm"]) and "(" in cell:
            flush_current_dilution()
            formulation_text = cell.lower()

            # Detect stability
            s8, s4 = np.nan, np.nan
            if "unstable concentrate" in formulation_text:
                s8 = False
                s4 = False

            values_to_check = [
                str(df.iat[row, col]).lower().strip()
                for col in range(1, min(11, df.shape[1]))
                if pd.notna(df.iat[row, col])
            ]
            for val in values_to_check:
                if "unstable concentrate" in val:
                    s4 = False
                    s8 = False
                if "unstable at 4c" in val:
                    s4 = False
                elif "stable at 4c" in val and s4 is not False:
                    s4 = True
                if "unstable at 8c" in val:
                    s8 = False
                elif "stable at 8c" in val and s8 is not False:
                    s8 = True

            dilution_has_8c = s8
            dilution_has_4c = s4

            last_formulation = parse_formulation(cell)
            if not last_formulation.get("SampleID"):
                fallback_id = f"Sample_{len(formulations)+1}"
                last_formulation["SampleID"] = fallback_id
            formulations[last_formulation["SampleID"]] = last_formulation

            # 🔍 Check next row for dilution
            next_row_text = ",".join(df.iloc[row + 1].fillna("").astype(str)).lower() if row + 1 < df.shape[0] else ""
            if not re.search(r"\d+\s*x", next_row_text):
                # If no dilution row follows, treat it as a single-row sample
                samples.append({
                    "SampleID": last_formulation["SampleID"],
                    "Dilution": None,
                    "Day": None,
                    "Foam (cc)": None,
                    "Foam Texture": None,
                    "Water (cc) (cc)": None,
                    "Zeta": None,
                    "Conductivity": None,
                    "Size": None,
                    "PI": None,
                    "Baseline": None,
                    "Date": None,
                    "Stable at 8C": s8,
                    "Stable at 4C": s4,
                    "Tube Volume (mL)": None
                })

            row += 1
            continue

        if re.match(r"Day\s*\d+", cell, re.IGNORECASE):
            row_data = {"SampleID": last_formulation["SampleID"]} if last_formulation else {}
            row_data["Dilution"] = last_dilution
            row_data["Day"] = cell.strip()
            stars = ["*" for i in range(column_map.get("Foam Texture", 0) + 1, df.shape[1]) if "*" in str(df.iat[row, i])]
            row_data["Baseline"] = ", ".join(stars) if stars else None
            for offset, label in enumerate(["Date", "Foam (cc)", "Foam Texture", "Water (cc)", "Zeta", "Conductivity", "Size", "PI"]):
                col_idx = column_map.get(label)
                if label == "Date" and col_idx is None:
                    col_idx = 1
                val = str(df.iat[row, col_idx]).strip() if col_idx is not None and col_idx < df.shape[1] else None
                if not val or val.lower() == "nan":
                    row_data[label] = None
                else:
                    if label in ["Foam (cc)", "Water (cc)", "Zeta", "Conductivity", "Size", "PI"]:
                        num = re.search(r"[-+]?\d+\.?\d*", val)
                        row_data[label] = float(num.group()) if num else None
                    else:
                        row_data[label] = val

            if row + 1 < df.shape[0]:
                next_row = df.iloc[row + 1]
                non_empty = next_row.dropna()
                if len(non_empty) == 1 and column_map.get("Foam Texture") in non_empty.index:
                    extra_texture = str(non_empty.values[0]).strip()
                    if extra_texture:
                        existing = row_data.get("Foam Texture", "")
                        row_data["Foam Texture"] = f"{existing}, {extra_texture}".strip(", ")
                    row += 1

            current_dilution_rows.append(row_data)
            row += 1
            continue

        row_text_combined = ",".join(df.iloc[row].fillna("").astype(str)).lower()
        dilution_search = re.search(r"(\d+\s*X)", row_text_combined, re.IGNORECASE)
        if dilution_search:
            flush_current_dilution()
            base_dilution = dilution_search.group(1).replace(" ", "").upper()
            extra_label = []
            tube_volume = ""

            for col in range(1, 6):
                if col < df.shape[1]:
                    val = str(df.iat[row, col]).strip()
                    if val and val.lower() != "nan":
                        if "ml" in val.lower():
                            tube_volume = val
                        else:
                            extra_label.append(val)

            last_dilution = base_dilution + (" " + " ".join(extra_label) if extra_label else "")
            last_tube_volume = tube_volume

            if "foam" in row_text_combined:
                header_row = df.iloc[row]
                row += 1
            elif row + 1 < df.shape[0] and "foam" in ",".join(df.iloc[row + 1].fillna("").astype(str)).lower():
                header_row = df.iloc[row + 1]
                row += 2
            else:
                row += 1
                continue

            column_map = {}
            for i, val in header_row.items():
                val = str(val).strip().lower()
                if "foam amount" in val or ("foam" in val and "cc" in val):
                    column_map["Foam (cc)"] = i
                elif "foam texture" in val or "texture" in val:
                    column_map["Foam Texture"] = i
                elif "zeta" in val:
                    column_map["Zeta"] = i
                elif "pi" in val:
                    column_map["PI"] = i
                elif "conductivity" in val:
                    column_map["Conductivity"] = i
                elif "size" in val:
                    column_map["Size"] = i
                elif "Water (cc)" in val:
                    column_map["Water (cc)"] = i
                elif "date" in val:
                    column_map["Date"] = i
            continue

        row += 1

    flush_current_dilution()
    return samples, formulations


def process_dilution(dilution):
    pilot = np.nan
    temp_foam = np.nan
    ini_foam = "5cc"
    ratio = np.nan
    sonic=np.nan

    if pd.isna(dilution):
        return pilot, temp_foam, ini_foam, dilution

    text = str(dilution)

    # Extract and remove AFC
    if "AFC" in text:
        pilot = "AFC"
        #text = re.sub(r"\bAFC\b", "", text, flags=re.IGNORECASE)

#   Extract sonicated
    match_sonic = re.search(r"sonicated", text, flags=re.IGNORECASE)
    if match_sonic:
        #sonic = match_sonic.group(1)
        sonic = True
    elif re.search(r"no\s*sonic", text, flags=re.IGNORECASE):
        sonic = False

#   Extract ratio
    match_ratio = re.search(r"\(?(\d):(\d)\)?\s*ratio", text, re.IGNORECASE)
    if match_ratio:
        x, y = int(match_ratio.group(1)), int(match_ratio.group(2))
        ratio = round(y / x, 3) if x != 0 else np.nan

    # Extract and remove Xcc -> Water
    xcc_match = re.search(r"(\d+)\s*cc", text, flags=re.IGNORECASE)
    if xcc_match:
        ini_foam = xcc_match.group(1).strip()
        #text = re.sub(r"(\d+)\s*cc", "", text, flags=re.IGNORECASE)

    # Extract and remove Xc (single c or C) -> Temp Foam Monitoring
    xc_match = re.search(r"(\d+)\s*c(?!c)", text, flags=re.IGNORECASE)
    if xc_match:
        temp_foam = int(xc_match.group(1))
        # text = re.sub(r"(\d+)\s*c(?!c)", "", text, flags=re.IGNORECASE) with Removing
    cleaned_text = text.strip(" ,;-").strip()
    return pilot, temp_foam, ini_foam, cleaned_text, ratio, sonic

def assign_pilot_column(df):
    df["Pilot"] = df["SampleID"].apply(lambda x: "AFC" if pd.notna(x) and "AFC" in str(x).upper() else None)
    df["Pilot"] = df["Dilution"].apply(lambda x: "AFC" if pd.notna(x) and "AFC" in str(x).upper() else None)

    return df

