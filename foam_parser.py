import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

import streamlit as st

# Simple password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["auth"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password")
        st.stop()

check_password()

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
            row_data["Tube Volume"] = last_tube_volume
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
                    "Water": None,
                    "Zeta": None,
                    "Conductivity": None,
                    "Size": None,
                    "PI": None,
                    "Baseline": None,
                    "Date": None,
                    "Stable at 8C": s8,
                    "Stable at 4C": s4,
                    "Tube Volume": None
                })

            row += 1
            continue

        if re.match(r"Day\s*\d+", cell, re.IGNORECASE):
            row_data = {"SampleID": last_formulation["SampleID"]} if last_formulation else {}
            row_data["Dilution"] = last_dilution
            row_data["Day"] = cell.strip()
            stars = ["*" for i in range(column_map.get("Foam Texture", 0) + 1, df.shape[1]) if "*" in str(df.iat[row, i])]
            row_data["Baseline"] = ", ".join(stars) if stars else None
            for offset, label in enumerate(["Date", "Foam (cc)", "Foam Texture", "Water", "Zeta", "Conductivity", "Size", "PI"]):
                col_idx = column_map.get(label)
                if label == "Date" and col_idx is None:
                    col_idx = 1
                val = str(df.iat[row, col_idx]).strip() if col_idx is not None and col_idx < df.shape[1] else None
                if not val or val.lower() == "nan":
                    row_data[label] = None
                else:
                    if label in ["Foam (cc)", "Water", "Zeta", "Conductivity", "Size", "PI"]:
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
            tube_volume = None

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
                elif "water" in val:
                    column_map["Water"] = i
                elif "date" in val:
                    column_map["Date"] = i
            continue

        row += 1

    flush_current_dilution()
    return samples, formulations

# After Extraction
def process_dilution(dilution):
    pilot = np.nan
    temp_foam = np.nan
    water = np.nan

    if pd.isna(dilution):
        return pilot, temp_foam, water, dilution

    text = str(dilution)

    # Extract and remove AFC
    if "AFC" in text:
        pilot = "AFC"
        text = re.sub(r"\bAFC\b", "", text, flags=re.IGNORECASE)

    # Extract and remove Xcc -> Water
    xcc_match = re.search(r"(\d+)\s*cc", text, flags=re.IGNORECASE)
    if xcc_match:
        water = int(xcc_match.group(1))
        text = re.sub(r"(\d+)\s*cc", "", text, flags=re.IGNORECASE)

    # Extract and remove Xc (single c or C) -> Temp Foam Monitoring
    xc_match = re.search(r"(\d+)\s*c(?!c)", text, flags=re.IGNORECASE)
    if xc_match:
        temp_foam = int(xc_match.group(1))
        # text = re.sub(r"(\d+)\s*c(?!c)", "", text, flags=re.IGNORECASE) with Removing
    cleaned_text = text.strip(" ,;:-").strip()
    return pilot, temp_foam, water, cleaned_text

def assign_pilot_column(df):
    df["Pilot"] = df["SampleID"].apply(lambda x: "AFC" if pd.notna(x) and "AFC" in str(x).upper() else None)
    return df

# === Streamlit Web App ===
st.set_page_config(page_title="Foam Sample Parser", layout="centered")
st.title("🧪 Foam Sample Data Extractor")
st.markdown("Upload a CSV file containing foam sample data. The app will extract and display structured results.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file, header=None)
        samples, formulations = extract_samples_complete_fixed(df_input)

        df_samples = pd.DataFrame(samples)
        df_formulations = pd.DataFrame.from_dict(formulations, orient="index")
        df_formulations["SampleID"] = df_formulations.index

        final_df = df_samples.merge(df_formulations, on="SampleID", how="left")
        final_df = final_df.loc[:, final_df.columns.map(lambda col: col.count('%') < 2)]
        final_df = final_df.loc[:, final_df.columns.map(lambda col: col.count('(') < 2)]
        final_df = final_df.drop_duplicates()

        # Create the new columns with default NaN
        final_df["Initial Foam Volume (cc)"] = 5  # Set default value
        final_df["Pilot"] = np.nan
        final_df["Temp Foam Monitoring"] = np.nan
        final_df["Initial Foam Temp"] = np.nan  # No logic yet, reserved
        final_df["Water (cc)"] = np.nan  # Ensure 'Water' column exists

        # Apply the processing
        final_df[["Pilot", "Temp Foam Monitoring", "Water (cc)", "Dilution"]] = final_df.apply(
            lambda row: pd.Series(process_dilution(row["Dilution"])),
            axis=1
        )
        final_df["Tube Volume (mL)"] = final_df["Tube Volume (mL)"].astype(str).str.replace(r"mL\s*tube", "", case=False, regex=True).str.strip()
        final_df = assign_pilot_column(final_df)
        final_df = final_df.drop_duplicates()

        final_df = final_df.replace({None: np.nan})

        st.success("✅ Parsing complete!")
        st.dataframe(final_df)
 
        # Prepare download
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Parsed Data",
            data=csv,
            file_name="Parsed_Foam_Data.csv",
            mime="text/csv"
        )
        # SampleID search box
        if "final_df" in locals():
            st.markdown("### 🔍 Search for a SampleID")
            search_id = st.text_input("Enter SampleID to search:")

            if search_id:
                filtered_df = final_df[final_df["SampleID"].str.contains(search_id, case=False, na=False)]
                if not filtered_df.empty:
                    st.dataframe(filtered_df)
                else:
                    st.warning(f"No matching SampleID found for: {search_id}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("👈 Upload a CSV file to begin.")
