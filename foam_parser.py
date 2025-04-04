import streamlit as st
import pandas as pd
import re
import os

def parse_formulation(text):
    data = {
        "SampleID": re.search(r"\((.*?)\)", text).group(1).strip() if re.search(r"\((.*?)\)", text) else None,
        "HS (%)": None, "Critic (%)": None, "CapB (%)": None,
        "CapB (ppm)": None, "APG (%)": None
    }
    for p in text.split(','):
        p = p.strip()
        if "ppm" in p.lower():
            m = re.match(r"(\d+\.?\d*)\s*ppm\s*(.*)", p, re.IGNORECASE)
            if m and "capb" in m.group(2).lower():
                data["CapB (ppm)"] = float(m.group(1))
        else:
            m = re.match(r"(\d+\.?\d*)%\s*(.*)", p)
            if m:
                val, chem = m.groups()
                chem = chem.lower().strip()
                if "hs" in chem:
                    data["HS (%)"] = float(val)
                elif "citric" in chem:
                    data["Critic (%)"] = float(val)
                elif "capb" in chem:
                    data["CapB (%)"] = float(val)
                elif "apg" in chem:
                    data["APG (%)"] = float(val)
    return data

def process_file(df):
    samples = []
    row = 0
    last_formulation = None
    last_dilution = None
    column_map = {}

    while row < df.shape[0]:
        cell = str(df.iat[row, 0])

        if re.search(r"\([A-Za-z0-9\s\-]+\)", cell):
            last_formulation = parse_formulation(cell)
            for offset in range(1, 5):
                if row + offset < df.shape[0]:
                    dline = str(df.iat[row + offset, 0]).strip()
                    if re.search(r"\d+\s*X", dline, re.IGNORECASE):
                        last_dilution = re.search(r"\d+\s*X", dline, re.IGNORECASE).group(0).replace(" ", "")
                        header_row_index = row + offset + 1
                        if header_row_index < df.shape[0]:
                            header_row = df.iloc[header_row_index]
                            column_map = {}
                            for i, val in header_row.items():
                                val = str(val).strip().lower()
                                if "foam amount" in val:
                                    column_map["Foam (cc)"] = i
                                elif "foam texture" in val:
                                    column_map["Foam Texture"] = i
                                elif "zeta" in val:
                                    column_map["Zeta"] = i
                                elif "pi" == val or "pi" in val:
                                    column_map["PI"] = i
                                elif "conductivity" in val:
                                    column_map["Conductivity"] = i
                                elif "size" in val:
                                    column_map["Size"] = i
                                elif "liquid" in val:
                                    column_map["Liquid Amount"] = i
                                elif "date" in val:
                                    column_map["Date"] = i
                        break
            row += 1
            continue

        if re.match(r"Day\s*\d+", cell, re.IGNORECASE):
            row_data = last_formulation.copy() if last_formulation else {}
            row_data["Dilution"] = last_dilution
            row_data["Day"] = cell.strip()

            for label in ["Date", "Foam (cc)", "Foam Texture", "Liquid Amount", "Zeta", "Conductivity", "Size", "PI"]:
                col_idx = column_map.get(label)
                val = str(df.iat[row, col_idx]).strip() if col_idx is not None and col_idx < df.shape[1] else None
                if not val or val.lower() == "nan":
                    row_data[label] = None
                else:
                    if label in ["Foam (cc)", "Liquid Amount", "Zeta", "Conductivity", "Size", "PI"]:
                        num = re.search(r"[-+]?\d+\.?\d*", val)
                        row_data[label] = float(num.group()) if num else None
                    else:
                        row_data[label] = val

            samples.append(row_data)

        row += 1

    columns = [
        "SampleID", "HS (%)", "Critic (%)", "CapB (%)", "CapB (ppm)", "APG (%)",
        "Dilution", "Day", "Date", "Foam (cc)", "Foam Texture", "Liquid Amount",
        "Zeta", "Conductivity", "Size", "PI"
    ]
    final_df = pd.DataFrame(samples, columns=columns)
    return final_df

# Streamlit GUI
st.title("🧪 Foam Sample CSV Parser")
st.write("Upload a messy foam sample CSV file to extract structured data.")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        result_df = process_file(df)
        st.success("✅ Extraction complete!")
        st.dataframe(result_df.head(50))

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Structured CSV", csv, "structured_foam_samples.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
