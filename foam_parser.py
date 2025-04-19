import streamlit as st
import pandas as pd
import numpy as np
import myUtility
import re
from io import BytesIO

st.set_page_config(page_title="Foam (no Oil) Sample Extractor")
st.title("Foam (no Oil) Sample Extractor")

# Upload section
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)  # default utf-8
    except UnicodeDecodeError:
        # Try with alternative encoding
        uploaded_file.seek(0)  # reset file pointer
        df = pd.read_csv(uploaded_file, encoding="cp1252")  # or "ISO-8859-1"
    
    st.write("Preview of the uploaded file:")
    st.dataframe(df)
if uploaded_file is not None:
    st.success("Parsing Complete")
    df_input = pd.read_csv(uploaded_file, header=None, encoding="cp1252")

    # Apply your utility functions
    samples, formulations = myUtility.extract_samples_complete_fixed(df_input)
    df_samples = pd.DataFrame(samples)
    df_formulations = pd.DataFrame.from_dict(formulations, orient="index")
    df_formulations["SampleID"] = df_formulations.index
    final_df = df_samples.merge(df_formulations, on="SampleID", how="left")

    final_df["Initial Foam Volume (cc)"] = "5cc"
    final_df["Pilot"] = np.nan
    final_df["Temp Foam Monitoring"] = np.nan
    final_df["Initial Foam Temp"] = np.nan
    final_df["Water (cc)"] = np.nan
    final_df["Sonicated"] = np.nan

    final_df[["Pilot", "Temp Foam Monitoring", "Initial Foam Volume (cc)", "Dilution", "Ratio", "Sonicated"]] = final_df.apply(
        lambda row: pd.Series(myUtility.process_dilution(row["Dilution"])), axis=1
    )

    final_df["Tube Volume (mL)"] = final_df["Tube Volume (mL)"].astype(str).str.replace(r"mL\\s*tube", "", case=False, regex=True).str.strip()
    final_df = myUtility.assign_pilot_column(final_df)
    final_df = final_df.drop_duplicates()

    final_df["time"] = None
    final_df = final_df.replace({None: np.nan})

    # Save to CSV in memory
    parsed_buffer = BytesIO()
    final_df.to_csv(parsed_buffer, index=False)
    parsed_buffer.seek(0)

    # Process for single line transformation
    df_input=final_df
    df_input = df_input[df_input["Day"].notna()]
    df_input["Day_Num"] = df_input["Day"].str.extract(r'(\\d+)').astype(int)
    max_day = df_input["Day_Num"].max()

    formulation_cols = [
        col for col in df_input.columns if col not in ["SampleID", "Day", "Day_Num", "Foam (cc)", "Foam Texture", "Date", "Baseline", "Pilot"]
    ]

    output_rows = []
    for (sample_id, dilution), group in df_input.groupby(["SampleID", "Dilution"]):
        row = {"SampleID": sample_id, "Dilution": dilution}

        for col in formulation_cols:
            if col in group.columns:
                row[col] = group[col].dropna().iloc[0] if not group[col].dropna().empty else np.nan

        day0_row = group[group["Day_Num"] == 0]
        row["Date"] = day0_row["Date"].iloc[0] if not day0_row.empty else np.nan
        row["Baseline"] = "*" if group["Baseline"].astype(str).str.contains(r"\\*").any() else ""
        pilot_val = group["Pilot"].dropna()
        row["Pilot"] = pilot_val.iloc[0] if not pilot_val.empty else ""

        for day in range(max_day + 1):
            day_row = group[group["Day_Num"] == day]
            row[f"Day {day} - Amount (cc)"] = day_row["Foam (cc)"].values[0] if not day_row.empty else np.nan
            row[f"Day {day} - Foam Texture"] = day_row["Foam Texture"].values[0] if not day_row.empty else ""

        output_rows.append(row)

    df_transformed_fixed = pd.DataFrame(output_rows)
    transformed_buffer = BytesIO()
    df_transformed_fixed.to_csv(transformed_buffer, index=False)
    transformed_buffer.seek(0)

    # Display table
    st.subheader("Transformed Parsed Data")
    st.dataframe(df_transformed_fixed)

    # Search
    st.subheader("Search Sample")
    search_term = st.text_input("Enter Sample ID")
    if search_term:
        st.dataframe(df_transformed_fixed[df_transformed_fixed["SampleID"].astype(str).str.contains(search_term, case=False)])

    # Download buttons
    st.download_button("Download Multiple Line", parsed_buffer, file_name="Parsed_Foam_Data.csv", mime="text/csv")
    st.download_button("Download Single Line", transformed_buffer, file_name="Transformed_Parsed_2.csv", mime="text/csv")
