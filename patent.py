import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io

st.set_page_config(page_title="Smart Patent Search", layout="wide")
st.title("🔍 Smart Patent Search")

st.markdown("### Upload CSV or Enter Patent URLs Below")
upload_option = st.radio("Choose input method:", ["📁 Upload CSV", "🔗 Enter URLs manually"])

url_input = ""

# Option 1: CSV Upload
if upload_option == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with 'result link' column", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, skiprows=1)  # skip the first row
            if "result link" not in df.columns:
                st.error("CSV must contain a 'result link' column.")
            else:
                urls = df["result link"].dropna().tolist()
                url_input = "\n".join(urls)
                st.success(f"{len(urls)} URLs extracted from CSV.")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# Option 2: Manual Entry
if upload_option == "🔗 Enter URLs manually":
    url_input = st.text_area("Enter Google Patent URLs (one per line):", height=200)

@st.cache_data(show_spinner=False)
def extract_claims_from_google_patent_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        claims = soup.find_all("div", {"class": "claim"})
        return [c.text.strip() for c in claims]
    except Exception as e:
        return [f"Error retrieving claims from {url}: {str(e)}"]

# Display extracted URLs if from CSV
if upload_option == "📁 Upload CSV" and url_input:
    st.text_area("Extracted Patent URLs from CSV:", url_input, height=200)

# Extract Claims Button
if st.button("Extract Claims"):
    urls = [line.strip() for line in url_input.strip().split("\n") if line.strip()]
    if not urls:
        st.warning("Please enter at least one patent URL.")
    else:
        with st.spinner("Extracting claims from provided patents..."):
            all_claims = []
            full_text = ""

            for idx, url in enumerate(urls):
                claims = extract_claims_from_google_patent_url(url)
                all_claims.append((url, claims))

                full_text += f"Patent {idx+1}: {url}\n"
                for i, c in enumerate(claims[:5]):
                    full_text += f"  Claim {i+1}: {c}\n"
                full_text += "\n"

            for url, claims in all_claims:
                st.markdown(f"### 🔗 [{url}]({url})")
                for i, c in enumerate(claims[:5]):
                    st.write(f"**Claim {i+1}:** {c}")

            st.download_button("📄 Download Claims as .txt", full_text.encode("utf-8"), file_name="patent_claims.txt")
