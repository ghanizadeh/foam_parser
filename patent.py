import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io

st.set_page_config(page_title="Smart Patent Search (only Claims)", layout="wide")
st.title("🔍 Smart Patent Search")

st.markdown("### Upload CSV or Enter Patent URLs Below")
upload_option = st.radio("Choose input method:", ["📁 Upload CSV", "🔗 Enter URLs manually"])

url_input = ""

# Option 1: CSV Upload
if upload_option == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, skiprows=1)
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
def extract_title_and_claims_from_google_patent_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract Title
        title_tag = soup.find("span", {"itemprop": "title"})
        title = title_tag.text.strip() if title_tag else "No title found"

        # Extract Claims
        claims = soup.find_all("div", {"class": "claim"})
        claim_texts = [c.text.strip() for c in claims]

        return title, claim_texts

    except Exception as e:
        return "Error", [f"Error retrieving data from {url}: {str(e)}"]

# Display extracted URLs if from CSV
if upload_option == "📁 Upload CSV" and url_input:
    st.text_area("Extracted Patent URLs from CSV:", url_input, height=200)

# Extract Claims Button
if st.button("Extract Claims"):
    urls = [line.strip() for line in url_input.strip().split("\n") if line.strip()]
    if not urls:
        st.warning("Please enter at least one patent URL.")
    else:
        with st.spinner("Extracting patent titles and claims..."):
            all_data = []
            full_text = ""

            for idx, url in enumerate(urls):
                title, claims = extract_title_and_claims_from_google_patent_url(url)
                all_data.append((url, title, claims))

                full_text += f"Patent {idx+1}: {url}\n"
                full_text += f"Title: {title}\n"
                for i, c in enumerate(claims[:100]):
                    full_text += f"  Claim {i+1}: {c}\n"
                full_text += "\n"

            for url, title, claims in all_data:
                st.markdown(f"### 🔗 [{url}]({url})")
                st.markdown(f"**Title: {title}**")
                for i, c in enumerate(claims[:100]):
                    st.write(f"**Claim {i+1}:** {c}")

            st.download_button("📄 Download Claims.txt", full_text.encode("utf-8"), file_name="patent_claims.txt")
