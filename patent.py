import streamlit as st
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Smart Patent Search", layout="wide")
st.title("🔍 Smart Patent Search")

# Step 1: Get user input
keyword = st.text_input("Enter keyword(s) to search patents:", "foam surfactant oilfield")
num_results = st.number_input("How many results to fetch?", min_value=1, max_value=50, value=10, step=1)

@st.cache_data(show_spinner=False)
def get_google_patents_links(keyword, max_results):
    query = keyword.replace(" ", "+")
    url = f"https://patents.google.com/?q={query}&num={max_results}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        if "/patent/" in a['href'] and "/en" in a['href']:
            full_url = "https://patents.google.com" + a['href']
            if full_url not in links:
                links.append(full_url)
            if len(links) >= max_results:
                break
    return links

@st.cache_data(show_spinner=False)
def extract_claims_from_google_patent_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    claims = soup.find_all("div", {"class": "claim"})
    return [c.text.strip() for c in claims]

if st.button("Search and Extract Claims"):
    with st.spinner("Fetching patents and extracting claims..."):
        links = get_google_patents_links(keyword, num_results)
        all_claims = []
        full_text = ""

        for idx, url in enumerate(links):
            claims = extract_claims_from_google_patent_url(url)
            all_claims.append((url, claims))

            full_text += f"Patent {idx+1}: {url}\n"
            for i, c in enumerate(claims[:5]):
                full_text += f"  Claim {i+1}: {c}\n"
            full_text += "\n"

        # Display the results
        for url, claims in all_claims:
            st.markdown(f"### 🔗 [{url}]({url})")
            for i, c in enumerate(claims[:5]):
                st.write(f"**Claim {i+1}:** {c}")

        # Save to file
        st.download_button("📄 Download Claims as .txt", full_text.encode("utf-8"), file_name="patent_claims.txt")
