import vid
import main_page
import report
import streamlit as st

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
PAGES = {
    "HOMEPAGE": main_page,
    "DEMO": vid,
    "REPORT" : report
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()