import streamlit as st

def main():
    st.title("LPBF AlSi10Mg Analysis")
    st.write("Welcome to the LPBF Analysis Tool")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

if __name__ == "__main__":
    main()
