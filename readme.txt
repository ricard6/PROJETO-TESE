#Correr o venv
.\.venv\Scripts\activate

#Correr a API (backend)#
uvicorn main:app --reload

#Correr frontend Streamlit#
streamlit run Discussion_Summary.py   