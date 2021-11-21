import random
import numpy as np
import pandas as pd
import streamlit as st

from src.cb_t import Solver
from src.config import CBSolverConfig

solver=Solver(CBSolverConfig)
df = None
st.subheader('Файл для обработки')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    solver.upload_df(uploaded_file)
    tdf=solver.preproc_df()
    df=solver.predict(tdf)


st.subheader('Обработанный файл')
if df is not None:
    st.dataframe(df, height=250)