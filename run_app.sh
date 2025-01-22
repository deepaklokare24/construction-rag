#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run streamlit_app.py --server.port 8501 --server.address localhost 