import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import sys


if __name__=="__main__":
    try:
        session = ort.InferenceSession("model.onnx")
    except Exception as e:
        print("Error: ",e)
        sys.exit(1)

    input_info = session.get_inputs()
    number_of_features = input_info[0].shape[-1] #TODO - do it better, more error proof. Only 1d supported
    # print(number_of_features)
    # for i in input_info:
    #     print(f"Nazwa wejścia: {i.name}")
    #     print(f"Kształt (shape): {i.shape}")
    #     print(f"Typ danych: {i.type}")

    # outputs = session.get_outputs()
    # for o in outputs:
    #     print(f"Nazwa wyjścia: {o.name}")
    #     print(f"Kształt wyjścia: {o.shape}")
    x_options = [x for x in range(number_of_features)]
    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("x axis feature",options=x_options)

    y_options = [opt for opt in x_options if opt != x_axis]

    with col2:
        y_axis = st.selectbox("y axis feature",options=y_options)

    remaining_options = [opt for opt in y_options if opt != y_axis]

    state_values = [0] * number_of_features
    
    for feature in remaining_options:
        key_min = f"min_{feature}"
        key_max = f"max_{feature}"
        key_sld = f"sld_{feature}"
        col1,col2,col3 = st.columns([1,2,1])

        with col1:
            min_v = st.number_input("Min",value=-1.0,key=key_min,step=0.1)
        with col3:
            max_v = st.number_input("Max",value=1.0,key=key_max,step=0.1)
        with col2:
            if min_v >= max_v:
                st.warning("Min >= Max")
                state_values[feature] = min_v
            else:
                state_values[feature] = st.slider(f"Feature {feature}",float(min_v),float(max_v),value=0.0,key=key_sld)