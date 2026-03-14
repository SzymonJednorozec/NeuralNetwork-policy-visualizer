import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import sys

SIZE = 50

if __name__=="__main__":
    try:
        session = ort.InferenceSession("model.onnx")
    except Exception as e:
        print("Error: ",e)
        sys.exit(1)

    input_info = session.get_inputs()[0]
    number_of_features = input_info.shape[-1] #TODO - do it better, more error proof. Only 1d supported
    input_name = input_info.name
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
        
    if st.button("Show graph"):
        x_range = np.linspace(0,1,SIZE)
        y_range = np.linspace(0,1,SIZE)
        xx, yy = np.meshgrid(x_range, y_range)
        flat_x, flat_y = xx.ravel(), yy.ravel()

        input_data = np.zeros(shape=(SIZE*SIZE,number_of_features))

        for i in range(number_of_features):
            if i==x_axis:
                input_data[:,i] = flat_x
            elif i==y_axis:
                input_data[:,i] = flat_y
            else:
                input_data[:,i] = state_values[i]

        input_data = input_data.astype(np.float32)
        outputs = session.run(None, {input_name: input_data})
        prediction = outputs[0]
        if prediction.shape[1] > 1:
            z_val = np.argmax(prediction,axis=1)
        else:
            z_val = prediction.flatten()
        z_grid = z_val.reshape(SIZE,SIZE)

        fig, ax = plt.subplots()
        c = ax.contourf(xx, yy, z_grid, cmap='RdYlBu')
        fig.colorbar(c)
        st.pyplot(fig)



            
            