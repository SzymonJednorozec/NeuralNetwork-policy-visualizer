import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import onnxruntime as ort
import sys

SIZE = 70

if __name__=="__main__":
    try:
        session = ort.InferenceSession("model.onnx")
    except Exception as e:
        print("Error: ",e)
        sys.exit(1)

    input_info = session.get_inputs()[0]
    number_of_features = input_info.shape[-1] #TODO - do it better, more error proof. Only 1d supported
    input_name = input_info.name

    with st.sidebar.expander("Name features"):
        feature_names = []
        for i in range(number_of_features):
            name = st.text_input(f"index {i}",value=f"feature {i}",key=f"feat_{i}")
            feature_names.append(name)
        
    num_outputs = session.get_outputs()[0].shape[-1]
    output_names = []

    if num_outputs > 1:
        with st.sidebar.expander("Name outputs"):
            for i in range(num_outputs):
                name = st.text_input(f"index {i}:", value=f"output {i}", key=f"out_{i}")
                output_names.append(name)
    else:
        output_names = ["Value"]

    col1, col2, col3, col4, col5, col6 = st.columns([1,3,1,1,3,1])
    with col1:
        min_x = st.number_input("Min x",value=-1.0,step=0.1)
    with col3:
        max_x = st.number_input("Max x",value=1.0,step=0.1)
    with col2:
        x_name = st.selectbox("x axis feature",options=feature_names)
        x_axis = feature_names.index(x_name)
    
    with col4:
        min_y = st.number_input("Min y",value=-1.0,step=0.1)
    with col6:
        max_y = st.number_input("Max y",value=1.0,step=0.1)
    with col5:
        y_options = [opt for opt in feature_names if opt != x_name]
        y_name = st.selectbox("y axis feature",options=y_options)
        y_axis = feature_names.index(y_name)

    remaining_indices = [i for i in range(number_of_features) if i != x_axis and i != y_axis]

    state_values = [0] * number_of_features
    
    for feature_idx in remaining_indices:
        key_min = f"min_{feature_idx}"
        key_max = f"max_{feature_idx}"
        key_sld = f"sld_{feature_idx}"
        col1,col2,col3 = st.columns([1,2,1])

        with col1:
            min_v = st.number_input("Min",value=-1.0,key=key_min,step=0.1)
        with col3:
            max_v = st.number_input("Max",value=1.0,key=key_max,step=0.1)
        with col2:
            if min_v >= max_v:
                st.warning("Min >= Max")
                state_values[feature_idx] = min_v
            else:
                state_values[feature_idx] = st.slider(feature_names[feature_idx],float(min_v),float(max_v),value=0.0,key=key_sld)
        

    x_range = np.linspace(min_x,max_x,SIZE)
    y_range = np.linspace(min_y,max_y,SIZE)
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
    
    flip_y = st.sidebar.checkbox("Invert Y axis", value=False)

    if num_outputs > 1:
        cmap = plt.get_cmap('tab10', num_outputs)
        im = ax.imshow(z_grid, extent=[min_x, max_x, min_y, max_y], 
                       origin='lower', cmap=cmap, interpolation='nearest', aspect='auto')
        
        cbar = fig.colorbar(im, ticks=range(num_outputs))
        cbar.ax.set_yticklabels(output_names)
    else:
        c = ax.contourf(xx, yy, z_grid, cmap='RdYlBu')
        fig.colorbar(c)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    if flip_y:
        ax.invert_yaxis()

    st.pyplot(fig)



            
            