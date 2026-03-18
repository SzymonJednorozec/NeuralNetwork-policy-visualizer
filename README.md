# NeuralNetwork-policy-visualizer

An interactive Explainable AI dashboard built with Streamlit to visualize and interpret neural network decision boundaries.
 
---
 
## 🎯 Overview
 
This simple tool is designed to provide a high-resolution heatmaps that map the agent's policy across a 2D spatial area. It was originally developed to debug and optimize a custom-built Deep-Q-Pong agent.
 
---
 
## 🚀 Key Features
 
- **Universal Model Support** - Compatible with any model exported to ONNX format, making it framework-agnostic (PyTorch, TensorFlow, etc.).
- **Dynamic Dimension Slicing** - Select any two input parameters as axes for the 2D heatmap.
- **Interactive Environment Control** - Real-time manipulation of non-spatial variables via dynamic UI sliders to observe policy shifts.
- **XAI Insights** - Helps identify dead zones, reward-shaping artifacts, and decision instabilities in trained agents.
 
---
 
## 🧠 How it Works
  
| Step | Description |
|---|---|
| **1. Input Parsing** | Automatically detects the number of input features from the ONNX graph |
| **2. Grid Generation** | Creates a dense grid of points for the selected X and Y coordinates |
| **3. Vectorized Inference** | The entire grid is processed in a single batch through the model |
| **4. Heatmap Rendering** | `contourf` or `imshow` from Matplotlib draws the decision boundaries |
 
---
 
## 🛠️ Tech Stack
 
| Category | Technology |
|---|---|
| **Interface** | Streamlit |
| **Inference Engine** | ONNX Runtime |
| **Visualization** | Matplotlib, NumPy |
| **Model Graph Parsing** | ONNX (Python API) |
 
---
 
## 🏗️ Installation & Usage
 

 
### 1. Install dependencies
 
```bash
pip install -r requirements.txt
```
### 2. Make sure that .onnx and .onnx.data model files are in the same directory with main.py
 
### 3. Run the app
 
```bash
streamlit run app.py
```
