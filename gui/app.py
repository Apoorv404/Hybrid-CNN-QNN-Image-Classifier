import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pennylane as qml


# ---- Custom CSS Styling ----
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #ffffff;
        }
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        [data-testid="stSidebar"] {
            background-color: rgba(15, 32, 39, 0.9);
        }
        h1, h2, h3 {
            color: #8ef6e4;
            text-align: center;
        }
        .predictionBox {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1CB5E0, #000851);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #76b852, #8DC26F);
            color: black;
        }
    </style>
""", unsafe_allow_html=True)


# ---- Quantum Circuit ----
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[1, 2])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ---- Hybrid CNN + QNN Model ----
class HybridCNNQNN(nn.Module):
    def __init__(self):
        super(HybridCNNQNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        dummy_input = torch.randn(1, 3, 128, 128)
        x = self.pool1(self.relu(self.conv1(dummy_input)))
        x = self.pool2(self.relu(self.conv2(x)))
        flattened_size = x.view(x.size(0), -1).size(1)

        self.fc_link = nn.Linear(flattened_size, 4)

        weight_shapes = {"weights": n_qubits}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.fc_out = nn.Linear(4, 3)  # 3 classes

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc_link(x)
        x = self.quantum_layer(x)
        return self.fc_out(x)


# ---- Load Model ----
@st.cache_resource
def load_model():
    model = HybridCNNQNN()
    model.load_state_dict(torch.load("hybrid_quantum_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()


# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class_labels = ["Disease", "Dry_Leaf", "Healthy"]


#  ---- Sidebar ----
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("## 🌿 Hybrid Quantum Classifier")
    st.markdown("""
    **Developed by:** Apoorv Jadhav  
    **Model:** Hybrid CNN + QNN  
    **Frameworks:** PyTorch + PennyLane  
    """)
    st.markdown("---")
    st.info("Navigate between tabs for prediction, architecture, and metrics.")


# ---- Main UI Tabs ----
st.title("🌿 Bay Leaf Classification using Hybrid CNN + QNN")

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "🧠 Model Architecture", "📈 Model Metrics"])

# ---- TAB 1: Prediction ----
with tab1:
    st.write("Upload one or more bay leaf images to classify them into **Healthy**, **Disease**, or **Dry Leaf**.")

    uploaded_files = st.file_uploader(
        "📷 Upload bay leaf image(s)...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.markdown(f"### 📸 File: `{uploaded_file.name}`")

            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            img_tensor = transform(image).unsqueeze(0)

            with st.spinner("Running quantum inference..."):
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    label = class_labels[predicted.item()]
                    confidence_score = confidence.item() * 100

            st.markdown("<div class='predictionBox'>", unsafe_allow_html=True)
            st.markdown(f"### ✅ Prediction: **{label}**")
            st.markdown(f"**Confidence:** {confidence_score:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            # 📊 Probabilities chart + JSON
            st.markdown("### 📊 Class Probabilities")
            chart_data = {class_labels[i]: probs[0][i].item() for i in range(len(class_labels))}
            st.bar_chart(chart_data)

            prob_dict = {class_labels[i]: f"{probs[0][i].item()*100:.2f}%" for i in range(len(class_labels))}
            st.subheader("📄 Probability Details (JSON)")
            st.json(prob_dict)

            st.download_button(
                f"💾 Download Probabilities for {uploaded_file.name}",
                str(prob_dict),
                file_name=f"{uploaded_file.name}_prediction.json"
            )


# ---- TAB 2: Model Architecture ----
with tab2:
    st.image("model_architecture.png", caption="Hybrid CNN + QNN Architecture", use_container_width=True)
    st.markdown("""
    **Description:**  
    - CNN feature extractor with two convolutional layers  
    - Fully connected bridge to Quantum Layer (4 qubits)  
    - Quantum circuit with Angle Embedding and RY rotations  
    - Final dense layer for 3-class classification  
    """)

# ---- TAB 3: Model Metrics ----
with tab3:
    st.image(
        ["training.png", "validation.png", "confusion_matrix.png"],
        caption=["Training Curve", "Validation Curve", "Confusion Matrix"],
        use_container_width=True
    )
    st.markdown("""
    **Metrics Overview:**  
    - Training and validation loss/accuracy plots  
    - Confusion matrix showing class-wise performance  
    """)
