import streamlit as st
import torch
import numpy as np
import pandas as pd
from mg import SimpleCNN, federated_averaging, get_non_iid_subsets, train_local_model, test_model

# --- VISUALIZATION LOGIC ---
def display_distribution(train_dataset, client_indices):
    """
    Higher-Order Visualization: Shows the 'Silos' of data.
    This proves to the professor that the data is truly Non-IID.
    """
    st.subheader("📊 Data Distribution Across Clients")
    dist_data = []
    
    for i, indices in enumerate(client_indices):
        # Get labels for this specific client
        labels = train_dataset.targets[indices].numpy()
        counts = np.bincount(labels, minlength=10)
        dist_data.append(counts)
    
    # Create a Relational Table for Streamlit
    df = pd.DataFrame(dist_data, 
                      index=[f"Client {i+1}" for i in range(len(client_indices))],
                      columns=[f"Digit {i}" for i in range(10)])
    
    st.bar_chart(df)
    st.caption("Each client 'sees' a biased subset of digits. This is the core Federated challenge.")

# --- DASHBOARD UI SETUP ---
st.set_page_config(page_title="FedAvg Research Lab", layout="wide")

st.title("🌐 Federated Learning: FedAvg Dashboard")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Experiment Settings")
num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
rounds = st.sidebar.slider("Communication Rounds", 1, 20, 5)

# --- THE EXECUTION ENGINE ---
if st.sidebar.button("🚀 Start Federation"):
    st.info(f"Starting {rounds} rounds of Federation with {num_clients} clients...")
    
    # 1. Relational Priming: Get Data and Show Distribution
    train_dataset, client_indices = get_non_iid_subsets(num_clients)
    display_distribution(train_dataset, client_indices)
    
    # 2. Initialize the Global 'Brain'
    global_model = SimpleCNN()
    accuracy_history = []
    
    # UI Placeholders for real-time updates
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    status_text = st.empty()

    # 3. The Communication Rounds
    for r in range(rounds):
        status_text.text(f"Round {r+1}: Training {num_clients} Clients...")
        local_weights_list = []
        client_sizes = []

        for i in range(num_clients):
            # Each client gets a fresh copy of the current global model
            local_model = SimpleCNN()
            local_model.load_state_dict(global_model.state_dict())
            
            # Train on the client's local, biased data
            weights = train_local_model(local_model, train_dataset, client_indices[i])
            
            local_weights_list.append(weights)
            client_sizes.append(len(client_indices[i]))
            
            # VRAM Guard: Keep your HP Victus memory clean
            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4. SERVER AGGREGATION: The FedAvg Logic
        new_global_weights = federated_averaging(local_weights_list, client_sizes)
        global_model.load_state_dict(new_global_weights)
        
        # 5. EVALUATION: Test the combined intelligence
        acc = test_model(global_model)
        accuracy_history.append(acc)
        
        # Update UI components
        progress_bar.progress((r + 1) / rounds)
        chart_placeholder.line_chart(accuracy_history)
        st.write(f"✅ Round {r+1} Accuracy: **{acc:.2f}%**")

    st.success("🎯 Federation complete! The model has synthesized knowledge from all silos.")
