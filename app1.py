import streamlit as st
import torch
import numpy as np
import pandas as pd
from mg import SimpleCNN, federated_averaging, get_non_iid_subsets, train_local_model, test_model

def display_distribution(train_dataset, client_indices):
    st.subheader("📊 Data Distribution Across Clients")
    dist_data = []
    for i, indices in enumerate(client_indices):
        labels = train_dataset.targets[indices].numpy()
        counts = np.bincount(labels, minlength=10)
        dist_data.append(counts)
    
    df = pd.DataFrame(dist_data, 
                      index=[f"Client {i+1}" for i in range(len(client_indices))],
                      columns=[f"Digit {i}" for i in range(10)])
    
    st.bar_chart(df)
    st.caption("This Non-IID 'Silo' structure is what makes Federated Learning a unique challenge.")

# --- DASHBOARD UI SETUP ---
st.set_page_config(page_title="FedAvg Research Lab", layout="wide")
st.title("🌐 Federated Learning: FedAvg Dashboard")
st.markdown("---")

st.sidebar.header("Experiment Settings")
num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
rounds = st.sidebar.slider("Communication Rounds", 1, 20, 5)

if st.sidebar.button("🚀 Start Federation"):
    # 1. UNPACKING SYNC: Now receiving 3 values
    train_ds, test_ds, client_indices = get_non_iid_subsets(num_clients)
    
    display_distribution(train_ds, client_indices)
    
    global_model = SimpleCNN()
    
    # TELEMETRY: Tracking both Accuracy and Loss
    accuracy_history = []
    loss_history = []
    
    # UI Layout: Side-by-Side Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Global Accuracy (%)")
        acc_chart = st.empty()
    with col2:
        st.subheader("📉 Global Loss")
        loss_chart = st.empty()
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 3. COMMUNICATION ROUNDS
    for r in range(rounds):
        status_text.text(f"Round {r+1}: Training {num_clients} Clients...")
        local_weights_list = []
        client_sizes = []
        round_losses = []

        for i in range(num_clients):
            local_model = SimpleCNN()
            local_model.load_state_dict(global_model.state_dict())
            
            # SYNC: train_local_model now returns (weights, loss)
            weights, loss = train_local_model(local_model, train_ds, client_indices[i])
            
            local_weights_list.append(weights)
            client_sizes.append(len(client_indices[i]))
            round_losses.append(loss)

            # VRAM Guard
            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4. AGGREGATION
        new_global_weights = federated_averaging(local_weights_list, client_sizes)
        global_model.load_state_dict(new_global_weights)
        
        # 5. EVALUATION: test_model now returns (acc, loss)
        acc, test_loss = test_model(global_model, test_ds)
        accuracy_history.append(acc)
        loss_history.append(test_loss)
        
        # 6. LIVE TELEMETRY UPDATE
        progress_bar.progress((r + 1) / rounds)
        acc_chart.line_chart(accuracy_history)
        loss_chart.line_chart(loss_history)
        
        st.write(f"✅ Round {r+1} | Acc: **{acc:.2f}%** | Loss: **{test_loss:.4f}**")

    st.success("🎯 Federation complete! The 'Brain' has successfully synthesized decentralized knowledge.")
