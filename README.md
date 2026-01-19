# GAGNN: A Geography-Aware Graph Neural Network for Citywide Commuting Flows Prediction

This project proposes a commute flow prediction model combining Graph Attention Networks (GAT) and multi-source urban data. By fusing geographic adjacency, semantic adjacency (bus network), and POI features, it achieves precise prediction of inter-regional commute flows.

## 1. Core Features
* Dual-Graph Fusion: Utilizes both geographic spatial adjacency and semantic (bus) network adjacency simultaneously.
* Attention Mechanism: Uses GAT to dynamically capture the importance of neighbor nodes.
* Distance Awareness: Explicitly introduces physical distance features between regions in the final regression layer.
* End-to-End Architecture: GAT extracts region representations + MLP performs flow regression.

## 2. Environment Dependencies
The following Python libraries need to be installed:
pip install torch numpy pandas scikit-learn matplotlib

## 3. Data Preparation
Please ensure the project directory structure is as follows (data files must include an index column):

Project_Root/
├── data/                       # Stores all data files
│   ├── adjacency_matrix.csv            # Geographic adjacency matrix (N×N)
│   ├── semantic_adjacency_matrix_bus.csv # Semantic adjacency matrix (N×N)
│   ├── poi_count_by_region.csv         # Regional POI features (N×F)
│   ├── commute_flow_matrix.csv         # Label: Commute flow matrix (N×N)
│   └── region_distance_matrix.csv      # Inter-regional distance matrix (N×N)
├── train.py                    # Main training script
└── README.txt                  # This readme file

## 4. Quick Start
1. Configure Paths: Ensure data files are located in the ./data/ directory (or modify the pd.read_csv paths in the code).
2. Start Training: python train.py
3. View Output: 
   - The console prints Train/Val Loss in real-time.
   - A Loss convergence curve plot is automatically generated after training ends.

## 5. Model Architecture Overview
The model processing flow is as follows:
1. Graph Construction: Superimpose the geographic graph and semantic graph (Adj_geo + Adj_sem).
2. Feature Encoding (GAT Layers):
   - Input: Regional POI features.
   - Operation: Multi-layer GAT performs neighborhood aggregation and feature update.
   - Output: High-level Region Embeddings.
3. Flow Prediction (MLP Head):
   - Concatenation: [Origin Embedding, Destination Embedding, Origin-to-CBD Distance, Destination-to-CBD Distance, OD Distance]
   - Regression: MLP outputs the predicted flow value.

## 6. Key Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| epochs | 700 | Training epochs |
| batch_size | 128 | Batch size |
| lr | 0.001 | Learning rate |
| gat_layers | 3 | Number of GAT layers |
| hidden_dim | 128 | MLP hidden layer dimension |

License: MIT
