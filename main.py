import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'./.venv/Lib/site-packages/PyQt5/Qt5/plugins'
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qt5agg')
from typing import Tuple, List, Optional, Union
import math
import networkx as nx
import warnings
import os

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='torch_geometric.nn.conv.gatv2_conv')
warnings.filterwarnings("ignore", category=UserWarning, module='torch_geometric.nn.pool.glob')

# --- Configuration ---
# Define constants based on README and user request
CSV_PATH = 'dataset.csv'  # Assumed path, replace if different
MIN_X, MAX_X = 0, 1000  # Example boundaries, adjust based on actual data range
MIN_Y, MAX_Y = 0, 1000  # Example boundaries, adjust based on actual data range
MIN_RSSI, MAX_RSSI = -120, -30  # As per README
MIN_DIST, MAX_DIST = 0, np.sqrt((MAX_X - MIN_X) ** 2 + (MAX_Y - MIN_Y) ** 2)  # Max possible distance in area
EMBEDDING_DIM = 16  # As requested by user
K_NEIGHBORS = 5  # As requested by user
BATCH_SIZE = 32  # Standard batch size, can be tuned
EPOCHS = 100  # Number of training epochs, adjust as needed
LEARNING_RATE = 0.001
ALPHA_LOSS = 0.9  # Weight for azimuth loss
BETA_LOSS = 0.5  # Weight for distance loss
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
print(f"Using device: {DEVICE}")


# --- 1. Data Loading and Preprocessing Module ---

class LocationDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for loading and preprocessing location data.
    Handles feature scaling, encoding, graph construction, and splitting.
    """

    def __init__(self, csv_path: str, train: bool = True, val_size: float = 0.15, test_size: float = 0.15,
                 random_state: int = 42):
        super().__init__(None, None, None)
        self.csv_path = csv_path
        self.train = train
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.data_list = self._process()
        self._split_data()

    def _positional_encoding(self, coord: torch.Tensor, d_model: int) -> torch.Tensor:
        """Applies positional encoding to normalized coordinates."""
        # Based on Transformer encoding
        position = coord.unsqueeze(1)  # Shape: (N, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: (d_model/2)
        pe = torch.zeros(coord.size(0), d_model)  # Shape: (N, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _encode_coordinates(self, x: np.ndarray, y: np.ndarray, d: int) -> torch.Tensor:
        """ Normalizes and applies positional encoding to coordinates. """
        # MinMax scaling
        x_scaled = (x - MIN_X) / (MAX_X - MIN_X)
        y_scaled = (y - MIN_Y) / (MAX_Y - MIN_Y)

        x_norm_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_norm_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        # Positional Encoding
        pe_x = self._positional_encoding(x_norm_tensor, d // 2)  # Split dim for x and y
        pe_y = self._positional_encoding(y_norm_tensor, d - d // 2)

        return torch.cat([pe_x, pe_y], dim=1)  # Shape: (N, d)

    def _encode_rssi(self, rssi: np.ndarray) -> torch.Tensor:
        """ Normalizes RSSI values. """
        # MinMax scaling
        rssi_scaled = (rssi - MIN_RSSI) / (MAX_RSSI - MIN_RSSI)
        return torch.tensor(rssi_scaled, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    def _encode_azimuth(self, azimuth: np.ndarray) -> torch.Tensor:
        """ Encodes azimuth using sin/cos. """
        # Sin/Cos encoding
        azimuth_rad = np.radians(azimuth)
        sin_azi = np.sin(azimuth_rad)
        cos_azi = np.cos(azimuth_rad)
        return torch.tensor(np.stack([sin_azi, cos_azi], axis=1), dtype=torch.float32)  # Shape: (N, 2)

    def _encode_distance(self, distance: np.ndarray) -> torch.Tensor:
        """ Normalizes distance values. """
        # MinMax scaling
        distance_scaled = (distance - MIN_DIST) / (MAX_DIST - MIN_DIST)
        return torch.tensor(distance_scaled, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    def _build_graph(self, group_df: pd.DataFrame) -> Data:
        """ Constructs a PyG Data object for a single source_id group. """
        num_nodes = len(group_df)
        if num_nodes < K_NEIGHBORS:
            print(
                f"Warning: source_id {group_df['source_id'].iloc[0]} has only {num_nodes} nodes, less than k={K_NEIGHBORS}. Using num_nodes-1 for k.")
            k_actual = max(1, num_nodes - 1)  # Need at least 1 neighbor
        else:
            k_actual = K_NEIGHBORS

        # 1. Node Features
        coords_embedded = self._encode_coordinates(group_df['longitude'].values, group_df['latitude'].values,
                                                   EMBEDDING_DIM)
        rssi_encoded = self._encode_rssi(group_df['rssi'].values)
        node_features = torch.cat([rssi_encoded, coords_embedded], dim=1)  # Shape: (N, 1 + d)

        # 2. Edge Index (Connectivity) - Using k-NN
        # Use original coordinates for distance calculation in k-NN
        pos = torch.tensor(group_df[['longitude', 'latitude']].values, dtype=torch.float32)
        # Calculate k-NN graph edges
        adj = torch.cdist(pos, pos)
        _, nn_indices = torch.topk(adj, k=k_actual + 1, dim=1,
                                   largest=False)  # +1 because a node is its own closest neighbor

        row, col = [], []
        for i in range(num_nodes):
            # Skip self-loop (index 0)
            neighbors = nn_indices[i, 1:]
            row.extend([i] * k_actual)
            col.extend(neighbors.tolist())

        edge_index = torch.tensor([row, col], dtype=torch.long)

        # Ensure undirected graph (add reverse edges) - Not strictly needed if using the alternative edge feature, but good practice
        # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        # edge_index = torch.unique(edge_index, dim=1) # Remove duplicate edges

        # 3. Edge Features (Alternative representation requested)
        # Calculate features for each edge defined in edge_index
        edge_features_list = []
        start_nodes, end_nodes = edge_index[0], edge_index[1]

        for i in range(edge_index.size(1)):
            src_idx, dst_idx = start_nodes[i].item(), end_nodes[i].item()

            # Use original coordinates for accurate distance/angle
            src_pos = pos[src_idx]
            dst_pos = pos[dst_idx]

            # Distance
            dist = torch.norm(src_pos - dst_pos)
            dist_scaled = (dist - MIN_DIST) / (MAX_DIST - MIN_DIST)  # Normalize

            # Angle (Azimuth) from src to dst
            delta_x = dst_pos[0] - src_pos[0]
            delta_y = dst_pos[1] - src_pos[1]
            angle_rad = torch.atan2(delta_x,
                                    delta_y)  # atan2(x, y) gives angle from North, clockwise positive in math -> needs care if strict North=0 is needed

            # Correcting atan2 for typical Azimuth (North=0, East=90)
            # angle_deg = (torch.rad2deg(angle_rad) + 360) % 360

            # Using sin/cos directly from atan2 result is often sufficient for models
            sin_a = torch.sin(angle_rad)
            cos_a = torch.cos(angle_rad)

            # Angle from dst to src (angle + pi or 180 deg)
            angle_rad_rev = angle_rad + math.pi
            sin_a_rev = torch.sin(angle_rad_rev)
            cos_a_rev = torch.cos(angle_rad_rev)

            # Feature vector [distance, sin(a), cos(a), sin(a + 180 deg), cos(a + 180 deg)]
            edge_vec = torch.tensor([dist_scaled, sin_a, cos_a, sin_a_rev, cos_a_rev], dtype=torch.float)
            edge_features_list.append(edge_vec)

        edge_features = torch.stack(edge_features_list, dim=0) if edge_features_list else torch.empty((0, 5),
                                                                                                      dtype=torch.float)  # Shape: (E, 5)

        # 4. Target Features
        target_azimuth = self._encode_azimuth(group_df['azimuth'].values)  # Shape: (N, 2)
        target_distance = self._encode_distance(group_df['distance'].values)  # Shape: (N, 1)

        # Store original positions and targets for visualization/evaluation if needed
        original_pos = torch.tensor(group_df[['longitude', 'latitude']].values, dtype=torch.float32)
        original_source_pos = torch.tensor(group_df[['source_longitude', 'source_latitude']].iloc[0].values,
                                           dtype=torch.float32)  # Same for all nodes in group
        original_target_azimuth = torch.tensor(group_df['azimuth'].values, dtype=torch.float32)
        original_target_distance = torch.tensor(group_df['distance'].values, dtype=torch.float32)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y_azimuth=target_azimuth,
            y_distance=target_distance,
            pos=original_pos,  # Keep original coordinates for potential use (e.g., visualization)
            source_pos=original_source_pos  # Keep original source coordinates
            # You might want to add original_target_azimuth and original_target_distance here too
        )

    def _process(self) -> List[Data]:
        """ Reads CSV and processes it into a list of Data objects. """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.csv_path}")
            # Create a dummy dataset.csv if it doesn't exist for demonstration
            print("Creating a dummy dataset.csv...")
            self._create_dummy_csv(self.csv_path)
            df = pd.read_csv(self.csv_path)

        # Group by source_id to process each training sample (graph)
        grouped = df.groupby('source_id')
        data_list = [self._build_graph(group) for _, group in grouped]
        return data_list

    def _create_dummy_csv(self, path: str, num_sources: int = 10, points_per_source: int = 20):
        """ Creates a dummy dataset file for demonstration. """
        data = []
        for i in range(num_sources):
            source_lon = np.random.uniform(MIN_X * 0.2, MAX_X * 0.8)
            source_lat = np.random.uniform(MIN_Y * 0.2, MAX_Y * 0.8)
            for _ in range(points_per_source):
                lon = np.random.uniform(MIN_X, MAX_X)
                lat = np.random.uniform(MIN_Y, MAX_Y)
                dx = source_lon - lon
                dy = source_lat - lat
                dist = np.sqrt(dx ** 2 + dy ** 2)
                # Simplified RSSI based on distance (add noise)
                rssi = MAX_RSSI - 10 * 2.5 * np.log10(dist / 1.0 + 1e-6) + np.random.normal(0,
                                                                                            5)  # Basic log-distance path loss model
                rssi = np.clip(rssi, MIN_RSSI, MAX_RSSI)
                # Azimuth calculation (angle from North)
                azimuth = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

                data.append([i, lon, lat, rssi, source_lon, source_lat, azimuth, dist])

        dummy_df = pd.DataFrame(data, columns=['source_id', 'longitude', 'latitude', 'rssi', 'source_longitude',
                                               'source_latitude', 'azimuth', 'distance'])
        dummy_df.to_csv(path, index=False)
        print(f"Dummy dataset created at {path}")

    def _split_data(self):
        """ Splits the data_list into train, validation, and test sets. """
        indices = list(range(len(self.data_list)))
        train_indices, test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )
        # Adjust validation size relative to the remaining training data
        relative_val_size = self.val_size / (1 - self.test_size)
        train_indices, val_indices = train_test_split(
            train_indices, test_size=relative_val_size, random_state=self.random_state
        )

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    def len(self) -> int:
        if self.train == 'train':
            return len(self.train_indices)
        elif self.train == 'val':
            return len(self.val_indices)
        elif self.train == 'test':
            return len(self.test_indices)
        else:  # Get all data
            return len(self.data_list)

    def get(self, idx: int) -> Data:
        if self.train == 'train':
            original_idx = self.train_indices[idx]
        elif self.train == 'val':
            original_idx = self.val_indices[idx]
        elif self.train == 'test':
            original_idx = self.test_indices[idx]
        else:  # Get from the full list
            original_idx = idx
        return self.data_list[original_idx]


# --- 2. Model Definition Module ---

class GATLocationModel(torch.nn.Module):
    """
    GNN model with GAT layers for predicting azimuth and distance.
    Based on architecture described in README.
    """

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim

        # GAT Layers - Using edge_attr requires GATv2Conv or modifying GATConv
        # We'll use standard GATConv and incorporate edge features later if needed,
        # or assume their effect is captured implicitly by node distances/neighbors.
        # For simplicity now, let's use GATConv without explicit edge_attr handling in the convolution step.
        # If edge_attr is crucial, GATv2Conv or a custom layer would be needed.
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False,
                             dropout=0.6)  # Last layer, no concat

        # Output Layers
        self.azimuth_out = torch.nn.Linear(hidden_dim, 2)  # sin, cos output between -1 and 1
        self.distance_out = torch.nn.Linear(hidden_dim, 1)  # distance output between 0 and 1

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        # edge_attr = data.edge_attr # Available if needed

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        # x now contains the final node embeddings after GAT layers

        # Predict azimuth (sin, cos) for each node
        # Tanh activation naturally outputs between -1 and 1
        azimuth_pred = torch.tanh(self.azimuth_out(x))

        # Predict distance for each node
        # Sigmoid activation naturally outputs between 0 and 1
        distance_pred = torch.sigmoid(self.distance_out(x))

        return azimuth_pred, distance_pred


# --- 3. Training Module with Visualization ---

def train(model: GATLocationModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[
    float, float]:
    """ Performs one training epoch. """
    model.train()
    total_loss = 0
    total_azi_loss = 0
    total_dist_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        azimuth_pred, distance_pred = model.forward(data)

        # Loss Calculation
        loss_azimuth = F.mse_loss(azimuth_pred, data.y_azimuth)
        loss_distance = F.mse_loss(distance_pred, data.y_distance)
        loss = ALPHA_LOSS * loss_azimuth + BETA_LOSS * loss_distance

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_azi_loss += loss_azimuth.item() * data.num_graphs
        total_dist_loss += loss_distance.item() * data.num_graphs

    num_total_graphs = len(loader.dataset)
    return total_loss / num_total_graphs, total_azi_loss / num_total_graphs, total_dist_loss / num_total_graphs


@torch.no_grad()
def evaluate(model: GATLocationModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """ Evaluates the model on the validation or test set. """
    model.eval()
    total_azi_loss = 0
    total_dist_loss = 0
    for data in loader:
        data = data.to(device)
        azimuth_pred, distance_pred = model(data)
        loss_azimuth = F.mse_loss(azimuth_pred, data.y_azimuth)
        loss_distance = F.mse_loss(distance_pred, data.y_distance)
        total_azi_loss += loss_azimuth.item() * data.num_graphs
        total_dist_loss += loss_distance.item() * data.num_graphs

    num_total_graphs = len(loader.dataset)
    return total_azi_loss / num_total_graphs, total_dist_loss / num_total_graphs


# --- Visualization Functions ---
plt.ion()  # Turn on interactive mode for real-time plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax_loss, ax_pred = axes

train_loss_hist_azi, train_loss_hist_dist = [], []
val_loss_hist_azi, val_loss_hist_dist = [], []


def update_loss_plot(epoch: int):
    """ Updates the loss plot. """
    ax_loss.clear()
    epochs_range = range(1, epoch + 2)
    ax_loss.plot(epochs_range, train_loss_hist_azi, label='Train Azimuth Loss', color='tab:blue', linestyle='--')
    ax_loss.plot(epochs_range, train_loss_hist_dist, label='Train Distance Loss', color='tab:orange', linestyle='--')
    ax_loss.plot(epochs_range, val_loss_hist_azi, label='Val Azimuth Loss', color='tab:blue')
    ax_loss.plot(epochs_range, val_loss_hist_dist, label='Val Distance Loss', color='tab:orange')
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True)


def plot_predictions(model: GATLocationModel, data: Data, device: torch.device):
    """ Plots predictions for a single test sample. """
    model.eval()
    # data = data.to(device)
    data = data.clone().to(device)  # Создаем клон и перемещаем его
    pred_azimuth, pred_distance = model(data)

    # Move data to CPU and detach for plotting
    pred_azimuth = pred_azimuth.cpu().detach().numpy()
    pred_distance = pred_distance.cpu().detach().numpy()
    true_pos = data.pos.cpu().numpy()  # Measurement points
    true_source_pos = data.source_pos.cpu().numpy()  # True source

    # Denormalize distance prediction
    pred_distance_denorm = pred_distance * (MAX_DIST - MIN_DIST) + MIN_DIST

    # Calculate predicted angles in degrees from sin/cos
    pred_angle_rad = np.arctan2(pred_azimuth[:, 0], pred_azimuth[:, 1])  # atan2(sin, cos) -> angle
    pred_angle_deg = (np.degrees(pred_angle_rad) + 360) % 360  # Convert to degrees [0, 360)

    ax_pred.clear()
    # Plot measurement points
    ax_pred.scatter(true_pos[:, 0], true_pos[:, 1], c='blue', label='Measurement Points', s=50, zorder=3)
    # Plot true source location
    ax_pred.scatter(true_source_pos[0], true_source_pos[1], c='red', marker='*', s=200, label='True Source', zorder=4)

    # Plot predicted direction arrows
    for i in range(len(true_pos)):
        start_point = true_pos[i]
        angle_rad_viz = np.radians(90 - pred_angle_deg[i])  # Convert Azimuth (North=0) to plot angle (East=0)
        arrow_len = pred_distance_denorm[i, 0] * 0.3  # Scale arrow length for visualization
        # Calculate end point based on angle and scaled distance
        # Note: This arrow length is arbitrary for visualization, not the predicted distance itself
        dx = arrow_len * np.cos(angle_rad_viz)
        dy = arrow_len * np.sin(angle_rad_viz)

        ax_pred.arrow(start_point[0], start_point[1], dx, dy,
                      head_width=max(10, arrow_len * 0.1), head_length=max(15, arrow_len * 0.15), fc='green',
                      ec='green', alpha=0.6, zorder=2)

    # Estimate source location (simple intersection/centroid - very basic)
    # A more robust method would be needed for accurate Step 3 prediction
    predicted_sources = []
    for i in range(len(true_pos)):
        angle_rad_viz = np.radians(90 - pred_angle_deg[i])
        px = true_pos[i, 0] + pred_distance_denorm[i, 0] * np.cos(angle_rad_viz)
        py = true_pos[i, 1] + pred_distance_denorm[i, 0] * np.sin(angle_rad_viz)
        predicted_sources.append([px, py])

    if predicted_sources:
        predicted_sources = np.array(predicted_sources)
        pred_source_centroid = np.mean(predicted_sources, axis=0)
        ax_pred.scatter(pred_source_centroid[0], pred_source_centroid[1], c='green', marker='x', s=150,
                        label='Predicted Source (Centroid)', zorder=4)

    ax_pred.set_xlabel("Longitude")
    ax_pred.set_ylabel("Latitude")
    ax_pred.set_title("Sample Prediction (Test Set)")
    ax_pred.legend()
    ax_pred.grid(True)
    ax_pred.axis('equal')  # Keep aspect ratio consistent

    # Set plot limits dynamically based on data points
    all_x = np.concatenate([true_pos[:, 0], true_source_pos[0:1], pred_source_centroid[0:1]])
    all_y = np.concatenate([true_pos[:, 1], true_source_pos[1:2], pred_source_centroid[1:2]])
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax_pred.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax_pred.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)


def run_training():
    """ Main function to orchestrate data loading, model training, and visualization. """
    # --- Data Loading (Optimized) ---
    print("Loading and processing dataset ONCE...")
    # 1. Load and process everything once
    full_dataset_processor = LocationDataset(csv_path=CSV_PATH, train=None) # train=None ensures it loads all data

    # Check if processing was successful
    if not hasattr(full_dataset_processor, 'data_list') or not full_dataset_processor.data_list:
         print("Error: Failed to load or process data. Exiting.")
         # Potentially add more specific error handling based on how LocationDataset signals failure
         # For example, check if self.data_list is empty or None after _process()
         # Or add a status flag in LocationDataset
         # if not getattr(full_dataset_processor, 'processing_successful', False): # Assuming you add such a flag
         #     print("Error: Data processing indicated failure.")
         #     return
         return # Exit if data loading/processing failed


    print(f"Dataset processed. Total samples: {len(full_dataset_processor.data_list)}")
    print(f"Split indices: Train={len(full_dataset_processor.train_indices)}, Val={len(full_dataset_processor.val_indices)}, Test={len(full_dataset_processor.test_indices)}")


    # 2. Create subsets using the processed data and indices
    # We can use index_select which is efficient for PyG datasets
    train_dataset = full_dataset_processor.index_select(full_dataset_processor.train_indices)
    val_dataset = full_dataset_processor.index_select(full_dataset_processor.val_indices)
    test_dataset = full_dataset_processor.index_select(full_dataset_processor.test_indices)

    # --- Sanity Check (Optional but recommended) ---
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print(f"Warning: One or more dataset splits are empty! Check dataset size and split ratios.")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        # Decide if you want to proceed or exit if splits are invalid
        # return

    print("Dataset subsets created for Train, Validation, and Test.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get one test sample for visualization (from the already created test_dataset subset)
    sample_test_data = test_dataset.get(0) if len(test_dataset) > 0 else None
    if sample_test_data:
         print(f"Using test sample (source_id group) with {sample_test_data.num_nodes} nodes for visualization.")
    else:
         print("Warning: No test data available for visualization.")


    # --- Model Initialization ---
    # Get dimensions from the original processor instance or any subset (they are the same)
    node_feat_dim = full_dataset_processor.num_node_features
    edge_feat_dim = full_dataset_processor.num_edge_features
    model = GATLocationModel(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model initialized:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print("Attempting to fetch first batch...")
    try:
        # Попробовать получить один бэтч из train_loader
        first_batch = next(iter(train_loader))
        print(f"Successfully fetched first batch with {first_batch.num_graphs} graphs.")
        # Можно добавить больше информации о бэтче при необходимости
        # print(first_batch)
    except StopIteration:
        print("Error: Train loader is empty! Cannot fetch any batches.")
        # Здесь нужно решить, что делать - возможно, выйти
        return
    except Exception as e:
        print(f"Error fetching first batch: {e}")
        import traceback
        traceback.print_exc()
        # Выйти или обработать ошибку
        return

    print("Starting training...")  # Переместить эту строку сюда
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        train_loss, train_azi_loss, train_dist_loss = train(model, train_loader, optimizer, DEVICE)
        val_azi_loss, val_dist_loss = evaluate(model, val_loader, DEVICE)

        train_loss_hist_azi.append(train_azi_loss)
        train_loss_hist_dist.append(train_dist_loss)
        val_loss_hist_azi.append(val_azi_loss)
        val_loss_hist_dist.append(val_dist_loss)

        print(
            f'Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} (Azi: {train_azi_loss:.4f}, Dist: {train_dist_loss:.4f}) | Val Loss => Azi: {val_azi_loss:.4f}, Dist: {val_dist_loss:.4f}')

        # --- Visualization Update ---
        update_loss_plot(epoch)
        if sample_test_data:
            plot_predictions(model, sample_test_data, DEVICE)
        plt.pause(0.1)  # Pause briefly to update the plot

    # (Rest of the training loop remains the same)
    # ... (Keep the existing loop for training, evaluation, plotting) ...

    plt.ioff() # Turn off interactive mode
    plt.show() # Keep the final plot window open

    # --- Final Evaluation (Optional) ---
    # Ensure test_loader is not empty before evaluating
    if len(test_loader) > 0:
        test_azi_loss, test_dist_loss = evaluate(model, test_loader, DEVICE)
        print(f'\nTraining Complete.')
        print(f'Final Test Loss => Azimuth: {test_azi_loss:.4f}, Distance: {test_dist_loss:.4f}')
    else:
        print(f'\nTraining Complete.')
        print("Skipping final test evaluation as the test set is empty.")


    # Save the model (Optional)
    # torch.save(model.state_dict(), 'gat_location_model.pth')
    # print("Model saved to gat_location_model.pth")


# Make sure the main execution block calls the modified run_training
if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} not found. Creating a dummy dataset for demonstration.")
        # Instantiate once just to trigger dummy creation if needed
        # The instance itself isn't used directly here, run_training will create its own.
        try:
             _ = LocationDataset(csv_path=CSV_PATH, train=None)
        except Exception as e:
             print(f"Error during dummy data creation or initial load: {e}")
             # Decide how to handle this - exit or proceed cautiously
             # exit() # Or maybe allow run_training to handle it

    run_training()