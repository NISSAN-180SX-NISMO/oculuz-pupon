import argparse
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from torch_geometric.nn import GATv2Conv # Используем GATv2Conv для лучшего внимания
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import defaultdict
import math
import os
from tqdm import tqdm
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'./.venv/Lib/site-packages/PyQt5/Qt5/plugins'
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import time

# --- Конфигурация ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Гиперпараметры (можно вынести в argparse)
NUM_EPOCHS = 50
BATCH_SIZE = 32 # Уменьшите, если не хватает памяти GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 512
POS_ENCODING_DIM = 64 # Размерность позиционного кодирования (должна быть четной)
NUM_GAT_LAYERS = 5
GAT_HEADS = 4 # Количество голов внимания в GAT
KNN_K = 7      # Количество соседей для KNN (k < кол-во точек в сессии)
LOSS_LAMBDA_DIR = 1.5
LOSS_LAMBDA_CONTAIN = 10 # Увеличим вес для contain loss
LOSS_LAMBDA_WIDTH = 0.4
ANGLE_THRESHOLD_DEGREES = 30.0 # Порог для метрики точности
ANGLE_THRESHOLD_RADIANS = math.radians(ANGLE_THRESHOLD_DEGREES)
PLOT_UPDATE_FREQ = 1 # Обновлять график предсказаний каждые N эпох

# Пути
CSV_FILE = 'resources/dataset-v3_ulight.csv' # Укажите ваш путь к файлу
MODEL_SAVE_PATH = 'model-v3.pth'
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# --- 1. Класс для хранения данных сессии (для визуализации) ---
class SessionData:
    def __init__(self, session_id, measurements, source):
        """
        Хранит ненормализованные данные одной сессии.
        Args:
            session_id: ID сессии.
            measurements (pd.DataFrame): DataFrame с колонками 'meas_x', 'meas_y', 'rssi'.
            source (tuple): Координаты источника (source_x, source_y).
        """
        self.session_id = session_id
        self.measurements = measurements
        self.source_x, self.source_y = source

# --- Вспомогательная функция: Позиционное кодирование ---
def get_positional_encoding(coords, encoding_dim):
    """
    Создает позиционное кодирование для 2D координат.
    Args:
        coords (torch.Tensor): Тензор координат [N, 2].
        encoding_dim (int): Размерность кодирования (должна быть четной).
    Returns:
        torch.Tensor: Тензор позиционного кодирования [N, encoding_dim].
    """
    if encoding_dim % 4 != 0:
        raise ValueError("encoding_dim must be divisible by 4")

    d_model_half = encoding_dim // 2
    div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half)).to(coords.device)

    pe = torch.zeros(coords.size(0), encoding_dim).to(coords.device)

    # Кодирование для X
    pe[:, 0:d_model_half:2] = torch.sin(coords[:, 0:1] * div_term)
    pe[:, 1:d_model_half:2] = torch.cos(coords[:, 0:1] * div_term)

    # Кодирование для Y
    pe[:, d_model_half::2] = torch.sin(coords[:, 1:2] * div_term)
    pe[:, d_model_half+1::2] = torch.cos(coords[:, 1:2] * div_term)

    return pe

# --- 1. Функция создания графа для одного семпла (сессии) ---
# --- 1. Функция создания графа для одного семпла (сессии) ---
def create_graph_sample(session_id, meas_df, source_coords, k, pos_encoding_dim, scalers):
    """
    Создает PyG Data объект для одной сессии.
    Использует KNN граф, если точек > k, иначе создает полный граф.
    Args:
        session_id: ID сессии.
        meas_df (pd.DataFrame): Данные измерений сессии ('meas_x', 'meas_y', 'rssi').
        source_coords (tuple): Координаты источника (x, y).
        k (int): Количество соседей для KNN.
        pos_encoding_dim (int): Размерность позиционного кодирования.
        scalers (dict): Словарь с обученными скейлерами {'coords': scaler, 'rssi': scaler, 'edge': scaler}.
    Returns:
        tuple: (torch_geometric.data.Data, SessionData) или (None, None) если точек <= 1.
    """
    n_points = len(meas_df)
    # --- ИЗМЕНЕНИЕ: Пропускаем только если точка всего одна (нельзя создать ребра) ---
    if n_points <= 1:
        print(f"Warning: Session {session_id} has {n_points} points. Skipping as no edges can be formed.")
        return None, None
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # Ненормализованные данные для SessionData
    raw_measurements = meas_df[['meas_x', 'meas_y', 'rssi']].copy()
    session_data_obj = SessionData(session_id, raw_measurements, source_coords)

    # --- Нормализация ---
    coords = meas_df[['meas_x', 'meas_y']].values
    rssi = meas_df['rssi'].values.reshape(-1, 1)

    norm_coords = scalers['coords'].transform(coords)
    norm_rssi = scalers['rssi'].transform(rssi)

    coords_tensor = torch.tensor(norm_coords, dtype=torch.float)
    rssi_tensor = torch.tensor(norm_rssi, dtype=torch.float)

    # --- Признаки узлов (Node Features) ---
    pos_encoding = get_positional_encoding(coords_tensor, pos_encoding_dim)
    node_features = torch.cat([pos_encoding, rssi_tensor], dim=1)

    # --- Построение графа ---
    edge_index_list = []

    # --- ИЗМЕНЕНИЕ: Логика выбора типа графа ---
    if n_points <= k:
        # Создаем ПОЛНЫЙ граф (все со всеми, кроме петель)
        print(f"Info: Session {session_id} has {n_points} points <= k={k}. Creating fully connected graph.")
        sources = []
        destinations = []
        for i in range(n_points):
            for j in range(n_points):
                if i != j: # Не добавляем петли (self-loops)
                    sources.append(i)
                    destinations.append(j)
        if not sources: # На всякий случай, если n_points=1 (хотя должно отсеяться выше)
             print(f"Warning: No edges created for fully connected graph in session {session_id} (n_points={n_points}). Skipping.")
             return None, None
        edge_index = torch.tensor([sources, destinations], dtype=torch.long)

    else:
        # Создаем KNN граф (как и было)
        dist_matrix = torch.cdist(coords_tensor, coords_tensor)
        # Находим k+1 ближайших, т.к. первый будет сам узел (расстояние 0)
        # Убедимся, что не запрашиваем больше соседей, чем есть точек (минус сам узел)
        actual_k_plus_1 = min(k + 1, n_points)
        _, indices = torch.topk(dist_matrix, actual_k_plus_1, largest=False, dim=1)

        # Убираем сам узел из соседей
        source_nodes = torch.arange(n_points)
        for i in range(n_points):
            # indices[i, 1:actual_k_plus_1] выбирает соседей, исключая self-loop
            neighbors = indices[i, 1:actual_k_plus_1]
            # Проверяем, есть ли соседи после исключения self-loop
            if neighbors.numel() > 0:
                 source_node_repeated = source_nodes[i].repeat(neighbors.numel())
                 edge_index_list.append(torch.stack([source_node_repeated, neighbors], dim=0))

        if not edge_index_list: # Если не удалось создать ни одного ребра (маловероятно при n_points > k)
            print(f"Warning: No KNN edges created for session {session_id} (n_points={n_points}, k={k}). Skipping.")
            return None, None
        edge_index = torch.cat(edge_index_list, dim=1)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---


    # --- Признаки ребер (Edge Features) - этот блок остается без изменений ---
    row, col = edge_index
    coord_i = coords_tensor[row] # Координаты исходящих узлов
    coord_j = coords_tensor[col] # Координаты входящих узлов
    rssi_i = rssi_tensor[row]
    rssi_j = rssi_tensor[col]

    # 1. Расстояние
    distances = torch.linalg.norm(coord_i - coord_j, dim=1, keepdim=True)
    # 2. Вектор смещения
    displacement = coord_j - coord_i # Shape: [num_edges, 2]
    # 3. Разница RSSI
    rssi_diff = rssi_i - rssi_j # Shape: [num_edges, 1]

    # Собираем признаки ребер и нормализуем
    raw_edge_features = torch.cat([distances, displacement, rssi_diff], dim=1).numpy()
    try:
        norm_edge_features = torch.tensor(scalers['edge'].transform(raw_edge_features), dtype=torch.float)
    except ValueError as e:
        print(f"Error transforming edge features for session {session_id}: {e}")
        print(f"Raw edge features shape: {raw_edge_features.shape}")
        # Возможно, проблема с пустыми признаками, если не было ребер
        return None, None


    # --- Таргет ---
    # Убедитесь, что форма target_tensor соответствует ожиданиям loss функции
    target_tensor = torch.tensor([[source_coords[0], source_coords[1]]], dtype=torch.float) # [1, 2]

    # Создаем объект Data
    data = Data(x=node_features, edge_index=edge_index, edge_attr=norm_edge_features, y=target_tensor)
    # Добавляем ненормализованные координаты для удобства в loss/plotting
    data.pos = torch.tensor(coords, dtype=torch.float)

    return data, session_data_obj

# --- 2. Функция чтения данных и создания графов ---
def load_data(csv_path, k, pos_encoding_dim):
    """
    Читает CSV, нормализует данные (обучая скейлеры на трейне),
    создает графы и объекты SessionData.
    Args:
        csv_path (str): Путь к CSV файлу.
        k (int): Параметр для KNN.
        pos_encoding_dim (int): Размерность позиционного кодирования.
    Returns:
        tuple: (list[Data], list[SessionData], dict)
               Список графов PyG, список объектов SessionData, словарь скейлеров.
    """
    df = pd.read_csv(csv_path)

    grouped = df.groupby('session_id')
    session_ids = list(grouped.groups.keys())

    # --- Разделение на train/val/test ДО нормализации ---
    train_ids, test_ids = train_test_split(session_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42) # 0.15 * 0.8 = 0.12

    print(f"Train sessions: {len(train_ids)}, Val sessions: {len(val_ids)}, Test sessions: {len(test_ids)}")

    # --- Обучение скейлеров ТОЛЬКО на тренировочных данных ---
    train_df = df[df['session_id'].isin(train_ids)]

    coord_scaler = MinMaxScaler()
    rssi_scaler = StandardScaler() # StandardScaler лучше для RSSI? Или MinMaxScaler
    edge_feature_placeholder = [] # Собрать признаки ребер со всех трейн сессий для обучения скейлера

    # Временный расчет признаков ребер на трейне для обучения скейлера
    temp_coords_for_edge_scaler = []
    temp_rssi_for_edge_scaler = []
    for session_id in tqdm(train_ids, desc="Fitting Scalers (Pass 1)"):
        session_df = grouped.get_group(session_id)
        if len(session_df) > k:
             # Нормализуем временно для расчета KNN
            coords = session_df[['meas_x', 'meas_y']].values
            rssi = session_df['rssi'].values.reshape(-1, 1)
            temp_coords_for_edge_scaler.append(coords)
            temp_rssi_for_edge_scaler.append(rssi)

    # Обучаем coord и rssi скейлеры
    all_train_coords = np.vstack(temp_coords_for_edge_scaler)
    all_train_rssi = np.vstack(temp_rssi_for_edge_scaler)
    coord_scaler.fit(all_train_coords)
    rssi_scaler.fit(all_train_rssi)

    # Теперь считаем признаки ребер и обучаем их скейлер
    for session_id in tqdm(train_ids, desc="Fitting Scalers (Pass 2)"):
         session_df = grouped.get_group(session_id)
         n_points = len(session_df)
         if n_points > k:
            coords = session_df[['meas_x', 'meas_y']].values
            rssi = session_df['rssi'].values.reshape(-1, 1)
            norm_coords = coord_scaler.transform(coords)
            norm_rssi = rssi_scaler.transform(rssi)
            coords_tensor = torch.tensor(norm_coords, dtype=torch.float)
            rssi_tensor = torch.tensor(norm_rssi, dtype=torch.float)

            dist_matrix = torch.cdist(coords_tensor, coords_tensor)
            _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
            edge_index_list = []
            source_nodes = torch.arange(n_points)
            for i in range(n_points):
                neighbors = indices[i, 1:]
                source_node_repeated = source_nodes[i].repeat(k)
                edge_index_list.append(torch.stack([source_node_repeated, neighbors], dim=0))
            edge_index = torch.cat(edge_index_list, dim=1)
            row, col = edge_index

            coord_i = coords_tensor[row]
            coord_j = coords_tensor[col]
            rssi_i = rssi_tensor[row]
            rssi_j = rssi_tensor[col]
            distances = torch.linalg.norm(coord_i - coord_j, dim=1, keepdim=True)
            displacement = coord_j - coord_i
            rssi_diff = rssi_i - rssi_j
            raw_edge_features = torch.cat([distances, displacement, rssi_diff], dim=1).numpy()
            edge_feature_placeholder.append(raw_edge_features)

    all_train_edge_features = np.vstack(edge_feature_placeholder)
    edge_scaler = StandardScaler() # StandardScaler для признаков ребер
    edge_scaler.fit(all_train_edge_features)

    scalers = {'coords': coord_scaler, 'rssi': rssi_scaler, 'edge': edge_scaler}
    print("Scalers fitted on training data.")

    # --- Создание графов для всех данных с использованием обученных скейлеров ---
    all_graphs = []
    all_session_data = []
    split_indices = {'train': [], 'val': [], 'test': []}
    current_index = 0

    for session_id in tqdm(session_ids, desc="Creating Graphs"):
        session_df = grouped.get_group(session_id)
        source_coords = (session_df['source_x'].iloc[0], session_df['source_y'].iloc[0])
        meas_df = session_df[['meas_x', 'meas_y', 'rssi']]

        graph_data, session_obj = create_graph_sample(session_id, meas_df, source_coords, k, pos_encoding_dim, scalers)

        if graph_data is not None:
            all_graphs.append(graph_data)
            all_session_data.append(session_obj) # Сохраняем для визуализации

            if session_id in train_ids:
                split_indices['train'].append(current_index)
            elif session_id in val_ids:
                split_indices['val'].append(current_index)
            else: # test_ids
                split_indices['test'].append(current_index)
            current_index += 1

    return all_graphs, all_session_data, scalers, split_indices

# --- 3. Класс Dataset для PyTorch Geometric ---
class SignalDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# --- 4. Описание модели GNN ---
class SignalGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, output_dim=3, num_layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        input_dim = node_feature_dim

        # GAT слои
        for i in range(num_layers):
            # В последнем слое выходная размерность может отличаться или усредняться
            is_last_layer = (i == num_layers - 1)
            out_dim = hidden_dim # if not is_last_layer else hidden_dim # Пример: можно менять
            current_heads = heads # if not is_last_layer else 1 # Пример: можно менять
            concat = True # if not is_last_layer else False # В последнем слое лучше не конкатенировать, а усреднять головы

            conv = GATv2Conv(input_dim, out_dim, heads=current_heads,
                             edge_dim=edge_feature_dim, dropout=dropout,
                             add_self_loops=True, # Важно для GAT
                             concat=concat) # Если concat=True, выходной размер = out_dim * heads
            self.layers.append(conv)

            # Обновляем input_dim для следующего слоя
            input_dim = out_dim * current_heads if concat else out_dim

        # Выходной MLP слой (применяется к каждому узлу)
        # Входная размерность - выход GAT слоя
        self.output_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, output_dim) # output_dim = 3 (sin, cos, raw_width)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
             x = layer(x, edge_index, edge_attr=edge_attr)
             if i < len(self.layers) - 1 : # Не применяем активацию после последнего GAT слоя перед MLP
                 x = F.leaky_relu(x)
             # Можно добавить BatchNorm или LayerNorm здесь, если нужно
             # x = F.dropout(x, p=0.2, training=self.training) # Доп. дропаут между слоями?

        # Применяем MLP к каждому узлу
        output = self.output_mlp(x) # Shape: [num_nodes_in_batch, 3]

        # Разделяем выход на sin, cos, raw_width
        pred_sin = torch.tanh(output[:, 0]) # tanh -> [-1, 1]
        pred_cos = torch.tanh(output[:, 1]) # tanh -> [-1, 1]
        pred_raw_width = output[:, 2]       # Без активации здесь, применим в loss/postprocessing

        return pred_sin, pred_cos, pred_raw_width

# --- 5. Гибридная функция ошибки ---
def compute_hybrid_loss(pred_sin, pred_cos, pred_raw_width, data, loss_weights, device):
    """
    Вычисляет гибридную функцию ошибки.
    Args:
        pred_sin, pred_cos, pred_raw_width: Выходы модели для узлов батча.
        data: Объект батча PyG (содержит data.pos - ненорм. коорд. узлов, data.y - коорд. источника).
        loss_weights (dict): Веса для компонентов ошибки {'dir': lambda1, 'contain': lambda2, 'width': lambda3}.
        device: Устройство ('cuda' или 'cpu').
    Returns:
        tuple: (total_loss, loss_components)
               Общая ошибка, словарь с компонентами ошибки (средние по батчу).
    """
    node_pos = data.pos.to(device) # Ненормированные координаты точек измерений [num_nodes, 2]
    # Координаты источника нужно "размножить" для каждого узла в батче
    # data.y содержит координаты источника для каждого графа
    # data.batch содержит индекс графа для каждого узла
    source_pos_expanded = data.y[data.batch].to(device) # [num_nodes, 2]

    # 1. Истинный вектор направления
    true_vector = source_pos_expanded - node_pos # [num_nodes, 2]
    # Избегаем деления на ноль, если точка измерения совпала с источником
    true_vector_norm = torch.linalg.norm(true_vector, dim=1, keepdim=True) + 1e-8
    true_unit_vector = true_vector / true_vector_norm # [num_nodes, 2]

    # 2. Предсказанный вектор направления (из sin, cos)
    # Важно: Нормализуем предсказанный вектор!
    pred_vector = torch.stack([pred_cos, pred_sin], dim=1) # [num_nodes, 2] (cos=x, sin=y)
    pred_vector_norm = torch.linalg.norm(pred_vector, dim=1, keepdim=True) + 1e-8
    pred_unit_vector = pred_vector / pred_vector_norm

    # 3. Предсказанная ширина угла (в радианах)
    # Используем softplus для гарантии положительной ширины
    pred_width_rad = F.softplus(pred_raw_width) # [num_nodes]
    # Альтернатива: sigmoid * max_angle, например sigmoid * pi
    # pred_width_rad = torch.sigmoid(pred_raw_width) * math.pi

    # --- Компоненты ошибки ---

    # L_dir: Ошибка направления (Cosine Similarity Loss)
    # Усредняем по всем узлам в батче
    cosine_sim = F.cosine_similarity(pred_unit_vector, true_unit_vector, dim=1)
    loss_dir = (1.0 - cosine_sim).mean()

    # L_contain: Ошибка покрытия (Hinge Loss)
    dot_product = torch.sum(pred_unit_vector * true_unit_vector, dim=1)
    # Ограничиваем для стабильности acos
    dot_product_clamped = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
    angle_diff_rad = torch.acos(dot_product_clamped) # Угол между предсказанным и истинным направлением [num_nodes]
    # Штраф, если угол больше половины предсказанной ширины
    containment_error = F.relu(angle_diff_rad - pred_width_rad )
    # containment_error = F.relu(pred_width_rad)
    loss_contain = containment_error.mean() # Усредняем по узлам

    # L_width: Регуляризация ширины
    loss_width = pred_width_rad.mean() # Усредняем по узлам

    # --- Итоговая ошибка ---
    total_loss = (loss_weights['dir'] * loss_dir +
                  loss_weights['contain'] * loss_contain +
                  loss_weights['width'] * loss_width)

    loss_components = {
        'dir': loss_dir.item(),
        'contain': loss_contain.item(),
        'width': loss_width.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_components

# --- 6. Функции для метрик ---
@torch.no_grad()
def calculate_metrics(pred_sin, pred_cos, pred_raw_width, data, angle_threshold_rad, device):
    """ Расчет метрик на батче """
    node_pos = data.pos.to(device)
    source_pos_expanded = data.y[data.batch].to(device)
    true_vector = source_pos_expanded - node_pos
    true_vector_norm = torch.linalg.norm(true_vector, dim=1, keepdim=True) + 1e-8
    true_unit_vector = true_vector / true_vector_norm

    pred_vector = torch.stack([pred_cos, pred_sin], dim=1)
    pred_vector_norm = torch.linalg.norm(pred_vector, dim=1, keepdim=True) + 1e-8
    pred_unit_vector = pred_vector / pred_vector_norm
    pred_width_rad = F.softplus(pred_raw_width)

    dot_product = torch.sum(pred_unit_vector * true_unit_vector, dim=1)
    dot_product_clamped = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
    angle_diff_rad = torch.acos(dot_product_clamped)

    # Метрика 1: Процент точек с углом <= порога И направлением внутри
    correct_direction = angle_diff_rad <= pred_width_rad / 2.0
    narrow_enough = pred_width_rad <= angle_threshold_rad
    accurate_and_narrow = correct_direction & narrow_enough
    accuracy_metric = accurate_and_narrow.float().mean().item()

    # Метрика 2: Процент точек с правильным направлением и их средняя ширина
    correct_direction_mask = correct_direction.float()
    percent_correct_direction = correct_direction_mask.mean().item()
    # Средняя ширина только для тех, где направление верно (избегаем деления на 0)
    if correct_direction_mask.sum() > 0:
        avg_width_correct = (pred_width_rad * correct_direction_mask).sum() / correct_direction_mask.sum()
        avg_width_correct = avg_width_correct.item()
    else:
        avg_width_correct = 0.0 # Или float('nan')

    return {
        'accuracy_narrow': accuracy_metric * 100.0, # В процентах
        'perc_correct_direction': percent_correct_direction * 100.0, # В процентах
        'avg_width_correct_deg': math.degrees(avg_width_correct)
    }


# --- 7. Функции для визуализации ---

# Глобальные переменные для графиков
fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
loss_history = {'train': defaultdict(list), 'val': defaultdict(list)}

fig_pred, ax_pred = plt.subplots(figsize=(10, 10))
# Сохраним оригинальные данные тестового семпла
sample_test_session_data = None

def init_plots():
    """ Инициализация графиков """
    # График потерь
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Losses")
    ax_loss.grid(True)

    # График предсказаний
    ax_pred.set_xlabel("X coordinate")
    ax_pred.set_ylabel("Y coordinate")
    ax_pred.set_title("Predictions for a Sample Test Graph")
    ax_pred.grid(True)
    ax_pred.set_aspect('equal', adjustable='box') # Равный масштаб осей

    plt.ion() # Включаем интерактивный режим
    plt.show()

def update_loss_plot(epoch):
    """ Обновление графика потерь """
    ax_loss.clear()
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Losses")
    ax_loss.grid(True)
    colors = plt.cm.tab10(np.linspace(0, 1, 8)) # Цвета для 4*2 линий

    i = 0
    for split in ['train', 'val']:
        for loss_name in ['total', 'dir', 'contain', 'width']:
            if loss_history[split][loss_name]: # Если есть данные
                epochs = range(1, len(loss_history[split][loss_name]) + 1)
                label = f"{split}_{loss_name}"
                ax_loss.plot(epochs, loss_history[split][loss_name], marker='.', linestyle='-', label=label, color=colors[i % len(colors)])
                i += 1

    ax_loss.legend(loc='upper right', fontsize='small')
    fig_loss.canvas.draw()
    fig_loss.canvas.flush_events() # Обновляем окно
    # Сохранение графика
    fig_loss.savefig(os.path.join(PLOT_DIR, f'loss_plot_epoch_{epoch}.png'))


def update_prediction_plot(model, test_sample_graph, test_session_data, epoch, device):
    """ Обновление графика предсказаний для тестового семпла """
    if test_sample_graph is None or test_session_data is None:
        return

    model.eval()
    with torch.no_grad():
        sample_graph_device = test_sample_graph.to(device)
        pred_sin, pred_cos, pred_raw_width = model(sample_graph_device)
        pred_width_rad = F.softplus(pred_raw_width).cpu().numpy()
        pred_sin = pred_sin.cpu().numpy()
        pred_cos = pred_cos.cpu().numpy()

    ax_pred.clear()
    ax_pred.set_xlabel("X coordinate")
    ax_pred.set_ylabel("Y coordinate")
    ax_pred.set_title(f"Predictions for Test Sample (Epoch {epoch})")
    ax_pred.grid(True)

    # Исходные данные
    meas_x = test_session_data.measurements['meas_x'].values
    meas_y = test_session_data.measurements['meas_y'].values
    source_x = test_session_data.source_x
    source_y = test_session_data.source_y

    # Рисуем точки измерений
    scatter = ax_pred.scatter(meas_x, meas_y, c=test_session_data.measurements['rssi'], cmap='viridis', label='Measurements (color=RSSI)')
    # fig_pred.colorbar(scatter, ax=ax_pred, label='RSSI')

    # Рисуем истинный источник
    ax_pred.plot(source_x, source_y, 'r*', markersize=15, label='True Source')

    # Рисуем предсказанные углы обзора (сектора)
    max_dim = max(np.ptp(meas_x), np.ptp(meas_y), 1.0) # Макс. размах для радиуса сектора
    vis_radius = max_dim * 1 # Визуальный радиус сектора

    for i in range(len(meas_x)):
        center_x, center_y = meas_x[i], meas_y[i]
        sin_i, cos_i = pred_sin[i], pred_cos[i]
        width_rad_i = pred_width_rad[i]

        # Угол центрального направления (от оси X против часовой стрелки)
        angle_center_rad = math.atan2(sin_i, cos_i) # atan2(y, x)
        angle_center_deg = math.degrees(angle_center_rad)

        # Углы границ сектора
        width_deg_i = math.degrees(width_rad_i)
        theta1 = angle_center_deg - width_deg_i / 2.0
        theta2 = angle_center_deg + width_deg_i / 2.0

        # Создаем сектор (Wedge patch)
        wedge = patches.Wedge(center=(center_x, center_y), r=vis_radius,
                              theta1=theta1, theta2=theta2,
                              alpha=0.1, # Полупрозрачность
                              color=plt.cm.viridis(test_session_data.measurements['rssi'].iloc[i] / test_session_data.measurements['rssi'].max())  # Цвет по RSSI
                              )
        ax_pred.add_patch(wedge)

        # Рисуем линию центрального направления для ясности
        end_x = center_x + vis_radius * 0.8 * cos_i
        end_y = center_y + vis_radius * 0.8 * sin_i
        ax_pred.plot([center_x, end_x], [center_y, end_y], color='black', linestyle='--', linewidth=0.6, alpha=0.6)


    # Настройка пределов осей для лучшего вида
    all_x = np.append(meas_x, source_x)
    all_y = np.append(meas_y, source_y)
    ax_pred.set_xlim(all_x.min() - max_dim*0.2, all_x.max() + max_dim*0.2)
    ax_pred.set_ylim(all_y.min() - max_dim*0.2, all_y.max() + max_dim*0.2)

    ax_pred.legend(loc='upper right', fontsize='small')
    ax_pred.set_aspect('equal', adjustable='box')
    fig_pred.canvas.draw()
    fig_pred.canvas.flush_events()
    # Сохранение графика
    fig_pred.savefig(os.path.join(PLOT_DIR, f'prediction_plot_epoch_{epoch}.png'))

# --- 8. Цикл обучения и оценки ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, loss_weights, num_epochs, device, test_sample_graph, test_session_data):
    """ Основной цикл обучения модели """
    best_val_loss = float('inf')
    init_plots() # Инициализация окон для графиков

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        model.train()
        total_train_loss = 0.0
        train_loss_components_agg = defaultdict(float)
        train_metrics_agg = defaultdict(float)
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for i, batch in enumerate(pbar):  # Added enumerate for batch index
            batch = batch.to(device)
            optimizer.zero_grad()

            # --- Debugging prints ---
            if i == 0 and epoch == 1:  # Print only for the first batch of the first epoch
                print("\n--- Debugging Batch Shapes (Epoch 1, Batch 0) ---")
                print(f"batch: {batch}")
                print(f"batch.x shape: {batch.x.shape}")
                print(f"batch.edge_index shape: {batch.edge_index.shape}")
                print(f"batch.edge_attr shape: {batch.edge_attr.shape}")
                print(f"batch.y shape: {batch.y.shape}")  # EXPECT [batch_size, 2]
                print(f"batch.pos shape: {batch.pos.shape}")  # EXPECT [num_nodes_in_batch, 2]
                print(f"batch.batch shape: {batch.batch.shape}")  # EXPECT [num_nodes_in_batch]
                print(f"Number of graphs in batch: {batch.num_graphs}")
                # Check the indexing directly
                try:
                    debug_expanded = batch.y[batch.batch]
                    print(
                        f"Shape after indexing batch.y[batch.batch]: {debug_expanded.shape}")  # EXPECT [num_nodes_in_batch, 2]
                except Exception as e:
                    print(f"Error during debug indexing: {e}")
                print("--- End Debugging ---")
            # --- End Debugging prints ---

            pred_sin, pred_cos, pred_raw_width = model(batch)
            loss, loss_components = loss_fn(pred_sin, pred_cos, pred_raw_width, batch, loss_weights, device)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            for k, v in loss_components.items():
                train_loss_components_agg[k] += v

            metrics = calculate_metrics(pred_sin, pred_cos, pred_raw_width, batch, ANGLE_THRESHOLD_RADIANS, device)
            for k, v in metrics.items():
                train_metrics_agg[k] += v

            num_train_batches += 1
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc(N)': f"{metrics['accuracy_narrow']:.1f}%",
                'Width(C)': f"{metrics['avg_width_correct_deg']:.2f}°"
                })

        avg_train_loss = total_train_loss / num_train_batches
        print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.4f}")
        for k in train_loss_components_agg:
            avg_val = train_loss_components_agg[k] / num_train_batches
            loss_history['train'][k].append(avg_val)
            if k == 'dir':
                print(f"  Avg Train Loss ({k}): {avg_val:.4f} => {(avg_val * LOSS_LAMBDA_DIR):.4f}")
            elif k == 'contain':
                print(f"  Avg Train Loss ({k}): {avg_val:.4f} => {(avg_val * LOSS_LAMBDA_CONTAIN):.4f}")
            elif k == 'width':
                print(f"  Avg Train Loss ({k}): {avg_val:.4f} => {(avg_val * LOSS_LAMBDA_WIDTH):.4f}")
            elif k == 'total':
                print(f"  Avg Train Loss ({k}): {avg_val:.4f} => {(avg_val):.4f}")
        for k in train_metrics_agg:
             avg_val = train_metrics_agg[k] / num_train_batches
             print(f"  Avg Train Metric ({k}): {avg_val:.2f}{'%' if 'perc' in k or 'acc' in k else ('rad' if 'rad' in k else '')}")


        # --- Валидация ---
        model.eval()
        total_val_loss = 0.0
        val_loss_components_agg = defaultdict(float)
        val_metrics_agg = defaultdict(float)
        num_val_batches = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
            for batch in pbar_val:
                batch = batch.to(device)
                pred_sin, pred_cos, pred_raw_width = model(batch)
                loss, loss_components = loss_fn(pred_sin, pred_cos, pred_raw_width, batch, loss_weights, device)

                total_val_loss += loss.item()
                for k, v in loss_components.items():
                    val_loss_components_agg[k] += v

                metrics = calculate_metrics(pred_sin, pred_cos, pred_raw_width, batch, ANGLE_THRESHOLD_RADIANS, device)
                for k, v in metrics.items():
                    val_metrics_agg[k] += v

                num_val_batches += 1
                pbar_val.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc(N)': f"{metrics['accuracy_narrow']:.1f}%",
                    'Width(C)': f"{metrics['avg_width_correct_deg']:.2f}°"
                    })

        avg_val_loss = total_val_loss / num_val_batches
        print(f"Epoch {epoch} Avg Validation Loss: {avg_val_loss:.4f}")
        for k in val_loss_components_agg:
            avg_val = val_loss_components_agg[k] / num_val_batches
            loss_history['val'][k].append(avg_val)
            if k == 'dir':
                print(f"  Avg Val Loss ({k}): {avg_val:.4f} => {(avg_val * LOSS_LAMBDA_DIR):.4f}")
            elif k == 'contain':
                print(f"  Avg Val Loss ({k}): {avg_val:.4f} => {(avg_val * LOSS_LAMBDA_CONTAIN):.4f}")
            elif k == 'width':
                print(f"  Avg Val Loss ({k}): {avg_val:.4f} => {(avg_val * LOSS_LAMBDA_WIDTH):.4f}")
            elif k == 'total':
                print(f"  Avg Val Loss ({k}): {avg_val:.4f} => {(avg_val):.4f}")


        for k in val_metrics_agg:
             avg_val = val_metrics_agg[k] / num_val_batches
             print(f"  Avg Val Metric ({k}): {avg_val:.2f}{'%' if 'perc' in k or 'acc' in k else ('rad' if 'rad' in k else '')}")

        # Обновление графиков
        update_loss_plot(epoch)
        if epoch % PLOT_UPDATE_FREQ == 0 or epoch == num_epochs:
             update_prediction_plot(model, test_sample_graph, test_session_data, epoch, device)

        # Уменьшение learning rate
        scheduler.step(avg_val_loss)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"*** Best model saved with validation loss: {best_val_loss:.4f} ***")

        plt.pause(0.1) # Даем время на отрисовку

    plt.ioff() # Выключаем интерактивный режим
    plt.show() # Показываем финальные графики


# --- Основной скрипт ---
# if __name__ == "__main__":
#     # 1. Загрузка и подготовка данных
#     all_graphs, all_session_data, scalers, split_indices = load_data(
#         CSV_FILE, KNN_K, POS_ENCODING_DIM
#     )
#
#     # 2. Выбор семпла для визуализации и фильтрация тестовых индексов
#     test_sample_graph = None
#     sample_test_session_data = None
#     random_test_index_in_list = -1 # Инициализация
#
#     if split_indices['test']: # Убедимся, что есть тестовые индексы
#          # Выбираем случайный индекс из *оригинального* списка тестовых индексов
#          random_test_index_in_list = random.choice(split_indices['test'])
#
#          # Получаем данные для визуализации по этому индексу
#          test_sample_graph = all_graphs[random_test_index_in_list]
#          sample_test_session_data = all_session_data[random_test_index_in_list] # Индексы совпадают
#          print(f"\nSelected session {sample_test_session_data.session_id} for prediction visualization.")
#
#          # --- ИЗМЕНЕНИЕ: Создаем отфильтрованный список тестовых индексов ---
#          # Исключаем индекс графика, выбранного для визуализации
#          filtered_test_indices = [idx for idx in split_indices['test'] if idx != random_test_index_in_list]
#          print(f"Original test indices: {len(split_indices['test'])}, Filtered test indices for evaluation: {len(filtered_test_indices)}")
#          # --- КОНЕЦ ИЗМЕНЕНИЯ ---
#
#          # Создаем датасеты, используя отфильтрованный список для теста
#          train_dataset = SignalDataset([all_graphs[i] for i in split_indices['train']])
#          val_dataset = SignalDataset([all_graphs[i] for i in split_indices['val']])
#          # Используем отфильтрованные индексы для тестового датасета!
#          test_dataset = SignalDataset([all_graphs[i] for i in filtered_test_indices])
#
#     else:
#          # Если тестовых данных нет
#          print("\nWarning: No test samples available for visualization or evaluation.")
#          train_dataset = SignalDataset([all_graphs[i] for i in split_indices['train']])
#          val_dataset = SignalDataset([all_graphs[i] for i in split_indices['val']])
#          test_dataset = SignalDataset([]) # Пустой тестовый датасет
#          filtered_test_indices = [] # На всякий случай
#
#
#     # 3. Создание даталоадеров
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#     # test_loader будет использовать test_dataset, который не содержит граф для визуализации
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#
#     # 4. Инициализация модели (проверяем, есть ли данные для обучения)
#     if train_dataset:
#         sample_data = train_dataset.get(0)
#         node_feature_dim = sample_data.num_node_features
#         edge_feature_dim = sample_data.num_edge_features
#         print(f"Node feature dimension: {node_feature_dim}")
#         print(f"Edge feature dimension: {edge_feature_dim}")
#
#         model = SignalGNN(
#             node_feature_dim=node_feature_dim,
#             edge_feature_dim=edge_feature_dim,
#             hidden_dim=HIDDEN_DIM,
#             output_dim=3,
#             num_layers=NUM_GAT_LAYERS,
#             heads=GAT_HEADS
#         ).to(DEVICE)
#         print(model)
#
#         # 5. Оптимизатор и планировщик
#         optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
#
#         # 6. Веса для функции ошибки
#         loss_weights = {
#             'dir': LOSS_LAMBDA_DIR,
#             'contain': LOSS_LAMBDA_CONTAIN,
#             'width': LOSS_LAMBDA_WIDTH
#         }
#
#         # 7. Запуск обучения
#         # Передаем оригинальные test_sample_graph и sample_test_session_data для графика
#         start_time = time.time()
#         train_model(
#             model, train_loader, val_loader, optimizer, scheduler,
#             compute_hybrid_loss, loss_weights, NUM_EPOCHS, DEVICE,
#             test_sample_graph, sample_test_session_data
#         )
#         end_time = time.time()
#         print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
#
#         # 8. Оценка на тестовом наборе (используя test_loader без графика для визуализации)
#         # Убедимся, что test_loader не пустой перед оценкой
#         if len(test_loader) > 0:
#             print("\nEvaluating on Test Set with the best model...")
#             # Загружаем лучшую модель (исправление с map_location уже должно быть применено)
#             model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
#             # model.to(DEVICE) # На всякий случай, или если map_location не используется
#             model.eval()
#             total_test_loss = 0.0
#             test_loss_components_agg = defaultdict(float)
#             test_metrics_agg = defaultdict(float)
#             num_test_batches = 0
#             with torch.no_grad():
#                  pbar_test = tqdm(test_loader, desc="Testing") # test_loader теперь без семпла для виз.
#                  for batch in pbar_test:
#                       # >>> Сюда ошибка больше не должна доходить <<<
#                       batch = batch.to(DEVICE)
#                       pred_sin, pred_cos, pred_raw_width = model(batch)
#                       loss, loss_components = compute_hybrid_loss(pred_sin, pred_cos, pred_raw_width, batch, loss_weights, DEVICE)
#                       total_test_loss += loss.item()
#                       for k, v in loss_components.items():
#                            test_loss_components_agg[k] += v
#                       metrics = calculate_metrics(pred_sin, pred_cos, pred_raw_width, batch, ANGLE_THRESHOLD_RADIANS, DEVICE)
#                       for k, v in metrics.items():
#                            test_metrics_agg[k] += v
#                       num_test_batches += 1
#
#             # Проверка деления на ноль, если test_loader был пуст (хотя мы проверили len > 0)
#             if num_test_batches > 0:
#                 avg_test_loss = total_test_loss / num_test_batches
#                 print(f"Average Test Loss: {avg_test_loss:.4f}")
#                 for k in test_loss_components_agg:
#                     avg_val = test_loss_components_agg[k] / num_test_batches
#                     print(f"  Avg Test Loss ({k}): {avg_val:.4f}")
#                 for k in test_metrics_agg:
#                      avg_val = test_metrics_agg[k] / num_test_batches
#                      print(f"  Avg Test Metric ({k}): {avg_val:.2f}{'%' if 'perc' in k or 'acc' in k else ('rad' if 'rad' in k else '')}")
#             else:
#                 print("Test loader was empty, skipping test evaluation metrics.")
#         else:
#             print("Test loader is empty, skipping evaluation on test set.")
#
#     else:
#         print("No training data available. Exiting.")




# --- НОВАЯ ФУНКЦИЯ 1: Подготовка и сохранение скейлеров ---

def plot_single_prediction(pred_sin, pred_cos, pred_width_rad, session_data, save_path): # <-- Функция из предыдущего ответа
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Predictions for Session {session_data.session_id}")
    ax.grid(True)
    meas_x = session_data.measurements['meas_x'].values; meas_y = session_data.measurements['meas_y'].values
    rssi = session_data.measurements['rssi'].values
    source_x = session_data.source_x; source_y = session_data.source_y
    rssi_min, rssi_max = rssi.min(), rssi.max()
    rssi_norm = (rssi - rssi_min) / (rssi_max - rssi_min + 1e-8) if rssi_max > rssi_min else np.zeros_like(rssi)
    scatter = ax.scatter(meas_x, meas_y, c=rssi, cmap='viridis', label='Measurements (color=RSSI)', zorder=5)
    try: # Добавим try-except для colorbar
      fig.colorbar(scatter, ax=ax, label='RSSI')
    except Exception as e:
      print(f"Warning: Could not draw colorbar: {e}")
    ax.plot(source_x, source_y, 'r*', markersize=15, label='True Source', zorder=10)
    max_dim_x = np.ptp(meas_x) if len(meas_x) > 1 else 1.0
    max_dim_y = np.ptp(meas_y) if len(meas_y) > 1 else 1.0
    max_dim = max(max_dim_x, max_dim_y, 1.0)
    vis_radius = max_dim * 0.8
    for i in range(len(meas_x)):
        center_x, center_y = meas_x[i], meas_y[i]
        sin_i, cos_i = pred_sin[i], pred_cos[i]
        width_rad_i = pred_width_rad[i]
        angle_center_rad = math.atan2(sin_i, cos_i)
        angle_center_deg = math.degrees(angle_center_rad)
        width_deg_i = math.degrees(width_rad_i)
        width_deg_i = min(width_deg_i, 359.9)
        theta1 = angle_center_deg - width_deg_i / 2.0
        theta2 = angle_center_deg + width_deg_i / 2.0
        wedge_color = plt.cm.viridis(rssi_norm[i]) if len(meas_x)>1 else plt.cm.viridis(0.5)
        wedge = patches.Wedge(center=(center_x, center_y), r=vis_radius,
                              theta1=theta1, theta2=theta2, alpha=0.25,
                              color=wedge_color, zorder=3)
        ax.add_patch(wedge)
        end_x = center_x + vis_radius * cos_i
        end_y = center_y + vis_radius * sin_i
        ax.plot([center_x, end_x], [center_y, end_y], color='black', linestyle='--', linewidth=0.7, alpha=0.5, zorder=4)
    all_x = np.append(meas_x, source_x); all_y = np.append(meas_y, source_y)
    x_min, x_max = all_x.min(), all_x.max(); y_min, y_max = all_y.min(), all_y.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    ax.legend(loc='best', fontsize='small')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def prepare_and_save_scalers(train_csv_path, scaler_save_path, k, pos_encoding_dim):
    """
    Загружает ОРИГИНАЛЬНЫЙ ТРЕНИРОВОЧНЫЙ датасет, ОБУЧАЕТ на нем скейлеры и СОХРАНЯЕТ их.
    Args:
        train_csv_path (str): Путь к CSV файлу, который использовался для обучения.
        scaler_save_path (str): Куда сохранить файл скейлеров (.joblib).
        k (int): Параметр KNN (должен совпадать с обучением).
        pos_encoding_dim (int): Размерность Positional Encoding (должна совпадать).
    Returns:
        dict: Словарь с обученными скейлерами.
    """
    print(f"--- Preparing and Saving Scalers from: {train_csv_path} ---")
    try:
        df = pd.read_csv(train_csv_path)
    except FileNotFoundError:
        print(f"Error: Training CSV file not found at {train_csv_path}")
        return None

    grouped = df.groupby('session_id')
    session_ids = list(grouped.groups.keys())

    # --- Обучение скейлеров на ВСЕХ данных из этого файла ---
    # (Предполагается, что этот файл СОДЕРЖИТ ТОЛЬКО ТРЕНИРОВОЧНЫЕ ДАННЫЕ,
    # или что для консистентности мы обучаем скейлеры на всем этом наборе)
    # Если при ОРИГИНАЛЬНОМ обучении было разделение train/test, то здесь
    # нужно воспроизвести ТОЛЬКО train часть для обучения скейлеров.
    # Для простоты примера, обучаем на всем содержимом train_csv_path.

    coord_scaler = MinMaxScaler()
    rssi_scaler = StandardScaler()
    edge_feature_placeholder = []
    coords_placeholder = []
    rssi_placeholder = []

    print("Fitting scalers on the provided dataset...")
    # Pass 1: Сбор данных для coord и rssi скейлеров
    for session_id in tqdm(session_ids, desc="Fitting Scalers (Pass 1)", leave=False):
        session_df = grouped.get_group(session_id)
        if len(session_df) > k:
            coords = session_df[['meas_x', 'meas_y']].values
            rssi = session_df['rssi'].values.reshape(-1, 1)
            coords_placeholder.append(coords)
            rssi_placeholder.append(rssi)

    if not coords_placeholder:
        print("Error: No valid sessions found in the dataset to fit scalers.")
        return None

    all_coords = np.vstack(coords_placeholder)
    all_rssi = np.vstack(rssi_placeholder)
    coord_scaler.fit(all_coords)
    rssi_scaler.fit(all_rssi)
    print("Coord and RSSI scalers fitted.")

    # Pass 2: Сбор данных для edge скейлера (используя уже обученные coord/rssi)
    for session_id in tqdm(session_ids, desc="Fitting Scalers (Pass 2)", leave=False):
         session_df = grouped.get_group(session_id)
         n_points = len(session_df)
         if n_points > k:
            coords = session_df[['meas_x', 'meas_y']].values
            rssi = session_df['rssi'].values.reshape(-1, 1)
            # Применяем обученные скейлеры
            norm_coords = coord_scaler.transform(coords)
            norm_rssi = rssi_scaler.transform(rssi)
            coords_tensor = torch.tensor(norm_coords, dtype=torch.float)
            rssi_tensor = torch.tensor(norm_rssi, dtype=torch.float)
            # Создаем ребра KNN
            dist_matrix = torch.cdist(coords_tensor, coords_tensor)
            _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
            edge_index_list = []
            source_nodes = torch.arange(n_points)
            for i in range(n_points):
                neighbors = indices[i, 1:]
                source_node_repeated = source_nodes[i].repeat(k)
                edge_index_list.append(torch.stack([source_node_repeated, neighbors], dim=0))
            if not edge_index_list: continue # Пропускаем, если нет ребер
            edge_index = torch.cat(edge_index_list, dim=1)
            row, col = edge_index
            # Считаем признаки ребер
            coord_i = coords_tensor[row]; coord_j = coords_tensor[col]
            rssi_i = rssi_tensor[row]; rssi_j = rssi_tensor[col]
            distances = torch.linalg.norm(coord_i - coord_j, dim=1, keepdim=True)
            displacement = coord_j - coord_i
            rssi_diff = rssi_i - rssi_j
            raw_edge_features = torch.cat([distances, displacement, rssi_diff], dim=1).numpy()
            edge_feature_placeholder.append(raw_edge_features)

    if not edge_feature_placeholder:
        print("Error: No valid edges found to fit edge scaler.")
        return None

    all_train_edge_features = np.vstack(edge_feature_placeholder)
    edge_scaler = StandardScaler()
    edge_scaler.fit(all_train_edge_features)
    print("Edge scaler fitted.")

    scalers = {'coords': coord_scaler, 'rssi': rssi_scaler, 'edge': edge_scaler}

    # Сохраняем скейлеры
    try:
        joblib.dump(scalers, scaler_save_path)
        print(f"Scalers successfully saved to {scaler_save_path}")
    except Exception as e:
        print(f"Error saving scalers: {e}")
        return None

    return scalers

# --- НОВАЯ ФУНКЦИЯ 2: Оценка на внешнем тестовом файле ---

def evaluate_external_test_set(model_path, test_csv_path, scaler_path, eval_plot_dir, config):
    """
    Загружает модель, ЗАГРУЖАЕТ скейлеры, оценивает на НОВОМ тестовом наборе,
    рисует и сохраняет графики, вычисляет метрики.

    Args:
        model_path (str): Путь к сохраненному файлу модели (.pth).
        test_csv_path (str): Путь к НОВОМУ CSV файлу с тестовыми данными.
        scaler_path (str): Путь к сохраненному файлу скейлеров (.joblib).
        eval_plot_dir (str): Папка для сохранения графиков предсказаний.
        config (dict): Словарь с конфигурационными параметрами.
    """
    print(f"--- Evaluating Model {model_path} on Test File: {test_csv_path} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(eval_plot_dir, exist_ok=True)

    # --- 1. Загрузка скейлеров ---
    try:
        scalers = joblib.load(scaler_path)
        print(f"Loaded scalers from {scaler_path}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}. Please run scaler preparation first.")
        return
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # --- 2. Загрузка и обработка ТЕСТОВОГО датасета ---
    try:
        df_test = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: Test CSV file not found at {test_csv_path}")
        return

    grouped_test = df_test.groupby('session_id')
    test_session_ids = list(grouped_test.groups.keys())

    test_graphs = []
    test_session_objs = []
    print(f"Processing {len(test_session_ids)} sessions from {test_csv_path}...")
    for session_id in tqdm(test_session_ids, desc="Creating Test Graphs"):
        session_df = grouped_test.get_group(session_id)
        source_coords = (session_df['source_x'].iloc[0], session_df['source_y'].iloc[0])
        meas_df = session_df[['meas_x', 'meas_y', 'rssi']]

        # Используем ЗАГРУЖЕННЫЕ скейлеры
        graph_data, session_obj = create_graph_sample(
            session_id, meas_df, source_coords,
            config['KNN_K'], config['POS_ENCODING_DIM'], scalers
        )
        if graph_data is not None:
            test_graphs.append(graph_data)
            test_session_objs.append(session_obj)

    if not test_graphs:
        print("No valid graphs could be created from the test file.")
        return

    print(f"Created {len(test_graphs)} test graphs.")

    # --- 3. Загрузка модели ---
    sample_data = test_graphs[0] # Берем первый для определения размерностей
    node_feature_dim = sample_data.num_node_features
    edge_feature_dim = sample_data.num_edge_features

    model = SignalGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=config['HIDDEN_DIM'],
        num_layers=config['NUM_GAT_LAYERS'],
        heads=config['GAT_HEADS']
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        # Дополнительная информация об ошибке state_dict
        print("This might happen if the model architecture in the script doesn't match the saved weights.")
        print("Check HIDDEN_DIM, NUM_GAT_LAYERS, GAT_HEADS etc.")
        return


    # --- 4. Оценка, визуализация и сбор метрик ---
    all_preds_sin = []
    all_preds_cos = []
    all_preds_raw_width = []
    all_batch_data_for_metrics = [] # Собираем данные для итогового расчета метрик

    loss_weights = {
        'dir': config['LOSS_LAMBDA_DIR'],
        'contain': config['LOSS_LAMBDA_CONTAIN'],
        'width': config['LOSS_LAMBDA_WIDTH']
    }
    total_test_loss = 0.0
    test_loss_components_agg = defaultdict(float)
    num_test_nodes_processed = 0

    print(f"Evaluating {len(test_graphs)} test samples...")
    with torch.no_grad():
        for i, graph_data in enumerate(tqdm(test_graphs, desc="Evaluating and Plotting Test Set")):
            session_obj = test_session_objs[i]
            graph_data = graph_data.to(device)

            pred_sin, pred_cos, pred_raw_width = model(graph_data)

            # Визуализация
            pred_width_rad_np = F.softplus(pred_raw_width).cpu().numpy()
            pred_sin_np = pred_sin.cpu().numpy()
            pred_cos_np = pred_cos.cpu().numpy()
            plot_savename = os.path.join(eval_plot_dir, f"test_prediction_session_{session_obj.session_id}.png")
            plot_single_prediction(pred_sin_np, pred_cos_np, pred_width_rad_np, session_obj, plot_savename)

            # Сбор данных для итоговых метрик
            all_preds_sin.append(pred_sin.cpu()) # Перемещаем на CPU для сбора
            all_preds_cos.append(pred_cos.cpu())
            all_preds_raw_width.append(pred_raw_width.cpu())
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long) # Добавляем batch индекс для одного графа
            all_batch_data_for_metrics.append(graph_data.cpu()) # Сохраняем на CPU

            # Расчет потерь для текущего графа (не обязательно, но можно)
            loss, loss_components = compute_hybrid_loss(pred_sin, pred_cos, pred_raw_width, graph_data, loss_weights, device)
            total_test_loss += loss.item() * graph_data.num_nodes
            for k, v in loss_components.items():
                test_loss_components_agg[k] += v * graph_data.num_nodes
            num_test_nodes_processed += graph_data.num_nodes

    print("Finished evaluation loop. Calculating final metrics...")

    # --- 5. Расчет и вывод итоговых метрик ---
    if not all_preds_sin or num_test_nodes_processed == 0:
        print("No predictions were generated. Cannot calculate final metrics.")
        return

    # Объединяем все предсказания и данные
    final_preds_sin = torch.cat(all_preds_sin, dim=0).to(device)
    final_preds_cos = torch.cat(all_preds_cos, dim=0).to(device)
    final_preds_raw_width = torch.cat(all_preds_raw_width, dim=0).to(device)

    # Собираем один большой батч
    try:
        final_batch = Batch.from_data_list(all_batch_data_for_metrics).to(device)

        # Пересчитываем потери на всем батче (опционально, для консистентности)
        # final_loss, final_loss_components = compute_hybrid_loss(
        #     final_preds_sin, final_preds_cos, final_preds_raw_width,
        #     final_batch, loss_weights, device
        # )

        # Считаем метрики на всем батче
        final_metrics = calculate_metrics(
            final_preds_sin, final_preds_cos, final_preds_raw_width,
            final_batch, config['ANGLE_THRESHOLD_RADIANS'], device
        )

        print("\n--- Final Metrics on External Test Set ---")
        # Выводим агрегированные потери (рассчитанные по отдельным графам)
        avg_test_loss = total_test_loss / num_test_nodes_processed
        print(f"Average Test Loss (Total, aggregated): {avg_test_loss:.4f}")
        for k in test_loss_components_agg:
            avg_val = test_loss_components_agg[k] / num_test_nodes_processed
            print(f"  Avg Test Loss ({k}, aggregated): {avg_val:.4f}")

        # Выводим итоговые метрики (рассчитанные на объединенном батче)
        print("\nMetrics calculated on the combined batch:")
        for k, v in final_metrics.items():
            unit = '%' if 'perc' in k or 'acc' in k else ('°' if 'deg' in k else '')
            print(f"  Metric ({k}): {v:.2f}{unit}")

    except RuntimeError as e:
         print(f"\nError creating final batch for metrics (possibly OOM): {e}")
         print("Metrics calculation might be incomplete or based only on aggregated losses.")
         avg_test_loss = total_test_loss / num_test_nodes_processed
         print(f"Average Test Loss (Total, aggregated): {avg_test_loss:.4f}")
         for k in test_loss_components_agg:
             avg_val = test_loss_components_agg[k] / num_test_nodes_processed
             print(f"  Avg Test Loss ({k}, aggregated): {avg_val:.4f}")
         print("Could not calculate aggregated percentage/width metrics due to memory constraints.")

    print(f"Evaluation plots saved to: {eval_plot_dir}")
    print("--- Evaluation Finished ---")


# --- Основной скрипт (модифицирован для вызова новых функций) ---
if __name__ == "__main__":

    ORIGINAL_TRAIN_CSV_FILE = CSV_FILE
    SCALER_SAVE_PATH = 'scalers.joblib'
    EVAL_PLOT_DIR = 'eval_plots'

    parser = argparse.ArgumentParser(description="Train GNN model or Evaluate on Test Set.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'prepare_scalers', 'evaluate'],
                        help="Operation mode: 'train' to train the model, "
                             "'prepare_scalers' to fit and save scalers from training data, "
                             "'evaluate' to evaluate model on a new test set.")
    parser.add_argument('--train_csv', type=str, default=ORIGINAL_TRAIN_CSV_FILE,
                        help="Path to the CSV file used for training (and for preparing scalers).")
    parser.add_argument('--test_csv', type=str, default='resources/aboba.csv', # Пример имени файла
                        help="Path to the external CSV file for evaluation mode.")
    parser.add_argument('--model_path', type=str, default='saved_models/' + MODEL_SAVE_PATH,
                        help="Path to save/load the model weights.")
    parser.add_argument('--scaler_path', type=str, default=SCALER_SAVE_PATH,
                        help="Path to save/load the scalers.")
    parser.add_argument('--eval_plots_dir', type=str, default=EVAL_PLOT_DIR,
                        help="Directory to save evaluation plots.")

    args = parser.parse_args()

    # Общая конфигурация (может быть переопределена аргументами, если добавить их в парсер)
    config = {
        'KNN_K': KNN_K,
        'POS_ENCODING_DIM': POS_ENCODING_DIM,
        'HIDDEN_DIM': HIDDEN_DIM,
        'NUM_GAT_LAYERS': NUM_GAT_LAYERS,
        'GAT_HEADS': GAT_HEADS,
        'LOSS_LAMBDA_DIR': LOSS_LAMBDA_DIR,
        'LOSS_LAMBDA_CONTAIN': LOSS_LAMBDA_CONTAIN,
        'LOSS_LAMBDA_WIDTH': LOSS_LAMBDA_WIDTH,
        'ANGLE_THRESHOLD_RADIANS': ANGLE_THRESHOLD_RADIANS,
        'DEVICE': DEVICE,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'NUM_EPOCHS': NUM_EPOCHS,
        'PLOT_UPDATE_FREQ': PLOT_UPDATE_FREQ,
        'PLOT_DIR': PLOT_DIR
    }

    if args.mode == 'train':
        print("--- Starting Training Mode ---")
        # --- Код обучения из оригинального train_main-v3.py ---
        # 1. Загрузка и подготовка данных (с обучением скейлеров и разделением)
        # Важно: Убедитесь, что load_data в режиме обучения делает то же самое,
        # что и в оригинальном скрипте (включая сохранение скейлеров, если нужно)
        # Примерный вызов load_data, как в оригинале:
        # all_graphs, all_session_data, scalers, split_indices = load_data(...) # Нужна оригинальная реализация

        # !!! Необходимо вставить сюда ВЕСЬ код блока if __name__ == "__main__":
        #     из оригинального train_main-v3.py, который отвечает за обучение,
        #     включая load_data, создание датасетов/лоадеров, инициализацию модели,
        #     оптимизатора, планировщика и вызов train_model(...)
        # !!! Вместо заглушки ниже:
        print("Training code should be placed here (copied from original script).")
        # Пример:
        # 1. Load data (using ORIGINAL_TRAIN_CSV_FILE, fit scalers, split)
        # 2. Create Datasets and DataLoaders (train, val, test)
        # 3. Initialize Model, Optimizer, Scheduler
        # 4. Call train_model(...)
        pass # Убрать pass и вставить код обучения


    elif args.mode == 'prepare_scalers':
        print("--- Starting Scaler Preparation Mode ---")
        prepare_and_save_scalers(
            train_csv_path=args.train_csv,
            scaler_save_path=args.scaler_path,
            k=config['KNN_K'],
            pos_encoding_dim=config['POS_ENCODING_DIM']
        )

    elif args.mode == 'evaluate':
        print("--- Starting Evaluation Mode ---")
        if not os.path.exists(args.model_path):
             print(f"Model file not found at {args.model_path}. Cannot evaluate.")
        elif not os.path.exists(args.scaler_path):
             print(f"Scaler file not found at {args.scaler_path}. Run --mode prepare_scalers first.")
        else:
            evaluate_external_test_set(
                model_path=args.model_path,
                test_csv_path=args.test_csv,
                scaler_path=args.scaler_path,
                eval_plot_dir=args.eval_plots_dir,
                config=config
            )

    else:
        print(f"Unknown mode: {args.mode}")