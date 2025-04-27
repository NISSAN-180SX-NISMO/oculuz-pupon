import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("qt5agg")

def visualize_chunk(df: pd.DataFrame, chunk_start: int):
    grouped = df.groupby('source_id')

    for source_id, group in grouped:
        plt.figure(figsize=(8, 8))
        plt.xlim(0, GRID_SIZE)
        plt.ylim(0, GRID_SIZE)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Source ID: {source_id}")

        if EMULATE_DD:
            x_meas = (group["longitude"] - BASE_COORDINATES[0]) / 0.0001
            y_meas = (group["latitude"] - BASE_COORDINATES[1]) / 0.0001
            x_src = (group["source_longitude"].iloc[0] - BASE_COORDINATES[0]) / 0.0001
            y_src = (group["source_latitude"].iloc[0] - BASE_COORDINATES[1]) / 0.0001
        else:
            x_meas = group["longitude"]
            y_meas = group["latitude"]
            x_src = group["source_longitude"].iloc[0]
            y_src = group["source_latitude"].iloc[0]

        # Измерения — синие точки
        plt.scatter(x_meas, y_meas, s=10, c='blue', alpha=0.6, label='Measurements')

        # Источник — красная точка
        plt.scatter(x_src, y_src, s=50, c='red', label='Source')

        # Азимуты — стрелки
        arrow_length = 20
        azimuths = np.radians(group["azimuth"])
        dx = arrow_length * np.sin(azimuths)
        dy = arrow_length * np.cos(azimuths)

        plt.quiver(
            x_meas, y_meas, dx, dy,
            angles='xy', scale_units='xy', scale=1,
            color='green', width=0.003, alpha=0.7, label='Azimuth'
        )

        plt.legend()
        plt.grid(True)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.tight_layout()
        plt.show()




# Конфигурация
GRID_SIZE = 1000
SOURCES_COUNT = 1000
MEASUREMENTS_PER_SOURCE = 100
BASE_COORDINATES = (50.0, 30.0)
CHUNK_SIZE = 100
OUTPUT_FILE = "dataset.csv"
NOISE_IMAGE_PATH = "noise_map/noise.png"
MAX_DISTANCE = 1000  # Максимальное расстояние для расчета RSSI
EMULATE_DD = False  # Флаг для выбора системы координат: True - географические, False - декартовы


def generate_sources(n: int) -> np.ndarray:
    """Генерация массива источников с отступами"""
    margin = 150
    return np.column_stack((
        np.random.randint(margin, GRID_SIZE - margin, n),
        np.random.randint(margin, GRID_SIZE - margin, n)
    ))


def calculate_rssi(distances: np.ndarray) -> np.ndarray:
    """Преобразование расстояний в RSSI по логарифмической шкале"""
    distances = np.clip(distances, 0, MAX_DISTANCE)
    # Логарифмическая формула: от 0 (на источнике) до -140 (на расстоянии 1000+)
    return -140 * np.log10(distances + 1) / np.log10(MAX_DISTANCE + 1)


def apply_noise_grid() -> np.ndarray:
    """Генерация шумовой карты в диапазоне ±5 dBm"""
    try:
        noise_img = Image.open(NOISE_IMAGE_PATH).convert('L')
        noise_img = noise_img.resize((GRID_SIZE, GRID_SIZE))
        noise = np.array(noise_img) / 255.0  # Нормализация 0-1
        return noise * 10 - 5  # Преобразование в диапазон -5..+5
    except FileNotFoundError:
        print("Using random noise")
        return np.random.uniform(-5, 5, (GRID_SIZE, GRID_SIZE))


def generate_measurements_batch(sources: np.ndarray,
                                measurements_per_source: int) -> Tuple[np.ndarray, np.ndarray]:
    """Векторизованная генерация измерений с распределением по сторонам многоугольника"""
    n = len(sources)
    measurements = []
    source_ids = []

    for src_idx in range(n):
        source = sources[src_idx]

        # Параметры многоугольника
        sides = np.random.choice([3, 4, 5])
        radius = np.random.uniform(300, 600)

        # Генерация вершин многоугольника
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        poly_x = source[0] + radius * np.cos(angles)
        poly_y = source[1] + radius * np.sin(angles)

        # Рассчитываем точки вдоль каждой стороны
        points_per_side, remainder = divmod(measurements_per_source, sides)
        coords = []

        for i in range(sides):
            # Определяем текущую и следующую вершины
            x1, y1 = poly_x[i], poly_y[i]
            x2, y2 = poly_x[(i + 1) % sides], poly_y[(i + 1) % sides]

            # Генерация точек на стороне с шумом
            num_points = points_per_side + (1 if i < remainder else 0)
            t = np.linspace(0, 1, num_points)

            # Базовые координаты на стороне
            base_x = x1 + t * (x2 - x1)
            base_y = y1 + t * (y2 - y1)

            # Добавляем перпендикулярный шум
            dx = x2 - x1
            dy = y2 - y1
            normal = np.array([-dy, dx], dtype=float)
            if np.linalg.norm(normal) > 0:
                normal /= np.linalg.norm(normal)

            noise = np.random.uniform(-50, 50, num_points)
            offset_x = noise * normal[0]
            offset_y = noise * normal[1]

            coords.append(np.column_stack([
                base_x + offset_x,
                base_y + offset_y
            ]))

        # Собираем все точки для источника
        src_measurements = np.vstack(coords)
        measurements.append(src_measurements)
        source_ids.extend([src_idx] * measurements_per_source)

    return np.vstack(measurements), np.array(source_ids)


def process_chunk(sources: np.ndarray,
                  noise_grid: np.ndarray,
                  chunk_start: int) -> pd.DataFrame:
    """Обработка чанка данных с добавлением целевых значений"""
    measurements, source_ids = generate_measurements_batch(
        sources, MEASUREMENTS_PER_SOURCE)

    meas_mask = (measurements[:, 0] >= 0) & (measurements[:, 1] >= 0)
    measurements = measurements[meas_mask]
    source_ids = source_ids[meas_mask]

    # Получаем координаты источников для текущего чанка
    source_coords = sources[source_ids]

    # Расчет расстояний (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    delta = measurements - source_coords
    distances = np.hypot(delta[:, 0], delta[:, 1])

    # Расчет азимута (в градусах 0-360)
    dx = source_coords[:, 0] - measurements[:, 0]
    dy = source_coords[:, 1] - measurements[:, 1]
    azimuth = np.degrees(np.arctan2(dx, dy))  # atan2(dx, dy) т.к. север это +Y
    azimuth = (azimuth + 360) % 360  # Нормализация в диапазон 0-360

    # Расчет RSSI
    rssi = calculate_rssi(distances)

    # Добавление шума
    y_coords = np.clip(measurements[:, 1].astype(int), 0, GRID_SIZE - 1)
    x_coords = np.clip(measurements[:, 0].astype(int), 0, GRID_SIZE - 1)
    rssi += noise_grid[y_coords, x_coords]

    # Ограничение значений
    rssi = np.clip(rssi, -140, 0)


    if EMULATE_DD:
        # Преобразование координат измерений в географические
        gps_coords = np.column_stack((
            BASE_COORDINATES[0] + measurements[:, 0] * 0.0001,
            BASE_COORDINATES[1] + measurements[:, 1] * 0.0001
        ))

        # Преобразование координат источников в географические
        source_gps = np.column_stack((
            BASE_COORDINATES[0] + source_coords[:, 0] * 0.0001,
            BASE_COORDINATES[1] + source_coords[:, 1] * 0.0001
        ))

        return pd.DataFrame({
            "source_id": source_ids + chunk_start,
            "longitude": np.round(gps_coords[:, 0], 6),
            "latitude": np.round(gps_coords[:, 1], 6),
            "rssi": np.round(rssi, 2),
            "source_longitude": np.round(source_gps[:, 0], 6),
            "source_latitude": np.round(source_gps[:, 1], 6),
            "azimuth": np.round(azimuth, 2),
            "distance": np.round(distances, 2)
        })
    else:
        # Использование декартовых координат
        return pd.DataFrame({
            "source_id": source_ids + chunk_start,
            "longitude": np.int16(measurements[:, 0]),
            "latitude": np.int16(measurements[:, 1]),
            "rssi": np.round(rssi, 2),
            "source_longitude": np.int16(source_coords[:, 0]),
            "source_latitude": np.int16(source_coords[:, 1]),
            "azimuth": np.round(azimuth, 2),
            "distance": np.round(distances, 2)  # Добавлено
        })


def generate_large_dataset():
    """Генерация полного датасета с новыми полями"""
    sources = generate_sources(SOURCES_COUNT)
    noise_grid = apply_noise_grid()

    # Инициализация файла с новыми колонками
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=[
            "source_id",
            "longitude",
            "latitude",
            "rssi",
            "source_longitude",
            "source_latitude",
            "azimuth",
            "distance"
        ]).to_csv(OUTPUT_FILE, index=False)

    # Обработка чанками
    for chunk_start in tqdm(range(0, SOURCES_COUNT, CHUNK_SIZE),
                            desc="Генерация данных"):
        chunk_end = min(chunk_start + CHUNK_SIZE, SOURCES_COUNT)
        chunk_sources = sources[chunk_start:chunk_end]

        chunk_df = process_chunk(chunk_sources, noise_grid, chunk_start)
        chunk_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

        # visualize_chunk(chunk_df, chunk_start)


if __name__ == "__main__":
    generate_large_dataset()
    print(f"Датасет сохранен в {OUTPUT_FILE}")
