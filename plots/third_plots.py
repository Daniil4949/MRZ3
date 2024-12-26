# Лабораторная работа 3 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация сети Джордана с визуализацией зависимости MAPE от размера скользящего окна
# Вариант 11
# Ссылки на источники:
# https://rep.bstu.by/bitstream/handle/data/30365/471-1999.pdf?sequence=1&isAllowed=y

import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Константы
LEAKY_RELU_ALPHA = 0.01
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 30  # Фиксированное количество нейронов на скрытом слое
INPUT_SIZE = 1
OUTPUT_SIZE = 1
MAX_EPOCHS = 5000  # Увеличено количество эпох
TARGET_LOSS = 1e-8
SEED = 42
TEST_SPLIT = 0.2  # 20% данных для тестирования
NUM_RUNS = 3  # Количество независимых тренировок


def generate_arithmetic_sequence(start: float, step: float, length: int) -> List[float]:
    """Генерирует арифметическую последовательность."""
    return [start + i * step for i in range(length)]


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную процентную ошибку (MAPE) в %.
    MAPE = 100/n * Σ(|(y_true - y_pred) / y_true|).
    """
    eps = 1e-8  # Чтобы избежать деления на ноль
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет среднюю абсолютную ошибку (MAE)."""
    return np.mean(np.abs(y_true - y_pred))


def normalize(seq: List[float]) -> Tuple[np.ndarray, float, float]:
    """Нормализует последовательность."""
    seq_array = np.array(seq)
    mean = seq_array.mean()
    std = seq_array.std()
    normalized_seq = (seq_array - mean) / std
    return normalized_seq, mean, std


def denormalize(seq_norm: List[float], mean: float, std: float) -> np.ndarray:
    """Денормализует последовательность."""
    seq_norm_array = np.array(seq_norm)
    return seq_norm_array * std + mean


def create_dataset(seq: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Создаёт обучающую выборку с использованием скользящего окна."""
    X, Y = [], []
    for i in range(len(seq) - window_size):
        X.append(seq[i: i + window_size])
        Y.append(seq[i + window_size])
    return np.array(X), np.array(Y)


class JordanNetwork:
    """Реализация рекуррентной нейронной сети Джордана."""

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            learning_rate: float = LEARNING_RATE,
            seed: int = SEED,
            alpha: float = LEAKY_RELU_ALPHA,
    ):
        """
        Инициализирует параметры сети.

        Args:
            input_size (int): Размер входного слоя.
            hidden_size (int): Размер скрытого слоя.
            output_size (int): Размер выходного слоя.
            learning_rate (float): Скорость обучения.
            seed (int): Сид для генератора случайных чисел.
            alpha (float): Параметр для Leaky ReLU.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.alpha = alpha

        # Фиксируем семя для воспроизводимости
        np.random.seed(seed)

        # Инициализация весов (He)
        self.W_ih = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.W_cy = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / output_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)

        # Инициализация смещений нулями
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прямой проход через сеть.

        Args:
            x (np.ndarray): Входной вектор (input_size, 1).
            context (np.ndarray): Контекстный вектор (output_size, 1).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Выход сети и скрытое состояние.
        """
        self.x = x
        self.context = context

        # Скрытое состояние
        self.h_raw = np.dot(self.W_ih, self.x) + np.dot(self.W_cy, self.context) + self.b_h
        self.h = self.leaky_relu(self.h_raw)

        # Выход
        self.y_raw = np.dot(self.W_hy, self.h) + self.b_y
        self.y = self.y_raw  # Линейный выход

        return self.y, self.h

    def backward(self, dy: np.ndarray):
        """
        Обратный проход (чистый SGD).

        Args:
            dy (np.ndarray): Градиент ошибки по выходу (output_size, 1).
        """
        # Градиенты выходного слоя
        dW_hy = np.dot(dy, self.h.T)
        db_y = dy

        # Градиент скрытого слоя
        dh = np.dot(self.W_hy.T, dy) * self.d_leaky_relu(self.h_raw)
        dW_ih = np.dot(dh, self.x.T)
        dW_cy = np.dot(dh, self.context.T)
        db_h = dh

        # Обновление параметров (чистый SGD)
        self.W_hy -= self.lr * dW_hy
        self.b_y -= self.lr * db_y
        self.W_ih -= self.lr * dW_ih
        self.W_cy -= self.lr * dW_cy
        self.b_h -= self.lr * db_h

    def leaky_relu(self, x: np.ndarray) -> np.ndarray:
        """Функция активации Leaky ReLU."""
        return np.where(x > 0, x, self.alpha * x)

    def d_leaky_relu(self, x: np.ndarray) -> np.ndarray:
        """Производная функции активации Leaky ReLU."""
        return np.where(x > 0, 1, self.alpha)

    def train(
            self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            X_val: np.ndarray,
            Y_val: np.ndarray,
            mean: float,
            std: float,
            window_size: int,
            max_epochs: int = MAX_EPOCHS,
            target_loss: float = TARGET_LOSS,
    ) -> Tuple[List[float], List[float]]:
        """
        Обучение сети до достижения целевой ошибки или максимального количества эпох.

        Args:
            X_train (np.ndarray): Обучающие входные окна, shape (num_samples, window_size).
            Y_train (np.ndarray): Обучающие целевые значения, shape (num_samples,).
            X_val (np.ndarray): Валидационные входные окна, shape (num_val_samples, window_size).
            Y_val (np.ndarray): Валидационные целевые значения, shape (num_val_samples,).
            mean (float): Среднее значение для денормализации.
            std (float): Стандартное отклонение для денормализации.
            window_size (int): Размер скользящего окна.
            max_epochs (int, optional): Максимальное количество эпох. Defaults to MAX_EPOCHS.
            target_loss (float, optional): Целевая ошибка. Defaults to TARGET_LOSS.

        Returns:
            Tuple[List[float], List[float]]: Список значений потерь и список MAPE на валидации по эпохам.
        """
        losses = []
        mape_history = []

        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            for window, target in zip(X_train, Y_train):
                # Обнуляем контекст перед каждым окном
                context = np.zeros((self.output_size, 1))

                # Прямой проход по окну
                for x_val in window:
                    x = np.array([[x_val]])  # (input_size, 1)
                    output, _ = self.forward(x, context)
                    context = output  # Обновляем контекст на основе текущего выхода

                # Вычисляем ошибку (MSE) на последнем выходе
                loss = np.mean((output - target) ** 2)
                total_loss += loss

                # Градиент ошибки по выходу
                dy = 2 * (output - target) / output.size

                # Обратный проход и обновление весов
                self.backward(dy)

            # Средняя ошибка за эпоху
            avg_loss = total_loss / len(X_train)
            losses.append(avg_loss)

            # Оценка на валидационной выборке
            y_val_pred = self.predict_batch(X_val, window_size)
            y_val_pred_denorm = denormalize(y_val_pred, mean, std)
            y_val_denorm = denormalize(Y_val, mean, std)
            current_mape = mean_absolute_percentage_error(y_val_denorm, y_val_pred_denorm)
            mape_history.append(current_mape)

            # Логирование каждые 100 эпох и на первой
            if epoch == 1 or epoch % 100 == 0:
                logging.info(f'Epoch {epoch}/{max_epochs}, Loss: {avg_loss:.6f}, MAPE: {current_mape:.2f}%')

            # Раннее завершение обучения
            if avg_loss <= target_loss:
                logging.info("Достигнут целевой уровень ошибки. Остановка обучения.")
                break

        return losses, mape_history

    def predict(
            self, initial_window: List[float], predict_steps: int, window_size: int
    ) -> List[float]:
        """
        Прогнозирование следующих значений на основе начального окна.

        Args:
            initial_window (List[float]): Начальное окно длиной window_size.
            predict_steps (int): Количество значений для прогнозирования.
            window_size (int): Размер окна.

        Returns:
            List[float]: Спрогнозированные значения.
        """
        # Убедимся, что initial_window является списком
        window = initial_window.copy() if isinstance(initial_window, list) else initial_window.tolist()

        predictions = []
        # Обнуляем контекст
        context = np.zeros((self.output_size, 1))

        for step in range(predict_steps):
            # Прогоняем последнее окно через сеть
            for x_val in window[-window_size:]:
                x_input = np.array([[x_val]])  # (input_size, 1)
                output, _ = self.forward(x_input, context)
                context = output  # Обновляем контекст

            # Получаем прогноз и добавляем его в список
            pred = output.item()
            predictions.append(pred)
            window.append(pred)  # Обновляем окно

        return predictions

    def predict_batch(self, X: np.ndarray, window_size: int) -> List[float]:
        """
        Прогнозирование для пакета входных окон.

        Args:
            X (np.ndarray): Входные окна, shape (num_samples, window_size).
            window_size (int): Размер окна.

        Returns:
            List[float]: Спрогнозированные значения.
        """
        predictions = []
        for window in X:
            pred = self.predict(window, 1, window_size)[0]  # Прогнозируем один шаг
            predictions.append(pred)
        return predictions


def main():
    """Основная функция для демонстрации использования сети Джордана с визуализацией зависимости MAPE от размера скользящего окна."""
    # Генерация арифметической последовательности (1, 4, 7, ..., 88)
    sequence = generate_arithmetic_sequence(start=1, step=3, length=30)  # [1, 4, 7, ..., 88]
    logging.info(f"Изначальная последовательность: {sequence}")

    # Нормализация последовательности
    sequence_norm, mean, std = normalize(sequence)
    logging.info(f"Нормализованная последовательность: {sequence_norm}")

    # Диапазон значений для WINDOW_SIZE
    window_sizes = list(range(3, 11))  # От 3 до 10 включительно

    # Список для хранения средних MAPE для каждого window_size
    avg_mapes = []

    for window_size in window_sizes:
        logging.info(f"\n=== Размер скользящего окна (WINDOW_SIZE) = {window_size} ===")
        mape_runs = []

        # Создание обучающей и тестовой выборок для текущего window_size
        X, Y = create_dataset(sequence_norm, window_size)
        logging.info(f"WINDOW_SIZE = {window_size}: X.shape = {X.shape}, Y.shape = {Y.shape}")

        # Разделение на обучающую и тестовую выборки
        split_index = int(len(X) * (1 - TEST_SPLIT))
        X_train, X_test = X[:split_index], X[split_index:]
        Y_train, Y_test = Y[:split_index], Y[split_index:]
        logging.info(
            f"WINDOW_SIZE = {window_size}: Обучающая выборка: X_train.shape = {X_train.shape}, Y_train.shape = {Y_train.shape}")
        logging.info(
            f"WINDOW_SIZE = {window_size}: Тестовая выборка: X_test.shape = {X_test.shape}, Y_test.shape = {Y_test.shape}")

        for run in range(1, NUM_RUNS + 1):
            logging.info(f"\n--- Тренировка {run} из {NUM_RUNS} для WINDOW_SIZE = {window_size} ---")
            # Инициализация сети Джордана с разными сидями для разнообразия
            net = JordanNetwork(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                output_size=OUTPUT_SIZE,
                learning_rate=LEARNING_RATE,
                seed=SEED + run,  # Изменяем сид для каждой тренировки
                alpha=LEAKY_RELU_ALPHA,
            )

            # Обучение сети
            losses, mape_history = net.train(
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_test,
                Y_val=Y_test,
                mean=mean,  # Передаём mean
                std=std,  # Передаём std
                window_size=window_size,  # Передаём window_size
                max_epochs=MAX_EPOCHS,
                target_loss=TARGET_LOSS,
            )

            # Последнее значение MAPE после обучения
            final_mape = mape_history[-1]
            logging.info(f"Тренировка {run} для WINDOW_SIZE = {window_size}: Final MAPE = {final_mape:.2f}%")
            mape_runs.append(final_mape)

        # Усреднение MAPE по трём тренировкам
        avg_mape = np.mean(mape_runs)
        avg_mapes.append(avg_mape)
        logging.info(f"WINDOW_SIZE = {window_size}: Средний MAPE = {avg_mape:.2f}%")

    # Построение графика зависимости MAPE от размера скользящего окна
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, avg_mapes, marker='o', linestyle='-', color='b', linewidth=2)
    plt.xlabel('Размер скользящего окна (WINDOW_SIZE)', fontsize=12)
    plt.ylabel('Средняя абсолютная процентная ошибка (MAPE) (%)', fontsize=12)
    plt.title('Зависимость средней абсолютной процентной ошибки (MAPE) от размера скользящего окна', fontsize=14)
    plt.xticks(window_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Прогнозирование и оценка модели после обучения для последнего window_size
    logging.info("\n=== Прогнозирование и оценка модели ===")
    mape_values = []  # Список для хранения MAPE

    # Используем последнюю обученную сеть для прогнозирования
    # net уже инициализирован и обучен для последнего window_size
    for idx, start_idx in enumerate(range(0, len(sequence_norm) - window_size, window_size), 1):
        window = sequence_norm[start_idx: start_idx + window_size]

        # Проверка достаточности данных
        if len(window) < window_size:
            logging.warning("Недостаточно данных для формирования окна. Пропуск.")
            continue

        # Денормализация текущего окна для отображения
        window_denorm = denormalize(window, mean, std)
        logging.info(f"\nОкно #{idx}: {window_denorm}")

        # Прогноз сети (window_size значений) — итеративный метод predict
        preds_norm = net.predict(window, predict_steps=window_size, window_size=window_size)
        preds = denormalize(preds_norm, mean, std)
        logging.info(f"  Прогноз сети (на {window_size} шагов): {preds}")

        # Ожидаемые значения (следующие window_size элементов из исходной последовательности)
        expected = sequence[start_idx + window_size: start_idx + window_size + window_size]
        logging.info(f"  Ожидаемые значения: {expected}")

        # Проверка достаточности ожидаемых значений
        if len(expected) < window_size:
            logging.warning("Недостаточно данных для сравнения. Пропуск.")
            continue

        # Вычисление MAE и MAPE
        local_mae = mean_absolute_error(expected, preds)
        local_mape = mean_absolute_percentage_error(expected, preds)

        mape_values.append(local_mape)
        logging.info(f"  MAE: {local_mae:.2f}, MAPE: {local_mape:.2f}%")

    # Вывод средней MAPE по всем окнам
    if mape_values:
        avg_mape_final = np.mean(mape_values)
        logging.info(f"\nСредняя абсолютная процентная ошибка (MAPE) по всем окнам: {avg_mape_final:.2f}%")
    else:
        logging.warning("Не было достаточно данных для формирования окон и прогноза.")


if __name__ == '__main__':
    main()
