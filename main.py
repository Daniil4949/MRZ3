# Лабораторная работа 3 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация сети Джордана
# Вариант 11
# Ссылки на источники:
# https://rep.bstu.by/bitstream/handle/data/30365/471-1999.pdf?sequence=1&isAllowed=y

import numpy as np

from src.jordan_network import JordanNetwork


def generate_arithmetic_sequence(start, step, length):
    return [start + i * step for i in range(length)]


def generate_geometric_sequence(a, r, length):
    if length <= 0:
        return []

    geom_sequence = [a]
    for i in range(1, length):
        geom_sequence.append(geom_sequence[-1] * r)

    return geom_sequence


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Вычисляет среднюю абсолютную процентную ошибку (MAPE) в %.
    MAPE = 100/n * Σ(|(y_true - y_pred) / y_true|).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def mean_absolute_error(y_true, y_pred):
    """
    Вычисляет среднюю абсолютную ошибку (MAE).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def normalize(seq):
    seq = np.array(seq)
    mean = seq.mean()
    std = seq.std()
    return (seq - mean) / std, mean, std


def denormalize(seq_norm, mean, std):
    seq_norm = np.array(seq_norm)
    return seq_norm * std + mean


def create_dataset(seq, window_size):
    X, Y = [], []
    for i in range(len(seq) - window_size):
        X.append(seq[i: i + window_size])
        Y.append(seq[i + window_size])
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    sequence = generate_arithmetic_sequence(1, 3, 30)
    print("\nИзначальная последовательность:", sequence)

    sequence_norm, mean, std = normalize(sequence)
    print("\nНормализованная последовательность:", sequence_norm)

    window_size = 5
    X, Y = create_dataset(sequence_norm, window_size)
    print("\nПримеры входных данных (X) и целевых значений (Y):")
    print("X:", X)
    print("Y:", Y)

    print("\nX.shape =", X.shape, "Y.shape =", Y.shape)

    net = JordanNetwork(input_size=1, hidden_size=10, output_size=1, learning_rate=1e-4, alpha=0.01)
    print("\nОбучение")
    losses = net.train(X, Y, max_epochs=1000, target_loss=1e-4)

    predict_steps = window_size
    mape_values = []  # список для хранения MAPE

    for idx, start_idx in enumerate(range(0, len(sequence_norm) - window_size, window_size), 1):
        window = sequence_norm[start_idx: start_idx + window_size]

        if len(window) < window_size:
            break

        print(f"\nОкно #{idx}: {denormalize(window, mean, std)}")

        preds_norm = net.predict(window, predict_steps, window_size)
        preds = denormalize(preds_norm, mean, std)
        print(f"  Прогноз сети (на {predict_steps} шагов): {preds}")

        expected = sequence[start_idx + window_size: start_idx + window_size + predict_steps]
        print(f"  Ожидаемые значения: {expected}")

        if len(expected) < predict_steps:
            print("  Недостаточно данных для сравнения. Пропуск.")
            continue

        local_mae = mean_absolute_error(expected, preds)
        local_mape = mean_absolute_percentage_error(expected, preds)

        mape_values.append(local_mape)
        print(f"  MAE: {local_mae:.2f}, MAPE: {local_mape:.2f}%")

    if mape_values:
        print(f"\nСредняя абсолютная процентная ошибка (MAPE) по всем окнам: {np.mean(mape_values):.2f}%")
    else:
        print("Не было достаточно данных для формирования окон и прогноза.")
