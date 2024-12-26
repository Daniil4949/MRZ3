# Лабораторная работа 3 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация сети Джордана
# Вариант 11
# Ссылки на источники:
# https://rep.bstu.by/bitstream/handle/data/30365/471-1999.pdf?sequence=1&isAllowed=y

import numpy as np


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


class JordanNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-2, seed=42, alpha=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.alpha = alpha

        np.random.seed(seed)

        self.W_ih = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.W_cy = np.random.randn(hidden_size, output_size) * np.sqrt(2. / output_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x, context):
        self.x = x
        self.context = context

        self.h_raw = np.dot(self.W_ih, self.x) + np.dot(self.W_cy, self.context) + self.b_h
        self.h = leaky_relu(self.h_raw, self.alpha)

        self.y_raw = np.dot(self.W_hy, self.h) + self.b_y
        self.y = self.y_raw

        return self.y, self.h

    def backward(self, dy):
        dW_hy = np.dot(dy, self.h.T)
        db_y = dy

        dh = np.dot(self.W_hy.T, dy) * d_leaky_relu(self.h_raw, self.alpha)
        dW_ih = np.dot(dh, self.x.T)
        dW_cy = np.dot(dh, self.context.T)
        db_h = dh

        self.W_hy -= self.lr * dW_hy
        self.b_y -= self.lr * db_y
        self.W_ih -= self.lr * dW_ih
        self.W_cy -= self.lr * dW_cy
        self.b_h -= self.lr * db_h

    def train(self, X, Y, max_epochs=10000, target_loss=1e-8):
        losses = []

        for epoch in range(max_epochs):
            total_loss = 0
            for i in range(len(X)):
                context = np.zeros((self.output_size, 1))

                window = X[i]
                target = np.array([[Y[i]]])

                for t in range(len(window)):
                    x = np.array([[window[t]]])
                    output, _ = self.forward(x, context)
                    context = output

                loss = np.mean((output - target) ** 2)
                total_loss += loss

                dy = 2 * (output - target) / output.size

                self.backward(dy)

            avg_loss = total_loss / len(X)
            losses.append(avg_loss)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.6f}')

            if avg_loss <= target_loss:
                print("Достигнут целевой уровень ошибки. Остановка обучения.")
                break

        return losses

    def predict(self, initial_window, predict_steps, window_size):
        if isinstance(initial_window, np.ndarray):
            window = initial_window.copy().tolist()
        else:
            window = initial_window.copy()

        predictions = []
        context = np.zeros((self.output_size, 1))

        for step in range(predict_steps):
            for x_val in window[-window_size:]:
                x_input = np.array([[x_val]])
                output, _ = self.forward(x_input, context)
                context = output

            pred = output.item()
            predictions.append(pred)
            window.append(pred)

        return predictions
