import os

import matplotlib.pyplot as plt


def generate_training_report(errors, iterations, output_dir="report"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(errors, iterations, marker='o', linestyle='-', color='b', label="Итерации обучения")
    plt.xscale('log')
    plt.xlabel("Допустимая ошибка", fontsize=12)
    plt.ylabel("Количество предсказанных элементов", fontsize=12)
    plt.title("Зависимость количества предсказанных элементов от допустимой ошибки", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "training_iterations_plot.png")
    plt.savefig(plot_path)
    plt.close()

    description = "График зависимости количества предсказанных элементов от допустимой ошибки:\n\n"
    description += "Допустимая ошибка | Количество предсказанных элементов\n"
    description += "------------------|----------------------------\n"
    for err, iter_count in zip(errors, iterations):
        description += f"{err:<18} | {iter_count:<26}\n"

    description += "\nГрафик сохранен в файл: training_iterations_plot.png"

    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(description)

    print(f"Отчет успешно создан в директории '{output_dir}'.")


errors = [0.5, 0.1, 0.01, 0.001, 0.0001]
iterations = [0, 0, 2, 4, 4]

generate_training_report(errors, iterations)
