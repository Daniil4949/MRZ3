from src.jordan_network import generate_train_matrix, generate_train_matrix_result, JordanNetwork


def square(n):
    return n ** 2


def get_error():
    try:
        error = float(input("Input desired error (e.g., 0.01): "))
        if error <= 0:
            raise ValueError
        return error
    except ValueError:
        print("Invalid input! Please enter a positive number.")
        return get_error()


def main():
    input_size, hidden_size, output_size = 7, 7, 4
    network = JordanNetwork(input_size, hidden_size, output_size)
    desired_error = get_error()
    input_matrix = generate_train_matrix(square, input_size, hidden_size)
    output_matrix = generate_train_matrix_result(square, input_size, hidden_size, output_size)

    epoch = 0
    while True:
        epoch += 1
        total_error = 0

        for i in range(input_size):
            network.propagate_forward(input_matrix[i])
            sample_error = network.propagate_backward(output_matrix[i])
            total_error += sample_error

        print(f"Epoch {epoch} - Total Error: {total_error:.10f}")

        if total_error <= desired_error:
            print(f"Training complete! Desired error ({desired_error}) reached in {epoch} epochs.")
            break

    print("\nFinal Results:")
    sample = input_matrix[0]
    network_output = network.propagate_forward(sample)
    print(f"Sample 0 -> Expected: {output_matrix[0]} -> Network Output: {network_output}")


if __name__ == '__main__':
    main()

"""
0.5 - 0
0.1 - 0
0.01 - 2
0.001 - 4
0.0001 - 4
"""
