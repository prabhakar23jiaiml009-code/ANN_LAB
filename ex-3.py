def perceptron_logic(gate, inputs):
    """
    Simulate basic logic gates using a perceptron model.
    Supported gates: AND, OR, NOT, NAND, NOR
    """
    n = len(inputs)
    gate = gate.upper()

    # Define weights and thresholds
    if gate in ["AND", "OR", "NAND", "NOR"]:
        w = [1] * n
        t = n if gate in ["AND", "NAND"] else 1
    elif gate == "NOT":
        if n != 1:
            return "Error: NOT gate only accepts 1 input."
        w = [-1]
        t = 0
    else:
        return "Error: Invalid gate selected."

    # Weighted sum
    summation = sum(i * j for i, j in zip(w, inputs))

    # Activation function (threshold logic)
    if gate in ["NAND", "NOR"]:
        output = int(summation < t)
    else:
        output = int(summation >= t)

    return output


def get_inputs(num_inputs):
    """
    Collects valid binary inputs from the user.
    Allows user to type 'exit' to quit.
    """
    inputs = []
    for i in range(num_inputs):
        while True:
            val = input(f"Enter input {i + 1} (0 or 1): ").strip().lower()
            if val == "exit":
                print("Exiting input collection.")
                return None
            if val in ["0", "1"]:
                inputs.append(int(val))
                break
            print("Invalid input. Only 0 or 1 are allowed.")
    return inputs


def main():
    print("🧠 Perceptron Logic Gate Simulator")
    print("-----------------------------------")

    while True:
        print("\nAvailable Gates: AND, OR, NOT, NAND, NOR")
        gate = input("Enter logic gate (or type 'exit' to quit): ").strip().upper()

        if gate == "EXIT":
            print("Program exited.")
            break

        if gate not in ["AND", "OR", "NOT", "NAND", "NOR"]:
            print("Invalid gate. Try again.")
            continue

        try:
            num_inputs = int(input("Enter number of inputs: "))
            if gate == "NOT" and num_inputs != 1:
                print("NOT gate only accepts 1 input.")
                continue
            elif num_inputs < 1:
                print("Number of inputs must be at least 1.")
                continue
        except ValueError:
            print("Invalid number of inputs.")
            continue

        inputs = get_inputs(num_inputs)
        if inputs is None:  # user typed 'exit'
            break

        result = perceptron_logic(gate, inputs)
        print(f"\n🧩 {gate} Gate: {inputs} → Output: {result}")

        again = input("\nDo you want to try again? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
