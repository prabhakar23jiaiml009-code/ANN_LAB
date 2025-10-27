

def perceptron_not(x):
    w = -1      # weight
    b = 0.5     # bias
    # Step activation function
    y_in = w * x + b
    if y_in >= 0:
        return 1
    else:
        return 0


x = int(input("Enter input (0 or 1): "))

# Validate input
if x not in [0, 1]:
    print("Invalid input! Please enter only 0 or 1.")
else:
    output = perceptron_not(x)
    print(f"Output (NOT {x}) = {output}")
