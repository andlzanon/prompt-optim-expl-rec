from engine import generate_recommendations
import sys

algorithm = sys.argv[1]                 # "bprmf"
numbers = list(map(int, sys.argv[2:]))  # 1, 5, 20

if any(num > 200 for num in numbers):
    print("Please, the value of k need to be at least 1 and at max 200!")
    sys.exit()

if any(num < 1 for num in numbers):
    print("Please, the value of k need to be at least 1 and at max 200!")
    sys.exit()

generate_recommendations(algorithm_name=algorithm, k_vector=numbers)



# k_values = [1, 5, 10, 20, 50, 100, 200]
# k_values = [5]


# generate_recommendations(algorithm_name="userknn", k_vector=k_values)
# generate_recommendations(algorithm_name="itemknn",  k_vector=k_values)
# generate_recommendations(algorithm_name="ncf",  k_vector=k_values)
# generate_recommendations(algorithm_name="bprmf", k_vector=k_values)

# generate_recommendations(algorithm_name="all",  k_vector=k_values)

# userknn itemknn ncf neumf bprmf