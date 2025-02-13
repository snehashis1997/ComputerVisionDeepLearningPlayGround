# import matplotlib.pyplot as plt

# # Line 1: Green circles connected by lines
# plt.plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)

# # Line 2: Red squares without connecting lines
# # plt.plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')

# # Adding legend
# plt.legend()

# # Adding labels
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# # Adding title
# plt.title('Example Plot')

# # Display the plot
# plt.show()

import numpy as np

# Example Fundamental Matrix
F = np.array([
    [0.000001, -0.00003, 0.004],
    [0.00002, 0.000002, -0.005],
    [-0.003, 0.005, 0.002]
])

# Perform SVD
U, S, Vt = np.linalg.svd(F)

# Enforce rank-2 constraint
S[2] = 0  # Set the smallest singular value to zero
F_enforced = U @ np.diag(S) @ Vt

print("Original F:\n", F)
print("Enforced Rank-2 F:\n", F_enforced)