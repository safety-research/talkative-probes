# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# Data: [best-of-N, variance_recovered±err, ...]
data = [
    [1, 0.7503],
    [2, 0.7600],
    [4, 0.7655],
    [8, 0.7697],
    [16, 0.7735],
    [32, 0.7772],
]

x = [row[0] for row in data]
y = [row[1] for row in data]

plt.figure()
plt.plot(x, y, marker='o')
plt.xscale('log')
plt.xlabel('best-of-N')
plt.ylabel('variance recovered')
plt.title('Inference scaling of NLAEs (gemma-3-27b)')
plt.tight_layout()
plt.show()

# %%
