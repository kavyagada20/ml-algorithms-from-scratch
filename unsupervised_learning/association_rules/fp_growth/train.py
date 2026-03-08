"""
Training script for FP-Growth from scratch
"""

import matplotlib.pyplot as plt

from model import FPGrowthScratch


# Example market basket dataset
transactions = [
    {"milk", "bread", "butter"},
    {"bread", "beer"},
    {"milk", "bread", "beer"},
    {"milk", "bread", "butter"},
    {"bread", "butter"},
    {"milk", "bread"},
]


# Train model
model = FPGrowthScratch(min_support=0.3)

model.fit(transactions)

patterns = model.predict()


print("Frequent Patterns:")

for itemset, support in patterns.items():
    print(set(itemset), support)


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("FP-Growth Frequent Patterns\n")

    for itemset, support in patterns.items():
        f.write(f"{set(itemset)} : {support}\n")


# Visualization
labels = [str(set(i)) for i in patterns.keys()]
values = list(patterns.values())


plt.figure(figsize=(8,5))

plt.bar(labels, values)

plt.xticks(rotation=45)

plt.ylabel("Support")

plt.title("FP-Growth Frequent Pattern Support")

plt.tight_layout()

plt.savefig("fp_growth_support_plot.png")

plt.show()