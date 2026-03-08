"""
Training script for Apriori from scratch
"""

import matplotlib.pyplot as plt

from model import AprioriScratch


# Example dataset (market basket style)
transactions = [
    {"milk", "bread", "butter"},
    {"beer", "bread"},
    {"milk", "bread", "beer"},
    {"milk", "bread", "butter"},
    {"bread", "butter"},
    {"milk", "bread"},
]


# Train model
model = AprioriScratch(min_support=0.3)

model.fit(transactions)

frequent_itemsets = model.predict()


print("Frequent Itemsets:")

for itemset, support in frequent_itemsets.items():
    print(itemset, support)


# Save metrics
with open("metrics.txt", "w") as f:

    f.write("Apriori Frequent Itemsets\n")

    for itemset, support in frequent_itemsets.items():
        f.write(f"{set(itemset)} : {support}\n")


# Visualization
labels = [str(set(i)) for i in frequent_itemsets.keys()]
values = list(frequent_itemsets.values())


plt.figure(figsize=(8,5))

plt.bar(labels, values)

plt.xticks(rotation=45)

plt.title("Frequent Itemset Support")

plt.ylabel("Support")

plt.tight_layout()

plt.savefig("support_plot.png")

plt.show()