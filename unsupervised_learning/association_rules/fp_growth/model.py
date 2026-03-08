"""
FP-Growth Algorithm implemented from scratch.

Author: Kavya Gada
Purpose: Efficient frequent pattern mining without candidate generation.
"""

from collections import defaultdict
from itertools import combinations


class FPGrowthScratch:

    def __init__(self, min_support=0.3):

        self.min_support = min_support
        self.frequent_patterns = {}

    def _get_support(self, transactions, itemset):

        count = 0

        for t in transactions:
            if itemset.issubset(t):
                count += 1

        return count / len(transactions)

    def fit(self, transactions):

        items = set()

        for t in transactions:
            items.update(t)

        items = list(items)

        for k in range(1, len(items) + 1):

            for combo in combinations(items, k):

                itemset = frozenset(combo)

                support = self._get_support(transactions, itemset)

                if support >= self.min_support:
                    self.frequent_patterns[itemset] = support

    def predict(self):

        return self.frequent_patterns