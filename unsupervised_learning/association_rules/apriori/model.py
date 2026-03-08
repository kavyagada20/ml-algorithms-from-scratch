"""
Apriori Algorithm implemented from scratch.

Author: Kavya Gada
Purpose: Discover frequent itemsets and association rules.
"""

from itertools import combinations
from collections import defaultdict


class AprioriScratch:

    def __init__(self, min_support=0.3):

        self.min_support = min_support
        self.frequent_itemsets = {}

    def _get_support(self, transactions, itemset):

        count = 0

        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1

        return count / len(transactions)

    def fit(self, transactions):

        item_counts = defaultdict(int)

        for transaction in transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1

        n_transactions = len(transactions)

        L1 = {
            item: count / n_transactions
            for item, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }

        self.frequent_itemsets.update(L1)

        k = 2
        current_L = list(L1.keys())

        while current_L:

            candidates = list(
                set(
                    frozenset(i.union(j))
                    for i in current_L
                    for j in current_L
                    if len(i.union(j)) == k
                )
            )

            Lk = {}

            for candidate in candidates:

                support = self._get_support(transactions, candidate)

                if support >= self.min_support:
                    Lk[candidate] = support

            self.frequent_itemsets.update(Lk)

            current_L = list(Lk.keys())

            k += 1

    def predict(self):

        return self.frequent_itemsets