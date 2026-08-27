from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            if len(tokens) < 2:
                break
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break
            
            max_pairs = max(pairs.values())
            best = sorted(pair for pair, count in pairs.items() if count == max_pairs)[0] #take first index to break ties
            merges.append([best[0], best[1]])

            updated_tokens =[]
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i+1] == best[1]:
                    updated_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    updated_tokens.append(tokens[i])
                    i += 1
            tokens = updated_tokens
        return merges


