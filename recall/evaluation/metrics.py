import math


def recall_at_k(retrieved: list[set[str]], relevant: set[str], k: int = 10) -> float:
    """recall@k: |union of keys in the top k hits ∩ relevant| / |relevant|.

    Each retrieved entry is the set of gold keys that hit satisfies (empty = none).
    Raises ValueError if relevant is empty.
    """
    if not relevant:
        raise ValueError("relevant set cannot be empty")
    found: set[str] = set()
    for keys in retrieved[:k]:
        found |= keys
    return len(found & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[set[str]], relevant: set[str]) -> float:
    """1/rank of the first hit satisfying any relevant key, else 0.0."""
    for i, keys in enumerate(retrieved):
        if keys & relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[set[str]], relevant: set[str], k: int = 10) -> float:
    """nDCG@k where a hit's gain is the number of relevant keys it newly satisfies.

    DCG = sum(gain_i / log2(i + 2)); IDCG assumes min(|relevant|, k) unit gains at the
    top positions. That is an approximation (a single hit carrying several keys can beat
    it), so the result is capped at 1.0.
    """
    dcg = 0.0
    seen: set[str] = set()
    for i, keys in enumerate(retrieved[:k]):
        new = (keys & relevant) - seen
        seen |= new
        dcg += len(new) / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if idcg == 0:
        return 0.0
    return min(1.0, dcg / idcg)


def percentile(values: list[float], p: float) -> float:
    """
    Calculate percentile using linear interpolation (numpy's default).
    Formula: rank = (p/100)*(n-1), then linear interpolation.

    Args:
        values: List of values
        p: Percentile (0-100)

    Returns:
        Percentile value

    Raises:
        ValueError: If values list is empty
    """
    if not values:
        raise ValueError("values list cannot be empty")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Linear interpolation formula: rank = (p/100)*(n-1)
    rank = (p / 100.0) * (n - 1)

    # Get the floor and ceiling indices
    lower_idx = int(rank)
    upper_idx = lower_idx + 1

    # If rank is exactly an integer, return that value
    if rank == lower_idx:
        return sorted_values[lower_idx]

    # Linear interpolation between the two values
    fraction = rank - lower_idx
    lower_value = sorted_values[lower_idx]
    upper_value = sorted_values[min(upper_idx, n - 1)]

    return lower_value + fraction * (upper_value - lower_value)
