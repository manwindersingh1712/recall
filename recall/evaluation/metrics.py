import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Calculate recall@k: |set(retrieved[:k]) ∩ relevant| / |relevant|

    Args:
        retrieved: List of retrieved items
        relevant: Set of relevant items
        k: Number of top items to consider

    Returns:
        Recall at k

    Raises:
        ValueError: If relevant set is empty
    """
    if not relevant:
        raise ValueError("relevant set cannot be empty")

    # Filter out empty strings from retrieved list
    filtered_retrieved = [item for item in retrieved[:k] if item != ""]
    hits = len(set(filtered_retrieved) & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Calculate reciprocal rank (MRR): 1/rank of first relevant item, else 0.0

    Args:
        retrieved: List of retrieved items
        relevant: Set of relevant items

    Returns:
        Reciprocal rank
    """
    for i, item in enumerate(retrieved):
        # Skip empty strings
        if item != "" and item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Calculate normalized Discounted Cumulative Gain at k.
    Uses binary gain (1 for relevant, 0 for irrelevant).
    Discount factor: 1/log2(i+2) for 0-based position i.
    A key counts only at its first occurrence.

    Args:
        retrieved: List of retrieved items
        relevant: Set of relevant items
        k: Number of top items to consider

    Returns:
        nDCG@k
    """
    # Calculate DCG
    dcg = 0.0
    seen = set()
    for i, item in enumerate(retrieved[:k]):
        if item != "" and item not in seen:
            if item in relevant:
                dcg += 1.0 / math.log2(i + 2)
            seen.add(item)

    # Calculate ideal DCG (IDCG)
    # Ideal ranking: all relevant items first
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    # Handle edge case where IDCG is 0
    if idcg == 0:
        return 0.0

    return dcg / idcg


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
