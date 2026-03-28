def hit_rate(results):
    """
    Calculates the Hit Rate across all queries.
    A hit means at least one of the expected pages was found
    among the retrieved chunks for a given query.

    Args:
        results: list of dicts with keys 'expected_pages' and 'retrieved_pages'

    Returns:
        float: proportion of queries where at least one correct page was retrieved
    """
    hits = 0
    for r in results:
        expected = set(r["expected_pages"])
        retrieved = set(r["retrieved_pages"])
        if expected & retrieved:
            hits += 1
    return hits / len(results) if results else 0.0


def mrr(results):
    """
    Calculates the Mean Reciprocal Rank (MRR) across all queries.
    Rewards retrievers that place the correct chunk in a higher position.

    Args:
        results: list of dicts with keys 'expected_pages' and 'retrieved_pages_ordered'
                 where 'retrieved_pages_ordered' is a list of pages in retrieval order

    Returns:
        float: mean of the reciprocal ranks (1/position of first correct hit)
    """
    reciprocal_ranks = []
    for r in results:
        expected = set(r["expected_pages"])
        rr = 0.0
        for rank, page in enumerate(r["retrieved_pages_ordered"], start=1):
            if page in expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def recall(results):
    """
    Calculates the mean Recall across all queries.
    Recall measures the fraction of expected pages that were actually retrieved.

    For each query:
        recall_i = |expected ∩ retrieved| / |expected|

    Args:
        results: list of dicts with keys 'expected_pages' and 'retrieved_pages'

    Returns:
        float: mean recall across all queries (0.0 – 1.0)
    """
    recall_scores = []
    for r in results:
        expected = set(r["expected_pages"])
        retrieved = set(r["retrieved_pages"])
        if expected:
            recall_scores.append(len(expected & retrieved) / len(expected))
        else:
            recall_scores.append(0.0)
    return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
