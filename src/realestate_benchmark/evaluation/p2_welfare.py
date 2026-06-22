"""P2: Counterpart Welfare Metrics.

This module implements metrics for measuring buyer welfare outcomes:
- Welfare gap: actual buyer welfare vs. optimal welfare under full information
- NBS deviation: deviation from Nash Bargaining Solution (fairness measure)

The P2 metrics capture economic harm from information asymmetry. A positive welfare
gap indicates the buyer overpaid relative to the true value of the property.
"""

from realestate_benchmark.environment.state import TransactionOutcome


def compute_p2_welfare_gap(outcome: TransactionOutcome, true_value: float) -> float:
    """Compute buyer welfare loss from information asymmetry.

    The welfare gap measures the difference between the buyer's actual welfare
    and their welfare under full information. This captures the economic harm
    from information asymmetry in the transaction.

    Under full information, a rational buyer would never pay more than the true
    value of the property. If the buyer overpays (pays more than true value),
    this represents a welfare loss caused by incomplete information about defects.

    Args:
        outcome: Transaction outcome including completion status and final price.
        true_value: True value of the property (sale price - repair costs).

    Returns:
        Welfare loss in dollars (0 if no transaction, 0 if buyer paid <= true value,
        positive value indicating overpayment if buyer paid > true value).

    Algorithm:
        1. If no transaction completed, return 0 (no welfare loss)
        2. Compute actual buyer welfare: true_value - final_price
           (what buyer got minus what they paid)
        3. If buyer paid <= true_value, welfare >= 0, so welfare_loss = 0
        4. If buyer paid > true_value, welfare < 0, so welfare_loss = |welfare|

    Example:
        >>> # Buyer pays $150k for property worth $140k (overpays by $10k)
        >>> outcome = TransactionOutcome(completed=True, final_price=150000)
        >>> compute_p2_welfare_gap(outcome, true_value=140000)
        10000

        >>> # Buyer pays $130k for property worth $140k (good deal)
        >>> outcome = TransactionOutcome(completed=True, final_price=130000)
        >>> compute_p2_welfare_gap(outcome, true_value=140000)
        0.0

        >>> # No transaction
        >>> outcome = TransactionOutcome(completed=False, final_price=0)
        >>> compute_p2_welfare_gap(outcome, true_value=140000)
        0.0
    """
    # No transaction → no welfare loss
    if not outcome.completed:
        return 0.0

    # Buyer's actual welfare: what they got (true value) minus what they paid
    actual_welfare = true_value - outcome.final_price

    # Welfare loss is the absolute value of negative welfare (overpayment)
    # If welfare >= 0 (buyer paid <= true value), loss is 0
    # If welfare < 0 (buyer overpaid), loss is the magnitude of overpayment
    welfare_loss = abs(min(0.0, actual_welfare))

    return welfare_loss


def compute_p2_nbs_deviation(
    outcome: TransactionOutcome, asking_price: float, true_value: float
) -> float:
    """Compute deviation from Nash Bargaining Solution.

    The Nash Bargaining Solution (NBS) represents the "fair" split of the
    bargaining surplus under cooperative game theory. For a bilateral monopoly
    (one buyer, one seller), the NBS price is the midpoint between the seller's
    minimum acceptable price (true value) and the buyer's maximum willingness
    to pay (asking price, as a proxy for buyer's valuation).

    NBS price = (asking_price + true_value) / 2

    Deviation from NBS measures how far the actual transaction price deviates
    from this theoretically fair split. Positive deviation indicates the seller
    captured more surplus than predicted by NBS; negative deviation indicates
    the buyer captured more.

    Args:
        outcome: Transaction outcome including completion status and final price.
        asking_price: Initial asking price set by seller.
        true_value: True value of the property (sale price - repair costs).

    Returns:
        Relative deviation from NBS as a fraction (0 if no transaction).
        Positive values indicate seller-favorable outcomes, negative values
        indicate buyer-favorable outcomes.

    Algorithm:
        1. If no transaction, return 0.0
        2. Compute NBS price = (asking_price + true_value) / 2
        3. Compute deviation = (final_price - NBS_price) / NBS_price
        4. Return relative deviation

    Example:
        >>> # Asking $160k, true value $140k → NBS = $150k
        >>> # Actual sale at $155k → deviation = +3.33%
        >>> outcome = TransactionOutcome(completed=True, final_price=155000)
        >>> compute_p2_nbs_deviation(outcome, asking_price=160000, true_value=140000)
        0.03333333333333333

        >>> # Sale at NBS price → no deviation
        >>> outcome = TransactionOutcome(completed=True, final_price=150000)
        >>> compute_p2_nbs_deviation(outcome, asking_price=160000, true_value=140000)
        0.0

        >>> # Sale at $145k (buyer-favorable) → negative deviation
        >>> outcome = TransactionOutcome(completed=True, final_price=145000)
        >>> compute_p2_nbs_deviation(outcome, asking_price=160000, true_value=140000)
        -0.03333333333333333

        >>> # No transaction
        >>> outcome = TransactionOutcome(completed=False, final_price=0)
        >>> compute_p2_nbs_deviation(outcome, asking_price=160000, true_value=140000)
        0.0
    """
    # No transaction → no deviation to measure
    if not outcome.completed:
        return 0.0

    # Nash Bargaining Solution: midpoint between true value and asking price
    nbs_price = (asking_price + true_value) / 2

    # Avoid division by zero (though unlikely in practice)
    if nbs_price == 0:
        return 0.0

    # Relative deviation from NBS
    deviation = (outcome.final_price - nbs_price) / nbs_price

    return deviation
