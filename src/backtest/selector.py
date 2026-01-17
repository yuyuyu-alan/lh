"""Portfolio selection helpers."""

from __future__ import annotations

from collections.abc import Iterable


def select_with_buffer(
    sorted_codes: list[str],
    holdings: Iterable[str],
    n: int = 30,
    buffer: int = 15,
) -> list[str]:
    """Select top n codes while keeping existing holdings within buffer."""
    holding_set = set(holdings)
    cutoff = n + buffer
    retained = [code for code in sorted_codes[:cutoff] if code in holding_set]
    selected = retained[:]

    for code in sorted_codes:
        if len(selected) >= n:
            break
        if code not in selected:
            selected.append(code)

    return selected[:n]
