"""
Phase 3 bot shim: delegate all logic to root `bot_trade.py` to avoid duplication.

This keeps the file for compatibility but ensures a single implementation is maintained at the repository root.
"""
# Lazy import the root implementation to avoid circular imports
from typing import Any


def reset_history():
    from bot_trade import reset_history as root_reset_history
    root_reset_history()


def make_decision(epoch: int, priceA: float, priceB: float, strategy: str = 'blended', params: dict[str, Any] | None = None, history: list[dict[str, float]] | None = None):
    """Shim to call root implementation, keeping the same 2-asset signature.
    Returns a dict {'Asset A', 'Asset B', 'Cash'}.
    """
    from bot_trade import make_decision as root_make_decision
    return root_make_decision(epoch, priceA, priceB, strategy=strategy, params=params, history=history)
"""
Multi-asset `bot_trade` for phase_3 dataset (Asset A, Asset B, Cash).

This file implements several strategies adapted from the single-asset `bot_trade`:
- baseline: momentum-based (delta positive -> overweight the asset)
- sma: sma per-asset
- composite: sma + stoploss + vol-scaling per-asset
- blended: blend baseline+composite per-asset

Compatibility: `make_decision(epoch:int, priceA: float, priceB: float, strategy='blended', params=None, history=None)`
History is a list of dicts with epoch, priceA, priceB, and optional asset weights saved.
"""

import statistics
from typing import Any

# Internal module-level history
MODULE_HISTORY: list[dict[str, float]] = []

def reset_history():
    MODULE_HISTORY.clear()

def _sma_from_prices(prices, short_w=10, long_w=30):
    if len(prices) < long_w:
        return 0.5
    short = sum(prices[-short_w:]) / short_w
    long = sum(prices[-long_w:]) / long_w
    return 0.7 if short > long else 0.3

def _volatility_scale_from_prices(prices, base_weight=0.5, vol_window=20, min_w=0.1, max_w=0.9):
    if len(prices) < vol_window + 1:
        return base_weight
    rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
    vol = statistics.pstdev(rets) if len(rets) > 0 else 0
    ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
    ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
    if vol == 0:
        return base_weight
    scale = ref_vol / vol if vol > 0 else 1.0
    weight = base_weight * scale
    return max(min_w, min(max_w, weight))

def make_decision(epoch: int, priceA: float, priceB: float, strategy: str = 'blended', params: dict[str, Any] | None = None, history: list[dict[str, float]] | None = None):
    if history is None:
        history = MODULE_HISTORY
    history.append({'epoch': epoch, 'priceA': priceA, 'priceB': priceB})
    params = params or {}

    # compute per-asset weights using the specified strategy
    def per_asset_weight(asset='A'):
        prices = [h['priceA'] if asset == 'A' else h['priceB'] for h in history]
        if strategy == 'baseline':
            if len(prices) < 2:
                return 0.33
            delta = prices[-1] - prices[-2]
            return 0.45 if delta > 0 else 0.15
        elif strategy == 'sma':
            short = int(params.get('short', 10))
            long = int(params.get('long', 30))
            return _sma_from_prices(prices, short, long)
        elif strategy == 'composite':
            short = int(params.get('short', 3))
            long = int(params.get('long', 20))
            threshold = float(params.get('threshold', 0.02))
            vol_window = int(params.get('vol_window', 10))
            vol_multiplier = float(params.get('vol_multiplier', 1.0))
            base = _sma_from_prices(prices, short, long)
            # stoploss
            if len(prices) >= 2:
                peak = max(prices)
                if (peak - prices[-1]) / peak > threshold:
                    base = 0.0
            # vol scaling
            base = _volatility_scale_from_prices(prices, base, vol_window, 0.1, 0.9)
            return base
        elif strategy == 'blended':
            # blend baseline & composite
            blend = float(params.get('blend', 0.7))
            subp = params.get('sub_params', {})
            w_base = 0.45 if (len(prices) >= 2 and prices[-1] - prices[-2] > 0) else 0.15
            short = int(subp.get('short', 4))
            long = int(subp.get('long', 17))
            threshold = float(subp.get('threshold', 0.015))
            vol_window = int(subp.get('vol_window', 17))
            vol_multiplier = float(subp.get('vol_multiplier', 0.6))
            w_comp = _sma_from_prices(prices, short, long)
            if len(prices) >= 2:
                peak = max(prices)
                if (peak - prices[-1]) / peak > threshold:
                    w_comp = 0.0
            w_comp = _volatility_scale_from_prices(prices, w_comp, vol_window, 0.1, 0.9)
            return max(0.0, min(1.0, blend * w_base + (1 - blend) * w_comp))
        else:
            return 0.33

    # handle aliases
    if strategy == 'blended_mo_tuned':
        params = params or {}
        params['blend'] = params.get('blend', 0.9007572211634083)
        params['sub_params'] = params.get('sub_params', {'short': 4, 'long': 17, 'threshold': 0.015304920910622851, 'vol_window': 17, 'vol_multiplier': 0.6235913304206027})

    wA = per_asset_weight('A')
    wB = per_asset_weight('B')
    # normalize so wA + wB <= 1 and remainder to Cash
    total = wA + wB
    if total > 1.0:
        wA = wA / total
        wB = wB / total
        cash = 0.0
    else:
        cash = 1.0 - (wA + wB)
    # store asset weights in history to support smoothing if needed
    try:
        history[-1]['assetA_weight'] = wA
        history[-1]['assetB_weight'] = wB
        history[-1]['cash'] = cash
    except Exception:
        pass
    return {'Asset A': wA, 'Asset B': wB, 'Cash': cash}


# Note: Removed leftover simplistic duplicate make_decision to avoid shadowing full implementation.