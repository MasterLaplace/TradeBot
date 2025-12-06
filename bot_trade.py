

import math
import statistics
from typing import Any

# The bot now supports multiple strategies via `strategy` parameter.
# make_decision signature: make_decision(epoch, price, strategy='blended_robust', params=None, history=None)
#
# Default strategy: 'blended_robust' (recommended)
# This is the most robust strategy found by the Optuna runs (trial 113), with tuned parameters:
#   blend = 0.9996436412271156
#   sub_params: short=10, long=27, threshold=0.0452921667498627, vol_window=13, vol_multiplier=0.7322341609310995
#
# Compatibility note for production testers:
# - Test platform calls `make_decision(epoch, price)` sequentially for each epoch.
# - We maintain an internal `MODULE_HISTORY` list to preserve state across calls when `history` is not provided.
# - For local testing, you can pass a `history` list to `make_decision` to keep control over state (e.g., in `main.py`).

# Internal module-level history stored by default when the testing harness calls `make_decision(epoch, price)`
MODULE_HISTORY: list[dict[str, float]] = []


def reset_history():
    """Reset the internal module-level history; useful for testing and for the online judge sessions between runs."""
    MODULE_HISTORY.clear()

def _get_delta(history: list[dict[str, float]]) -> float:
    return history[-1]["price"] - history[-2]["price"]

def _sma(history: list[dict[str, float]], short_w: int = 10, long_w: int = 30) -> float:
    prices = [h['price'] for h in history]
    if len(prices) < long_w:
        return 0.5
    short = sum(prices[-short_w:]) / short_w
    long = sum(prices[-long_w:]) / long_w
    return 0.7 if short > long else 0.3

def _volatility_scale(history: list[dict[str, float]], base_weight: float = 0.5, vol_window: int = 20, min_w: float = 0.1, max_w: float = 0.9) -> float:
    prices = [h['price'] for h in history]
    if len(prices) < vol_window + 1:
        return base_weight
    # compute daily returns
    rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
    vol = statistics.pstdev(rets) if len(rets) > 0 else 0
    if vol == 0:
        return base_weight
    # Reference volatility is the median of historical vol over entire history (fallback simple)
    ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
    ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
    scale = ref_vol / vol if vol > 0 else 1.0
    weight = base_weight * scale
    return max(min_w, min(max_w, weight))

def _stoploss(history: list[dict[str, float]], base_weight: float = 0.7, threshold: float = 0.05) -> float:
    # threshold is fractional drop from peak (e.g., 0.05 => 5%). If price is down > threshold from peak, shrink exposure.
    prices = [h['price'] for h in history]
    if len(prices) < 2:
        return 0.5
    peak = max(prices)
    current = prices[-1]
    if (peak - current) / peak > threshold:
        # sell down to cash
        return 0.0
    else:
        return base_weight

def make_decision(epoch: int, price: float, priceB: float | None = None, strategy: str = 'blended_robust_ensemble', params: dict[str, Any] | None = None, history: list[dict[str, float]] | None = None):
    """Return the portfolio allocation for the given epoch and price.

    Compatibility: the function can be called as `make_decision(epoch, price)` by the online judge.
    If `history` is omitted, the function will append to an internal module-level `MODULE_HISTORY` list
    so the state is preserved across calls.

    For local runs, you may provide your own `history` list to keep full control over the state.
    """
    # Use module-level history by default (ensures compatibility with site testing harness)
    if history is None:
        history = MODULE_HISTORY

    # Default params
    params = params or {}

    # If passed a second price, run multi-asset logic (Asset A and Asset B)
    if priceB is not None:
        # ensure history exists and uses A/B keys
        if history is None:
            history = MODULE_HISTORY
        history.append({'epoch': epoch, 'priceA': price, 'priceB': priceB})

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

        # If running the ensemble strategy for multi-asset, compute averaged allocations early
        if strategy == 'blended_robust_ensemble':
            ensemble_candidates = [
                {'blend': 0.9996436412271156, 'sub_params': {'short': 10, 'long': 27, 'threshold': 0.0452921667498627, 'vol_window': 13, 'vol_multiplier': 0.7322341609310995}},
                {'blend': 0.9995821862855174, 'sub_params': {'short': 9, 'long': 23, 'threshold': 0.0431405472291731, 'vol_window': 14, 'vol_multiplier': 0.7285051684602486}},
                {'blend': 0.99956027510904, 'sub_params': {'short': 9, 'long': 51, 'threshold': 0.0439471735236884, 'vol_window': 10, 'vol_multiplier': 0.6304819173928731}},
                {'blend': 0.9994743275641822, 'sub_params': {'short': 10, 'long': 23, 'threshold': 0.0440539482092565, 'vol_window': 18, 'vol_multiplier': 0.7221503215581743}},
                {'blend': 0.9994293508702308, 'sub_params': {'short': 8, 'long': 29, 'threshold': 0.0491851548056736, 'vol_window': 12, 'vol_multiplier': 0.6668045796733401}},
            ]
            # candidate histories are initialized from the global history WITHOUT the current epoch so indicators have context
            candidate_histories = [list(history[:-1]) for _ in ensemble_candidates]
            allocsA = []
            allocsB = []
            for i, cand in enumerate(ensemble_candidates):
                cand_hist = list(candidate_histories[i])
                alloc = make_decision(epoch, price, priceB, strategy='blended', params=cand, history=cand_hist)
                allocsA.append(alloc.get('Asset A', 0.0))
                allocsB.append(alloc.get('Asset B', 0.0))
                candidate_histories[i].append({'epoch': epoch, 'priceA': price, 'priceB': priceB})
            wA = sum(allocsA) / len(allocsA)
            wB = sum(allocsB) / len(allocsB)
            total = wA + wB
            if total > 1.0:
                wA = wA / total
                wB = wB / total
                cash = 0.0
            else:
                cash = 1.0 - (wA + wB)
            try:
                history[-1]['assetA_weight'] = wA
                history[-1]['assetB_weight'] = wB
                history[-1]['cash'] = cash
            except Exception:
                pass
            return {'Asset A': wA, 'Asset B': wB, 'Cash': cash}

        def weight_for(prices, strat):
            # per-asset single strategy decision (copied from phase3 implementation)
            if strat == 'baseline':
                if len(prices) < 2:
                    return 0.33
                delta = prices[-1] - prices[-2]
                return 0.45 if delta > 0 else 0.15
            elif strat == 'sma':
                short_w = int(params.get('short', 10))
                long_w = int(params.get('long', 30))
                return _sma_from_prices(prices, short_w, long_w)
            elif strat == 'composite':
                short = int(params.get('short', 3))
                long = int(params.get('long', 20))
                threshold = float(params.get('threshold', 0.02))
                vol_window = int(params.get('vol_window', 10))
                vol_multiplier = float(params.get('vol_multiplier', 1.0))
                base = _sma_from_prices(prices, short, long)
                if len(prices) >= 2:
                    peak = max(prices)
                    if (peak - prices[-1]) / peak > threshold:
                        base = 0.0
                base = _volatility_scale_from_prices(prices, base, vol_window, 0.1, 0.9)
                return base
            elif strat in ('blended', 'blended_mo_tuned', 'blended_tuned', 'blended_robust'):
                blend = float(params.get('blend', 0.5))
                subp = params.get('sub_params', {})
                short = int(subp.get('short', 5))
                long = int(subp.get('long', 16))
                threshold = float(subp.get('threshold', 0.0175))
                vol_window = int(subp.get('vol_window', 16))
                vol_multiplier = float(subp.get('vol_multiplier', 0.72))
                # If we are using the robust strategy and no params provided, set tuned defaults
                if strat == 'blended_robust' and ('blend' not in params or 'sub_params' not in params):
                    blend = 0.9996436412271156
                    short = 10
                    long = 27
                    threshold = 0.0452921667498627
                    vol_window = 13
                    vol_multiplier = 0.7322341609310995
                w_base = 0.45 if (len(prices) >= 2 and prices[-1] - prices[-2] > 0) else 0.15
                w_comp = _sma_from_prices(prices, short, long)
                if len(prices) >= 2:
                    peak = max(prices)
                    if (peak - prices[-1]) / peak > threshold:
                        w_comp = 0.0
                w_comp = _volatility_scale_from_prices(prices, w_comp, vol_window, 0.1, 0.9)
                return max(0.0, min(1.0, blend * w_base + (1 - blend) * w_comp))
            else:
                return 0.33

        pricesA = [h['priceA'] for h in history]
        pricesB = [h['priceB'] for h in history]
        wA = weight_for(pricesA, strategy)
        wB = weight_for(pricesB, strategy)
        total = wA + wB
        # If this safe mode has a max total exposure (max_total_exposure), apply scaling to reduce total exposure
        if strategy == 'blended_robust_safe':
            max_total_exposure = float(params.get('max_total_exposure', 0.9)) if params else 0.9
            if total > max_total_exposure and total > 0:
                scale = max_total_exposure / total
                wA *= scale
                wB *= scale
                total = wA + wB
        if total > 1.0:
            # normalize to 1.0 overall
            wA = wA / total
            wB = wB / total
            cash = 0.0
        else:
            cash = 1.0 - (wA + wB)
        try:
            history[-1]['assetA_weight'] = wA
            history[-1]['assetB_weight'] = wB
            history[-1]['cash'] = cash
        except Exception:
            pass
        return {'Asset A': wA, 'Asset B': wB, 'Cash': cash}

    # Ensemble multi-asset strategy: average allocations from multiple top candidates
    if priceB is not None and strategy == 'blended_robust_ensemble':
        # Use top 5 robust candidates (hard-coded to avoid file IO in judge)
        ensemble_candidates = [
            {'blend': 0.9996436412271156, 'sub_params': {'short': 10, 'long': 27, 'threshold': 0.0452921667498627, 'vol_window': 13, 'vol_multiplier': 0.7322341609310995}},
            {'blend': 0.9995821862855174, 'sub_params': {'short': 9, 'long': 23, 'threshold': 0.0431405472291731, 'vol_window': 14, 'vol_multiplier': 0.7285051684602486}},
            {'blend': 0.99956027510904, 'sub_params': {'short': 9, 'long': 51, 'threshold': 0.0439471735236884, 'vol_window': 10, 'vol_multiplier': 0.6304819173928731}},
            {'blend': 0.9994743275641822, 'sub_params': {'short': 10, 'long': 23, 'threshold': 0.0440539482092565, 'vol_window': 18, 'vol_multiplier': 0.7221503215581743}},
            {'blend': 0.9994293508702308, 'sub_params': {'short': 8, 'long': 29, 'threshold': 0.0491851548056736, 'vol_window': 12, 'vol_multiplier': 0.6668045796733401}},
        ]
        # create per-candidate histories to avoid cross-talk
        # start each candidate history from the global history WITHOUT the current epoch so indicators have context
        candidate_histories = [list(history[:-1]) for _ in ensemble_candidates]
        allocsA = []
        allocsB = []
        # compute each candidate's allocation at this epoch by calling make_decision with candidate history
        for i, cand in enumerate(ensemble_candidates):
            cand_hist = list(candidate_histories[i])
            # call make_decision for the candidate's blended config to obtain per-asset allocation
            alloc = make_decision(epoch, price, priceB, strategy='blended', params=cand, history=cand_hist)
            allocsA.append(alloc.get('Asset A', 0.0))
            allocsB.append(alloc.get('Asset B', 0.0))
            # append current prices to candidate history copy for next epoch (not the module history)
            candidate_histories[i].append({'epoch': epoch, 'priceA': price, 'priceB': priceB})
        wA = sum(allocsA) / len(allocsA)
        wB = sum(allocsB) / len(allocsB)
        total = wA + wB
        if total > 1.0:
            wA = wA / total
            wB = wB / total
            cash = 0.0
        else:
            cash = 1.0 - (wA + wB)
        try:
            history[-1]['assetA_weight'] = wA
            history[-1]['assetB_weight'] = wB
            history[-1]['cash'] = cash
        except Exception:
            pass
        return {'Asset A': wA, 'Asset B': wB, 'Cash': cash}

    # append single-asset price entry to history for single-asset processing
    history.append({'epoch': epoch, 'price': price})

    if strategy == 'baseline':
        if len(history) < 2:
            return {'Asset B': 0.5, 'Cash': 0.5}
        if _get_delta(history) > 0:
            return {'Asset B': 0.7, 'Cash': 0.3}
        else:
            return {'Asset B': 0.3, 'Cash': 0.7}
    elif strategy == 'sma':
        short_w = int(params.get('short', 10))
        long_w = int(params.get('long', 30))
        asset_w = _sma(history, short_w, long_w)
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'volscale':
        base_weight = float(params.get('base', 0.5))
        vol_window = int(params.get('vol_window', 20))
        min_w = float(params.get('min', 0.1))
        max_w = float(params.get('max', 0.9))
        asset_w = _volatility_scale(history, base_weight, vol_window, min_w, max_w)
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'stoploss':
        base_weight = float(params.get('base', 0.7))
        threshold = float(params.get('threshold', 0.05))
        asset_w = _stoploss(history, base_weight, threshold)
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'sma_stoploss':
        # combine sma and stoploss: target asset weight via sma, then apply stoploss threshold
        short_w = int(params.get('short', 3))
        long_w = int(params.get('long', 20))
        threshold = float(params.get('threshold', 0.02))
        target = _sma(history, short_w, long_w)
        # apply stoploss: if drawdown from peak > threshold, set weight to 0
        if len(history) >= 2:
            peak = max([h['price'] for h in history])
            if (peak - history[-1]['price']) / peak > threshold:
                asset_w = 0.0
            else:
                asset_w = target
        else:
            asset_w = target
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'sma_volfilter':
        short_w = int(params.get('short', 3))
        long_w = int(params.get('long', 20))
        vol_window = int(params.get('vol_window', 10))
        vol_threshold = float(params.get('vol_threshold', 0.02))
        # target weight by sma
        target = _sma(history, short_w, long_w)
        # compute vol
        prices = [h['price'] for h in history]
        if len(prices) < vol_window + 1:
            vol = 0
        else:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
        if vol > vol_threshold:
            # reduce exposure
            asset_w = 0.3
        else:
            asset_w = target
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'sma_smooth_stop':
        # SMA target with smoothing (gradual position changes), partial stop-loss and vol-scaling
        short_w = int(params.get('short', 3))
        long_w = int(params.get('long', 20))
        alpha = float(params.get('alpha', 0.2))  # smoothing factor
        stop_threshold = float(params.get('stop_threshold', 0.05))
        stop_level = float(params.get('stop_level', 0.2))
        vol_window = int(params.get('vol_window', 10))
        vol_threshold = float(params.get('vol_threshold', 0.02))
        vol_scale_factor = float(params.get('vol_scale_factor', 0.7))

        target = _sma(history, short_w, long_w)
        # get previous asset weight (if any) from history decisions; fall back to 0.5
        prev_w = None
        if len(history) >= 2 and isinstance(history[-2], dict):
            prev_w = history[-2].get('asset_weight')
        if prev_w is None:
            prev_w = 0.5

        # smoothing
        asset_w = prev_w + alpha * (target - prev_w)

        # volatility scaling
        prices = [h['price'] for h in history]
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            if vol > vol_threshold:
                asset_w *= vol_scale_factor

        # partial stop-loss based on peak drawdown
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > stop_threshold:
                asset_w = min(asset_w, stop_level)

        # clamp
        asset_w = max(0.0, min(1.0, asset_w))
        # store current chosen weight in history for next smoothing step
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'adaptive_baseline':
        # baseline adjusted by risk gating: apply baseline allocation but scale it by risk factor
        # risk factor is product of volatility gate and stop loss gate
        if len(history) < 2:
            base_weight = 0.5
        else:
            base_weight = 0.7 if _get_delta(history) > 0 else 0.3
        # params
        vol_window = int(params.get('vol_window', 10))
        vol_threshold = float(params.get('vol_threshold', 0.02))
        vol_scale_factor = float(params.get('vol_scale_factor', 0.7))
        stop_threshold = float(params.get('stop_threshold', 0.05))
        stop_scale = float(params.get('stop_scale', 0.4))

        # vol gate
        prices = [h['price'] for h in history]
        vol_gate = 1.0
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            if vol > vol_threshold:
                vol_gate = vol_scale_factor

        # stop gate
        stop_gate = 1.0
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > stop_threshold:
                stop_gate = stop_scale

        risk_factor = vol_gate * stop_gate
        asset_w = max(0.0, min(1.0, base_weight * risk_factor))
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'blended':
        # Blend baseline with composite (or another strategy) to get risk-managed baseline
        # params: blend (0..1), composite sub-params
        blend = float(params.get('blend', 0.5))
        # calculate baseline weight
        if len(history) < 2:
            baseline_w = 0.5
        else:
            baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
        # calculate composite-like weight using a simplified composite (SMA + stoploss + vol)
        sub_params = params.get('sub_params', {})
        short_w = int(sub_params.get('short', 5))
        long_w = int(sub_params.get('long', 16))
        threshold = float(sub_params.get('threshold', 0.0175))
        vol_window = int(sub_params.get('vol_window', 16))
        vol_multiplier = float(sub_params.get('vol_multiplier', 0.72))
        # compute composite weight: sma
        target = _sma(history, short_w, long_w)
        # stoploss
        prices = [h['price'] for h in history]
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > threshold:
                target = 0.0
        # vol scaling
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
            if vol > 0:
                scale = ref_vol / vol
            else:
                scale = 1.0
            target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))

        asset_w = blend * baseline_w + (1.0 - blend) * target
        asset_w = max(0.0, min(1.0, asset_w))
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'blended_tuned':
        # Pre-configured tuned parameters (from optuna walk-forward tuning)
        tuned_params = {
            'blend': 0.5004034618154071,
            'sub_params': {
                'short': 6,
                'long': 39,
                'threshold': 0.010008502229651432,
                'vol_window': 9,
                'vol_multiplier': 0.6940552234824028,
            },
        }
        # Implement blended behavior directly (avoid recursion to prevent double-appending to history)
        blend = float(tuned_params.get('blend', 0.5))
        sub_params = tuned_params.get('sub_params', {})
        short_w = int(sub_params.get('short', 5))
        long_w = int(sub_params.get('long', 16))
        threshold = float(sub_params.get('threshold', 0.0175))
        vol_window = int(sub_params.get('vol_window', 16))
        vol_multiplier = float(sub_params.get('vol_multiplier', 0.72))
        # compute baseline & target like the original blended
        if len(history) < 2:
            baseline_w = 0.5
        else:
            baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
        target = _sma(history, short_w, long_w)
        prices = [h['price'] for h in history]
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > threshold:
                target = 0.0
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
            if vol > 0:
                scale = ref_vol / vol
            else:
                scale = 1.0
            target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))
        asset_w = blend * baseline_w + (1.0 - blend) * target
        asset_w = max(0.0, min(1.0, asset_w))
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'blended_mo_tuned':
        # Multi-objective tuned parameters (from optuna_blended_mo)
        tuned_params = {
            'blend': 0.9007572211634083,
            'sub_params': {
                'short': 4,
                'long': 17,
                'threshold': 0.015304920910622851,
                'vol_window': 17,
                'vol_multiplier': 0.6235913304206027,
            },
        }
        # Implement blended logic directly (avoid recursion)
        blend = float(tuned_params.get('blend', 0.5))
        sub_params = tuned_params.get('sub_params', {})
        short_w = int(sub_params.get('short', 5))
        long_w = int(sub_params.get('long', 16))
        threshold = float(sub_params.get('threshold', 0.0175))
        vol_window = int(sub_params.get('vol_window', 16))
        vol_multiplier = float(sub_params.get('vol_multiplier', 0.72))
        # compute baseline & target like the original blended
        if len(history) < 2:
            baseline_w = 0.5
        else:
            baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
        target = _sma(history, short_w, long_w)
        prices = [h['price'] for h in history]
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > threshold:
                target = 0.0
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
            if vol > 0:
                scale = ref_vol / vol
            else:
                scale = 1.0
            target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))
        asset_w = blend * baseline_w + (1.0 - blend) * target
        asset_w = max(0.0, min(1.0, asset_w))
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'blended_robust':
        # Robust tuned parameters discovered by robust Optuna (trial 113)
        tuned_params = {
            'blend': 0.9996436412271156,
            'sub_params': {
                'short': 10,
                'long': 27,
                'threshold': 0.0452921667498627,
                'vol_window': 13,
                'vol_multiplier': 0.7322341609310995,
            },
        }
        # Implement the blended logic directly with tuned params
        blend = float(tuned_params.get('blend', 0.5))
        sub_params = tuned_params.get('sub_params', {})
        short_w = int(sub_params.get('short', 5))
        long_w = int(sub_params.get('long', 16))
        threshold = float(sub_params.get('threshold', 0.0175))
        vol_window = int(sub_params.get('vol_window', 16))
        vol_multiplier = float(sub_params.get('vol_multiplier', 0.72))
        # compute baseline & target like the original blended
        if len(history) < 2:
            baseline_w = 0.5
        else:
            baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
        target = _sma(history, short_w, long_w)
        prices = [h['price'] for h in history]
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > threshold:
                target = 0.0
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
            if vol > 0:
                scale = ref_vol / vol
            else:
                scale = 1.0
            target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))
        asset_w = blend * baseline_w + (1.0 - blend) * target
        asset_w = max(0.0, min(1.0, asset_w))
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'blended_robust_safe':
        # Same as blended_robust but with max exposure cap to reduce drawdown
        tuned_params = {
            'blend': 0.9996436412271156,
            'sub_params': {
                'short': 10,
                'long': 27,
                'threshold': 0.0452921667498627,
                'vol_window': 13,
                'vol_multiplier': 0.7322341609310995,
            },
        }
        max_exposure = float(params.get('max_exposure', 0.5)) if params else 0.5
        # compute as blended_robust
        blend = float(tuned_params.get('blend', 0.5))
        sub_params = tuned_params.get('sub_params', {})
        short_w = int(sub_params.get('short', 5))
        long_w = int(sub_params.get('long', 16))
        threshold = float(sub_params.get('threshold', 0.0175))
        vol_window = int(sub_params.get('vol_window', 16))
        vol_multiplier = float(sub_params.get('vol_multiplier', 0.72))
        if len(history) < 2:
            baseline_w = 0.5
        else:
            baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
        target = _sma(history, short_w, long_w)
        prices = [h['price'] for h in history]
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > threshold:
                target = 0.0
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0
            ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
            if vol > 0:
                scale = ref_vol / vol
            else:
                scale = 1.0
            target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))
        asset_w = blend * baseline_w + (1.0 - blend) * target
        asset_w = max(0.0, min(1.0, asset_w))
        # Apply cap
        if asset_w > max_exposure:
            asset_w = max_exposure
        try:
            history[-1]['asset_weight'] = asset_w
        except Exception:
            pass
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}
    elif strategy == 'composite':
        # Composite strategy: SMA signal, apply stoploss threshold, and apply volatility scaling multiplier
        short_w = int(params.get('short', 3))
        long_w = int(params.get('long', 20))
        threshold = float(params.get('threshold', 0.02))
        vol_window = int(params.get('vol_window', 10))
        vol_multiplier = float(params.get('vol_multiplier', 1.0))
        base = _sma(history, short_w, long_w)
        # stoploss
        prices = [h['price'] for h in history]
        if len(prices) >= 2:
            peak = max(prices)
            if (peak - prices[-1]) / peak > threshold:
                base = 0.0
        # volatility scaling (contract exposure when vol is high)
        if len(prices) >= vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
            vol = statistics.pstdev(rets)
            # apply multiplier: if vol > ref then scale down
            ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
            ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
            if vol > 0:
                scale = ref_vol / vol
            else:
                scale = 1.0
            base = max(0.0, min(1.0, base * (1 + (scale - 1) * vol_multiplier)))
        asset_w = base
        return {'Asset B': asset_w, 'Cash': 1 - asset_w}

    elif strategy == 'blended_robust_ensemble':
        # Ensemble of top 5 candidates averaged for single-asset mode
        ensemble_candidates = [
            {'blend': 0.9996436412271156, 'sub_params': {'short': 10, 'long': 27, 'threshold': 0.0452921667498627, 'vol_window': 13, 'vol_multiplier': 0.7322341609310995}},
            {'blend': 0.9995821862855174, 'sub_params': {'short': 9, 'long': 23, 'threshold': 0.0431405472291731, 'vol_window': 14, 'vol_multiplier': 0.7285051684602486}},
            {'blend': 0.99956027510904, 'sub_params': {'short': 9, 'long': 51, 'threshold': 0.0439471735236884, 'vol_window': 10, 'vol_multiplier': 0.6304819173928731}},
            {'blend': 0.9994743275641822, 'sub_params': {'short': 10, 'long': 23, 'threshold': 0.0440539482092565, 'vol_window': 18, 'vol_multiplier': 0.7221503215581743}},
            {'blend': 0.9994293508702308, 'sub_params': {'short': 8, 'long': 29, 'threshold': 0.0491851548056736, 'vol_window': 12, 'vol_multiplier': 0.6668045796733401}},
        ]
        # start each candidate history from the current full history so indicators have context
        candidate_histories = [list(history) for _ in ensemble_candidates]
        allocs = []
        for i, cand in enumerate(ensemble_candidates):
            cand_hist = list(candidate_histories[i])
            # call make_decision for the candidate's blended config to obtain allocation
            alloc = make_decision(epoch, price, strategy='blended', params=cand, history=cand_hist)
            asset_w = alloc.get('Asset B', 0.0)
            blend = float(cand.get('blend', 0.5))
            subp = cand.get('sub_params', {})
            short_w = int(subp.get('short', 5))
            long_w = int(subp.get('long', 16))
            threshold = float(subp.get('threshold', 0.0175))
            vol_window = int(subp.get('vol_window', 16))
            vol_multiplier = float(subp.get('vol_multiplier', 0.72))
            # baseline & target
            if len(candidate_histories[i]) < 1:
                baseline_w = 0.5
            else:
                # safe delta check
                if len(history) < 2:
                    baseline_w = 0.5
                else:
                    baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
            target = _sma(candidate_histories[i] + [{'epoch': epoch, 'price': price}], short_w, long_w) if len(candidate_histories[i]) >= 0 else _sma(history, short_w, long_w)
            prices_local = [h['price'] for h in candidate_histories[i]] + [price]
            if len(prices_local) >= 2:
                peak = max(prices_local)
                if (peak - prices_local[-1]) / peak > threshold:
                    target = 0.0
            if len(prices_local) >= vol_window + 1:
                rets = [(prices_local[j] / prices_local[j-1] - 1.0) for j in range(-vol_window, 0)]
                vol = statistics.pstdev(rets) if len(rets) > 0 else 0
                ref_rets = [(prices_local[j] / prices_local[j-1] - 1.0) for j in range(1, len(prices_local))]
                ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
                if vol > 0:
                    scale = ref_vol / vol
                else:
                    scale = 1.0
                target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))
            asset_w = blend * baseline_w + (1.0 - blend) * target
            asset_w = max(0.0, min(1.0, asset_w))
            allocs.append(asset_w)
            candidate_histories[i].append({'epoch': epoch, 'price': price})
        avg_alloc = sum(allocs) / len(allocs)
        try:
            history[-1]['asset_weight'] = avg_alloc
        except Exception:
            pass
        return {'Asset B': avg_alloc, 'Cash': 1 - avg_alloc}
    elif strategy == 'ensemble':
        # Ensemble that chooses between baseline and blended/composite based on volatility and drawdown
        switch_vol_window = int(params.get('switch_vol_window', 10))
        switch_vol_threshold = float(params.get('switch_vol_threshold', 0.03))
        fallback = params.get('fallback', 'baseline')
        preferred = params.get('preferred', 'blended')
        # compute vol
        prices = [h['price'] for h in history]
        vol = 0.0
        if len(prices) >= switch_vol_window + 1:
            rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-switch_vol_window, 0)]
            vol = statistics.pstdev(rets) if len(rets) > 0 else 0.0
        if vol > switch_vol_threshold:
            # if fallback is baseline, compute baseline decision
            if fallback == 'baseline':
                if len(history) < 2:
                    return {'Asset B': 0.5, 'Cash': 0.5}
                if _get_delta(history) > 0:
                    return {'Asset B': 0.7, 'Cash': 0.3}
                else:
                    return {'Asset B': 0.3, 'Cash': 0.7}
            else:
                # fallback to cash
                return {'Asset B': 0.0, 'Cash': 1.0}
        else:
            # use preferred strategy (we implement blended inline to avoid recursive history appends)
            if preferred == 'blended':
                p = params.get('preferred_params', {})
                blend = float(p.get('blend', params.get('blend', 0.5)))
                sub = p.get('sub_params', params.get('sub_params', {}))
                short_w = int(sub.get('short', 5))
                long_w = int(sub.get('long', 16))
                threshold = float(sub.get('threshold', 0.0175))
                vol_window = int(sub.get('vol_window', 16))
                vol_multiplier = float(sub.get('vol_multiplier', 0.72))
                # blended logic copy
                if len(history) < 2:
                    baseline_w = 0.5
                else:
                    baseline_w = 0.7 if _get_delta(history) > 0 else 0.3
                target = _sma(history, short_w, long_w)
                prices = [h['price'] for h in history]
                if len(prices) >= 2:
                    peak = max(prices)
                    if (peak - prices[-1]) / peak > threshold:
                        target = 0.0
                if len(prices) >= vol_window + 1:
                    rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
                    vol = statistics.pstdev(rets) if len(rets) > 0 else 0
                    ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
                    ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
                    if vol > 0:
                        scale = ref_vol / vol
                    else:
                        scale = 1.0
                    target = max(0.0, min(1.0, target * (1 + (scale - 1) * vol_multiplier)))
                asset_w = blend * baseline_w + (1.0 - blend) * target
                asset_w = max(0.0, min(1.0, asset_w))
                try:
                    history[-1]['asset_weight'] = asset_w
                except Exception:
                    pass
                return {'Asset B': asset_w, 'Cash': 1 - asset_w}
            elif preferred == 'composite':
                # implement local composite logic to avoid recursion
                p = params.get('preferred_params', {})
                short_w = int(p.get('short', 3))
                long_w = int(p.get('long', 20))
                threshold = float(p.get('threshold', 0.02))
                vol_window = int(p.get('vol_window', 10))
                vol_multiplier = float(p.get('vol_multiplier', 1.0))
                base = _sma(history, short_w, long_w)
                prices = [h['price'] for h in history]
                if len(prices) >= 2:
                    peak = max(prices)
                    if (peak - prices[-1]) / peak > threshold:
                        base = 0.0
                if len(prices) >= vol_window + 1:
                    rets = [(prices[i] / prices[i-1] - 1.0) for i in range(-vol_window, 0)]
                    vol = statistics.pstdev(rets)
                    ref_rets = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]
                    ref_vol = statistics.median([abs(r) for r in ref_rets]) if len(ref_rets) > 0 else vol
                    if vol > 0:
                        scale = ref_vol / vol
                    else:
                        scale = 1.0
                    base = max(0.0, min(1.0, base * (1 + (scale - 1) * vol_multiplier)))
                asset_w = base
                return {'Asset B': asset_w, 'Cash': 1 - asset_w}
            else:
                if len(history) < 2:
                    return {'Asset B': 0.5, 'Cash': 0.5}
                if _get_delta(history) > 0:
                    return {'Asset B': 0.7, 'Cash': 0.3}
                else:
                    return {'Asset B': 0.3, 'Cash': 0.7}
    else:
        # unknown strategy -> fallback to baseline
        return make_decision(epoch, price, strategy='baseline', params=params, history=history)
