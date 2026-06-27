"""
Reporting — journal summaries and the weekly newsletter (SPEC.md §3.7/§3.9).

Pure formatting over journal events + live portfolio state. Produces plain
text suitable for the console and for posting to Discord.
"""

from typing import List

from ..engine import TradeBotEngine


def _trade_lines(trades: List[dict]) -> List[str]:
    lines = []
    for t in trades:
        sign = "🟢" if "BUY" in t.get("action", "") else "🔴"
        lines.append(
            f"  {sign} {t['action']} {t['quantity']} {t['symbol']} "
            f"@ ${t['price']:.2f}  ({t.get('reason', '')})"
        )
    return lines


def _portfolio_block(engine: TradeBotEngine, pm) -> List[str]:
    held = pm.held_symbols()
    prices = engine.live_prices(held)
    mtm = pm.mark_to_market(prices)
    if mtm["track_cash"]:
        head = (f"{pm.label}: total ${mtm['total']:,.2f} "
                f"(cash ${mtm['cash']:,.2f} + holdings ${mtm['holdings_value']:,.2f}), "
                f"unrealized P&L ${mtm['unrealized_pnl']:+,.2f}")
    else:
        head = (f"{pm.label}: holdings ${mtm['holdings_value']:,.2f} "
                f"(invested ${mtm['cost_basis']:,.2f}), "
                f"unrealized P&L ${mtm['unrealized_pnl']:+,.2f}")
    lines = [head]
    for sym, p in mtm["positions"].items():
        lines.append(
            f"  • {sym}: {p['quantity']} @ ${p['entry']:.2f} → ${p['price']:.2f} "
            f"({p['pnl']:+.2f}, {p['pnl_pct']:+.1f}%)"
        )
    return lines


def build_report(engine: TradeBotEngine, last_n: int = 30) -> str:
    """Compact report of recent journal activity + both portfolios."""
    events = engine.journal.tail(last_n)
    trades = [e for e in events if e.get("type") == "trade"]
    signals = [e for e in events if e.get("type") == "signal"]

    lines = ["📋 TradeBot report", ""]
    lines += _portfolio_block(engine, engine.real)
    lines.append("")
    lines += _portfolio_block(engine, engine.sim)
    lines.append("")

    if trades:
        lines.append(f"Recent trades ({len(trades)}):")
        lines += _trade_lines(trades)
        lines.append("")

    actionable = [s for s in signals if s.get("direction") in ("BUY", "SELL")]
    if actionable:
        actionable.sort(key=lambda s: abs(s.get("composite", 0)), reverse=True)
        lines.append("Top recent signals:")
        for s in actionable[:5]:
            lines.append(
                f"  {s['direction']} {s['symbol']} "
                f"(conf {s['confidence']:.0%}, composite {s['composite']:+.2f})"
            )
    return "\n".join(lines)


def build_weekly_newsletter(engine: TradeBotEngine) -> str:
    """End-of-week summary over the last 7 days (Discord newsletter)."""
    events = engine.journal.since(7)
    trades = [e for e in events if e.get("type") == "trade"]
    signals = [e for e in events if e.get("type") == "signal"]
    equity = [e for e in events if e.get("type") == "equity"]

    lines = ["📰 **TradeBot — weekly newsletter**", ""]

    # Simulated portfolio performance over the week
    if equity:
        start, end = equity[0], equity[-1]
        delta = end["total"] - start["total"]
        pct = (delta / start["total"] * 100) if start["total"] else 0.0
        lines.append(
            f"Simulated equity: ${start['total']:,.2f} → ${end['total']:,.2f} "
            f"({delta:+,.2f}, {pct:+.2f}%)"
        )
    lines += _portfolio_block(engine, engine.real)
    lines.append("")

    buys = [t for t in trades if "BUY" in t.get("action", "")]
    sells = [t for t in trades if "SELL" in t.get("action", "")]
    lines.append(f"Trades this week: {len(buys)} buys, {len(sells)} sells.")
    if trades:
        lines += _trade_lines(trades)
    lines.append("")

    actionable = [s for s in signals if s.get("direction") in ("BUY", "SELL")]
    if actionable:
        best = max(actionable, key=lambda s: s.get("composite", 0))
        worst = min(actionable, key=lambda s: s.get("composite", 0))
        lines.append(
            f"Strongest BUY: {best['symbol']} ({best['composite']:+.2f}) | "
            f"Strongest SELL: {worst['symbol']} ({worst['composite']:+.2f})"
        )
    lines.append(f"\n{len(signals)} signals generated over the week.")
    return "\n".join(lines)
