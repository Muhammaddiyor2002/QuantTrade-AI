"""Generate synthetic OHLCV files for offline demos."""

from __future__ import annotations

from app.data.storage import save_ohlcv
from app.data.synthetic import generate_ohlcv
from app.utils.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def main() -> None:
    pairs = [
        ("BTC/USDT", "1h", 2_000, 30_000.0),
        ("ETH/USDT", "1h", 2_000, 1_800.0),
        ("SOL/USDT", "1h", 2_000, 25.0),
    ]
    for symbol, tf, n, base in pairs:
        df = generate_ohlcv(n=n, timeframe=tf, initial_price=base, seed=hash(symbol) % 2**31)
        path = save_ohlcv(df, symbol, tf)
        log.info("Wrote %s (%d rows)", path, len(df))


if __name__ == "__main__":
    main()
