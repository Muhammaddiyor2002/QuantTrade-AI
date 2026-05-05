"""Main Streamlit entrypoint for the QuantTrade AI dashboard."""

from __future__ import annotations

import streamlit as st

from app.config import get_settings


def main() -> None:
    st.set_page_config(
        page_title="QuantTrade AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = get_settings()
    st.title("QuantTrade AI 📈")
    st.caption(
        "AI-powered trading + backtesting platform — vectorized indicators, "
        "rule-based and ML strategies, walk-forward optimization."
    )
    st.markdown(
        """
        **Use the sidebar to navigate between modules:**

        1. **Market Overview** - fetch OHLCV, plot candles, attach indicators.
        2. **Strategy Builder** - pick a rule-based strategy and tune parameters.
        3. **Backtesting** - run a strategy and inspect equity curve + metrics.
        4. **AI Predictions** - train an XGBoost model and see probability signals.
        5. **Portfolio** - multi-strategy / multi-asset comparison.
        6. **Risk Analysis** - stop-loss / take-profit / VaR / CVaR.
        """
    )

    with st.expander("Active configuration"):
        st.json(
            {
                "env": settings.env,
                "exchange": settings.exchange,
                "default_symbol": settings.default_symbol,
                "default_timeframe": settings.default_timeframe,
                "initial_capital": settings.initial_capital,
                "commission_bps": settings.commission_bps,
                "slippage_bps": settings.slippage_bps,
                "risk_per_trade": settings.risk_per_trade,
            }
        )


if __name__ == "__main__":
    main()
