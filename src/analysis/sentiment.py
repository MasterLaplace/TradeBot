"""
Ollama Sentiment Analysis Module.

Performs AI-powered semantic analysis of financial news articles
using a locally hosted Ollama model (qwen3 or gemma4).

Features:
- Structured JSON output constrained by Pydantic schema
- Deterministic results (temperature=0)
- Graceful fallback if Ollama is not available
- Extracts: company name, ticker, indicators, risks, sentiment polarity, confidence

Usage:
    analyzer = OllamaSentimentAnalyzer(model="qwen3")
    report = analyzer.analyze(article_text)
"""

from datetime import datetime
from typing import List, Optional
import json
import logging

from pydantic import BaseModel, Field

from ..core.models import SentimentReport, Position

logger = logging.getLogger(__name__)

# Optional dependency
try:
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


# =============================================================================
# PYDANTIC SCHEMA FOR STRUCTURED OUTPUT
# =============================================================================

class SentimentSchema(BaseModel):
    """JSON schema sent to Ollama to constrain output structure."""
    company_name: str = Field(description="Exact name of the identified company")
    ticker: str = Field(description="Stock ticker symbol or ISIN")
    key_indicators: List[str] = Field(
        description="Key financial indicators mentioned: revenue, margin, debt, etc."
    )
    identified_risks: List[str] = Field(
        description="Market, operational, or regulatory risks identified"
    )
    sentiment_polarity: float = Field(
        description="Semantic polarity: -1.0 (extremely bearish) to +1.0 (extremely bullish)"
    )
    confidence: float = Field(
        description="Analysis confidence score between 0.0 and 1.0"
    )
    portfolio_advice: str = Field(
        description="Actionable advice considering current portfolio position (e.g., Take Profit, Average Down, Hold).",
        default="No advice"
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = (
    "You are a quantitative financial analyst for an investment fund. "
    "Examine the provided article to extract fundamental metrics, "
    "operational risks, and evaluate the overall semantic polarity. "
    "If portfolio context is provided by the user, provide actionable advice "
    "(e.g., 'Take Profit', 'Hold', 'Accumulate') considering the current position. "
    "You must respond exclusively as a valid JSON object conforming to the imposed schema. "
    "Be precise and factual. Base your sentiment on concrete data mentioned in the article, "
    "not on speculation. If the article lacks financial data, set confidence to a low value."
)


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================

class OllamaSentimentAnalyzer:
    """
    Analyze financial news sentiment using a local Ollama model.

    The analyzer sends structured prompts and constrains the output
    via a JSON schema to ensure deterministic, parseable results.
    """

    def __init__(
        self,
        model: str = "qwen3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def analyze(
        self,
        article_text: str,
        source_url: str = "",
        current_position: Optional[Position] = None,
    ) -> SentimentReport:
        """Analyze a single article and return a structured SentimentReport.

        Args:
            article_text: Full text or summary of the financial article.
            source_url: URL of the source article (for traceability).
            current_position: Optional current portfolio holding context.

        Returns:
            SentimentReport with extracted sentiment and financial data.
            Returns a neutral report if Ollama is unavailable or analysis fails.
        """
        if not HAS_OLLAMA:
            logger.warning("Ollama not installed. Returning neutral sentiment.")
            return self._neutral_report(source_url)

        if not article_text or len(article_text.strip()) < 20:
            logger.debug("Article text too short for meaningful analysis")
            return self._neutral_report(source_url)

        try:
            # Build user context
            user_prompt = article_text
            if current_position:
                user_prompt = (
                    f"PORTFOLIO CONTEXT: You currently hold {current_position.quantity} shares "
                    f"of {current_position.symbol} at an average entry price of ${current_position.average_entry_price:.2f}. "
                    f"Consider this context when generating the portfolio_advice.\n\n"
                    f"ARTICLE TEXT:\n{article_text}"
                )

            response = ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format=SentimentSchema.model_json_schema(),
                options={"temperature": self.temperature},
            )

            content = response["message"]["content"]
            parsed = SentimentSchema.model_validate_json(content)

            return SentimentReport(
                company_name=parsed.company_name,
                ticker=parsed.ticker,
                key_indicators=parsed.key_indicators,
                identified_risks=parsed.identified_risks,
                sentiment_polarity=max(-1.0, min(1.0, parsed.sentiment_polarity)),
                confidence=max(0.0, min(1.0, parsed.confidence)),
                portfolio_advice=parsed.portfolio_advice,
                source_url=source_url,
                analyzed_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Ollama sentiment analysis failed: {e}")
            return self._neutral_report(source_url)

    def analyze_batch(
        self,
        articles: List[dict],
        current_position: Optional[Position] = None,
    ) -> List[SentimentReport]:
        """Analyze multiple articles sequentially.

        Args:
            articles: List of dicts with 'text' and optional 'url' keys.
            current_position: Optional current portfolio holding context.

        Returns:
            List of SentimentReport objects.
        """
        reports = []
        for article in articles:
            text = article.get("text", "")
            url = article.get("url", "")
            report = self.analyze(text, source_url=url, current_position=current_position)
            reports.append(report)
        return reports

    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is loaded."""
        if not HAS_OLLAMA:
            return False
        try:
            models = ollama_client.list()
            model_names = [m.get("name", "").split(":")[0] for m in models.get("models", [])]
            return self.model in model_names
        except Exception:
            return False

    @staticmethod
    def _neutral_report(source_url: str = "") -> SentimentReport:
        """Return a neutral sentiment report (fallback)."""
        return SentimentReport(
            company_name="Unknown",
            ticker="",
            key_indicators=[],
            identified_risks=[],
            sentiment_polarity=0.0,
            confidence=0.0,
            source_url=source_url,
            analyzed_at=datetime.now(),
        )
