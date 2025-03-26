from dataclasses import dataclass



@dataclass
class Asset:
    returns: float
    volatility: float
    ticker: str | None = None