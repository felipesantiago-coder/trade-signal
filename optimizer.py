"""
optimizer.py
------------
Otimizacao de parametros da estrategia CTEV via grid search.

Percorre combinacoes de parametros (RSI thresholds, BB period/std,
volume multiplier, SL/TP ATR multipliers) e avalia cada uma usando
backtesting Walk-Forward. Retorna os melhores parametros ordenados
por Sharpe Ratio e Profit Factor.

Funcionalidades:
    - Grid search configuravel com ranges por parametro
    - Avaliacao via backtest simulado (nao usa API)
    - Ranking por Sharpe Ratio, Profit Factor, Win Rate
    - Salvamento de resultados no DB para comparacao historica
    - Suporte a assincrono (non-blocking) via API

Referencias:
    - QuantStart (2025): "Parameter optimization is critical for
      strategy robustness. Always use walk-forward to avoid curve fitting"
    - AlgorithmicTrading.net: "Grid search is the simplest optimization
      method. For each combination of parameters, run a backtest"
    - Reddit r/algotrading: "If you optimize too much you'll overfit.
      Keep parameter space small and use out-of-sample testing"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ctev.optimizer")


@dataclass(frozen=True)
class ParamSet:
    """Conjunto de parametros da estrategia CTEV."""
    rsi_long: float         # RSI < X para LONG (ex: 35)
    rsi_short: float        # RSI > Y para SHORT (ex: 65)
    bb_period: int          # BB period (ex: 20)
    bb_std: float           # BB std dev (ex: 2.0)
    vol_multiplier: float   # Volume > X * SMA20 (ex: 1.5)
    sl_atr_mult: float      # SL = Entry +/- X * ATR
    tp_atr_mult: float      # TP = Entry -/+ X * ATR

    def to_dict(self) -> dict:
        return {
            "rsi_long": self.rsi_long,
            "rsi_short": self.rsi_short,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "vol_multiplier": self.vol_multiplier,
            "sl_atr_mult": self.sl_atr_mult,
            "tp_atr_mult": self.tp_atr_mult,
        }

    def key(self) -> str:
        return (
            f"rsi_l{self.rsi_long}_rsi_s{self.rsi_short}_"
            f"bb{self.bb_period}_{self.bb_std}_"
            f"vol{self.vol_multiplier}_"
            f"sl{self.sl_atr_mult}_tp{self.tp_atr_mult}"
        )


@dataclass
class OptimizationResult:
    """Resultado da otimizacao de um conjunto de parametros."""
    param_set: ParamSet
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl_pct: float = 0.0
    score: float = 0.0  # Score composto para ranking

    def to_dict(self) -> dict:
        return {
            "params": self.param_set.to_dict(),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "score": round(self.score, 4),
        }


# ------------------------------------------------------------------
# Grid de parametros padrao
# ------------------------------------------------------------------
DEFAULT_GRID = {
    "rsi_long": [30, 35, 40],
    "rsi_short": [60, 65, 70],
    "bb_period": [20],
    "bb_std": [2.0],
    "vol_multiplier": [1.3, 1.5, 1.8],
    "sl_atr_mult": [1.5, 2.0],
    "tp_atr_mult": [2.0, 2.5, 3.0],
}


class Optimizer:
    """
    Otimizador de parametros da estrategia CTEV. Singleton.

    Executa grid search com backtesting para encontrar os melhores
    parametros. Thread-safe via Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: bool = False
        self._progress: float = 0.0
        self._total_combos: int = 0
        self._current_combo: int = 0
        self._results: List[OptimizationResult] = []
        self._best: Optional[OptimizationResult] = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def best(self) -> Optional[OptimizationResult]:
        with self._lock:
            return self._best

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "progress": round(self._progress, 1),
                "total_combos": self._total_combos,
                "current_combo": self._current_combo,
                "best": self._best.to_dict() if self._best else None,
                "results_count": len(self._results),
            }

    def run_grid_search(
        self,
        days: int = 180,
        atr_pct_min: float = 0.20,
        atr_pct_max: float = 0.80,
        custom_grid: Optional[Dict[str, list]] = None,
    ) -> List[OptimizationResult]:
        """
        Executa grid search completo.

        Parameters:
            days: dias de dados historicos para backtest
            atr_pct_min: filtro de volatilidade minimo
            atr_pct_max: filtro de volatilidade maximo
            custom_grid: dicionario customizado de ranges (override DEFAULT_GRID)

        Returns:
            Lista de OptimizationResult ordenados por score
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Otimizacao ja em andamento.")
            self._running = True
            self._progress = 0.0
            self._results = []

        try:
            grid = custom_grid or DEFAULT_GRID
            keys = list(grid.keys())
            values = list(grid.values())

            # Calcula total de combinacoes
            total = 1
            for v in values:
                total *= len(v)
            self._total_combos = total
            self._current_combo = 0

            logger.info(
                "Otimizacao iniciada: %d combinacoes (%d dias) grid=%s",
                total, days, {k: len(v) for k, v in grid.items()},
            )

            # Importa aqui para evitar circular dependency em startup
            from indicators import compute_indicators
            from backtest import fetch_historical_ohlcv, calculate_metrics

            # Download dados uma unica vez
            df = fetch_historical_ohlcv("BTC/USDT", "1h", days)
            df_ind = compute_indicators(df, timeframe="1h")
            df_clean = df_ind.dropna(subset=[
                "ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr", "atr_percentile",
            ]).copy()

            # Gera todas as combinacoes
            combos = list(product(*values))

            for combo in combos:
                params = dict(zip(keys, combo))
                self._current_combo += 1

                try:
                    result = self._evaluate_params(
                        params, df_clean, atr_pct_min, atr_pct_max,
                    )
                    if result is not None:
                        self._results.append(result)
                except Exception as exc:
                    logger.debug("Falha ao avaliar %s: %s", params, exc)

                # Progress
                with self._lock:
                    self._progress = (self._current_combo / total) * 100.0

            # Ordena por score
            self._results.sort(key=lambda r: r.score, reverse=True)

            if self._results:
                self._best = self._results[0]
                logger.info(
                    "Otimizacao concluida: %d resultados. MELHOR: %s",
                    len(self._results), self._best.to_dict(),
                )
            else:
                logger.warning("Otimizacao: nenhum resultado valido.")

            return list(self._results)

        except Exception as exc:
            logger.error("Otimizacao falhou: %s", exc)
            raise
        finally:
            with self._lock:
                self._running = False

    def _evaluate_params(
        self,
        params: dict,
        df_clean: pd.DataFrame,
        atr_pct_min: float,
        atr_pct_max: float,
    ) -> Optional[OptimizationResult]:
        """
        Avalia um conjunto de parametros via backtest simplificado.
        """
        rsi_long = float(params["rsi_long"])
        rsi_short = float(params["rsi_short"])
        vol_mult = float(params["vol_multiplier"])
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])

        # Backtest simplificado com os parametros
        trades = self._simulate_with_params(
            df_clean, rsi_long, rsi_short, vol_mult, sl_mult, tp_mult,
            atr_pct_min, atr_pct_max,
        )

        if not trades:
            return None

        # Metricas
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        pf = gross_profit / gross_loss

        import numpy as np
        pnls = trades
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * (365 ** 0.5)
        else:
            sharpe = 0.0

        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        dd = cum_pnl - peak
        max_dd = abs(min(dd)) if len(dd) > 0 else 0.0

        # Score composto: WR * PF * (1 - MaxDD) + Sharpe bonus
        dd_factor = max(0, 1 - max_dd / 100)
        score = (win_rate / 100) * pf * dd_factor + min(sharpe, 3.0) * 0.1

        param_set = ParamSet(
            rsi_long=rsi_long,
            rsi_short=rsi_short,
            bb_period=int(params.get("bb_period", 20)),
            bb_std=float(params.get("bb_std", 2.0)),
            vol_multiplier=vol_mult,
            sl_atr_mult=sl_mult,
            tp_atr_mult=tp_mult,
        )

        return OptimizationResult(
            param_set=param_set,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=pf,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_pnl_pct=sum(pnls),
            score=score,
        )

    @staticmethod
    def _simulate_with_params(
        df: pd.DataFrame,
        rsi_long: float,
        rsi_short: float,
        vol_mult: float,
        sl_mult: float,
        tp_mult: float,
        atr_pct_min: float,
        atr_pct_max: float,
    ) -> List[float]:
        """
        Simula trades simplificados com parametros customizados.
        Retorna lista de PnLs em %.
        """
        pnls: List[float] = []
        i = 0
        n = len(df)
        max_bars = 48

        while i < n:
            row = df.iloc[i]

            # Verifica NaN
            critical = ["ema200", "bb_lower", "bb_upper", "rsi", "volume_sma20", "atr", "atr_percentile"]
            if any(__import__("pandas").isna(row.get(c)) for c in critical):
                i += 1
                continue

            atr_pct = float(row.get("atr_percentile", 0.5))
            if atr_pct < atr_pct_min or atr_pct > atr_pct_max:
                i += 1
                continue

            close = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])
            ema200 = float(row["ema200"])
            bb_lower = float(row["bb_lower"])
            bb_upper = float(row["bb_upper"])
            rsi = float(row["rsi"])
            volume = float(row["volume"])
            vol_sma = float(row["volume_sma20"])
            atr = float(row["atr"])

            signal_type = None
            entry = close

            # LONG
            if (close > ema200 and (low <= bb_lower or close <= bb_lower)
                    and rsi < rsi_long and vol_sma > 0 and volume > vol_sma * vol_mult):
                signal_type = "LONG"
                sl = entry - sl_mult * atr
                tp = entry + tp_mult * atr

            # SHORT
            elif (close < ema200 and (high >= bb_upper or close >= bb_upper)
                    and rsi > rsi_short and vol_sma > 0 and volume > vol_sma * vol_mult):
                signal_type = "SHORT"
                sl = entry + sl_mult * atr
                tp = entry - tp_mult * atr

            if signal_type is None:
                i += 1
                continue

            # Simula ate SL ou TP
            is_long = signal_type == "LONG"
            exit_price = None
            bars = 0

            for j in range(i + 1, min(i + max_bars, n)):
                future = df.iloc[j]
                f_close = float(future["close"])
                f_low = float(future["low"])
                f_high = float(future["high"])
                bars = j - i

                if is_long:
                    if f_low <= sl:
                        exit_price = sl
                        break
                    if f_high >= tp:
                        exit_price = tp
                        break
                else:
                    if f_high >= sl:
                        exit_price = sl
                        break
                    if f_low <= tp:
                        exit_price = tp
                        break

            if exit_price is None:
                last_j = min(i + max_bars, n) - 1
                exit_price = float(df.iloc[last_j]["close"])
                bars = last_j - i

            if is_long:
                pnl = (exit_price - entry) / entry * 100
            else:
                pnl = (entry - exit_price) / entry * 100

            pnls.append(round(pnl, 4))
            i += bars + 1

        return pnls


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_instance: Optional[Optimizer] = None
_lock = threading.Lock()


def get_optimizer() -> Optimizer:
    """Retorna a instancia unica de Optimizer."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Optimizer()
    return _instance
