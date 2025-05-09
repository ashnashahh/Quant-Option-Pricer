from __future__ import annotations

import argparse
import time
from math import log, sqrt, exp
from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _validate_option_type(opt_type: str) -> OptionType:
    if opt_type.lower() not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return opt_type.lower()  # type: ignore[return-value]


def d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    d1: float = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2: float = d1 - sigma * sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
) -> float:
    option_type = _validate_option_type(option_type)
    d1, d2 = d1_d2(S, K, r, sigma, T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def monte_carlo_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
    paths: int = 100_000,
    seed: int | None = None,
) -> float:
    option_type = _validate_option_type(option_type)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discounted_payoff = np.exp(-r * T) * payoff
    return float(np.mean(discounted_payoff))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black-Scholes / Monte-Carlo option pricer")
    p.add_argument("--model", choices=["bs", "mc"], default="bs")
    p.add_argument("--S", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--type", choices=["call", "put"], default="call")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()

    if args.model == "bs":
        price = black_scholes_price(args.S, args.K, args.r, args.sigma, args.T, args.type)
    else:
        price = monte_carlo_price(
            args.S,
            args.K,
            args.r,
            args.sigma,
            args.T,
            args.type,
            paths=args.paths,
            seed=args.seed,
        )

    elapsed = time.perf_counter() - start
    print(f"{args.model.upper()} price = {price:.4f}  (computed in {elapsed*1e3:.1f} ms)")


if __name__ == "__main__":
    main()
