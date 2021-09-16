import numpy as np
from scipy.stats import norm


class Base:
    def __init__(self, market_params, stock_params):
        self.r = market_params.r

        self.S0 = stock_params.S0
        self.sigma = stock_params.sigma

    def d1(self, K, tau):
        return (np.log(self.S0 / K) + (0.5 * self.sigma * self.sigma + self.r) * tau) / (self.sigma * np.sqrt(tau))

    def d2(self, K, tau):
        return self.d1(K=K, tau=tau) - self.sigma * np.sqrt(tau)

    def df(self, tau):
        return np.exp(-self.r * tau)


class Pricer(Base):
    def price(self, payoff):
        if payoff.kind.lower() == "option":
            price = self.option_price(payoff)
            return price
        elif payoff.kind.lower() == "digital":
            price = self.digital_price(payoff)
            return price
        else:
            raise KeyError("Unknown option kind")

    def option_price(self, payoff):
        K = payoff.params.K
        tau = payoff.params.tau

        if payoff.type.lower() == "call":
            price = self.S0 * norm.cdf(self.d1(K=K, tau=tau)) - K * self.df(tau=tau) * norm.cdf(self.d2(K=K, tau=tau))
            return price
        elif payoff.type.lower() == "put":
            price = K * self.df(tau=tau) * norm.cdf(-self.d2(K=K, tau=tau)) - self.S0 * norm.cdf(-self.d1(K=K, tau=tau))
            return price
        else:
            raise KeyError("Unknown option type")

    def digital_price(self, payoff):
        K = payoff.params.K
        tau = payoff.params.tau

        if payoff.type.lower() == "call":
            price = self.df(tau) * norm.cdf(self.d2(K=K, tau=tau))
            return price
        elif payoff.type.lower() == "put":
            price = self.df(tau) * (1 - norm.cdf(self.d2(K=K, tau=tau)))
            return price
        else:
            raise KeyError("Unknown option type")


class Delta(Base):
    def delta(self, payoff):
        if payoff.kind.lower() == "option":
            delta = self.option_delta(payoff)
            return delta
        elif payoff.kind.lower() == "digital":
            delta = self.digital_delta(payoff)
            return delta
        else:
            raise KeyError("Unknown option kind")

    def option_delta(self, payoff):
        K = payoff.params.K
        tau = payoff.params.tau

        if payoff.type.lower() == "call":
            delta = norm.cdf(self.d1(K=K, tau=tau))
            return delta
        elif payoff.type.lower() == "put":
            delta = norm.cdf(self.d1(K=K, tau=tau)) - 1
            return delta
        else:
            raise KeyError("Unknown option value")

    def digital_delta(self, payoff):
        K = payoff.params.K
        tau = payoff.params.tau

        delta = self.df(tau=tau) * norm.pdf(self.d2(K=K, tau=tau)) / (K * self.sigma * np.sqrt(tau))

        if payoff.type.lower() == "call":
            return delta
        elif payoff.type.lower() == "put":
            return -delta
        else:
            raise KeyError("Unknown option value")


class BSCalculator(Pricer, Delta):
    pass
