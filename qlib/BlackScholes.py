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

        delta = self.df(tau=tau) * norm.pdf(self.d2(K=K, tau=tau)) / (self.S0 * self.sigma * np.sqrt(tau))

        if payoff.type.lower() == "call":
            return delta
        elif payoff.type.lower() == "put":
            return -delta
        else:
            raise KeyError("Unknown option value")


class Gamma(Base):
    def gamma(self, payoff):
        if payoff.kind.lower() == "option":
            gamma = self.option_gamma(payoff)
            return gamma
        elif payoff.kind.lower() == "digital":
            gamma = self.digital_gamma(payoff)
            return gamma
        else:
            raise KeyError("Unknown option kind")

    def option_gamma(self, payoff):
        K = payoff.params.K
        tau = payoff.params.tau

        if payoff.type.lower() in ["call", "put"]:
            gamma = norm.pdf(self.d1(K=K, tau=tau)) / (self.S0 * self.sigma * np.sqrt(tau))
            return gamma
        else:
            raise KeyError("Unknown option value")

    def digital_gamma(self, payoff):
        K = payoff.params.K
        tau = payoff.params.tau

        gamma = -self.df(tau=tau) * self.d1(K=K, tau=tau) * norm.pdf(self.d2(K=K, tau=tau)) / (np.square(self.S0 * self.sigma) * tau)

        if payoff.type.lower() == "call":
            return gamma
        elif payoff.type.lower() == "put":
            return -gamma
        else:
            raise KeyError("Unknown option value")


class CDVBias(Base):
    def delta(self, payoff, eps):
        if payoff.kind.lower() == "option":
            gamma = self.delta_option(payoff, eps)
            return gamma
        elif payoff.kind.lower() == "digital":
            gamma = self.delta_digital(payoff, eps)
            return gamma
        else:
            raise KeyError("Unknown option kind")

    def delta_option(self, payoff, eps):
        K = payoff.params.K
        tau = payoff.params.tau

        if payoff.type.lower() == "call":
            work1 = norm.pdf(self.d1(K=K, tau=tau)) / (self.S0 * self.S0 * self.sigma * np.sqrt(tau))
            work2 = 1 + self.d1(K=K, tau=tau) / (self.sigma * np.sqrt(tau))
            bias = -(1.0 / 6.0) * work1 * work2 * np.square(eps)
            return bias
        else:
            raise KeyError("Unknown option value")

    def delta_digital(self, payoff, eps):
        K = payoff.params.K
        tau = payoff.params.tau

        if payoff.type.lower() == "call":
            d1 = self.d1(K=K, tau=tau)
            d2 = self.d2(K=K, tau=tau)
            s_sq_tau = self.sigma * tau

            work1 = d2 / (pow(self.S0, 3) * np.square(self.sigma) * tau)

            work21 = 2.0 * d1
            work22 = -1.0 / s_sq_tau
            work23 = d1 * d2 / s_sq_tau

            bias = (1.0 / 6.0) * self.df(tau=tau) * work1 * (work21 + work22 + work23) * np.square(eps)

            return bias
        else:
            raise KeyError("Unknown option value")


class BSCalculator(Pricer, Delta, Gamma):
    def __init__(self, market_params, stock_params):
        super().__init__(market_params, stock_params)
        self.cdv_bias = CDVBias(market_params, stock_params)
