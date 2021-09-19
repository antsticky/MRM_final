import numpy as np
import matplotlib.pyplot as plt


class Paths:
    def __init__(self, name, rnd_seed=-1):
        self.name = name
        self._paths = None
        self._df = None

        if rnd_seed != -1:
            np.random.seed(rnd_seed)

    @classmethod
    def european_lognormal(cls, rnd_seed, stock_params, market_params):
        lognormal_cls = cls(name="European Lognormal pricer", rnd_seed=rnd_seed)

        lognormal_cls.S0 = stock_params.S0
        lognormal_cls.r = market_params.r
        lognormal_cls.sigma = stock_params.sigma

        return lognormal_cls

    @property
    def paths(self):
        if self._paths is None:
            raise KeyError("Generate paths first")
        else:
            return self._paths

    @property
    def df(self):
        if self._df is None:
            raise KeyError("Generate paths first")
        else:
            return self._df

    def generate_path(self, **kwargs):
        if self.name == "European Lognormal pricer":
            self.generate_eu_lognormal_paths(T=kwargs["T"], size=kwargs["size"])
        else:
            KeyError("Not implemented yet")

    def generate_eu_lognormal_paths(self, T, size):
        def show_checks():
            double_average = True if len(self.paths.shape) != 1 else False

            EST = self.paths.mean(axis=0).mean() if double_average else self.paths.mean(axis=0)
            VAR = self.paths.var(axis=0).mean() if double_average else self.paths.var(axis=0)
            work1 = np.exp(self.sigma * self.sigma * T) - 1

            print("MC paths\n--------")
            print(f"E[ST] = {EST}")
            print(f"S0/DF = {self.S0 / self.df}")
            print(f"Var[ST] (MC) = {VAR}")
            print(f"Var[ST] (Ex) = {EST*EST*work1}")

        WT = np.random.normal(loc=0.0, scale=np.sqrt(T), size=size)

        work1 = (self.r - self.sigma * self.sigma / 2) * T
        work2 = self.sigma * WT

        self._df = np.exp(-self.r * T)
        self._paths = self.S0 * np.exp(work1 + work2)
        self.tau = T

        # TODO: eliminate this
        debug = False
        if debug == True:
            show_checks()

        return self._paths

    @staticmethod
    def _histogram_plot(paths, title=""):
        a_y, a_x, _ = plt.hist(x=paths, bins="auto", color="#0504aa", alpha=0.7)

        mean = paths.mean()
        plt.axvline(x=mean, color="red", label="mean")

        median = np.median(paths)
        plt.axvline(x=median, color="orange", label="median")

        mode = float(a_x[np.where(a_y == a_y.max())])
        plt.axvline(x=mode, color="yellow", label="mode")

        std = paths.std()
        plt.axvline(x=mean + std / 2, color="green", label="mean + std/2")
        plt.axvline(x=mean - std / 2, color="green", label="mean - std/2")

        plt.xlabel("ST")
        plt.ylabel("Frequency")
        plt.title(f"{title} ST hisztogram")
        plt.legend()
        plt.show()

    @staticmethod
    def show_paths(paths_class):
        if "European" in paths_class.name:
            paths = paths_class.paths.flatten()
            Paths._histogram_plot(paths, title=paths_class.name)
        else:
            raise KeyError("It works only for European paths")


class Pricer(Paths):
    def price(self, payoff):
        return self.df * payoff.F(ST=self.paths, payoff_params=payoff.params).mean(axis=0)


class CDV(Paths):
    def delta(self, eps, payoff):
        if self.name == "European Lognormal pricer":
            return self.calc_eu_lognormal_delta(eps=eps, payoff=payoff)
        else:
            raise KeyError("Not implemented (handle the bump with care)")

    def gamma(self, eps, payoff):
        if self.name == "European Lognormal pricer":
            return self.calc_eu_lognormal_gamma(eps=eps, payoff=payoff)
        else:
            raise KeyError("Not implemented (handle the bump with care)")

    def calc_eu_lognormal_delta(self, eps, payoff):
        def show_checks():
            # TODO:
            # improve the code with double_average check
            # check averaging
            print("\nUP bump\n-------")
            print(f"E[ST]  = {up_bumped_paths.mean(axis = 0).mean()}")
            print(f"Sup/DF = {(self.S0 + eps) / self.df}")
            print(f"Var[ST] (MC) = {up_bumped_paths.var(axis = 0).mean()}")
            sigma_shit = np.exp(self.sigma * self.sigma * self.tau) - 1
            print(f"Var[ST] (Ex) = {np.square((self.S0 + eps) / self.df) * sigma_shit} ")

            print("\nDOWN bump\n---------")
            print(f"E[ST]    = {down_bumped_paths.mean(axis = 0).mean()}")
            print(f"Sdown/DF = {(self.S0 - eps) / self.df}")
            print(f"Var[ST] (MC) = {down_bumped_paths.var(axis = 0).mean()}")
            print(f"Var[ST] (Ex) = {np.square((self.S0 - eps) / self.df) * sigma_shit} ")

        up_bumped_paths = self.paths / self.S0 * (self.S0 + eps)
        up_price = self.df * payoff.F(ST=up_bumped_paths, payoff_params=payoff.params)

        down_bumped_paths = self.paths / self.S0 * (self.S0 - eps)
        down_price = self.df * payoff.F(ST=down_bumped_paths, payoff_params=payoff.params)

        # TODO: eliminate this
        debug = False
        if debug == True:
            show_checks()

        # return (up_price - down_price).mean(axis=0) / (2 * eps)
        return (up_price.mean(axis=0) - down_price.mean(axis=0)) / (2 * eps)

    def calc_eu_lognormal_gamma(self, eps, payoff):
        prices = self.df * payoff.F(ST=self.paths, payoff_params=payoff.params)

        up_bumped_paths = self.paths / self.S0 * (self.S0 + eps)
        up_price = self.df * payoff.F(ST=up_bumped_paths, payoff_params=payoff.params)

        down_bumped_paths = self.paths / self.S0 * (self.S0 - eps)
        down_price = self.df * payoff.F(ST=down_bumped_paths, payoff_params=payoff.params)

        return (up_price.mean(axis=0) + down_price.mean(axis=0) - 2.0*prices.mean() ) / (eps * eps)

    def bias(self, eps, payoff, target):
        delta = self.delta(eps=eps, payoff=payoff)
        return delta.mean() - target

    def var(self, eps, payoff):
        delta = self.delta(eps=eps, payoff=payoff)
        mean = delta.mean()
        return np.square(delta - mean).mean()

    def MSE(self, eps, payoff, target, do_sanity_check=False):
        delta = self.delta(eps=eps, payoff=payoff)
        MSE = np.square(delta - target).mean()

        if do_sanity_check:
            check_value = np.abs(MSE - (self.var(eps=eps, payoff=payoff) + np.square(self.bias(eps=eps, payoff=payoff, target=target))))
            assert check_value < 0.000001, f"Too inaccurate (diff = {check_value})"

        return MSE


class MCCalculator(Pricer, CDV):
    pass
