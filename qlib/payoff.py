import numbers

import numpy as np
import matplotlib.pyplot as plt


class Payoff:
    def __init__(self, payoff_name, payoff_type, kind, payoff_function, params):
        self.name = payoff_name
        self.type = payoff_type
        self.kind = kind
        self.F = payoff_function
        self.params = params

    def __str__(self):
        ret_str = f"{self.name}\n"
        ret_str += "-" * len(self.name)

        for key, value in self.params.items():
            ret_str += f"\n{key}: {value}"

        return ret_str

    @classmethod
    def european_option(cls, option_type, params):
        def put(ST, payoff_params):
            K = payoff_params.K
            if isinstance(ST, numbers.Number):
                return np.max([K - ST, 0])
            else:
                return np.where(ST > K, 0, K - ST)

        def call(ST, payoff_params):
            K = payoff_params.K
            if isinstance(ST, numbers.Number):
                return np.max([ST - K, 0])
            else:
                return np.where(ST > K, ST - K, 0)

        F = put if option_type.lower() == "put" else call

        option_cls = cls(payoff_name=f"European {option_type} option", payoff_type=option_type, kind="option", payoff_function=F, params=params)

        return option_cls

    @classmethod
    def european_digital(cls, option_type, params):
        def put(ST, payoff_params):
            K = payoff_params.K
            if isinstance(ST, numbers.Number):
                return 1 if K - ST > 0 else 0
            else:
                return np.where(ST > K, 0, 1)

        def call(ST, payoff_params):
            K = payoff_params.K
            if isinstance(ST, numbers.Number):
                return 1 if ST - K > 0 else 0
            else:
                return np.where(ST > K, 1, 0)

        F = put if option_type.lower() == "put" else call

        option_cls = cls(payoff_name=f"European digital {option_type}", payoff_type=option_type, kind="digital", payoff_function=F, params=params)

        return option_cls

    @staticmethod
    def show(payoff, start_ST=0, stop_ST=2, nb_steps=100):
        ST = np.arange(start=start_ST, stop=stop_ST, step=(stop_ST - start_ST) / nb_steps)
        FT = payoff.F(ST, payoff.params)

        plt.plot(ST, FT, "-.")
        plt.title(payoff.name)
        plt.xlabel("ST")
        plt.ylabel("FT")
        plt.show()
