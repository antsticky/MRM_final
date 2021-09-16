import numpy as np
import matplotlib.pyplot as plt

from qlib.payoff import Payoff
from qlib.MonteCarlo import MCCalculator
from qlib.BlackScholes import BSCalculator

from qlib.misc import read_config


def show_fig(X, Y1s, Y2s):
    Y1_bais, Y1_var, Y1_mse = Y1s
    Y2_bais, Y2_var, Y2_mse = Y2s

    fig, axs = plt.subplots(3)

    fig.suptitle("MSE plots")

    # -----------------------------
    axs[0].plot(X, Y1_bais, label=f"Digital Call", marker=".", color="red", linewidth=1)
    axs[0].set_ylabel("Digital Call Bias", color="red")

    axs02 = axs[0].twinx()
    axs02.plot(X, Y2_bais, label=f"Standard Call", color="green", marker=".", linewidth=1)
    axs02.set_ylabel("Standard Call Bias", color="green")

    # -----------------------------
    axs[1].plot(X, Y1_var, label=f"Digital Call", marker=".", color="red", linewidth=1)
    axs[1].set_ylabel("Digital Call Var", color="red")

    axs12 = axs[1].twinx()
    axs12.plot(X, Y2_var, label=f"Standard Call", color="green", marker=".", linewidth=1)
    axs12.set_ylabel("Standard Call Var", color="green")

    # -----------------------------
    axs[2].plot(X, Y1_mse, label=f"Digital Call", marker=".", color="red", linewidth=1)
    axs[2].set_ylabel("Digital Call MSE", color="red")

    axs22 = axs[2].twinx()
    axs22.plot(X, Y2_mse, label=f"Standard Call", color="green", marker=".", linewidth=1)
    axs22.set_ylabel("Standard Call MSE", color="green")

    plt.show()


if __name__ == "__main__":
    run_config = read_config(config_path="config.yml")

    # generate MC paths
    numerical_calculator = MCCalculator.european_lognormal(rnd_seed=run_config.mc_params.rnd_seed, market_params=run_config.market_params, stock_params=run_config.stock_params)
    numerical_calculator.generate_path(T=run_config.option_params.tau, size=(run_config.mc_params.nb_paths, run_config.mc_params.nb_realizations), double_average=True)

    # define payoffs
    digital_call_payoff = Payoff.european_digital(option_type="call", params=run_config.option_params)
    call_option_payoff = Payoff.european_option(option_type="call", params=run_config.option_params)

    # calculate analyitical deltas
    analytical_calculator = BSCalculator(market_params=run_config.market_params, stock_params=run_config.stock_params)
    digital_delta_BS = analytical_calculator.delta(digital_call_payoff)
    option_delta_BS = analytical_calculator.delta(call_option_payoff)

    # calc MSE
    X = []
    Y1_bais = []
    Y1_var = []
    Y1_mse = []

    Y2_bais = []
    Y2_var = []
    Y2_mse = []

    for eps in np.arange(start=0.0001, stop=0.1 + 0.01, step=0.01):
        X.append(eps)

        digital_call_bias = numerical_calculator.bias(eps=eps, payoff=digital_call_payoff, target=digital_delta_BS)
        Y1_bais.append(digital_call_bias)
        digital_call_var = numerical_calculator.var(eps=eps, payoff=digital_call_payoff)
        Y1_var.append(digital_call_var)
        digital_call_mse = numerical_calculator.MSE(eps=eps, payoff=digital_call_payoff, target=digital_delta_BS)
        Y1_mse.append(digital_call_mse)
        # TODO: Normalization

        call_option_bias = numerical_calculator.bias(eps=eps, payoff=call_option_payoff, target=option_delta_BS)
        Y2_bais.append(call_option_bias)
        call_option_var = numerical_calculator.var(eps=eps, payoff=call_option_payoff)
        Y2_var.append(call_option_var)
        call_option_mse = numerical_calculator.MSE(eps=eps, payoff=call_option_payoff, target=option_delta_BS)
        Y2_mse.append(call_option_mse)
        # TODO: Normalization

    show_fig(X, (Y1_bais, Y1_var, Y1_mse), (Y2_bais, Y2_var, Y2_mse))
