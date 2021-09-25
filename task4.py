import numpy as np
import matplotlib.pyplot as plt

from qlib.payoff import Payoff
from qlib.BlackScholes import BSCalculator

from qlib.misc import read_config


def show_fig(X, Y1s, Y2s):
    Y1_bias, Y1_var, Y1_mse = Y1s
    Y2_bias, Y2_var, Y2_mse = Y2s

    fig, axs = plt.subplots(3)

    fig.suptitle("MSE plots")

    # -----------------------------
    axs[0].plot(X, Y1_bias, label=f"Digital Call", marker=".", color="red", linewidth=1)
    axs[0].set_ylabel("Digital Call Bias", color="red")

    axs02 = axs[0].twinx()
    axs02.plot(X, Y2_bias, label=f"Standard Call", color="green", marker=".", linewidth=1)
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

    # define payoffs
    digital_call_payoff = Payoff.european_digital(option_type="call", params=run_config.option_params)
    call_option_payoff = Payoff.european_option(option_type="call", params=run_config.option_params)

    # calculate analyitical deltas
    analytical_calculator = BSCalculator(market_params=run_config.market_params, stock_params=run_config.stock_params)
    digital_delta_BS = analytical_calculator.delta(digital_call_payoff)
    option_delta_BS = analytical_calculator.delta(call_option_payoff)

    # calc MSE
    X = []
    Y1_bias = []
    Y1_var = []
    Y1_mse = []

    Y2_bias = []
    Y2_var = []
    Y2_mse = []

    step_size = 0.01
    for eps in np.arange(start=0.01, stop=0.25 + step_size, step=step_size):
        X.append(eps)

        digital_bias = analytical_calculator.cdv_bias.delta(digital_call_payoff, eps=eps)
        digital_var = analytical_calculator.cdv_var.delta(digital_call_payoff, eps=eps, N=run_config.mc_params.nb_paths)
        digital_MSE = np.square(digital_bias) + digital_var

        Y1_bias.append(digital_bias)
        Y1_var.append(digital_var)
        Y1_mse.append(digital_MSE / digital_delta_BS)

        option_bias = analytical_calculator.cdv_bias.delta(call_option_payoff, eps=eps)
        option_var = analytical_calculator.cdv_var.delta(call_option_payoff, eps=eps, N=run_config.mc_params.nb_paths)
        option_MSE = np.square(option_bias) + option_var

        Y2_bias.append(option_bias)
        Y2_var.append(option_var)
        Y2_mse.append(option_MSE / option_delta_BS)

    if run_config.display.show_fig:
        show_fig(X, (Y1_bias, Y1_var, Y1_mse), (Y2_bias, Y2_var, Y2_mse))
