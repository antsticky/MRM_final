import numpy as np

from qlib.payoff import Payoff
from qlib.MonteCarlo import MCCalculator
from qlib.BlackScholes import BSCalculator

from qlib.misc import read_config

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
    # print("\n\nMSE\n---")
    X = []
    Y1 = []
    Y2 = []
    for eps in np.arange(start=0.5, stop=10 + 0.001, step=0.5):
        X.append(eps)

        digital_call_MSE = numerical_calculator.bias(eps=eps, payoff=digital_call_payoff, target=digital_delta_BS)
        # digital_call_MSE = numerical_calculator.var(eps=eps, payoff=digital_call_payoff)
        Y1.append(digital_call_MSE / digital_delta_BS * 100)

        # call_option_MSE = numerical_calculator.bias(eps=eps, payoff=call_option_payoff, target=option_delta_BS)
        # call_option_MSE = numerical_calculator.var(eps=eps, payoff=call_option_payoff)
        # Y2.append(call_option_MSE / option_delta_BS * 100)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ax.set_xlabel("Size of Increment")

    ax.plot(X, Y1, label=f"Digital", marker=".", color="red", linewidth=1)
    #ax2.plot(X, Y2, label=f"Standard Call", color="green", marker=".", linewidth=1)

    ax.set_ylabel("Digital", color="red")
    ax2.set_ylabel("Standard Call", color="green")

    # plt.xlabel("Size of Increment")
    # plt.ylabel(r"MSE / $\delta$ [%]")
    # plt.legend()

    # ax.set_yscale('log')
    # ax.set_xscale('log')

    plt.show()
