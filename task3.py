import numpy as np

from qlib.payoff import Payoff
from qlib.BlackScholes import BSCalculator

from qlib.misc import read_config

if __name__ == "__main__":
    run_config = read_config(config_path="config.yml")

    # define calculators
    analytical_calculator = BSCalculator(market_params=run_config.market_params, stock_params=run_config.stock_params)

    # European digital
    digital_call_payoff = Payoff.european_digital(option_type="call", params=run_config.option_params)
    digital_bias = analytical_calculator.cdv_bias.delta(digital_call_payoff, eps=run_config.mc_params.eps)
    digital_var = analytical_calculator.cdv_var.delta(digital_call_payoff, eps=run_config.mc_params.eps, N=run_config.mc_params.nb_paths)

    print(f"\n{digital_call_payoff}")
    print(f"\nbias (BS) = {digital_bias}")
    print(f"var (BS)  = {digital_var}")
    print(f"MSE (BS)  = {np.square(digital_bias) + digital_var}")

    # European option
    call_option_payoff = Payoff.european_option(option_type="call", params=run_config.option_params)
    option_bias = analytical_calculator.cdv_bias.delta(call_option_payoff, eps=run_config.mc_params.eps)
    option_var = analytical_calculator.cdv_var.delta(call_option_payoff, eps=run_config.mc_params.eps, N=run_config.mc_params.nb_paths)

    print(f"\n{call_option_payoff}")
    print(f"\nbias (BS) = {option_bias}")
    print(f"var (BS)  = {option_var}")
    print(f"MSE (BS)  = {np.square(option_bias) + option_var}")
