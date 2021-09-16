from qlib.payoff import Payoff
from qlib.MonteCarlo import MCCalculator
from qlib.BlackScholes import BSCalculator

from qlib.misc import read_config

if __name__ == "__main__":
    run_config = read_config(config_path="config.yml")

    # define calculators
    analytical_calculator = BSCalculator(market_params=run_config.market_params, stock_params=run_config.stock_params)

    numerical_calculator = MCCalculator.european_lognormal(rnd_seed=run_config.mc_params.rnd_seed, market_params=run_config.market_params, stock_params=run_config.stock_params)
    numerical_calculator.generate_path(T=run_config.option_params.tau, size=(run_config.mc_params.nb_paths, run_config.mc_params.nb_realizations), double_average=True)

    # European digital
    digital_call_payoff = Payoff.european_digital(option_type="call", params=run_config.option_params)
    digital_delta_BS = analytical_calculator.delta(digital_call_payoff)
    digital_call_MSE = numerical_calculator.MSE(eps=run_config.mc_params.eps, payoff=digital_call_payoff, target=digital_delta_BS, do_sanity_check=True)

    print(f"\n\n{digital_call_payoff}")
    print(f"\nMSE = {digital_call_MSE}")

    # European option
    call_option_payoff = Payoff.european_option(option_type="call", params=run_config.option_params)
    option_delta_BS = analytical_calculator.delta(call_option_payoff)
    call_option_MSE = numerical_calculator.MSE(eps=run_config.mc_params.eps, payoff=call_option_payoff, target=option_delta_BS, do_sanity_check=True)

    print(f"\n{call_option_payoff}")
    print(f"\nMSE = {call_option_MSE}")
