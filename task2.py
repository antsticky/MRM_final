from qlib.payoff import Payoff
from qlib.MonteCarlo import MCCalculator
from qlib.BlackScholes import BSCalculator

from qlib.misc import read_config

if __name__ == "__main__":
    run_config = read_config(config_path="config.yml")

    # Define calculators
    analytical_calculator = BSCalculator(market_params=run_config.market_params, stock_params=run_config.stock_params)

    numerical_calculator = MCCalculator.european_lognormal(rnd_seed=run_config.mc_params.rnd_seed, market_params=run_config.market_params, stock_params=run_config.stock_params)
    numerical_calculator.generate_path(T=run_config.option_params.tau, size=run_config.mc_params.nb_paths)
    MCCalculator.show_paths(numerical_calculator, show_fig=run_config.display.show_fig)

    # European digital
    digital_call_payoff = Payoff.european_digital(option_type="call", params=run_config.option_params)

    digital_price_BS = analytical_calculator.price(digital_call_payoff)  # 0.46397559
    digital_delta_BS = analytical_calculator.delta(digital_call_payoff)  # 1.98212847
    digital_gamma_BS = analytical_calculator.gamma(digital_call_payoff)  # - 1.11495

    digital_price_MC = numerical_calculator.price(digital_call_payoff)
    digital_delta_MC = numerical_calculator.delta(payoff=digital_call_payoff, eps=run_config.mc_params.eps)
    digital_gamma_MC = numerical_calculator.gamma(payoff=digital_call_payoff, eps=run_config.mc_params.eps)

    option_bias = analytical_calculator.cdv_bias.delta(digital_call_payoff, eps=run_config.mc_params.eps)

    print(f"\n{digital_call_payoff}")

    print(f"\nprice (BS) = {digital_price_BS}")
    print(f"price (MC) = {digital_price_MC}")

    print(f"\ndelta (BS) = {digital_delta_BS}")
    print(f"delta (MC) = {digital_delta_MC}")

    print(f"\ngamma (BS) = {digital_gamma_BS}")
    print(f"gamma (MC) = {digital_gamma_MC}")

    print(f"\nbias (BS) = {option_bias}")
    print(f"bias (MC) = {digital_delta_MC - digital_delta_BS}")

    # European option
    call_option_payoff = Payoff.european_option(option_type="call", params=run_config.option_params)

    option_price_BS = analytical_calculator.price(call_option_payoff)  # 0.08081098
    option_delta_BS = analytical_calculator.delta(call_option_payoff)  # 0.54478657
    option_gamma_BS = analytical_calculator.gamma(call_option_payoff)  # 1.98213

    option_price_MC = numerical_calculator.price(call_option_payoff)
    option_delta_MC = numerical_calculator.delta(payoff=call_option_payoff, eps=run_config.mc_params.eps)
    option_gamma_MC = numerical_calculator.gamma(payoff=call_option_payoff, eps=run_config.mc_params.eps)

    option_bias = analytical_calculator.cdv_bias.delta(call_option_payoff, eps=run_config.mc_params.eps)

    print(f"\n\n{call_option_payoff}")
    print(f"\nprice (BS) = {option_price_BS}")
    print(f"price (MC) = {option_price_MC}")
    print(f"\ndelta (BS) = {option_delta_BS}")
    print(f"delta (MC) = {option_delta_MC}")

    print(f"\ngamma (BS) = {option_gamma_BS}")
    print(f"gamma (MC) = {option_gamma_MC}")

    print(f"\nbias (BS) = {option_bias}")
    print(f"bias (MC) = {option_delta_MC - option_delta_BS}")
