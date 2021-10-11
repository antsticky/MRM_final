# Quantitative Finance Library

This library is well applicable on a market where the stock price assumed to follow a log-normal diﬀusion process on the interval [0,T]:

<p align=center>
<img src="https://render.githubusercontent.com/render/math?math=S_t = S_t ( r dt %2B  \sigma d W_t ) .">
 </p>

This approximation of the real world let us to use the Black-Scholes formulas for analytical calculations. But, one can also use numerical methods, e.g. Monte-Carlo with central estimator.

In this case, both, the exact and the numerical method can be compared by a direct calculation as well where the inaccruate of the MC is eye-catching in the case of the European Digital Call, but it gives a good proxy for European Call Option.

For the full theoretical background see [README.pdf](README.pdf) and for a short demo see `examples*.py`.
