# Optimal step size for estimating $\Delta_{\textrm{C}}$ in different financial derivatives

## Summary

In this document we will focus on a market where the stock price assumed to follow a log-normal diﬀusion process on the interval $[0,T]$:
\begin{equation}
\textrm{d} S_t = S_t \left( r \textrm{d} t + \sigma \textrm{d} W_t\right),
\label{eq:log_norm_diff}
\end{equation}
and we will discuss the precision of the best Monte-Carlo model (with fix $N=5000$ paths) for two derivatives which behaves very differently. The comparison will be made analytically and for numerically on a concrete parameter set:
\begin{equation}
r = 0.01, S_0 = 1, \sigma = 0.4, T = 0.25\textrm{Y} \, .
\end{equation}

The document can be divided in six main section. The first section defines the payoff for digital call and for call option. In the next section we calculate the main quantities for these option in the analytical framework. The third section pictures that how well can the analytical values of the above quantities approximated for the two derivatives. The section \textit{Bias and Variance} and section \textit{Mean Square Error and optimal} $\epsilon$ help to understand the limit of the numerical precision for a given $N$ and $\epsilon$.