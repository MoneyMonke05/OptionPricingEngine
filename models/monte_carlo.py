import numpy as np


def monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations, num_steps, option_type='call'):
    """
    Calculates option price using Monte Carlo Simulation.

    Parameters:
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    num_simulations (int): Number of price paths to simulate
    num_steps (int): Number of time steps in each simulation path
    option_type (str): 'call' for call option, 'put' for put option

    Returns:
    float: The calculated option price
    """
    if T <= 0 or num_simulations <= 0 or num_steps <= 0:
        return 0

    dt = T / num_steps

    # Generate random numbers for each step and simulation
    # Using normal distribution for log returns
    z = np.random.standard_normal((num_steps, num_simulations))

    # Calculate stock prices at each step
    # S_t = S_0 * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    daily_returns = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    # Cumulative product to get final stock prices
    price_paths = S * np.cumprod(daily_returns, axis=0)

    # The final prices are the last row of price_paths
    ST = price_paths[-1, :]

    # Calculate payoffs at maturity
    if option_type == 'call':
        payoffs = np.maximum(0, ST - K)
    elif option_type == 'put':
        payoffs = np.maximum(0, K - ST)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount average payoff back to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price