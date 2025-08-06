import numpy as np
import scipy.stats

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price and Greeks.

    Parameters:
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    option_type (str): 'call' for call option, 'put' for put option

    Returns:
    tuple: (price, delta, gamma, vega, theta, rho)
    """
    if T <= 0:
        # Handle cases where T is zero or negative
        # For practical purposes, options with T <= 0 are expired or at expiry
        return 0, 0, 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate price
    if option_type == 'call':
        price = S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Calculate Greeks
    N_prime_d1 = scipy.stats.norm.pdf(d1)

    # Delta
    if option_type == 'call':
        delta = scipy.stats.norm.cdf(d1)
    else: # put
        delta = scipy.stats.norm.cdf(d1) - 1

    # Gamma
    gamma = N_prime_d1 / (S * sigma * np.sqrt(T))

    # Vega (often scaled by 100 for a 1% change in volatility)
    vega = S * N_prime_d1 * np.sqrt(T) / 100

    # Theta (often scaled by 365 for daily decay)
    if option_type == 'call':
        theta = (-S * N_prime_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)) / 365
    else: # put
        theta = (-S * N_prime_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2)) / 365

    # Rho (often scaled by 100 for a 1% change in interest rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * scipy.stats.norm.cdf(d2) / 100
    else: # put
        rho = -K * T * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) / 100

    return price, delta, gamma, vega, theta, rho
