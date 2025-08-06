import numpy as np

def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    """
    Calculates option price using the Binomial (Cox-Ross-Rubinstein) Model.

    Parameters:
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    N (int): Number of steps in the binomial tree
    option_type (str): 'call' for call option, 'put' for put option

    Returns:
    float: The calculated option price
    """
    if T <= 0 or N <= 0:
        return 0

    dt = T / N # Time step
    u = np.exp(sigma * np.sqrt(dt)) # Up factor
    d = 1 / u # Down factor
    p = (np.exp(r * dt) - d) / (u - d) # Risk-neutral probability

    # Initialize asset prices at maturity (N steps)
    ST = np.zeros(N + 1)
    for j in range(N + 1):
        ST[j] = S * (u**j) * (d**(N - j))

    # Calculate option values at maturity
    option_values = np.zeros(N + 1)
    for j in range(N + 1):
        if option_type == 'call':
            option_values[j] = max(0, ST[j] - K)
        elif option_type == 'put':
            option_values[j] = max(0, K - ST[j])

    # Work backwards through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
            # For American options, add early exercise check here:
            # if option_type == 'call':
            #     option_values[j] = max(option_values[j], S * (u**j) * (d**(i - j)))
            # elif option_type == 'put':
            #     option_values[j] = max(option_values[j], K - S * (u**j) * (d**(i - j)))

    return option_values[0]
