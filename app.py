import streamlit as st
import numpy as np
import scipy.stats # Changed from 'from scipy.stats import norm'

# --- Black-Scholes Option Pricing Model ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price.

    Parameters:
    S (float): Current stock price
    K (float): Option strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)
    option_type (str): 'call' for call option, 'put' for put option

    Returns:
    float: The calculated option price
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        # Call option price
        # Changed norm.cdf to scipy.stats.norm.cdf
        price = S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    elif option_type == 'put':
        # Put option price
        # Changed norm.cdf to scipy.stats.norm.cdf
        price = K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

# --- Streamlit Application Layout ---
st.set_page_config(layout="centered", page_title="Black-Scholes Option Pricing")

st.title("💰 Black-Scholes Option Pricing Engine")
st.markdown("""
    This application calculates the theoretical price of European call and put options
    using the Black-Scholes model.
    Adjust the parameters below to see how they affect option prices.
""")

st.markdown("---")

# --- Input Parameters ---
st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    S = st.number_input(
        "Current Stock Price (S)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="The current market price of the underlying asset."
    )
    K = st.number_input(
        "Strike Price (K)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="The price at which the option holder can buy (call) or sell (put) the underlying asset."
    )
    T = st.number_input(
        "Time to Expiration (T, in years)",
        min_value=0.01,
        value=1.0,
        step=0.01,
        format="%.2f",
        help="The remaining time until the option expires, expressed in years (e.g., 0.5 for 6 months)."
    )

with col2:
    r = st.number_input(
        "Risk-Free Rate (r, annual %)",
        min_value=0.0,
        value=5.0,
        step=0.1,
        format="%.2f",
        help="The annual risk-free interest rate (e.g., U.S. Treasury bill rate). Enter as a percentage (e.g., 5 for 5%)."
    ) / 100.0  # Convert percentage to decimal
    sigma = st.number_input(
        "Volatility (σ, annual %)",
        min_value=0.01,
        value=20.0,
        step=0.1,
        format="%.2f",
        help="The annualized standard deviation of the underlying asset's returns. Enter as a percentage (e.g., 20 for 20%)."
    ) / 100.0  # Convert percentage to decimal

st.markdown("---")

# --- Calculation and Results ---
if st.button("Calculate Option Prices"):
    try:
        call_price = black_scholes(S, K, T, r, sigma, option_type='call')
        put_price = black_scholes(S, K, T, r, sigma, option_type='put')

        st.header("Calculated Option Prices")
        st.success(f"**European Call Option Price:** ${call_price:.2f}")
        st.info(f"**European Put Option Price:** ${put_price:.2f}")

        st.markdown("---")
        st.subheader("Interpretation:")
        st.markdown(f"""
            - A **Call Option** gives the holder the right, but not the obligation, to buy the underlying asset.
              Its calculated theoretical value is **${call_price:.2f}**.
            - A **Put Option** gives the holder the right, but not the obligation, to sell the underlying asset.
              Its calculated theoretical value is **${put_price:.2f}**.
        """)

    except ValueError as e:
        st.error(f"Error in calculation: {e}. Please check your inputs.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

