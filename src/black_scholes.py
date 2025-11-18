import math
from scipy.stats import norm

def bs_call_price(S, K, T, r, sigma):
    if T==0: return max(S-K, 0)
    d1 = (math.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    if T==0: return max(K-S, 0)
    d1 = (math.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type):
    if T==0: return 0,0,0,0,0
    d1 = (math.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    delta = norm.cdf(d1) if option_type=='call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*math.sqrt(T))
    vega = S*norm.pdf(d1)*math.sqrt(T)
    theta = -(S*norm.pdf(d1)*sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*(norm.cdf(d2) if option_type=='call' else norm.cdf(-d2))
    rho = K*T*math.exp(-r*T)*(norm.cdf(d2) if option_type=='call' else -norm.cdf(-d2))
    return delta, gamma, vega, theta, rho
