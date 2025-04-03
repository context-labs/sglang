prior = 0.01
false_pos_rate = 0.05

# assumes that spoofers always spoof and genuine operators never spoof
from math import comb

from scipy import integrate
from scipy.stats import beta


def compute_posterior_1(flagged, total_responses, false_pos_rate, prior):
    """
    Computes the posterior probability P(genuine | n) given:
    - n: Number of responses labeled as spoofs
    - N: Total number of responses
    - r: False positive rate of the detector
    - pi_g: Prior probability that an operator is genuine
    """
    n, N, r, p = flagged, total_responses, false_pos_rate, prior

    # PMF(n) of Binomial(r,N)
    P_n_genuine = comb(N, n) * (r**n) * (1 - r) ** (N - n)

    # Prior probability of being genuine
    P_genuine = 1 - p
    # Prior probability of being spoofer
    P_spoofer = p

    # Detector has FNR of 0, so if n < N then it's def not a spoofer and vice versa
    P_n_spoofer = 1 if n == N else 0

    # Law of total probability to compute denominator
    P_n = P_n_genuine * P_genuine + P_n_spoofer * P_spoofer

    P_genuine_n = (P_n_genuine * P_genuine) / P_n

    return P_genuine_n


# Assumes that spoofers spoof at some rate s and genuine operators never spoof.
# Thanks ChatGPT! I'll have to think about whether this is correct.
def compute_posterior_2(
    num_flagged, total_responses, prior, false_pos_rate, alpha=2, beta_param=2
):
    """
    Computes the posterior probability P(genuine | n) given:
    - n: Number of responses labeled as spoofs
    - N: Total number of responses
    - r: False positive rate of the detector
    - pi_g: Prior probability that an operator is genuine
    """
    n, N, r, p = num_flagged, total_responses, false_pos_rate, prior

    # PMF(n) of Binomial(r,N)
    P_n_genuine = comb(N, n) * (r**n) * (1 - r) ** (N - n)

    # Prior probability of being genuine
    P_genuine = 1 - p
    # Prior probability of being spoofer
    P_spoofer = p

    # Detector has FNR of 0, so if n < N then it's def not a spoofer and vice versa
    P_n_spoofer = 1 if n == N else 0

    P_n_and_spoofer = integrate.quad(
        lambda s: (
            comb(N, n)
            * s**n
            * (1 - s) ** (N - n)
            * beta.pdf(s, alpha, beta_param)
            * P_spoofer
        ),
        0,
        1,
    )[0]

    # Law of total probability to compute denominator
    P_n = P_n_genuine * P_genuine + P_n_and_spoofer

    P_genuine_n = (P_n_genuine * P_genuine) / P_n

    return P_genuine_n


# Example usage:


print(compute_posterior_1(30, 100, prior, false_pos_rate))
