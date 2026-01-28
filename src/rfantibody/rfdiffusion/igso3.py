"""SO(3) diffusion methods."""
import numpy as np
import torch

L_default = 2000

def igso3_expansion(omega, sigma, L=L_default):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, sigma =
    sqrt(2) * eps, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=sigma^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        sigma: std of IGSO(3).
        L: Truncation level
    """
    p = 0
    for l in range(L):
        p += (2*l + 1) * np.exp(-l*(l+1)*sigma**2/2) * np.sin(omega*(l+1/2)) / np.sin(omega/2)
    return p

def igso3_expansion_torch(omega, sigma, L=L_default):
    """Truncated sum of IGSO(3) distribution implemented in torch

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        sigma: std of IGSO(3).
        L: Truncation level
    """
    p = 0
    for l in range(L):
        p += (2*l + 1) * torch.exp(-l*(l+1)*sigma**2/2) * torch.sin(omega*(l+1/2)) / torch.sin(omega/2)
    return p


def density(expansion, omega, marginal=True):
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1-np.cos(omega))/np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi**2


def calc_score_norm(exp, omega, sigma, L=L_default):  # score of density over SO(3)
    """score uses the quotient rule to compute the scaling factor for the score
    of the IGSO(3) density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp: truncated expansion of the power series in the IGSO(3) density
        omega: length of an Euler vector (i.e. angle of rotation)
        sigma: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        L: truncation level

    Returns:
        The d/d omega log IGSO3(omega; sigma)/(1-cos(omega))

    """
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    return dSigma / exp

def calc_score_norm_torch(omega, sigma, L=L_default):  # score of density over SO(3)
    """calc_score_norm_torch is a differentiable torch implementation of calc_score_norm

    Args:
        omega: length of an Euler vector (i.e. angle of rotation)
        sigma: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        L: truncation level

    Returns:
        The d/d omega log IGSO3(omega; sigma)/(1-cos(omega))
    """
    
    # first compute truncation of the power series expansion
    exp = igso3_expansion_torch(omega, sigma, L=L)

    dSigma = 0
    for l in range(L):
        hi = torch.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * torch.cos(omega * (l + 1 / 2))
        lo = torch.sin(omega / 2)
        dlo = 1 / 2 * torch.cos(omega / 2)
        dSigma += (2 * l + 1) * torch.exp(-l * (l + 1) * sigma**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    return dSigma / exp

def calculate_igso3(*, num_sigma, num_omega, min_sigma, max_sigma, L=L_default):
    """calculate_igso3 pre-computes numerical approximations to the IGSO3 cdfs
    and score norms and expected squared score norms.

    Args:
        num_sigma: number of different sigmas for which to compute igso3
            quantities.
        num_omega: number of point in the discretization in the angle of
            rotation.
        min_sigma, max_sigma: the upper and lower ranges for the angle of
            rotation on which to consider the IGSO3 distribution.  This cannot
            be too low or it will create numerical instability.
    """
    # Discretize omegas for calculating CDFs. Skip omega=0.
    discrete_omega = np.linspace(0, np.pi, num_omega+1)[1:]

    # Exponential noise schedule.  This choice is closely tied to the
    # scalings used when simulating the reverse time SDE. For each step n,
    # discrete_sigma[n] = min_eps^(1-n/num_eps) * max_eps^(n/num_eps)
    discrete_sigma = 10 ** np.linspace(
        np.log10(min_sigma),
        np.log10(max_sigma),
        num_sigma + 1)[1:]
    exp_vals = np.asarray(
        [igso3_expansion(discrete_omega, sigma, L=L) for sigma in discrete_sigma])

    # Compute the pdf and cdf values for the marginal distribution of the angle
    # of rotation (which is needed for sampling)
    pdf_vals = np.asarray(
        [density(x, discrete_omega, marginal=True) for x in exp_vals])
    cdf_vals = np.asarray(
        [pdf.cumsum() / num_omega * np.pi for pdf in pdf_vals])

    # Compute the norms of the scores.  This are used to scale the rotation axis when
    # computing the score as a vector.
    score_norm = np.asarray(
        [calc_score_norm(exp_vals[i], discrete_omega, x) for i, x in enumerate(discrete_sigma)])

    # Compute the standard deviation of the score norm for each sigma
    exp_score_norms = np.sqrt(
        np.sum(
            score_norm**2 * pdf_vals, axis=1) / np.sum(
                pdf_vals, axis=1))
    return {
        'cdf': cdf_vals,
        'score_norm': score_norm,
        'exp_score_norms': exp_score_norms,
        'discrete_omega': discrete_omega,
        'discrete_sigma': discrete_sigma,
    }
