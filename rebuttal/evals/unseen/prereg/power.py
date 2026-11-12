# Power of a two-sided Spearman test (alpha .05) for N unseen models, under a bivariate normal with Pearson r.
import numpy as np
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
def crit(n, sims=200000):
    x = rng.standard_normal((sims, n)); y = rng.standard_normal((sims, n))
    rx = x.argsort(1).argsort(1); ry = y.argsort(1).argsort(1)
    rho = 1 - 6 * ((rx - ry) ** 2).sum(1) / (n * (n * n - 1))
    return np.quantile(np.abs(rho), 0.95)
print('N  crit|rho|  power at r=0.5 0.6 0.7 0.8')
for n in [6, 8, 10, 12, 16, 20, 24]:
    c = crit(n)
    pw = []
    for r in [0.5, 0.6, 0.7, 0.8]:
        sims = 40000
        x = rng.standard_normal((sims, n)); y = r * x + np.sqrt(1 - r * r) * rng.standard_normal((sims, n))
        rx = x.argsort(1).argsort(1); ry = y.argsort(1).argsort(1)
        rho = 1 - 6 * ((rx - ry) ** 2).sum(1) / (n * (n * n - 1))
        pw.append(round(float((np.abs(rho) > c).mean()), 2))
    print(n, round(float(c), 3), pw)
