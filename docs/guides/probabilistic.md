# Probabilistic View & Pseudorandomness

Boolean functions are deterministic, but the *inputs* need not be. When inputs are drawn from a probability distribution, a Boolean function becomes a **random variable** — and the entire machinery of probability, statistics, and information theory applies.

This guide covers:

1. [Boolean functions as random variables](#boolean-functions-as-random-variables)
2. [P-biased measures and threshold phenomena](#p-biased-measures)
3. [When to estimate vs compute exactly](#estimation-vs-exact-computation)
4. [Connection to pseudorandomness](#pseudorandomness)
5. [Connection to the invariance principle](#invariance-principle)

**O'Donnell references**: Chapters 1-3 (Fourier, expectations), Chapter 6 (pseudorandomness), Chapter 8.4 (p-biased analysis, Russo's formula).

---

## Boolean Functions as Random Variables

Under the **uniform distribution**, each input $x \in \{-1,+1\}^n$ is equally likely. A Boolean function $f$ then has:

- **Expectation**: $\mathbb{E}[f] = \hat{f}(\emptyset)$ (the empty Fourier coefficient)
- **Variance**: $\text{Var}[f] = \sum_{S \neq \emptyset} \hat{f}(S)^2$ (Parseval's identity)
- **Influence of variable $i$**: $\text{Inf}_i[f] = \Pr_x[f(x) \neq f(x^{\oplus i})]$

The key insight: **Fourier coefficients are expectations**:

$$\hat{f}(S) = \mathbb{E}_x[f(x) \chi_S(x)]$$

This means they can be *estimated* by sampling, not just computed exactly.

### API

```python
from boofun.analysis.sampling import (
    RandomVariableView, SpectralDistribution,
    estimate_fourier_coefficient, estimate_influence,
)

# Unified exact + Monte Carlo interface
rv = RandomVariableView(f, p=0.5)
rv.expectation()                          # Exact E[f]
rv.estimate_expectation(n_samples=10000)  # Monte Carlo E[f]
rv.variance()                             # Exact Var[f]
rv.total_influence()                      # Exact I[f]
rv.validate_estimates(n_samples=10000)    # Cross-check exact vs estimated

# Spectral distribution: Pr[S] = f_hat(S)^2
sd = SpectralDistribution.from_function(f)
sd.entropy()                              # Shannon entropy of spectrum
sd.weight_at_degree(k)                    # Fourier weight at degree k
```

---

## P-Biased Measures

Under the **p-biased distribution** $\mu_p$, each bit is 1 independently with probability $p$. This generalizes the uniform case ($p = 1/2$).

### Why p-biased matters

1. **Threshold phenomena**: Monotone functions exhibit sharp phase transitions at a critical $p_c$ where $\Pr_p[f = 1]$ jumps from near 0 to near 1.

2. **Russo's formula** (Margulis 1974, Russo 1981): For monotone $f$,
   $$\frac{d}{dp}\mu_p(f) = I_p[f]$$
   The slope of the threshold curve equals the total p-biased influence. High influence = sharp threshold.

3. **Friedgut-Kalai theorem**: Monotone functions with *coarse* thresholds (gradual transitions) are close to juntas — they essentially depend on few variables.

### API

```python
from boofun.analysis.p_biased import (
    p_biased_expectation,       # Exact E_{mu_p}[f]
    p_biased_total_influence,   # Exact I_p[f]
    PBiasedAnalyzer,            # Full p-biased analysis
)
from boofun.analysis.global_hypercontractivity import (
    threshold_curve,            # mu_p(f) over a range of p
    find_critical_p,            # Binary search for p_c
)

# Exact p-biased analysis
analyzer = PBiasedAnalyzer(f, p=0.3)
analyzer.expectation()        # E_{mu_0.3}[f]
analyzer.total_influence()    # I^{0.3}[f]
analyzer.summary()            # Full report

# Threshold curve (Monte Carlo for large n)
import numpy as np
p_range = np.linspace(0.01, 0.99, 100)
curve = threshold_curve(f, p_range, samples=5000)

# Critical probability
pc = find_critical_p(f, samples=5000)
```

### Built-in function conventions

The `bf.tribes(k, n)` function creates the **dual tribes** (AND-of-ORs) convention:

$$\text{Tribes}_{k,n}(x) = \bigwedge_{j=1}^{\lceil n/k \rceil} \bigvee_{i \in T_j} x_i$$

where $k$ is the tribe size and $n$ is the total number of variables. The textbook tribes (O'Donnell Ch. 4) uses OR-of-ANDs; the two are related by negation.

```python
bf.tribes(3, 12)  # 4 tribes of 3 on 12 variables: AND(OR(x0,x1,x2), ..., OR(x9,x10,x11))
bf.tribes(2, 6)   # 3 tribes of 2 on 6 variables: AND(OR(x0,x1), OR(x2,x3), OR(x4,x5))
```

---

## Estimation vs Exact Computation

The exact Fourier transform enumerates all $2^n$ inputs. The tradeoff:

| $n$ | Truth table size | Exact | Monte Carlo (10K samples) |
|-----|-----------------|-------|---------------------------|
| $\leq 14$ | $\leq$ 16K | Fast (ms) | Unnecessary |
| $14$–$20$ | 16K–1M | Feasible (s) | Faster alternative |
| $> 20$ | $> 1$M | **Infeasible** | **Only option** |

For large $n$, use oracle-based functions with Monte Carlo estimation:

```python
# Oracle: no truth table materialized
f_large = bf.create(lambda x: 1 if sum(x) > n // 2 else 0, n=30)

# Estimate Fourier coefficient via 10K samples
est, stderr = estimate_fourier_coefficient(f_large, S=1, n_samples=10000,
                                           return_confidence=True)
# Error scales as O(1/sqrt(N)) by the CLT
```

The `RandomVariableView` class provides both exact and estimated methods on the same object, making it easy to cross-validate on small functions and then trust the estimates on large ones.

---

## Pseudorandomness

A central question in computational complexity: **can we replace truly random bits with "pseudorandom" bits that are cheaper to generate, without affecting the computation?**

Boolean function analysis provides the theoretical foundation for this, through a key insight:

> **Functions with bounded Fourier weight at high degrees are "foolable" by distributions with limited independence.**

### The mechanism

1. **Spectral concentration**: If most of $f$'s Fourier weight is at low degrees ($\sum_{|S| \leq d} \hat{f}(S)^2 \approx 1$), then $f$ is well-approximated by its degree-$d$ truncation $f^{\leq d}$.

2. **Limited independence fools low degree**: A $d$-wise independent distribution (which requires only $O(d \log n)$ random bits) cannot be distinguished from uniform by any degree-$d$ polynomial.

3. **Therefore**: Functions with spectral concentration at degree $d$ are "fooled" by $d$-wise independent distributions.

### Results using this framework

| Result | Statement | Connection |
|--------|-----------|------------|
| **Linial-Mansour-Nisan (1993)** | AC$^0$ circuits have Fourier concentration at degree $O(\log n)^{d-1}$ | PRGs for AC$^0$ |
| **Hastad (2001), Tal (2017)** | Tight bounds on AC$^0$ Fourier tails | Optimal PRGs |
| **Viola (2009)** | $\mathbb{F}_2$-polynomials of degree $d$ fooled by $2^{O(d)} \log(n/\epsilon)$ bits | Fooling polynomials |
| **Chattopadhyay-Hatami-Lovett-Tal (2019)** | PRGs from second-level Fourier structure | PRGs for AC$^0$ with parity gates |

### Using boofun for spectral concentration

```python
from boofun.analysis.fourier import truncate_to_degree, fourier_weight_distribution

# How much Fourier weight is at low degrees?
weights = fourier_weight_distribution(f)
# weights[k] = sum_{|S|=k} f_hat(S)^2

# Cumulative weight at degree <= d
cumulative = [sum(weights[:d+1]) for d in range(len(weights))]

# Truncate to low-degree approximation
f_approx = truncate_to_degree(f, d=3)
```

Functions with weight concentrated at low degrees (like Majority, Tribes) are foolable by low-independence distributions. Functions with weight at high degrees (like Parity) require full independence — they are the "hardest to fool."

### Connection to cryptography

The cryptographic analysis module (`analysis.cryptographic`) approaches "randomness" from the dual perspective: rather than asking "can we fool this function?", it asks "how random does this function look?" See the [Cryptographic Analysis guide](cryptographic.md) for nonlinearity, bent functions, and Walsh spectrum analysis.

---

## Invariance Principle

The invariance principle (O'Donnell Ch. 11) connects p-biased analysis to Gaussian analysis:

> **Boolean functions with low influences behave the same whether inputs are drawn from the hypercube or from Gaussian space.**

Formally, if $f$ has $\max_i \text{Inf}_i[f] \leq \epsilon$, then for smooth test functions $\psi$:

$$|\mathbb{E}[\psi(f(x))] - \mathbb{E}[\psi(\tilde{f}(G))]| = O(\epsilon^{1/4})$$

where $\tilde{f}$ is the multilinear extension and $G \sim N(0, I_n)$.

### Implications

- **Majority is Stablest**: Among balanced, low-influence functions, Majority maximizes noise stability. This implies the UGC-hardness of approximating MAX-CUT (Khot-Kindler-Mossel-O'Donnell 2007).
- **Berry-Esseen for Boolean functions**: Low-influence functions have distributions close to Gaussian.
- **Pseudorandomness connection**: The invariance principle says low-influence functions can't distinguish different input distributions — a form of "fooling."

### API

```python
from boofun.analysis.invariance import InvarianceAnalyzer
from boofun.analysis.gaussian import GaussianAnalyzer, multilinear_extension

# Invariance analysis
inv = InvarianceAnalyzer(f)
inv.invariance_bound()          # O(max Inf^{1/4}) bound
inv.compare_domains()           # Boolean vs Gaussian stats
inv.noise_stability_deficit(rho)  # Gap from Majority
inv.is_stablest_candidate()     # Check MiS conditions

# Gaussian analysis
ga = GaussianAnalyzer(f)
ga.berry_esseen()               # Berry-Esseen bound
ga.is_approximately_gaussian()  # Quick check

# Multilinear extension: extend f from hypercube to R^n
p = multilinear_extension(f)
p(np.random.randn(n))          # Evaluate on Gaussian input
```

---

## Further Reading

- O'Donnell, *Analysis of Boolean Functions* (2014): Chapters 1-3, 6, 8, 10-11
- Tal, "Tight bounds on the Fourier spectrum of AC0" (CCC 2017)
- Chattopadhyay-Hatami-Lovett-Tal, "Pseudorandom generators from the second Fourier level" (ITCS 2019)
- Mossel-O'Donnell-Oleszkiewicz, "Noise stability of functions with low influences" (Annals of Mathematics 2010)
- Viola, "The sum of d small-bias generators fools polynomials of degree d" (CCC 2009)

### Related guides

- [Spectral Analysis](spectral_analysis.md): Fourier transform, influences, noise stability
- [Hypercontractivity](hypercontractivity.md): KKL theorem, Bonami's lemma, threshold phenomena
- [Cryptographic Analysis](cryptographic.md): Nonlinearity, bent functions, "close to random"
- [Advanced Topics](advanced.md): Gaussian analysis, invariance principle, communication complexity
