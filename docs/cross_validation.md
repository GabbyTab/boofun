# Cross-Validation

BooFun treats verifiability as a design commitment: every cross-validation
claim made in the documentation or the JOSS paper links to an executable
test, and states the reference (with version), the function families and
parameter ranges covered, and the tolerance used. If a claim is not in the
matrix below, BooFun does not claim it.

All test modules live in
[`tests/cross_validation/`](https://github.com/GabbyTab/boofun/tree/main/tests/cross_validation).

## Claim matrix

| Claim | Reference (version) | Families / range | Tolerance | Test |
|---|---|---|---|---|
| Walsh spectra match SageMath entry-by-entry (signed) | SageMath 10.9, pinned Docker fixtures¹ | full corpus² (303 functions) | exact integers | [`test_sagemath.py::test_walsh_spectrum`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_sagemath.py) |
| Nonlinearity matches SageMath | SageMath 10.9¹ | full corpus² | exact | [`test_sagemath.py::test_nonlinearity`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_sagemath.py) |
| Algebraic (ANF) degree matches SageMath | SageMath 10.9¹ | full corpus² | exact³ | [`test_sagemath.py::test_algebraic_degree`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_sagemath.py) |
| Correlation immunity matches the Siegenthaler order derived from SageMath's spectra | SageMath 10.9¹ | full corpus² | exact⁴ | [`test_sagemath.py::test_correlation_immunity`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_sagemath.py) |
| Balancedness and bent detection match SageMath | SageMath 10.9¹ | full corpus² | exact | [`test_sagemath.py`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_sagemath.py) |
| Canalization (is_canalizing, depth, essential variables, monotonicity, symmetry groups) matches BoolForge | BoolForge v1.0.1, commit `adae76b`⁵ | AND/OR/parity/majority, n = 2–5; constants; dictators | exact | [`test_boolforge.py`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_boolforge.py) |
| Influences match BoolForge exact activities and closed forms; total influence matches BoolForge exact average sensitivity | BoolForge v1.0.1⁵; O'Donnell 2014 Ch. 2 | parity(4), AND(4), majority(5) | float, atol 1e-10 (`exact=True`; tight enough to reject silent Monte Carlo fallback) | [`test_boolforge.py::TestBoolForgeSpectral`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_boolforge.py) |
| AES S-box: differential uniformity 4, component nonlinearity 112 | Nyberg, EUROCRYPT 1993; FIPS 197 | all 8 single-bit components, n = 8 | exact | [`test_published_values.py::TestAESSBox`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_published_values.py) |
| AES S-box: linearity 32, LAT/DDT spot values, component degree 7 | Daemen & Rijmen 2002 | n = 8 | exact | [`test_published_values.py::TestAESSBox`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_published_values.py) |
| PRESENT S-box: differential uniformity 4, linearity 8 | Bogdanov et al., CHES 2007 | n = 4 | exact | [`test_published_values.py::TestPRESENTSBox`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_published_values.py) |
| Known bent functions attain the Rothaus bound 2^(n−1) − 2^(n/2−1) | Rothaus 1976; thomasarmel README tables | n = 4, 6, 8 | exact | [`test_published_values.py::TestBentFunctions`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_published_values.py) |
| Fourier coefficients via FWHT match direct correlation sums (independent code paths) | internal redundant path | 7 standard functions, n = 3–4, all 2^n coefficients | 1e-10 | [`test_internal_consistency.py::test_fwht_vs_direct_correlation_sums`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_internal_consistency.py) |
| Influences (5 paths), total influence (9 paths), sensitivity (3 modules), degree, noise stability, variance, bias, certificates, decision-tree depth agree across modules | internal redundant paths | standard 3–4 var functions | 1e-10 / exact | [`test_internal_consistency.py`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_internal_consistency.py) |
| Huang's sensitivity theorem, Nisan–Szegedy, the s ≤ bs ≤ C ≤ D chain, D(AND_n) = D(OR_n) = n, property-testing accept/reject behavior | Huang 2019; Nisan & Szegedy 1994; Buhrman & de Wolf 2002 | AND/OR/majority/parity, n = 2–5; seeds pinned | exact | [`test_theoretical_bounds.py`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_theoretical_bounds.py) |
| Closed-form family values (majority influence asymptotics, parity spectra, tribes, noise stability formulas) | O'Donnell 2014 | families to n ≈ 21 | stated per test (asymptotics ≤ 15% rel.) | [`tests/test_theoretical_validation.py`](https://github.com/GabbyTab/boofun/blob/main/tests/test_theoretical_validation.py) |
| Exhaustive census: monotone (Dedekind), unate, canalizing, and bent counts over **all** truth tables of n variables | OEIS A000372, A245079, A102449, A004491 | all 2^(2^n) functions, n = 2–3 per PR; n = 4 on main/full-matrix runs⁶ | exact integer counts | [`test_census.py`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_census.py) |
| Canalizing-depth histogram over all four-variable functions ({0: 62024, 1: 2184, 2: 336, 3: 256, 4: 736}) | He & Macauley 2016⁷ | all 65,536 functions, n = 4⁶ | exact | [`test_census.py::TestFullFourVariableCensus`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/test_census.py) |

**Footnotes**

1. SageMath reference values are *pinned fixtures*: generated inside the
   `sagemath/sagemath:10.9` Docker image (digest recorded in
   [`tests/cross_validation/fixtures/sagemath.json`](https://github.com/GabbyTab/boofun/blob/main/tests/cross_validation/fixtures/sagemath.json)
   metadata, along with the generation date and exact command) by
   [`scripts/generate_sage_fixtures.py`](https://github.com/GabbyTab/boofun/blob/main/scripts/generate_sage_fixtures.py),
   which is boofun-free. The fixture tests are plain pytest and run on every
   pull request.
2. The Sage corpus: **all** 16 two-variable and **all** 256 three-variable
   functions; parity(2–8); majority(3, 5, 7); threshold and tribes families
   to n = 8 (tables asserted identical to `bf.parity`/`bf.majority`/
   `bf.tribes` constructors); inner-product bent functions (n = 4, 6, 8);
   and the 8 AES S-box component functions (S-box computed from the FIPS 197
   definition with spot-value self-checks).
3. Degree convention: Sage reports −1 for the zero function (degree of the
   zero polynomial); BooFun returns 0 for constants. Documented and
   converted in `sage_degree_to_boofun`.
4. Correlation-immunity convention: Sage's `correlation_immunity()` scans
   the a = 0 Walsh coefficient and therefore returns −1 for every unbalanced
   function; BooFun implements the textbook Siegenthaler order (ignoring
   a = 0). BooFun is validated against the textbook order derived from
   Sage's own recorded spectra on all 303 functions, and directly against
   Sage's value on every balanced function. The convention difference itself
   is kept executable in `TestFixtureIntegrity::test_sage_ci_convention`.
5. BoolForge runs *live* (not from fixtures) in the
   [Cross-Validation workflow](https://github.com/GabbyTab/boofun/blob/main/.github/workflows/cross-validation.yml):
   on every push to `main`, weekly, and on demand, pinned to commit
   `adae76be218eb8761e02d3c14a1d994764441102` (v1.0.1). A red run files or
   pings a tracking issue. Every Monte-Carlo-capable BoolForge API is called
   with `exact=True`, so no RNG seeds are involved.
6. Census cadence: the n ≤ 3 censuses (at most 256 functions) run as plain
   pytest on every pull request. The full n = 4 census (65,536 functions,
   ~15 s locally) runs when `BOOFUN_FULL_CENSUS=1`, which CI sets on pushes
   to `main`, release tags, and manual runs. Counts are of truth tables,
   not NPN-equivalence classes.
7. Constant-function conventions (deliberately different, both pinned):
   `is_canalizing` counts constants as trivially canalizing, matching
   OEIS A102449 (e.g. a(1) = 4); `get_canalizing_depth` assigns constants
   depth 0, matching He & Macauley — so each depth-0 bucket equals the
   non-canalizing count plus 2, asserted in
   `test_census.py::test_depth_zero_bucket_convention`.

## Convention conversions

All comparisons state their `{0,1}` ↔ `{−1,+1}` handling explicitly —
there are no `abs()` dodges:

- **Encoding**: BooFun follows O'Donnell — Boolean 0 → +1, 1 → −1, i.e.
  transforms `(-1)^f`. SageMath 10.9's `walsh_hadamard_transform()` uses the
  same encoding, verified by a runtime assertion in the fixture generator
  (the dictator's coefficient must be +4 at mask 1); spectra are compared
  with exact *signed* equality.
- **Truth-table indexing**: `t[x] = f(x)` with variable *i* in bit *i* of
  *x* (variable 0 = least significant bit), in both libraries; also
  asserted at generation time.
- Degree and correlation-immunity conventions: footnotes 3 and 4 above.

## What BooFun does *not* claim

- **Avishay Tal's scripts**: BooFun's API was partly motivated by Tal's
  course scripts, and the
  {doc}`migration guide <guides/migration_from_tal>` documents API
  correspondences and convention differences — but there is no executable
  comparison against those scripts, so BooFun does not claim numerical
  cross-validation against them.
- **Mathematica / Wolfram**: not used as a reference; earlier drafts of this
  document sketched Mathematica comparisons that were never implemented.
- **sboxU**: no executable comparison yet (planned alongside first-class
  vectorial Boolean functions).

## Regenerating the references

- **SageMath fixtures**: `./scripts/generate_sage_fixtures.sh` (requires
  Docker). To bump the pinned Sage version, edit `SAGE_TAG` in the wrapper,
  regenerate, and commit the fixture diff — the metadata header makes the
  provenance change reviewable.
- **BoolForge pin**: bump the commit SHA in the `pip install` line of
  `.github/workflows/cross-validation.yml` and recompile
  `requirements/boolforge.txt` if BoolForge's dependencies changed (see
  `requirements/README.md`).

## References

1. O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge
   University Press.
2. NIST FIPS 197 (2001). *Advanced Encryption Standard (AES)*.
3. Nyberg, K. (1993). Differentially uniform mappings for cryptography.
   *EUROCRYPT 1993*.
4. Daemen, J., & Rijmen, V. (2002). *The Design of Rijndael*. Springer.
5. Bogdanov, A., et al. (2007). PRESENT: An ultra-lightweight block cipher.
   *CHES 2007*.
6. Rothaus, O. (1976). On "bent" functions. *J. Combinatorial Theory A* 20.
7. Huang, H. (2019). Induced subgraphs of hypercubes and a proof of the
   Sensitivity Conjecture. *Annals of Mathematics* 190.
8. Nisan, N., & Szegedy, M. (1994). On the degree of Boolean functions as
   real polynomials. *Computational Complexity* 4.
9. He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean
   functions by canalizing depth. *Physica D* 314.
   <https://doi.org/10.1016/j.physd.2015.09.016>
10. OEIS Foundation. Sequences A000372 (Dedekind numbers), A245079 (unate),
    A102449 (canalizing), A004491 (bent). <https://oeis.org/>
11. SageMath Documentation: <https://doc.sagemath.org/>
12. BoolForge: <https://github.com/ckadelka/BoolForge>
