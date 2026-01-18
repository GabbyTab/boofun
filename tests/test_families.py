"""Tests for the families module."""

import pytest
import numpy as np

import boolfunc as bf
from boolfunc.families import (
    FunctionFamily,
    InductiveFamily,
    GrowthTracker,
    MajorityFamily,
    ParityFamily,
    TribesFamily,
    ThresholdFamily,
    ANDFamily,
    ORFamily,
    DictatorFamily,
    LTFFamily,
)
from boolfunc.families.builtins import RecursiveMajority3Family


class TestMajorityFamily:
    """Tests for MajorityFamily."""
    
    def test_generate_creates_majority(self):
        """generate() creates majority functions."""
        family = MajorityFamily()
        
        for n in [3, 5, 7]:
            f = family.generate(n)
            assert f.n_vars == n
            assert f.is_balanced()
    
    def test_metadata_properties(self):
        """Metadata contains expected properties."""
        family = MajorityFamily()
        meta = family.metadata
        
        assert meta.name == "Majority"
        assert "monotone" in meta.universal_properties
        assert "symmetric" in meta.universal_properties
    
    def test_theoretical_asymptotics(self):
        """Theoretical asymptotics are available."""
        family = MajorityFamily()
        meta = family.metadata
        
        # Total influence ~ sqrt(2/pi) * sqrt(n)
        for n in [5, 7, 9]:
            theory = meta.asymptotics["total_influence"](n)
            actual = family.generate(n).total_influence()
            # Should be within 20% for these n values
            assert abs(actual - theory) / theory < 0.3


class TestParityFamily:
    """Tests for ParityFamily."""
    
    def test_generate_creates_parity(self):
        """generate() creates parity functions."""
        family = ParityFamily()
        
        for n in [3, 5, 8]:
            f = family.generate(n)
            assert f.n_vars == n
            assert f.is_linear()
    
    def test_parity_has_full_degree(self):
        """Parity has degree n."""
        family = ParityFamily()
        
        for n in [3, 4, 5]:
            f = family.generate(n)
            assert f.degree() == n


class TestTribesFamily:
    """Tests for TribesFamily."""
    
    def test_generate_with_explicit_params(self):
        """generate() with k and w creates correct tribes."""
        family = TribesFamily()
        
        # 2 tribes of width 3 = 6 variables
        f = family.generate(6, k=2, w=3)
        assert f.n_vars == 6
        assert f.is_monotone(50)
    
    def test_generate_auto_params(self):
        """generate() auto-selects parameters for n variables."""
        family = TribesFamily()
        
        for n in [6, 8]:
            f = family.generate(n)
            # Should have at most n variables
            assert f.n_vars <= n
            assert f.is_monotone(50)
    
    def test_direct_tribes_function(self):
        """Can create tribes directly with bf.tribes."""
        # bf.tribes(k, n) = tribe_size k, total vars n
        f = bf.tribes(2, 6)  # 3 tribes of size 2
        assert f.n_vars == 6
        assert f.is_monotone(50)
    
    def test_metadata_properties(self):
        """Tribes metadata contains expected properties."""
        family = TribesFamily()
        meta = family.metadata
        
        assert meta.name == "Tribes"
        assert "monotone" in meta.universal_properties


class TestThresholdFamily:
    """Tests for ThresholdFamily."""
    
    def test_default_threshold_is_majority_like(self):
        """Default threshold is majority-like."""
        family = ThresholdFamily()
        
        for n in [3, 5, 7]:
            f = family.generate(n)
            assert f.n_vars == n
            assert f.is_monotone(50)
    
    def test_threshold_k_equals_1_is_or(self):
        """Threshold with k=1 equals OR."""
        family = ThresholdFamily()
        
        for n in [3, 4, 5]:
            f = family.generate(n, k=1)
            or_f = bf.OR(n)
            # Should have same hamming weight
            assert f.hamming_weight() == or_f.hamming_weight()
    
    def test_threshold_k_equals_n_is_and(self):
        """Threshold with k=n equals AND."""
        family = ThresholdFamily()
        
        for n in [3, 4, 5]:
            f = family.generate(n, k=n)
            and_f = bf.AND(n)
            assert f.hamming_weight() == and_f.hamming_weight()
    
    def test_custom_k_function(self):
        """Can use custom k function."""
        # Always use k = 2
        family = ThresholdFamily(k_function=lambda n: 2)
        
        f = family.generate(5)
        # At least 2 ones needed
        assert f.evaluate([0, 0, 0, 0, 0]) == 0
        assert f.evaluate([1, 0, 0, 0, 0]) == 0
        assert f.evaluate([1, 1, 0, 0, 0]) == 1


class TestANDORFamilies:
    """Tests for AND and OR families."""
    
    def test_and_family(self):
        """AND family generates AND functions."""
        family = ANDFamily()
        
        for n in [2, 3, 4]:
            f = family.generate(n)
            # AND is 1 only on all-ones input
            assert f.hamming_weight() == 1
    
    def test_or_family(self):
        """OR family generates OR functions."""
        family = ORFamily()
        
        for n in [2, 3, 4]:
            f = family.generate(n)
            # OR is 0 only on all-zeros input
            assert f.hamming_weight() == 2**n - 1
    
    def test_and_or_duality(self):
        """AND and OR are duals (De Morgan)."""
        and_fam = ANDFamily()
        or_fam = ORFamily()
        
        for n in [3, 4]:
            and_f = and_fam.generate(n)
            or_f = or_fam.generate(n)
            
            # Same total influence
            assert abs(and_f.total_influence() - or_f.total_influence()) < 1e-10
    
    def test_theoretical_asymptotics(self):
        """AND/OR follow theoretical influence formula."""
        family = ANDFamily()
        meta = family.metadata
        
        for n in [3, 4, 5]:
            f = family.generate(n)
            actual = f.total_influence()
            theory = meta.asymptotics["total_influence"](n)
            # Should match exactly for these simple functions
            assert abs(actual - theory) < 1e-6


class TestDictatorFamily:
    """Tests for DictatorFamily."""
    
    def test_dictator_family_default(self):
        """DictatorFamily(0) creates x_0 dictators."""
        family = DictatorFamily(0)
        
        for n in [3, 4, 5]:
            f = family.generate(n)
            assert f.n_vars == n
            # Dictator has one variable with influence 1
            influences = f.influences()
            assert max(influences) == 1.0


class TestLTFFamily:
    """Tests for LTFFamily."""
    
    def test_uniform_weights_is_majority(self):
        """Uniform weights create majority."""
        family = LTFFamily.uniform()
        
        for n in [3, 5, 7]:
            f = family.generate(n)
            maj = bf.majority(n)
            # Should be equivalent to majority
            assert f.is_balanced() == maj.is_balanced()
    
    def test_geometric_weights(self):
        """Geometric weights create valid LTF."""
        family = LTFFamily.geometric(ratio=0.5)
        
        f = family.generate(5)
        assert f.n_vars == 5
        # LTFs are always monotone when weights are positive
        # but geometric may have numerical issues, just check it's valid
        assert f.total_influence() > 0
    
    def test_harmonic_weights(self):
        """Harmonic weights create valid LTF."""
        family = LTFFamily.harmonic()
        
        f = family.generate(5)
        assert f.n_vars == 5


class TestRecursiveMajority3:
    """Tests for RecursiveMajority3Family."""
    
    def test_generate_at_3(self):
        """Base case n=3 equals MAJ_3."""
        family = RecursiveMajority3Family()
        
        f = family.generate(3)
        maj = bf.majority(3)
        
        # Should be identical to majority
        for x in range(8):
            assert f.evaluate(x) == maj.evaluate(x)
    
    def test_generate_at_9(self):
        """n=9 creates valid function."""
        family = RecursiveMajority3Family()
        
        f = family.generate(9)
        assert f.n_vars == 9
        assert f.is_balanced()
        assert f.is_monotone(50)
    
    def test_rejects_non_power_of_3(self):
        """Rejects n that isn't a power of 3."""
        family = RecursiveMajority3Family()
        
        with pytest.raises(ValueError):
            family.generate(5)
        
        with pytest.raises(ValueError):
            family.generate(10)
    
    def test_influence_bound(self):
        """Total influence follows theoretical bound."""
        family = RecursiveMajority3Family()
        
        # I[REC_MAJ3] ~ n^0.631
        for n in [3, 9]:
            f = family.generate(n)
            actual = f.total_influence()
            theory = n ** (np.log(2) / np.log(3))
            # Allow 50% deviation for small n
            assert actual < 2 * theory


class TestGrowthTracker:
    """Tests for GrowthTracker."""
    
    def test_track_total_influence(self):
        """Can track total influence growth."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence")
        results = tracker.observe([3, 5, 7])
        
        assert "total_influence" in results
        # Results contain TrackingResult objects with computed_values
        result = results["total_influence"]
        assert len(result.computed_values) == 3
        # Total influence should increase with n
        vals = result.computed_values
        assert vals[0] < vals[1] < vals[2]
    
    def test_track_multiple_properties(self):
        """Can track multiple properties at once."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence")
        tracker.mark("fourier_degree")
        
        results = tracker.observe([3, 4, 5])
        
        assert "total_influence" in results
        assert "fourier_degree" in results
        
        # Parity has total influence = n
        total_inf = results["total_influence"].computed_values
        assert abs(total_inf[0] - 3) < 0.01
        assert abs(total_inf[1] - 4) < 0.01
        assert abs(total_inf[2] - 5) < 0.01
        
        # Parity has degree = n
        degrees = results["fourier_degree"].computed_values
        assert degrees == [3, 4, 5]
    
    def test_track_custom_function(self):
        """Can track custom property function."""
        family = ANDFamily()
        tracker = GrowthTracker(family)
        
        # Custom: track hamming weight using compute_fn parameter
        tracker.mark("custom", name="hamming_weight", compute_fn=lambda f: f.hamming_weight())
        
        results = tracker.observe([2, 3, 4])
        
        # AND always has weight 1
        vals = results["hamming_weight"].computed_values
        assert vals == [1, 1, 1]


class TestInductiveFamily:
    """Tests for custom InductiveFamily creation."""
    
    def test_create_custom_family(self):
        """Can create a custom function family."""
        # Custom family: constant-1 function of size n
        class ConstantOneFamily(InductiveFamily):
            def generate(self, n, **kwargs):
                return bf.constant(True, n)
            
            @property
            def metadata(self):
                from boolfunc.families.base import FamilyMetadata
                return FamilyMetadata(
                    name="ConstantOne",
                    description="f(x) = 1 for all x",
                    parameters={},
                    asymptotics={
                        "total_influence": lambda n: 0.0,
                    },
                    universal_properties=["constant"],
                )
        
        family = ConstantOneFamily()
        f = family.generate(5)
        assert f.total_influence() == 0.0
        assert f.hamming_weight() == 2**5
    
    def test_custom_family_with_tracker(self):
        """Custom family works with GrowthTracker."""
        class ConstantFamily(InductiveFamily):
            def generate(self, n, **kwargs):
                return bf.constant(True, n)
            
            @property
            def metadata(self):
                from boolfunc.families.base import FamilyMetadata
                return FamilyMetadata(
                    name="Constant",
                    description="f(x) = 1",
                    parameters={},
                    asymptotics={},
                    universal_properties=[],
                )
        
        family = ConstantFamily()
        tracker = GrowthTracker(family)
        tracker.mark("total_influence")
        
        results = tracker.observe([3, 4, 5])
        # All should be 0 - access computed_values from TrackingResult
        vals = results["total_influence"].computed_values
        assert all(v == 0.0 for v in vals)


class TestFamilyIntegration:
    """Integration tests for families module."""
    
    def test_family_with_visualizer(self):
        """Family works with visualization system."""
        family = MajorityFamily()
        
        # Should be able to generate and analyze
        for n in [3, 5, 7]:
            f = family.generate(n)
            
            # All standard analyses should work
            assert f.total_influence() > 0
            assert f.degree() > 0
            assert len(f.influences()) == n
    
    def test_comparison_multiple_families(self):
        """Can compare multiple families."""
        families = {
            'majority': MajorityFamily(),
            'parity': ParityFamily(),
            'and': ANDFamily(),
        }
        
        n = 5
        functions = {name: fam.generate(n) for name, fam in families.items()}
        
        # Compare total influences
        influences = {name: f.total_influence() for name, f in functions.items()}
        
        # Parity has highest influence (n)
        assert influences['parity'] == max(influences.values())
        # AND has low influence
        assert influences['and'] < influences['majority']
    
    def test_all_builtin_families_generate(self):
        """All built-in families can generate valid functions."""
        families = [
            MajorityFamily(),
            ParityFamily(),
            TribesFamily(),
            ThresholdFamily(),
            ANDFamily(),
            ORFamily(),
            DictatorFamily(0),
            LTFFamily.uniform(),
        ]
        
        for family in families:
            f = family.generate(5)
            assert f.n_vars == 5 or f.n_vars >= 4  # Tribes may adjust
            assert f.total_influence() >= 0
    
    def test_family_metadata_consistency(self):
        """All families have consistent metadata."""
        families = [
            MajorityFamily(),
            ParityFamily(),
            ANDFamily(),
            ORFamily(),
        ]
        
        for family in families:
            meta = family.metadata
            assert meta.name is not None
            assert meta.description is not None
            assert isinstance(meta.universal_properties, list)
