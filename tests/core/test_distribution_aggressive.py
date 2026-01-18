"""
Aggressive tests for distribution representation.

These tests are designed to find bugs by testing:
- Edge cases and boundary conditions
- Invalid inputs and error handling
- Numerical stability
- Statistical properties
"""

import pytest
import numpy as np

from boolfunc.core.representations.distribution import (
    BooleanDistribution, DistributionRepresentation
)


class TestBooleanDistributionValidation:
    """Test validation in BooleanDistribution."""
    
    def test_mismatched_truth_table_length(self):
        """Truth table length must match domain size."""
        with pytest.raises(ValueError, match="Truth table length"):
            BooleanDistribution(
                truth_table=np.array([0, 1, 0]),  # Wrong length
                input_distribution=np.array([0.25, 0.25, 0.25, 0.25]),
                n_vars=2,
                domain_size=4
            )
    
    def test_mismatched_distribution_length(self):
        """Distribution length must match domain size."""
        with pytest.raises(ValueError, match="Input distribution length"):
            BooleanDistribution(
                truth_table=np.array([0, 1, 0, 1]),
                input_distribution=np.array([0.5, 0.5]),  # Wrong length
                n_vars=2,
                domain_size=4
            )
    
    def test_negative_probabilities(self):
        """Probabilities must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            BooleanDistribution(
                truth_table=np.array([0, 1, 0, 1]),
                input_distribution=np.array([0.5, -0.1, 0.3, 0.3]),  # Negative!
                n_vars=2,
                domain_size=4
            )
    
    def test_unnormalized_distribution_gets_normalized(self):
        """Non-normalized distribution gets normalized."""
        dist = BooleanDistribution(
            truth_table=np.array([0, 1, 0, 1]),
            input_distribution=np.array([1.0, 2.0, 3.0, 4.0]),  # Sum = 10
            n_vars=2,
            domain_size=4
        )
        
        # Should be normalized to sum to 1
        assert np.allclose(np.sum(dist.input_distribution), 1.0)
    
    def test_zero_distribution_raises(self):
        """All-zero distribution should cause issues."""
        # This creates a divide-by-zero situation during normalization
        with pytest.raises((ValueError, RuntimeWarning, ZeroDivisionError)):
            BooleanDistribution(
                truth_table=np.array([0, 1, 0, 1]),
                input_distribution=np.array([0.0, 0.0, 0.0, 0.0]),
                n_vars=2,
                domain_size=4
            )


class TestBooleanDistributionEvaluate:
    """Test evaluate method edge cases."""
    
    def test_evaluate_with_integer_index(self):
        """Evaluate with integer index."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        assert dist.evaluate(0) == False
        assert dist.evaluate(1) == True
        assert dist.evaluate(2) == True
        assert dist.evaluate(3) == False
    
    def test_evaluate_with_binary_vector(self):
        """Evaluate with binary vector."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        assert dist.evaluate([0, 0]) == False
        assert dist.evaluate([0, 1]) == True
    
    def test_evaluate_out_of_range_index(self):
        """Index out of range should raise."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        with pytest.raises(ValueError, match="out of range"):
            dist.evaluate(100)
        
        with pytest.raises(ValueError, match="out of range"):
            dist.evaluate(-1)
    
    def test_evaluate_wrong_vector_length(self):
        """Wrong binary vector length should raise."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        with pytest.raises(ValueError, match="Input length"):
            dist.evaluate([0, 0, 0])  # Too many


class TestBooleanDistributionSampling:
    """Test sampling methods."""
    
    def test_sample_input_deterministic(self):
        """Sampling with seed is deterministic."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        samples1 = dist.sample_input(100, random_state=42)
        samples2 = dist.sample_input(100, random_state=42)
        
        assert np.array_equal(samples1, samples2)
    
    def test_sample_outputs_matches_truth_table(self):
        """Sampled outputs should match truth table."""
        # Constant True function
        dist = BooleanDistribution.uniform(
            np.array([True, True, True, True]), n_vars=2
        )
        
        outputs = dist.sample_outputs(100, random_state=42)
        assert all(outputs)  # All should be True
    
    def test_sample_with_biased_distribution(self):
        """Biased sampling concentrates on high-probability inputs."""
        truth_table = np.array([False, False, False, True])  # Only index 3 is True
        dist = BooleanDistribution(
            truth_table=truth_table,
            input_distribution=np.array([0.0, 0.0, 0.0, 1.0]),  # Only index 3 has mass
            n_vars=2,
            domain_size=4
        )
        
        samples = dist.sample_input(100, random_state=42)
        assert all(s == 3 for s in samples)


class TestBooleanDistributionStatistics:
    """Test statistical methods."""
    
    def test_output_probability_balanced(self):
        """Balanced function should have p=0.5."""
        # XOR is balanced
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2  # XOR
        )
        
        p_true = dist.output_probability(True)
        p_false = dist.output_probability(False)
        
        assert abs(p_true - 0.5) < 0.01
        assert abs(p_false - 0.5) < 0.01
    
    def test_output_probability_constant(self):
        """Constant True function has p(True)=1."""
        dist = BooleanDistribution.uniform(
            np.array([True, True, True, True]), n_vars=2
        )
        
        assert dist.output_probability(True) == 1.0
        assert dist.output_probability(False) == 0.0
    
    def test_entropy_balanced(self):
        """Balanced function has entropy 1 bit."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2  # XOR
        )
        
        entropy = dist.entropy()
        assert abs(entropy - 1.0) < 0.01  # Should be 1 bit
    
    def test_entropy_constant(self):
        """Constant function has entropy 0."""
        dist = BooleanDistribution.uniform(
            np.array([True, True, True, True]), n_vars=2
        )
        
        entropy = dist.entropy()
        assert entropy == 0.0
    
    def test_mutual_information_out_of_range(self):
        """Variable index out of range should raise."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        with pytest.raises(ValueError, match="Variable index"):
            dist.mutual_information(5)
    
    def test_mutual_information_independent(self):
        """Constant function has zero MI with all variables."""
        dist = BooleanDistribution.uniform(
            np.array([True, True, True, True]), n_vars=2
        )
        
        mi_0 = dist.mutual_information(0)
        mi_1 = dist.mutual_information(1)
        
        assert abs(mi_0) < 0.01
        assert abs(mi_1) < 0.01
    
    def test_mutual_information_dictator(self):
        """Dictator function has high MI with its variable."""
        # f = x_0 (dictator on first variable)
        dist = BooleanDistribution.uniform(
            np.array([False, False, True, True]), n_vars=2  # x_0
        )
        
        mi_0 = dist.mutual_information(0)
        mi_1 = dist.mutual_information(1)
        
        # High MI with x_0, zero with x_1
        assert mi_0 > 0.9
        assert abs(mi_1) < 0.01
    
    def test_correlation_out_of_range(self):
        """Variable index out of range should raise."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        with pytest.raises(ValueError, match="Variable index"):
            dist.correlation(5)
    
    def test_correlation_constant_function(self):
        """Constant function has zero correlation with all variables."""
        dist = BooleanDistribution.uniform(
            np.array([True, True, True, True]), n_vars=2
        )
        
        corr = dist.correlation(0)
        assert abs(corr) < 0.01
    
    def test_moments_order(self):
        """Can request different moment orders."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        m1 = dist.moments(order=1)
        assert len(m1) == 1  # Just mean
        
        m4 = dist.moments(order=4)
        assert len(m4) == 4  # Mean, var, skewness, kurtosis


class TestBooleanDistributionConditional:
    """Test conditional probability."""
    
    def test_conditional_impossible_condition(self):
        """Condition that's never satisfied returns 0."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        # Condition: input must be > 100 (impossible)
        prob = dist.conditional_probability(True, lambda i: i > 100)
        assert prob == 0.0
    
    def test_conditional_always_true_condition(self):
        """Condition that's always true gives marginal probability."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2  # XOR
        )
        
        # Condition: always True
        prob = dist.conditional_probability(True, lambda i: True)
        assert abs(prob - 0.5) < 0.01


class TestBooleanDistributionFactoryMethods:
    """Test factory methods."""
    
    def test_uniform_creates_valid_distribution(self):
        """Uniform creates valid distribution."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        assert np.allclose(dist.input_distribution, [0.25, 0.25, 0.25, 0.25])
    
    def test_biased_p0_gives_all_zeros(self):
        """Bias=0 concentrates on all-zeros input."""
        dist = BooleanDistribution.biased(
            np.array([False, True, True, False]), n_vars=2, bias=0.0
        )
        
        # Only input 00 (index 0) should have probability
        assert dist.input_distribution[0] == 1.0
        assert sum(dist.input_distribution[1:]) == 0.0
    
    def test_biased_p1_gives_all_ones(self):
        """Bias=1 concentrates on all-ones input."""
        dist = BooleanDistribution.biased(
            np.array([False, True, True, False]), n_vars=2, bias=1.0
        )
        
        # Only input 11 (index 3) should have probability
        assert dist.input_distribution[3] == 1.0
        assert sum(dist.input_distribution[:3]) == 0.0


class TestBooleanDistributionSerialization:
    """Test serialization."""
    
    def test_roundtrip(self):
        """Distribution survives serialization."""
        original = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        data = original.to_dict()
        restored = BooleanDistribution.from_dict(data)
        
        assert np.array_equal(restored.truth_table, original.truth_table)
        assert np.allclose(restored.input_distribution, original.input_distribution)
        assert restored.n_vars == original.n_vars


class TestDistributionRepresentation:
    """Test DistributionRepresentation class."""
    
    def test_evaluate_scalar(self):
        """Evaluate with scalar input."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        rep = DistributionRepresentation()
        result = rep.evaluate(np.array(1), dist, None, 2)
        
        assert result == True
    
    def test_evaluate_1d_array(self):
        """Evaluate with 1D array of indices."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        rep = DistributionRepresentation()
        result = rep.evaluate(np.array([0, 1, 2, 3]), dist, None, 2)
        
        assert list(result) == [False, True, True, False]
    
    def test_evaluate_2d_batch(self):
        """Evaluate with 2D batch of binary vectors."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        rep = DistributionRepresentation()
        batch = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        result = rep.evaluate(batch, dist, None, 2)
        
        assert len(result) == 4
    
    def test_evaluate_invalid_shape(self):
        """3D input should raise."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        rep = DistributionRepresentation()
        
        with pytest.raises(ValueError, match="Unsupported input shape"):
            rep.evaluate(np.zeros((2, 2, 2)), dist, None, 2)
    
    def test_create_empty(self):
        """Create empty distribution."""
        rep = DistributionRepresentation()
        dist = rep.create_empty(3)
        
        assert dist.n_vars == 3
        assert all(not v for v in dist.truth_table)  # All False
    
    def test_is_complete(self):
        """Check completeness."""
        rep = DistributionRepresentation()
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        assert rep.is_complete(dist)
    
    def test_dump(self):
        """Dump distribution to dict."""
        dist = BooleanDistribution.uniform(
            np.array([False, True, True, False]), n_vars=2
        )
        
        rep = DistributionRepresentation()
        data = rep.dump(dist)
        
        assert data['type'] == 'distribution'
        assert 'truth_table' in data
