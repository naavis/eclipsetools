import pytest
import click
from eclipsetools.commands.filter import validate_sigma_parameters


class TestValidateSigmaParameters:
    """Tests for the validate_sigma_parameters function."""

    def test_valid_sigma_only(self):
        """Test that providing only sigma is valid."""
        # Should not raise any exception
        validate_sigma_parameters(sigma=2.0, sigma_tangent=None, sigma_radial=None)

    def test_valid_both_tangent_and_radial(self):
        """Test that providing both sigma_tangent and sigma_radial is valid."""
        # Should not raise any exception
        validate_sigma_parameters(sigma=None, sigma_tangent=1.5, sigma_radial=2.5)

    def test_valid_sigma_with_tangent_and_radial(self):
        """Test that providing all three parameters is not valid."""
        with pytest.raises(click.BadParameter):
            validate_sigma_parameters(sigma=1.0, sigma_tangent=1.5, sigma_radial=2.5)

    def test_invalid_no_parameters(self):
        """Test that providing no parameters raises BadParameter."""
        with pytest.raises(click.BadParameter):
            validate_sigma_parameters(sigma=None, sigma_tangent=None, sigma_radial=None)

    def test_invalid_only_sigma_tangent(self):
        """Test that providing only sigma_tangent raises BadParameter."""
        with pytest.raises(click.BadParameter):
            validate_sigma_parameters(sigma=None, sigma_tangent=1.5, sigma_radial=None)

    def test_invalid_only_sigma_radial(self):
        """Test that providing only sigma_radial raises BadParameter."""
        with pytest.raises(click.BadParameter):
            validate_sigma_parameters(sigma=None, sigma_tangent=None, sigma_radial=2.5)

    def test_invalid_mixed_incomplete_parameters(self):
        """Test various invalid combinations of parameters."""
        # sigma with only one of tangent/radial
        with pytest.raises(click.BadParameter):
            validate_sigma_parameters(sigma=1.0, sigma_tangent=1.5, sigma_radial=None)

        with pytest.raises(click.BadParameter):
            validate_sigma_parameters(sigma=1.0, sigma_tangent=None, sigma_radial=2.5)

    def test_zero_values(self):
        """Test that zero values for sigma, sigma_tangent, or sigma_radial are valid."""
        validate_sigma_parameters(sigma=0.0, sigma_tangent=None, sigma_radial=None)
        validate_sigma_parameters(sigma=None, sigma_tangent=0.0, sigma_radial=0.0)
        validate_sigma_parameters(sigma=None, sigma_tangent=1.0, sigma_radial=0.0)
        validate_sigma_parameters(sigma=None, sigma_tangent=0.0, sigma_radial=1.0)
