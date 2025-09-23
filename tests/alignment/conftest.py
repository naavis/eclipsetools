import numpy as np

# Generate a fixed test seed to ensure reproducible tests
TEST_SEED = 122807528840384100672342137672332424406


def pytest_generate_tests(metafunc):
    """Generate test cases for test_align_parametrized and test_translate_parametrized functions."""
    # Generate parameters for test_align_parametrized
    if "align_params" in metafunc.fixturenames:
        # Use the same random number generator as in the original test
        rng = np.random.default_rng(TEST_SEED)
        num_tests = 10

        # Generate the same test parameters as in the original test
        offsets = rng.uniform(-20.0, 20.0, (num_tests, 2))
        rotations = rng.uniform(-85.0, 85.0, num_tests)
        scales = rng.uniform(0.8, 1.2, num_tests)

        # Create a list of test cases with IDs
        test_cases = []
        ids = []

        for i in range(num_tests):
            test_cases.append((offsets[i], rotations[i], scales[i]))
            ids.append(
                f"offset=({offsets[i][0]:.2f}, {offsets[i][1]:.2f}) rot={rotations[i]:.2f} scale={scales[i]:.2f}"
            )

        metafunc.parametrize("align_params", test_cases, ids=ids)

    # Generate parameters for test_translate_parametrized
    if "translate_params" in metafunc.fixturenames:
        rng = np.random.default_rng(TEST_SEED)
        num_tests = 10

        # Generate the same offsets as in the original test_translate
        offsets = rng.uniform(-40.0, 40.0, (num_tests, 2))

        # Create a list of test cases with IDs
        test_cases = []
        ids = []

        for i in range(num_tests):
            test_cases.append(offsets[i])
            ids.append(f"offset=({offsets[i][0]:.1f}, {offsets[i][1]:.1f})")

        metafunc.parametrize("translate_params", test_cases, ids=ids)
