# Cascade Prediction Tests

This directory contains comprehensive tests for the `cascade_prediction` package.

## Test Structure

```
test/
├── conftest.py              # Shared pytest fixtures and utilities
├── test_dataset.py          # Tests for CascadeDataset class
├── test_collation.py        # Tests for batch collation
├── test_preprocessing.py    # Tests for preprocessing modules
├── test_generators.py       # Tests for data generator modules
└── README.md               # This file
```

## Running Tests

### Run all tests
```bash
pytest test/ -v
```

### Run specific test file
```bash
pytest test/test_dataset.py -v
pytest test/test_collation.py -v
pytest test/test_preprocessing.py -v
pytest test/test_generators.py -v
```

### Run specific test class
```bash
pytest test/test_dataset.py::TestCascadeDataset -v
```

### Run specific test method
```bash
pytest test/test_dataset.py::TestCascadeDataset::test_dataset_initialization -v
```

### Run with coverage
```bash
pytest test/ --cov=cascade_prediction --cov-report=html
```

### Run only fast tests (skip slow tests)
```bash
pytest test/ -v -m "not slow"
```

### Run with verbose output
```bash
pytest test/ -vv
```

### Run and stop at first failure
```bash
pytest test/ -x
```

## Test Coverage

### test_dataset.py
Tests for `cascade_prediction/data/dataset.py`:
- Dataset initialization and loading
- Scenario loading (cascade and normal)
- Normalization and preprocessing
- Edge mask creation
- Graph properties extraction
- Metadata caching
- Error handling (empty directories, corrupted files)

### test_collation.py
Tests for `cascade_prediction/data/collation.py`:
- Single and multiple item collation
- Variable length sequence handling
- Edge index sharing across batch
- Graph properties batching
- 4D tensor handling (images)
- Edge mask padding
- Missing keys handling

### test_preprocessing.py
Tests for `cascade_prediction/data/preprocessing/`:
- Power normalization (MW to p.u.)
- Frequency normalization (Hz to p.u.)
- Truncation window calculation
- Sequence truncation
- Edge mask creation from failures
- Tensor conversion
- Integration tests

### test_generators.py
Tests for `cascade_prediction/data/generator/`:
- **Topology**: Grid generation, connectivity, symmetry
- **Node Properties**: Type distribution, capacity sizing, thresholds
- **Power Flow**: AC power flow computation, failure handling
- **Frequency Dynamics**: Frequency updates, imbalance handling
- **Thermal Dynamics**: Temperature updates, cooling
- **Cascade**: State checking, propagation, adjacency lists
- **Environmental**: Satellite imagery, weather, threat indicators
- **Robotic**: Visual feeds, thermal cameras, sensor arrays
- **Simulator**: Complete scenario generation
- **Orchestrator**: Batch generation, train/val/test splitting
- **Utils**: Memory monitoring, file I/O, validation

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for file operations
- `mock_grid_topology`: Mock grid topology (30 nodes, 50 edges)
- `mock_node_properties`: Mock node properties
- `mock_timestep_data`: Mock data for single timestep
- `mock_cascade_scenario`: Complete cascade scenario
- `mock_normal_scenario`: Complete normal scenario
- `mock_scenario_file`: Pickle file with scenario
- `mock_data_dir`: Directory with multiple scenarios
- `simple_3node_grid`: Simple 3-node grid for basic tests

## Helper Functions

Assertion helpers in `conftest.py`:

- `assert_tensor_shape(tensor, expected_shape, name)`
- `assert_tensor_dtype(tensor, expected_dtype, name)`
- `assert_tensor_range(tensor, min_val, max_val, name)`
- `assert_numpy_shape(array, expected_shape, name)`
- `assert_numpy_dtype(array, expected_dtype, name)`
- `assert_numpy_range(array, min_val, max_val, name)`

## Test Markers

Custom markers for organizing tests:

- `@pytest.mark.slow`: Marks slow tests (can skip with `-m "not slow"`)
- `@pytest.mark.integration`: Marks integration tests
- `@pytest.mark.unit`: Marks unit tests

## Dependencies

Required packages for testing:
```bash
pip install pytest pytest-cov numpy torch
```

## Writing New Tests

When adding new tests:

1. Use fixtures from `conftest.py` when possible
2. Follow the existing naming convention: `test_<module>_<functionality>`
3. Add docstrings to test methods
4. Use assertion helpers for cleaner error messages
5. Mark slow tests with `@pytest.mark.slow`
6. Group related tests in classes

Example:
```python
import pytest

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self, mock_grid_topology):
        """Test basic functionality."""
        # Arrange
        data = mock_grid_topology
        
        # Act
        result = process_data(data)
        
        # Assert
        assert result is not None
        assert_tensor_shape(result, (30, 10), "result")
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure all tests pass before merging:

```bash
# Run all tests with coverage
pytest test/ --cov=cascade_prediction --cov-report=term-missing

# Check for minimum coverage (e.g., 80%)
pytest test/ --cov=cascade_prediction --cov-fail-under=80
```

## Troubleshooting

### Import errors
If you get import errors, ensure the package is installed:
```bash
pip install -e .
```

### Slow tests
Some tests (especially generator tests) can be slow. Skip them during development:
```bash
pytest test/ -m "not slow"
```

### Memory issues
Generator tests create large arrays. If you run into memory issues:
```bash
pytest test/ --maxfail=1  # Stop after first failure
```

### PyPSA warnings
Power flow tests may show PyPSA warnings. These are expected for edge cases and don't indicate test failures.
