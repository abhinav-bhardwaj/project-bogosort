# Flask App Test Suite

This directory contains comprehensive tests for the Bogosort Model Evaluation Dashboard Flask application.

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with coverage
```bash
pytest --cov=app --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_routes/test_api.py
```

### Run specific test class
```bash
pytest tests/test_routes/test_api.py::TestApiModelsEndpoint
```

### Run specific test
```bash
pytest tests/test_routes/test_api.py::TestApiModelsEndpoint::test_list_models_success
```

### Run with verbose output
```bash
pytest -v
```

### Run with detailed failure output
```bash
pytest -vv --tb=long
```

## Test Organization

### Directory Structure

```
app/tests/
├── conftest.py                 # Pytest fixtures and app setup
├── test_app.py                 # App creation and initialization tests
├── test_config.py              # Configuration tests
├── test_db/
│   ├── __init__.py
│   └── test_queries.py         # Database query tests
└── test_routes/
    ├── __init__.py
    ├── test_main.py            # Main route tests
    ├── test_api.py             # API endpoint tests
    ├── test_dashboard.py       # Dashboard route tests
    └── test_bogosort.py        # Bogosort route and helper function tests
```

## Test Categories

### Unit Tests
- **test_config.py**: Configuration class tests
- **test_db/test_queries.py**: Database query function tests
- **test_routes/test_bogosort.py**: Helper function tests (is_sorted, load_shuffled_toxic_words, etc.)

### Integration Tests
- **test_app.py**: App creation and blueprint registration
- **test_routes/test_main.py**: Main route integration
- **test_routes/test_api.py**: API route integration
- **test_routes/test_dashboard.py**: Dashboard route integration
- **test_routes/test_bogosort.py**: Bogosort route integration

### API Tests
- **test_routes/test_api.py**: All API endpoint tests

## Test Coverage

Current test suite covers:

- ✅ Configuration management (development, testing, production)
- ✅ Database queries (load and retrieve model evaluations)
- ✅ Main route redirects
- ✅ API endpoints (models list, model evaluation, default evaluation)
- ✅ Dashboard routes (standard and nerdy dashboards)
- ✅ Bogosort helper functions (sorting logic, data loading, snapshot generation)
- ✅ Bogosort routes (GET/POST handling, state management)
- ✅ Flask app initialization and blueprint registration

## Key Fixtures

Available in `conftest.py`:

- **app**: Flask test application
- **client**: Test client for making requests
- **sample_evaluations**: Sample model evaluation data
- **evaluations_file**: Temporary JSON file with evaluations
- **monkeypatch_data_path**: Fixture that patches the DATA_PATH for queries
- **mock_toxic_words**: Sample toxic words list for bogosort tests

## Mocking

The test suite uses `unittest.mock` for mocking:

- File I/O operations (JSON loading, image saving)
- NumPy operations (loading .npy files)
- Threading (background sorting)
- Matplotlib operations (plot generation)

## Adding New Tests

1. Create test file following naming convention: `test_*.py`
2. Import fixtures from `conftest.py`
3. Use appropriate markers if needed:
   ```python
   @pytest.mark.unit
   def test_something():
       pass
   ```
4. Run tests to verify:
   ```bash
   pytest tests/test_file.py::TestClass::test_function
   ```

## Troubleshooting

### Tests fail with import errors
- Ensure you're running pytest from the `app/` directory
- Verify all Flask app imports use relative imports

### Template tests fail (404 on dashboard routes)
- Check that template files exist in `app/templates/`
- Verify template names match in route handlers

### Bogosort tests fail
- These tests use mocking for file I/O and matplotlib
- Ensure `unittest.mock` is available

## Coverage Goals

Target coverage by component:
- Config: 100%
- Database queries: 95%
- Routes: 90%
- Helper functions: 95%

Generate coverage report:
```bash
pytest --cov=app --cov-report=term-missing
```
