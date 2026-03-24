"""Smoke test for the example script."""

from examples.basic_usage import main


def test_main_does_not_raise():
    """Importing and calling main() should complete without errors."""
    main()
