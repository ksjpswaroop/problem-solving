"""Phase 0: Basic import and version tests."""


def test_import_problemsolving() -> None:
    """Package is importable."""
    import problemsolving

    assert problemsolving is not None


def test_version_is_string() -> None:
    """Version is a semantic version string."""
    from problemsolving import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
