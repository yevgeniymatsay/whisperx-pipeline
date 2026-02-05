def pytest_configure(config):  # noqa: D401
    """Disable incompatible globally-installed plugins.

    This repo does not use asyncio tests, but some environments may have an
    incompatible `pytest_asyncio` version installed globally that breaks
    collection under newer pytest versions.
    """
    pm = config.pluginmanager
    for plugin in list(pm.get_plugins()):
        if getattr(plugin, "__name__", "") == "pytest_asyncio.plugin":
            pm.unregister(plugin)

