import sys
import types

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "pyPhoCoreHelpers"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


# Keep `from pyphocorehelpers.pprint import ...` working after renaming pprint.py → pretty.py.
# A top-level pprint.py shadows stdlib pprint when this package directory is on PYTHONPATH
# (e.g. IDE multi-root workspace folders used by the Python Test Adapter).
class _LazyPprintAlias(types.ModuleType):
    def __getattr__(self, name):
        from pyphocorehelpers import pretty as _pretty
        sys.modules[self.__name__] = _pretty
        return getattr(_pretty, name)


sys.modules[__name__ + '.pprint'] = _LazyPprintAlias(__name__ + '.pprint')
