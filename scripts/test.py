from deriva_ml.demo_catalog import DemoML
#ml_instance = DemoML()

from deriva_ml.demo_catalog import DemoML
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("test")
except PackageNotFoundError:
    # package is not installed
    pass

#ml_instance = DemoML()
# content of my_module/__init__.py

from pathlib import Path

# you can use os.path and open() as well
__version__ = Path(__file__).parent.joinpath("VERSION").read_text()


if __name__ == "__main__":
    print(__version__)
