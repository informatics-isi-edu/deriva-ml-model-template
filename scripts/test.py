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

if __name__ == "__main__":
    print
    print(__version__)