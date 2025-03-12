from deriva_ml.demo_catalog import DemoML
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    pass

#ml_instance = DemoML()

if __name__ == "__main__":
    print(__version__)