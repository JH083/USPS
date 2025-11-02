from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent

# Load requirements if the file exists, otherwise leave empty
requirements = []
req_file = here / "requirements.txt"
if req_file.exists():
	requirements = [
		line.strip()
		for line in req_file.read_text(encoding="utf-8").splitlines()
		if line.strip() and not line.strip().startswith("#")
	]

setup(
	name="USPS",
	version="0.0.1",
	description="USPS robot learning package",
	packages=find_packages(exclude=("tests", "docs")),
	install_requires=requirements,
	include_package_data=True,
	zip_safe=False,
)

