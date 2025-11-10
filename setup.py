from pathlib import Path

from setuptools import find_namespace_packages, setup


def _read_requirements(filename: str) -> list[str]:
    path = Path(__file__).parent / filename
    if not path.exists():
        return []

    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


setup(
    name="mindin_kor_engine",
    version="0.1.0",
    description="MindIn KOR Engine",
    packages=find_namespace_packages(include=["core", "core.*", "runtime", "runtime.*", "offline", "offline.*"]),
    include_package_data=True,
    install_requires=_read_requirements("requirements.txt"),
    python_requires=">=3.10",
)

