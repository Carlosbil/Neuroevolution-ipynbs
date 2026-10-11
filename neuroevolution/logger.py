"""
Custom logging utilities for notebook execution.

Redirects print() output to log files. Notebook/console echo is disabled by
default so long runs keep their output in artifacts instead of notebook cells.
"""

import builtins
import logging
import os
import shlex
import sys
from importlib import metadata


def setup_notebook_logging(
    log_dir: str,
    log_filename: str = "execution_log.txt",
    echo_to_console: bool = False,
) -> logging.Logger:
    """
    Sets up logging to redirect print() to a file.
    
    Args:
        log_dir: Directory to store log files, or full path to a .txt log file
        log_filename: Name of the log file when log_dir is a directory
        echo_to_console: Also print messages to stdout/stderr when True
    
    Returns:
        Configured logger instance
    """
    log_dir = os.fspath(log_dir)
    log_basename = os.path.basename(log_dir).lower()
    if log_filename == "execution_log.txt" and log_basename.endswith((".txt", ".log")):
        log_dir, log_filename = os.path.split(log_dir)
        log_dir = log_dir or "."

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("neuroevolution_notebook")
    
    # Clear existing handlers
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        os.path.join(log_dir, log_filename),
        mode="a",
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    
    original_print = getattr(builtins, "_neuroevolution_original_print", builtins.print)
    if not hasattr(builtins, "_neuroevolution_original_print"):
        builtins._neuroevolution_original_print = original_print

    # Override print() to write to logger without filling notebook output cells.
    def notebook_print(*args, sep=" ", end="\n", file=None, flush=False, **kwargs):
        if echo_to_console or file not in (None, sys.stdout, sys.stderr):
            original_print(*args, sep=sep, end=end, file=file, flush=flush, **kwargs)
        message = sep.join(str(arg) for arg in args)
        if end not in ("", "\n"):
            message += end
        logger.info(message)
    
    builtins.print = notebook_print
    return logger


def _dependency_issues(required_packages: list) -> list:
    """Inspect requirements and Torchvision's declared Torch dependency, without pip."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    issues = []
    requested_names = set()
    for package in required_packages:
        requirement = Requirement(package)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        requested_names.add(canonicalize_name(requirement.name))
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            issues.append(f"Missing dependency: {package}")
            continue
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            issues.append(f"{requirement.name} {installed} does not satisfy {requirement.specifier}")

    if 'torchvision' in requested_names:
        try:
            torch_version = metadata.version('torch')
            vision_version = metadata.version('torchvision')
            for declaration in metadata.requires('torchvision') or []:
                requirement = Requirement(declaration)
                if canonicalize_name(requirement.name) != 'torch':
                    continue
                if requirement.marker is not None and not requirement.marker.evaluate():
                    continue
                if requirement.specifier and not requirement.specifier.contains(torch_version, prereleases=True):
                    issues.append(f"torchvision {vision_version} requires {requirement}; installed torch is {torch_version}")
        except metadata.PackageNotFoundError:
            pass  # Missing packages are already reported above.
    return issues


def _require_dependencies(required_packages: list) -> None:
    issues = _dependency_issues(required_packages)
    if issues:
        command = [sys.executable, '-m', 'pip', 'install', '--upgrade', *required_packages]
        if os.name == 'nt':
            repair_command = '& ' + ' '.join("'" + argument.replace("'", "''") + "'" for argument in command)
            terminal = 'a PowerShell terminal'
        else:
            repair_command = shlex.join(command)
            terminal = 'a terminal'
        raise RuntimeError(
            'Dependency check failed:\n- ' + '\n- '.join(issues) +
            '\nNo packages were installed or replaced. Shut down all kernels using this '
            f'environment and run this command from {terminal}:\n' + repair_command + '\n'
            'Then start a new kernel. Requirements are declared in neuroevolution/config.py.'
        )


def install_package(package: str) -> None:
    """Legacy name: check a dependency without modifying the active environment."""
    _require_dependencies([package])
    print(f"OK dependency: {package}")


def _runtime_version_matches(loaded: str, installed: str) -> bool:
    """Metadata may omit the local build tag reported by the imported module."""
    from packaging.version import Version

    loaded_version, installed_version = Version(loaded), Version(installed)
    return loaded_version == installed_version or (
        loaded_version.public == installed_version.public
        and (loaded_version.local is None or installed_version.local is None)
    )


def verify_dependencies(required_packages: list = None) -> None:
    """Check versions, Torch/Torchvision compatibility and imports without running pip.

    Installation belongs in a terminal before starting the notebook kernel.
    Both install_packages and install_package retain their legacy public names,
    but no longer replace imported libraries in the running interpreter.
    """
    from .config import REQUIRED_PACKAGES
    from packaging.requirements import Requirement

    required_packages = list(REQUIRED_PACKAGES if required_packages is None else required_packages)
    _require_dependencies(required_packages)
    names = {Requirement(package).name.lower() for package in required_packages}
    if 'torch' not in names and 'torchvision' not in names:
        print('All dependencies satisfy the configured requirements; environment unchanged')
        return

    try:
        import torch
        disk_version = metadata.version('torch')
        if not _runtime_version_matches(torch.__version__, disk_version):
            raise RuntimeError(f"Kernel has torch {torch.__version__} loaded, but disk contains {disk_version}")
        if 'torchvision' in names:
            import torchvision
            vision_disk_version = metadata.version('torchvision')
            if not _runtime_version_matches(torchvision.__version__, vision_disk_version):
                raise RuntimeError(f"Kernel has torchvision {torchvision.__version__} loaded, but disk contains {vision_disk_version}")
            print(f"Torchvision {torchvision.__version__} imported correctly")
        print(f"PyTorch {torch.__version__} imported correctly")
        print(f"CUDA available: {'Yes' if torch.cuda.is_available() else 'No'}")
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    except Exception as error:
        raise RuntimeError(
            f'PyTorch/Torchvision runtime check failed: {error}. '
            'The installation may be incomplete or the kernel may still hold old libraries. '
            'Shut down all kernels using this environment, repair the installation from '
            'a terminal, and start a new kernel. No automatic installation was attempted.'
        ) from error
    print('All dependencies verified; environment unchanged')
