"""The notebook must never replace its imported binary libraries with pip."""

import os
import shlex
import subprocess
import sys

import pytest

from neuroevolution import logger
from neuroevolution.config import REQUIRED_PACKAGES


@pytest.fixture(autouse=True)
def forbid_installation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Dependency verification launched a subprocess')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'check_call', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def mock_metadata(monkeypatch, versions, vision_requires=None):
    def version(name):
        if name not in versions:
            raise logger.metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(logger.metadata, 'version', version)
    monkeypatch.setattr(logger.metadata, 'requires', lambda name: vision_requires or [])


def test_existing_torch_28_and_vision_023_do_not_need_an_upgrade(monkeypatch):
    mock_metadata(monkeypatch, {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128'}, ['torch==2.8.0'])
    assert logger._dependency_issues(['torch>=2.8.0', 'torchvision>=0.23.0']) == []
    logger.install_package('torch>=2.8.0')


def test_incompatible_native_pair_is_detected_before_import_or_install(monkeypatch):
    mock_metadata(monkeypatch, {'torch': '2.11.0', 'torchvision': '0.23.0'}, ['torch==2.8.0'])
    with pytest.raises(RuntimeError, match='torchvision 0.23.0 requires torch==2.8.0'):
        logger.verify_dependencies(['torch>=2.8.0', 'torchvision>=0.23.0'])


def test_config_declares_a_compatible_reproducible_native_pair(monkeypatch):
    binary_requirements = [package for package in REQUIRED_PACKAGES
                           if package.startswith(('torch==', 'torchvision=='))]
    assert binary_requirements == ['torch==2.11.0', 'torchvision==0.26.0']
    mock_metadata(monkeypatch, {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128'}, ['torch==2.11.0'])
    assert logger._dependency_issues(binary_requirements) == []


def test_incompatible_pair_gets_a_repair_command_for_the_kernel_interpreter(monkeypatch):
    mock_metadata(monkeypatch, {'torch': '2.11.0', 'torchvision': '0.23.0'}, ['torch==2.8.0'])
    monkeypatch.setattr(sys, 'executable', '/opt/jupyter env/bin/python')
    with pytest.raises(RuntimeError) as caught:
        logger.verify_dependencies(['torch==2.11.0', 'torchvision==0.26.0', 'numpy>=1.21.0'])
    message = str(caught.value)
    assert '/opt/jupyter env/bin/python' in message
    command = message.split('terminal:\n', 1)[1].splitlines()[0]
    if os.name == 'nt':
        assert command == ("& '/opt/jupyter env/bin/python' '-m' 'pip' 'install' '--upgrade' "
                           "'torch==2.11.0' 'torchvision==0.26.0' 'numpy>=1.21.0'")
    else:
        assert shlex.split(command) == ['/opt/jupyter env/bin/python', '-m', 'pip', 'install',
                                       '--upgrade', 'torch==2.11.0', 'torchvision==0.26.0', 'numpy>=1.21.0']
    assert 'torch==2.11.0' in message and 'torchvision==0.26.0' in message
    assert 'Shut down all kernels' in message
    assert 'Then start a new kernel' in message


@pytest.mark.parametrize('versions', [{}, {'torch': '2.7.0'}])
def test_missing_or_outdated_library_is_not_automatically_installed(monkeypatch, versions):
    mock_metadata(monkeypatch, versions)
    with pytest.raises(RuntimeError, match='No packages were installed or replaced'):
        logger.install_package('torch>=2.8.0')


def test_preflight_reports_all_missing_dependencies(monkeypatch):
    mock_metadata(monkeypatch, {})
    with pytest.raises(RuntimeError) as caught:
        logger.verify_dependencies(['torch>=2.8.0', 'torchvision>=0.23.0'])
    assert 'Missing dependency: torch>=2.8.0' in str(caught.value)
    assert 'Missing dependency: torchvision>=0.23.0' in str(caught.value)


def test_kernel_with_old_loaded_torch_is_detected(monkeypatch):
    import torch
    monkeypatch.setattr(torch, '__version__', '2.8.0')
    mock_metadata(monkeypatch, {'torch': '2.11.0'})
    with pytest.raises(RuntimeError, match='Kernel has torch 2.8.0 loaded, but disk contains 2.11.0'):
        logger.verify_dependencies(['torch>=2.8.0'])


@pytest.mark.parametrize('loaded,installed,expected', [
    ('2.11.0+cu128', '2.11.0', True),
    ('2.11.0+cpu', '2.11.0', True),
    ('2.11.0', '2.11.0+cu128', True),
    ('2.11.0+cu128', '2.11.0+cu128', True),
    ('2.11.0+cpu', '2.11.0+cu128', False),
    ('2.11.0+cu126', '2.11.0+cu128', False),
    ('2.8.0+cu128', '2.11.0', False),
    ('2.11.0rc1', '2.11.0', False),
])
def test_runtime_version_comparison_handles_missing_build_metadata(loaded, installed, expected):
    assert logger._runtime_version_matches(loaded, installed) is expected


def test_non_binary_dependency_check_does_not_touch_torch(monkeypatch):
    mock_metadata(monkeypatch, {'packaging': '25.0'})
    logger.verify_dependencies(['packaging>=23.0'])


def test_actual_torch_and_torchvision_runtime_check():
    logger.verify_dependencies(['torch>=2.8.0', 'torchvision>=0.23.0'])
