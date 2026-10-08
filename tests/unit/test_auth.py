"""Unit tests for authentication functions."""

import subprocess
from unittest.mock import Mock, patch

import pytest
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from dell_ai.auth import check_model_access, login_docker
from dell_ai.exceptions import (
    AuthenticationError,
    GatedRepoAccessError,
    ResourceNotFoundError,
)


def _mock_hf_response():
    """Create a minimal mock response object for huggingface_hub exceptions."""
    response = Mock()
    response.headers = {}
    return response


@patch("dell_ai.auth.shutil.which", return_value="/usr/bin/docker")
@patch("dell_ai.auth.subprocess.run")
def test_docker_login_uses_stdin(mock_run, mock_which):
    token = "hf_docker_test_token"
    assert login_docker(token, "test-user") is True
    mock_run.assert_called_once_with(
        ["docker", "login", "cr.hf.co", "--username", "test-user", "--password-stdin"],
        input=f"{token}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert token not in " ".join(mock_run.call_args.args[0])
    # Registry login is a client operation; no daemon or image pull is invoked.
    mock_which.assert_called_once_with("docker")


@patch("dell_ai.auth.shutil.which", return_value=None)
@patch("dell_ai.auth.subprocess.run")
def test_docker_login_skips_missing_cli(mock_run, mock_which):
    assert login_docker("hf_test", "test-user") is False
    mock_run.assert_not_called()


@pytest.mark.parametrize("failure", ["stderr", "stdout", "os_error"])
@patch("dell_ai.auth.shutil.which", return_value="/usr/bin/docker")
@patch("dell_ai.auth.subprocess.run")
def test_docker_login_errors_redact_token(mock_run, mock_which, failure):
    token = "hf_docker_secret"
    detail = f"Credential helper rejected {token}"
    if failure == "os_error":
        mock_run.side_effect = OSError(detail)
    else:
        mock_run.side_effect = subprocess.CalledProcessError(
            1,
            ["docker", "login"],
            stderr=detail if failure == "stderr" else None,
            output=detail if failure == "stdout" else None,
        )
    with pytest.raises(AuthenticationError) as error:
        login_docker(token, "test-user")
    assert "HF token is saved" in str(error.value)
    assert "cr.hf.co" in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert token not in str(error.value)


def test_check_model_access_success():
    """Test successful model access check."""
    with patch("dell_ai.auth.hf_auth_check") as mock_auth_check:
        mock_auth_check.return_value = True  # Access is granted

        result = check_model_access("test-org/accessible-model", token="test-token")

        assert result is True
        mock_auth_check.assert_called_once_with(
            repo_id="test-org/accessible-model", token="test-token"
        )


def test_check_model_access_no_token():
    """Test model access check without a token."""
    with patch("dell_ai.auth.get_token") as mock_get_token:
        mock_get_token.return_value = None

        with pytest.raises(AuthenticationError) as exc_info:
            check_model_access("test-org/some-model")

        assert "No authentication token found" in str(exc_info.value)


def test_check_model_access_gated_repo():
    """Test model access check for a gated repository."""
    with patch("dell_ai.auth.hf_auth_check") as mock_auth_check:
        # Simulate a GatedRepoError from huggingface_hub
        mock_auth_check.side_effect = GatedRepoError(
            "Access denied to gated repo", response=_mock_hf_response()
        )

        with pytest.raises(GatedRepoAccessError) as exc_info:
            check_model_access("meta-llama/gated-model", token="test-token")

        error = exc_info.value
        assert error.model_id == "meta-llama/gated-model"
        assert "Access denied" in str(error)
        assert "https://huggingface.co/meta-llama/gated-model" in str(error)


def test_check_model_access_nonexistent_repo():
    """Test model access check for a nonexistent repository."""
    with patch("dell_ai.auth.hf_auth_check") as mock_auth_check:
        # Simulate a RepositoryNotFoundError from huggingface_hub
        mock_auth_check.side_effect = RepositoryNotFoundError(
            "Repository not found", response=_mock_hf_response()
        )

        with pytest.raises(ResourceNotFoundError) as exc_info:
            check_model_access("test-org/nonexistent-model", token="test-token")

        error = exc_info.value
        assert error.resource_type == "model"
        assert error.resource_id == "test-org/nonexistent-model"


def test_check_model_access_other_error():
    """Test model access check with an unexpected error."""
    with patch("dell_ai.auth.hf_auth_check") as mock_auth_check:
        # Simulate a generic error
        mock_auth_check.side_effect = Exception("Network error")

        with pytest.raises(AuthenticationError) as exc_info:
            check_model_access("test-org/some-model", token="test-token")

        assert "Failed to check model access" in str(exc_info.value)
