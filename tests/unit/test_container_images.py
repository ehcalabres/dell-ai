"""Registry routing, HF extension setup, and authentication during deployment."""

import json
import os
import shlex
import subprocess
from unittest.mock import Mock

import pytest

from dell_ai import DellAIClient, auth, container_images, deployments, env, resources
from dell_ai.exceptions import AuthenticationError

XET_IMAGE = "cr.hf.co/hf-rev/nvidia-nemotron-3-diarization-pr:dell-gpu-latest-pr261"
TOKEN = "hf_test_image_token"


@pytest.fixture
def isolated_auth(monkeypatch):
    monkeypatch.setattr(env, "load_all_env_to_os", lambda: None)
    monkeypatch.setattr(auth, "hf_get_token", lambda: None)
    monkeypatch.setattr(auth, "validate_token", lambda token: True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


@pytest.fixture
def commands(monkeypatch):
    mock_run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, stdout="container-id\n", stderr=""
        )
    )
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(container_images.shutil, "which", lambda name: f"/bin/{name}")
    return mock_run


@pytest.mark.parametrize("detach", [True, False])
@pytest.mark.parametrize("source", ["sdk", "cache", "env"])
def test_xet_pull_before_docker_run(
    isolated_auth, commands, monkeypatch, detach, source
):
    if source == "cache":
        monkeypatch.setattr(auth, "hf_get_token", lambda: TOKEN)
    elif source == "env":
        monkeypatch.setenv("HF_TOKEN", TOKEN)
    client = DellAIClient(token=TOKEN if source == "sdk" else None)
    result = client._execute_snippet(
        f'docker run -it -p 8080:80 -e HF_TOKEN=$$_TOKEN_$$ "{XET_IMAGE}"',
        detach=detach,
    )

    assert result["success"] is True
    assert result["engine"] == "docker"
    assert len(commands.call_args_list) == 3
    probe, pull, run = commands.call_args_list
    assert probe.args[0] == ["hf", "image", "pull", "--help"]
    assert pull.args[0] == ["hf", "image", "pull", XET_IMAGE]
    assert pull.kwargs["env"]["HF_TOKEN"] == TOKEN
    assert TOKEN not in pull.args[0]
    assert os.environ.get("HF_TOKEN") == (TOKEN if source == "env" else None)
    run_tokens = shlex.split(run.args[0])
    assert run_tokens[:3] == ["docker", "run", "--pull=never"]
    assert XET_IMAGE in run_tokens
    assert ("-d" in run_tokens) is detach
    assert ("-it" in run_tokens) is not detach
    assert TOKEN not in json.dumps(result)
    if detach:
        assert result["container_id"] == "container-id"
        assert result["endpoint"] == "http://localhost:8080"


def test_xet_requires_login_without_placeholder(isolated_auth, commands):
    with pytest.raises(AuthenticationError, match="dell-ai login"):
        DellAIClient()._execute_snippet(f"docker run {XET_IMAGE}")
    commands.assert_not_called()


@pytest.mark.parametrize(
    "image",
    [
        "registry.dell.huggingface.co/enterprise-dell-inference-test:latest",
        "ubuntu:latest",
    ],
)
def test_other_registries_keep_docker_behavior(isolated_auth, commands, image):
    result = DellAIClient()._execute_snippet(f"docker run {image}")
    assert result["success"] is True
    commands.assert_called_once()
    assert shlex.split(commands.call_args.args[0]) == ["docker", "run", "-d", image]


def test_missing_extension_installed_before_pull(isolated_auth, commands):
    commands.side_effect = [
        subprocess.CompletedProcess([], 2, stderr="No such command 'image'"),
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0, stdout="container-id\n", stderr=""),
    ]
    result = DellAIClient(token=TOKEN)._execute_snippet(f"docker run {XET_IMAGE}")
    assert result["success"] is True
    assert [call.args[0] for call in commands.call_args_list[:4]] == [
        ["hf", "image", "pull", "--help"],
        ["hf", "extensions", "install", "hf-image"],
        ["hf", "image", "pull", "--help"],
        ["hf", "image", "pull", XET_IMAGE],
    ]


def test_missing_hf_cli_reports_setup_instructions(
    isolated_auth, commands, monkeypatch
):
    monkeypatch.setattr(container_images.shutil, "which", lambda name: None)
    result = DellAIClient(token=TOKEN)._execute_snippet(f"docker run {XET_IMAGE}")
    assert result["success"] is False
    assert "Hugging Face CLI (hf)" in result["error"]
    assert "hf extensions install hf-image" in result["error"]
    commands.assert_not_called()


@pytest.mark.parametrize("stage", ["install", "verify", "pull"])
def test_hf_failures_stop_deployment_and_redact_tokens(isolated_auth, commands, stage):
    def execute(command, **kwargs):
        if command == ["hf", "image", "pull", "--help"]:
            if stage == "install":
                return subprocess.CompletedProcess(command, 2)
            if stage == "verify":
                if commands.call_count == 1:
                    return subprocess.CompletedProcess(command, 2)
                raise subprocess.CalledProcessError(
                    1, command, stderr=f"broken {TOKEN}"
                )
        if (stage == "install" and command[1] == "extensions") or (
            stage == "pull" and command[-1] == XET_IMAGE
        ):
            raise subprocess.CalledProcessError(1, command, stderr=f"denied {TOKEN}")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    commands.side_effect = execute
    result = DellAIClient(token=TOKEN)._execute_snippet(f"docker run {XET_IMAGE}")
    assert result["success"] is False
    assert TOKEN not in json.dumps(result)
    assert "$$_TOKEN_$$" in result["error"]
    assert all(isinstance(call.args[0], list) for call in commands.call_args_list)


@pytest.mark.parametrize("flag", ["--platform linux/arm64", "--platform=linux/arm64"])
@pytest.mark.parametrize("pull_flag", ["--pull always", "--pull=always"])
def test_xet_platform_and_pull_policy(isolated_auth, commands, flag, pull_flag):
    result = DellAIClient(token=TOKEN)._execute_snippet(
        f"docker run {flag} {pull_flag} {XET_IMAGE} --pull=application-option"
    )
    assert result["success"] is True
    assert commands.call_args_list[1].args[0] == [
        "hf",
        "image",
        "pull",
        XET_IMAGE,
        "--platform",
        "linux/arm64",
    ]
    tokens = shlex.split(commands.call_args.args[0])
    assert "--pull=never" in tokens
    assert "--pull=always" not in tokens
    assert "--pull=application-option" in tokens


@pytest.mark.parametrize("suffix", ["", ":old", "@sha256:1234"])
@pytest.mark.parametrize("engine", ["docker", "kubernetes"])
def test_tag_injection_supports_xet(suffix, engine):
    image = f"cr.hf.co/hf-rev/nvidia-nemotron-3-diarization-pr{suffix}"
    snippet = f"docker run {image}" if engine == "docker" else f'image: "{image}"'
    result = resources.inject_image_tag(snippet, "dell-gpu-latest-pr261")
    expected = (
        f"docker run {XET_IMAGE}" if engine == "docker" else f'image: "{XET_IMAGE}"'
    )
    assert result == expected


@pytest.mark.parametrize("mount", ["local", "cache"])
@pytest.mark.parametrize("quote", ["", '"', "'"])
def test_weight_mounts_support_xet(tmp_path, mount, quote):
    snippet = f"docker run -e MODEL_ID=org/model {quote}{XET_IMAGE}{quote}"
    if mount == "local":
        result = resources.inject_local_dir(snippet, str(tmp_path))
        assert f"-v {tmp_path}:/data" in result
        assert "MODEL_ID=/data" in result
    else:
        result = resources.inject_hf_cache_dir(snippet, str(tmp_path))
        assert f"-v {tmp_path}:/root/.cache/huggingface" in result
        assert "HF_HUB_CACHE=/root/.cache/huggingface" in result
        assert "MODEL_ID=org/model" in result
    assert XET_IMAGE in result
    assert XET_IMAGE in shlex.split(result)


def test_weight_mount_quotes_paths_with_spaces(tmp_path):
    weights = tmp_path / "my weights"
    snippet = resources.inject_local_dir(f'docker run "{XET_IMAGE}"', str(weights))
    tokens = shlex.split(snippet)
    assert tokens[tokens.index("-v") + 1] == f"{weights}:/data"
    assert tokens[-1] == XET_IMAGE


def test_discovery_recognizes_both_registries_and_preserves_tracked_xet(
    commands, tmp_path, monkeypatch
):
    local_path = tmp_path / "local.json"
    monkeypatch.setattr(deployments, "get_local_deployments_path", lambda: local_path)
    monkeypatch.setattr(
        deployments, "get_global_deployments_path", lambda: tmp_path / "global.json"
    )
    deployments.save_deployment(
        "tracked-model", {"engine": "docker", "container_id": "tracked-id"}
    )
    images = [
        ("legacy-id", "registry.dell.huggingface.co/enterprise-dell-inference-test"),
        ("xet-id", XET_IMAGE),
        ("tracked-id", "cr.hf.co/hf-rev/another-model:custom-tag"),
        ("unrelated-id", "cr.hf.co/infra-workloads/hf-image-helper:latest"),
    ]
    commands.return_value.stdout = "\n".join(
        json.dumps({"ID": cid, "Image": image, "Ports": "0.0.0.0:8080->80/tcp"})
        for cid, image in images
    )
    result = deployments.list_deployments()
    assert {meta["container_id"] for meta in result.values()} == {
        "legacy-id",
        "xet-id",
        "tracked-id",
    }
    assert "tracked-model" in result
    assert (
        result["hf-rev/nvidia-nemotron-3-diarization-pr"]["endpoint"]
        == "http://localhost:8080"
    )
