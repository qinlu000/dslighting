from __future__ import annotations

import socket

import pytest

from dslighting.utils import host_overrides


@pytest.fixture
def isolated_host_overrides(monkeypatch: pytest.MonkeyPatch):
    observed: list[tuple[object, ...]] = []

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        observed.append((host, port, family, type, proto, flags))
        return [(family, type, proto, "", (str(host), int(port)))]

    monkeypatch.setattr(host_overrides, "_original_getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(host_overrides, "_host_overrides", {})
    monkeypatch.setattr(host_overrides, "_installed", False)
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setenv("no_proxy", "localhost")
    return observed


def test_activate_api_host_overrides_maps_only_target_and_bypasses_proxy(
    isolated_host_overrides,
) -> None:
    host_overrides.activate_api_host_overrides(
        '{"Private-API.Example.edu.": "10.20.30.40"}'
    )

    socket.getaddrinfo("private-api.example.edu", 443)
    socket.getaddrinfo("public.example.edu", 443)

    assert isolated_host_overrides[0][0] == "10.20.30.40"
    assert isolated_host_overrides[1][0] == "public.example.edu"
    assert "private-api.example.edu" in host_overrides.os.environ["NO_PROXY"].split(",")
    assert "private-api.example.edu" in host_overrides.os.environ["no_proxy"].split(",")


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        "{}",
        '{"private-api.example.edu": "not-an-ip"}',
        '{"bad host": "10.20.30.40"}',
    ],
)
def test_activate_api_host_overrides_rejects_invalid_configuration(
    raw: str,
    isolated_host_overrides,
) -> None:
    with pytest.raises(ValueError):
        host_overrides.activate_api_host_overrides(raw)


def test_activate_api_host_overrides_is_idempotent(isolated_host_overrides) -> None:
    raw = '{"private-api.example.edu": "10.20.30.40"}'
    assert host_overrides.activate_api_host_overrides(raw) == {
        "private-api.example.edu": "10.20.30.40"
    }
    assert host_overrides.activate_api_host_overrides(raw) == {
        "private-api.example.edu": "10.20.30.40"
    }

    socket.getaddrinfo("private-api.example.edu.", 443)
    assert isolated_host_overrides[-1][0] == "10.20.30.40"
