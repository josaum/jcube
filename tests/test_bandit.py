"""Tests for BanditClient and CascadeBandit.

All HTTP calls are mocked via ``unittest.mock.patch`` so no live
Mycelia instance is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from event_jepa_cube.bandit import BanditClient, BanditError, CascadeBandit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return BanditClient(
        base_url="https://test.mycelia.local",
        api_key="test-key",
        namespace="test-ns",
    )


def _mock_response(body: dict | list | None = None, status: int = 200):
    """Create a mock urllib response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode() if body is not None else b""
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_headers_with_auth_and_namespace(self, client):
        h = client._headers()
        assert h["Authorization"] == "Bearer test-key"
        assert h["X-Namespace"] == "test-ns"
        assert h["Content-Type"] == "application/json"

    def test_headers_without_auth(self):
        c = BanditClient("https://x.local")
        h = c._headers()
        assert "Authorization" not in h
        assert "X-Namespace" not in h


# ---------------------------------------------------------------------------
# Policy lifecycle
# ---------------------------------------------------------------------------


class TestPolicyLifecycle:
    @patch("urllib.request.urlopen")
    def test_create_policy(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"name": "p1", "strategy": "linucb"})
        result = client.create_policy("p1", strategy="linucb", alpha=0.5)
        assert result["name"] == "p1"
        assert result["strategy"] == "linucb"

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["name"] == "p1"
        assert body["strategy"] == "linucb"
        assert body["alpha"] == 0.5

    @patch("urllib.request.urlopen")
    def test_get_policy(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"name": "p1", "strategy": "thompson", "arms": 5})
        result = client.get_policy("p1")
        assert result["name"] == "p1"
        assert result["arms"] == 5

    @patch("urllib.request.urlopen")
    def test_list_policies(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response([{"name": "p1"}, {"name": "p2"}])
        result = client.list_policies()
        assert len(result) == 2

    @patch("urllib.request.urlopen")
    def test_delete_policy(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response(status=204)
        client.delete_policy("p1")
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "DELETE"
        assert "/v2/bandits/p1" in req.full_url


# ---------------------------------------------------------------------------
# Arm management
# ---------------------------------------------------------------------------


class TestArmManagement:
    @patch("urllib.request.urlopen")
    def test_add_arms(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"added": 2})
        arms = [
            {"id": "a1", "embedding": [0.1, 0.2]},
            {"id": "a2", "data": {"text": "hello"}},
        ]
        result = client.add_arms("p1", arms)
        assert result["added"] == 2

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert len(body["arms"]) == 2
        assert body["arms"][0]["id"] == "a1"

    @patch("urllib.request.urlopen")
    def test_refresh_arms(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"refreshed": 3})
        result = client.refresh_arms("p1")
        assert result["refreshed"] == 3

    @patch("urllib.request.urlopen")
    def test_remove_arm(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response(status=204)
        client.remove_arm("p1", "a1")
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "DELETE"
        assert "/v2/bandits/p1/arms/a1" in req.full_url


# ---------------------------------------------------------------------------
# Online operations
# ---------------------------------------------------------------------------


class TestOnlineOperations:
    @patch("urllib.request.urlopen")
    def test_select(self, mock_urlopen, client):
        selections = [
            [{"arm_id": "a1", "score": 0.9, "rank": 1}],
        ]
        mock_urlopen.return_value = _mock_response({"selections": selections})
        result = client.select("p1", contexts=[[0.5, 0.6]], k=1)
        assert len(result) == 1
        assert result[0][0]["arm_id"] == "a1"

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["contexts"] == [[0.5, 0.6]]
        assert body["k"] == 1

    @patch("urllib.request.urlopen")
    def test_select_multiple_contexts(self, mock_urlopen, client):
        selections = [
            [{"arm_id": "a1", "score": 0.9, "rank": 1}],
            [{"arm_id": "a2", "score": 0.8, "rank": 1}],
        ]
        mock_urlopen.return_value = _mock_response({"selections": selections})
        result = client.select("p1", contexts=[[0.1, 0.2], [0.3, 0.4]], k=1)
        assert len(result) == 2

    @patch("urllib.request.urlopen")
    def test_reward(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"status": "ok"})
        result = client.reward("p1", arm_id="a1", context=[0.5, 0.6], reward=1.0)
        assert result["status"] == "ok"

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["arm_id"] == "a1"
        assert body["context"] == [0.5, 0.6]
        assert body["reward"] == 1.0

    @patch("urllib.request.urlopen")
    def test_diagnostics(self, mock_urlopen, client):
        diag = {"total_rewards": 100, "avg_reward": 0.75, "arm_details": []}
        mock_urlopen.return_value = _mock_response(diag)
        result = client.diagnostics("p1")
        assert result["total_rewards"] == 100
        assert result["avg_reward"] == 0.75


# ---------------------------------------------------------------------------
# Offline training
# ---------------------------------------------------------------------------


class TestOfflineTraining:
    @patch("urllib.request.urlopen")
    def test_train_offline(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"task_id": "train-123"})
        result = client.train_offline("p1", data_source="s3://bucket/logs.parquet")
        assert result["task_id"] == "train-123"

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["data_source"] == "s3://bucket/logs.parquet"
        assert body["importance_sampling"] is False

    @patch("urllib.request.urlopen")
    def test_train_offline_with_importance_sampling(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response({"task_id": "train-456"})
        result = client.train_offline("p1", data_source="data.parquet", importance_sampling=True)
        assert result["task_id"] == "train-456"

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["importance_sampling"] is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @patch("urllib.request.urlopen")
    def test_http_error_raises_bandit_error(self, mock_urlopen, client):
        import io
        import urllib.error

        err_body = io.BytesIO(b'{"detail": "policy not found"}')
        mock_urlopen.side_effect = urllib.error.HTTPError("https://x", 404, "Not Found", {}, err_body)
        with pytest.raises(BanditError, match="HTTP 404"):
            client.get_policy("missing")

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_bandit_error(self, mock_urlopen, client):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(BanditError, match="Connection error"):
            client.list_policies()

    @patch("urllib.request.urlopen")
    def test_204_returns_none(self, mock_urlopen, client):
        mock_urlopen.return_value = _mock_response(status=204)
        result = client.delete_policy("p1")
        assert result is None


# ---------------------------------------------------------------------------
# CascadeBandit
# ---------------------------------------------------------------------------


class TestCascadeBandit:
    @pytest.fixture
    def cascade_bandit(self, client):
        return CascadeBandit(client, policy_name="test_cascade")

    @pytest.fixture
    def mock_cascade(self):
        cascade = MagicMock()
        cascade.levels = ["patient", "department", "financial"]
        return cascade

    @patch("urllib.request.urlopen")
    def test_setup_from_cascade_creates_policy(self, mock_urlopen, cascade_bandit, mock_cascade):
        # get_policy raises BanditError (404), then create_policy, then add_arms
        import io
        import urllib.error

        err_body = io.BytesIO(b'{"detail": "not found"}')
        get_err = urllib.error.HTTPError("https://x", 404, "Not Found", {}, err_body)
        create_resp = _mock_response({"name": "test_cascade", "strategy": "linucb"})
        add_resp = _mock_response({"added": 3})

        mock_urlopen.side_effect = [get_err, create_resp, add_resp]
        cascade_bandit.setup_from_cascade(mock_cascade)

        # Verify add_arms was called with correct arms
        last_req = mock_urlopen.call_args_list[-1][0][0]
        body = json.loads(last_req.data)
        assert len(body["arms"]) == 3
        assert body["arms"][0]["id"] == "patient"
        assert body["arms"][1]["id"] == "department"
        assert body["arms"][2]["id"] == "financial"
        assert body["arms"][0]["data"]["index"] == 0

    @patch("urllib.request.urlopen")
    def test_setup_from_cascade_policy_exists(self, mock_urlopen, cascade_bandit, mock_cascade):
        # get_policy succeeds (policy exists), then add_arms
        get_resp = _mock_response({"name": "test_cascade", "strategy": "linucb"})
        add_resp = _mock_response({"added": 3})

        mock_urlopen.side_effect = [get_resp, add_resp]
        cascade_bandit.setup_from_cascade(mock_cascade)

        # Should only have 2 calls (get + add_arms), no create
        assert mock_urlopen.call_count == 2

    @patch("urllib.request.urlopen")
    def test_select_levels(self, mock_urlopen, cascade_bandit):
        selections = [
            [
                {"arm_id": "department", "score": 0.9, "rank": 1},
                {"arm_id": "patient", "score": 0.7, "rank": 2},
            ]
        ]
        mock_urlopen.return_value = _mock_response({"selections": selections})
        levels = cascade_bandit.select_levels(context=[0.1, 0.2, 0.3], k=2)
        assert levels == ["department", "patient"]

    @patch("urllib.request.urlopen")
    def test_select_levels_empty(self, mock_urlopen, cascade_bandit):
        mock_urlopen.return_value = _mock_response({"selections": []})
        levels = cascade_bandit.select_levels(context=[0.1, 0.2, 0.3], k=2)
        assert levels == []

    @patch("urllib.request.urlopen")
    def test_record_outcome(self, mock_urlopen, cascade_bandit):
        mock_urlopen.return_value = _mock_response({"status": "ok"})
        cascade_bandit.record_outcome("patient", context=[0.1, 0.2], reward=0.85)

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["arm_id"] == "patient"
        assert body["context"] == [0.1, 0.2]
        assert body["reward"] == 0.85
