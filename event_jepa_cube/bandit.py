"""Contextual bandit client for adaptive decision-making.

Wraps the Mycelia bandit API to provide policy lifecycle management,
arm management with embedding-based semantics, online arm selection,
reward feedback for closed-loop learning, and offline batch training.

Use cases in jcube:

- Select which cascade level to prioritise
- Choose prediction model based on sequence characteristics
- Adaptive alert threshold tuning

Zero required dependencies (uses ``urllib`` from stdlib).

Example::

    client = BanditClient("https://api.getjai.com", api_key="...")
    client.create_policy("my_policy", strategy="linucb")
    client.add_arms("my_policy", [{"id": "arm1", "embedding": [0.1, 0.2]}])
    selections = client.select("my_policy", contexts=[[0.5, 0.6]], k=1)
    client.reward("my_policy", arm_id="arm1", context=[0.5, 0.6], reward=1.0)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class BanditError(Exception):
    """Error from the Mycelia bandit API."""


class BanditClient:
    """Contextual bandit client for adaptive model/level selection.

    Wraps the Mycelia bandit API to provide:

    - Policy lifecycle management
    - Arm management with embedding-based semantics
    - Online arm selection given context vectors
    - Reward feedback for closed-loop learning
    - Offline batch training from logged interactions

    Use cases in jcube:

    - Select which cascade level to prioritise
    - Choose prediction model based on sequence characteristics
    - Adaptive alert threshold tuning

    Args:
        base_url: Mycelia API base URL (e.g. ``"https://api.getjai.com"``).
        api_key: API key or Bearer token for authentication.
        namespace: Optional namespace scope for multi-tenant deployments.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        namespace: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._namespace = namespace
        self._timeout = timeout

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._namespace:
            headers["X-Namespace"] = self._namespace
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        """Make an HTTP request to the Mycelia bandit API."""
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 204:
                    return None
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise BanditError(f"HTTP {e.code} on {method} {path}: {body_text}") from e
        except urllib.error.URLError as e:
            raise BanditError(f"Connection error on {method} {path}: {e.reason}") from e

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, body)

    def _delete(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("DELETE", path, body)

    # ------------------------------------------------------------------
    # Policy lifecycle
    # ------------------------------------------------------------------

    def create_policy(self, name: str, strategy: str = "linucb", **kwargs: Any) -> dict[str, Any]:
        """Create a new bandit policy.

        Args:
            name: Unique policy name.
            strategy: Algorithm to use — ``"linucb"``, ``"thompson"``,
                or ``"neural"``.
            **kwargs: Additional strategy-specific parameters forwarded
                to the API (e.g. ``alpha``, ``exploration_rate``).

        Returns:
            Policy details dict from the API.
        """
        body: dict[str, Any] = {"name": name, "strategy": strategy, **kwargs}
        result = self._post("/v2/bandits", body)
        logger.info("Created bandit policy %r (strategy=%s)", name, strategy)
        return result

    def get_policy(self, name: str) -> dict[str, Any]:
        """Get policy details.

        Args:
            name: Policy name.

        Returns:
            Policy details dict.
        """
        return self._get(f"/v2/bandits/{name}")

    def list_policies(self) -> list[Any]:
        """List all bandit policies.

        Returns:
            List of policy summary dicts.
        """
        return self._get("/v2/bandits")

    def delete_policy(self, name: str) -> None:
        """Delete a policy and all its data.

        Args:
            name: Policy name.
        """
        self._delete(f"/v2/bandits/{name}")
        logger.info("Deleted bandit policy %r", name)

    # ------------------------------------------------------------------
    # Arm management
    # ------------------------------------------------------------------

    def add_arms(self, policy: str, arms: list[dict[str, Any]]) -> dict[str, Any]:
        """Add arms to a policy.

        Each arm dict should contain at least an ``id`` key, plus either
        an ``embedding`` vector or ``data`` for server-side embedding.

        Args:
            policy: Policy name.
            arms: List of arm dicts, e.g.
                ``[{"id": "a1", "embedding": [0.1, 0.2]}]``.

        Returns:
            Result dict with insertion details.
        """
        return self._post(f"/v2/bandits/{policy}/arms", {"arms": arms})

    def refresh_arms(self, policy: str) -> dict[str, Any]:
        """Idempotently re-embed and upsert arm semantics.

        Triggers server-side re-encoding of all arm data.

        Args:
            policy: Policy name.

        Returns:
            Refresh result dict.
        """
        return self._post(f"/v2/bandits/{policy}/arms/refresh")

    def remove_arm(self, policy: str, arm_id: str) -> None:
        """Remove an arm from a policy.

        Args:
            policy: Policy name.
            arm_id: Arm identifier to remove.
        """
        self._delete(f"/v2/bandits/{policy}/arms/{arm_id}")

    # ------------------------------------------------------------------
    # Online operations
    # ------------------------------------------------------------------

    def select(self, policy: str, contexts: list[list[float]], k: int = 1) -> list[Any]:
        """Select best arm(s) for given context embeddings.

        Args:
            policy: Policy name.
            contexts: List of context embedding vectors, one per query.
            k: Number of arms to return per context.

        Returns:
            List of selections per context, each containing
            ``arm_id``, ``score``, and ``rank``.
        """
        body: dict[str, Any] = {"contexts": contexts, "k": k}
        result = self._post(f"/v2/bandits/{policy}/select", body)
        return result.get("selections", result) if isinstance(result, dict) else result

    def reward(self, policy: str, arm_id: str, context: list[float], reward: float) -> dict[str, Any]:
        """Record reward feedback for an arm-context interaction.

        Args:
            policy: Policy name.
            arm_id: The arm that was selected.
            context: The context vector used during selection.
            reward: Observed reward value (higher is better).

        Returns:
            Acknowledgement dict from the API.
        """
        body: dict[str, Any] = {
            "arm_id": arm_id,
            "context": context,
            "reward": reward,
        }
        return self._post(f"/v2/bandits/{policy}/reward", body)

    def diagnostics(self, policy: str) -> dict[str, Any]:
        """Get policy diagnostics, arm statistics, and reward history.

        Args:
            policy: Policy name.

        Returns:
            Dict with ``stats``, ``arm_details``, ``reward_history``, etc.
        """
        return self._get(f"/v2/bandits/{policy}/diagnostics")

    # ------------------------------------------------------------------
    # Offline training
    # ------------------------------------------------------------------

    def train_offline(
        self,
        policy: str,
        data_source: str,
        importance_sampling: bool = False,
    ) -> dict[str, Any]:
        """Run offline batch training from logged interactions.

        Accepts a Parquet data source URI for counterfactual training.

        Args:
            policy: Policy name.
            data_source: URI or path to logged interaction data (Parquet).
            importance_sampling: Whether to apply importance-weighted
                corrections for off-policy data.

        Returns:
            Dict with ``task_id`` for async progress tracking.
        """
        body: dict[str, Any] = {
            "data_source": data_source,
            "importance_sampling": importance_sampling,
        }
        result = self._post(f"/v2/bandits/{policy}/train", body)
        logger.info("Started offline training for policy %r (task=%s)", policy, result.get("task_id"))
        return result


class CascadeBandit:
    """Bandit-powered adaptive cascade level selection.

    Wraps a :class:`BanditClient` and a :class:`ForecastCascade` to
    adaptively decide which cascade levels to prioritise based on
    observed outcomes.

    Each cascade level becomes a bandit arm.  Context is derived from
    the current sequence's representation.  Rewards come from prediction
    accuracy or alert relevance.

    Example::

        bandit = BanditClient("https://api.getjai.com", api_key="...")
        cb = CascadeBandit(bandit, policy_name="my_cascade")
        cb.setup_from_cascade(cascade)
        levels = cb.select_levels(context=[0.1, 0.2, 0.3], k=2)
        cb.record_outcome("patient", context=[0.1, 0.2, 0.3], reward=0.9)
    """

    def __init__(self, bandit: BanditClient, policy_name: str = "cascade_selector") -> None:
        self.bandit = bandit
        self.policy_name = policy_name

    def setup_from_cascade(self, cascade: Any) -> None:
        """Initialise bandit arms from cascade levels.

        Creates one arm per level, using the level name and its index as
        features.  The policy is created if it doesn't already exist.

        Args:
            cascade: A :class:`ForecastCascade` instance with a
                ``levels`` attribute listing level names.
        """
        try:
            self.bandit.get_policy(self.policy_name)
        except BanditError:
            self.bandit.create_policy(self.policy_name, strategy="linucb")

        arms = [{"id": name, "data": {"level_name": name, "index": i}} for i, name in enumerate(cascade.levels)]
        self.bandit.add_arms(self.policy_name, arms)
        logger.info("Set up %d cascade arms for policy %r", len(arms), self.policy_name)

    def select_levels(self, context: list[float], k: int = 2) -> list[str]:
        """Select top-k cascade levels to prioritise for a given context.

        Args:
            context: Sequence representation vector.
            k: Number of levels to select.

        Returns:
            List of level names, ordered by bandit score (best first).
        """
        selections = self.bandit.select(self.policy_name, contexts=[context], k=k)
        if not selections:
            return []
        # The API returns a list of selections per context; take the first.
        per_context = selections[0] if isinstance(selections[0], list) else selections
        return [s["arm_id"] for s in per_context]

    def record_outcome(self, level: str, context: list[float], reward: float) -> None:
        """Record prediction/alert outcome as reward signal.

        Args:
            level: The cascade level name (arm id).
            context: The context vector used during selection.
            reward: Observed outcome quality (higher is better).
        """
        self.bandit.reward(self.policy_name, arm_id=level, context=context, reward=reward)
