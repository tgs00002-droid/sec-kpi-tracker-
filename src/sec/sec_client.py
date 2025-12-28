import time
from typing import Dict, Any

import requests
from requests.adapters import HTTPAdapter, Retry


class SECClient:
    """
    SEC-friendly client with retries + throttling.
    SEC prefers a descriptive User-Agent with contact info.
    """
    def __init__(self, user_agent: str, throttle_s: float = 0.35):
        self.user_agent = user_agent
        self.throttle_s = throttle_s
        self.session = requests.Session()

        retries = Retry(
            total=6,
            backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def get_json(self, url: str) -> Dict[str, Any]:
        time.sleep(self.throttle_s)
        r = self.session.get(url, timeout=30)

        if r.status_code == 403:
            raise RuntimeError(
                "SEC returned 403. Make sure your User-Agent includes a real email."
            )
        if r.status_code == 429:
            raise RuntimeError(
                "SEC returned 429. Increase throttle (try 0.75–1.25s) and refresh less."
            )
        if r.status_code != 200:
            raise RuntimeError(f"SEC request failed ({r.status_code}): {r.text[:250]}")

        return r.json()
