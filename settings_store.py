"""API keys and preferences saved locally in data/settings.json (never sent anywhere except the matching API)."""
import json

from config import DATA_DIR

PATH = DATA_DIR / "settings.json"

KEYS = {
    "openalex_key": ("OpenAlex", "Optional. Avoids rate limits on busy days.", "https://openalex.org/rest-api"),
    "semantic_scholar_key": ("Semantic Scholar", "Optional. Without it, Semantic Scholar is often rate limited.",
                             "https://www.semanticscholar.org/product/api#api-key-form"),
    "ieee_key": ("IEEE Xplore", "Free key. Searches IEEE Xplore directly (metadata + abstracts).",
                 "https://developer.ieee.org/"),
    "elsevier_key": ("Elsevier (Scopus + ScienceDirect)", "Free key. Full Scopus search needs your university network "
                     "or an institutional token.", "https://dev.elsevier.com/"),
    "elsevier_insttoken": ("Elsevier institutional token", "Optional. Ask your university library.", ""),
    "springer_key": ("Springer Nature", "Free key.", "https://dev.springernature.com/"),
    "core_key": ("CORE", "Free key. 30M+ open-access full texts.", "https://core.ac.uk/services/api"),
}


def load() -> dict:
    try:
        return json.loads(PATH.read_text())
    except (OSError, ValueError):
        return {}


def save(values: dict) -> None:
    PATH.write_text(json.dumps(values, indent=2))
    try:
        PATH.chmod(0o600)
    except OSError:
        pass


def get(key: str) -> str:
    return (load().get(key) or "").strip()
