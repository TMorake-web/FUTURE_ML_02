"""Simple, explainable High/Medium/Low ticket priority rules.

The Zenodo dataset has category labels but no priority labels. These rules are
therefore a transparent prototype, not a trained or evaluated priority model.
"""

import re


# These phrases point to severe service-wide or security problems.
HIGH_PATTERNS = [
    r"\b(outage|service down|system down|production down)\b",
    r"\b(security breach|data breach|account compromised|hacked)\b",
    r"\b(emergency|urgent|critical)\b",
    r"\b(data loss|many users affected|business stopped)\b",
]

# These phrases describe problems that need attention but are not automatically
# treated as emergencies. Anything not matching either list defaults to Low.
MEDIUM_PATTERNS = [
    r"\b(error|not working|does not work|doesn't work|failed|failure)\b",
    r"\b(unable to access|cannot access|can't access|access denied)\b",
    r"\b(problem|issue|broken|cannot log in|can't log in)\b",
]


def predict_priority(text: str) -> str:
    """Return a rule-based priority; high severity terms take precedence."""
    normalized = text.lower()
    # Check High first so an urgent outage is not reduced to Medium by "error".
    if any(re.search(pattern, normalized) for pattern in HIGH_PATTERNS):
        return "High"
    if any(re.search(pattern, normalized) for pattern in MEDIUM_PATTERNS):
        return "Medium"
    return "Low"
