import json
import re
from typing import Any, Dict, Optional, Tuple

from cv_field_cleanup import normalize_phone_e164


PLACEHOLDER_VALUES = {'', 'missing', 'null', 'undefined', 'n/a', 'na', 'none', 'not provided'}

EXPLICIT_NATIONALITY_MAP = {
    'pakistan': 'Pakistan',
    'pakistani': 'Pakistan',
    'india': 'India',
    'indian': 'India',
    'bangladesh': 'Bangladesh',
    'bangladeshi': 'Bangladesh',
    'nepal': 'Nepal',
    'nepali': 'Nepal',
    'sri lanka': 'Sri Lanka',
    'sri lankan': 'Sri Lanka',
    'uae': 'UAE',
    'united arab emirates': 'UAE',
    'saudi arabia': 'Saudi Arabia',
    'saudi': 'Saudi Arabia',
    'qatar': 'Qatar',
    'kuwait': 'Kuwait',
    'oman': 'Oman',
    'bahrain': 'Bahrain',
}

PASSPORT_PREFIX_MAP = {
    'PA': 'Pakistan',
    'AB': 'Pakistan',
    'IN': 'India',
    'BD': 'Bangladesh',
    'NP': 'Nepal',
    'LK': 'Sri Lanka',
    'GB': 'United Kingdom',
}

# Only home countries of migrant workers. A Gulf number (+966, +971, ...) shows
# where someone works, not their nationality: Pakistani workers in Saudi Arabia
# use Saudi SIMs.
PHONE_PREFIX_MAP = {
    '92': 'Pakistan',
    '91': 'India',
    '880': 'Bangladesh',
    '977': 'Nepal',
    '94': 'Sri Lanka',
}

GCC_NATIONALITIES = {'UAE', 'Saudi Arabia', 'Qatar', 'Kuwait', 'Oman', 'Bahrain'}

# Words that show the CV is from someone living/studying/working in that country.
# A phone code counts only together with one of these.
COUNTRY_MENTION_TERMS = {
    'Pakistan': ('pakistan', 'pakistani', 'karachi', 'lahore', 'islamabad', 'rawalpindi', 'peshawar',
                 'faisalabad', 'multan', 'quetta', 'sialkot', 'gujranwala', 'hyderabad', 'punjab', 'sindh',
                 'khyber pakhtunkhwa', 'balochistan', 'azad kashmir'),
    'India': ('india', 'indian', 'delhi', 'mumbai', 'chennai', 'kerala', 'hyderabad', 'bangalore',
              'bengaluru', 'kolkata', 'punjab'),
    'Bangladesh': ('bangladesh', 'bangladeshi', 'dhaka', 'chittagong'),
    'Nepal': ('nepal', 'nepali', 'nepalese', 'kathmandu'),
    'Sri Lanka': ('sri lanka', 'sri lankan', 'colombo'),
}


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in PLACEHOLDER_VALUES:
        return None
    return text


def _ensure_confidence_dict(cv_data: Dict[str, Any]) -> Dict[str, Any]:
    confidence = cv_data.get('extraction_confidence')
    if not isinstance(confidence, dict):
        confidence = {}
        cv_data['extraction_confidence'] = confidence
    return confidence


def _normalize_explicit_nationality(value: Any) -> Optional[str]:
    text = _clean_text(value)
    if not text:
        return None
    return EXPLICIT_NATIONALITY_MAP.get(text.lower(), text)


def _detect_cnic_country(cv_data: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
    cnic = _clean_text(cv_data.get('cnic') or cv_data.get('cnic_normalized') or cv_data.get('document_number'))
    if not cnic:
        return None
    digits = re.sub(r'\D', '', cnic)
    if len(digits) == 13:
        return 'Pakistan', 'cnic', 0.98
    return None


def _detect_passport_country(cv_data: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
    passport = _clean_text(cv_data.get('passport') or cv_data.get('passport_no'))
    if not passport:
        return None
    normalized = re.sub(r'[^A-Za-z0-9]', '', passport).upper()
    if len(normalized) < 2:
        return None
    country = PASSPORT_PREFIX_MAP.get(normalized[:2])
    if country:
        return country, 'passport_prefix', 0.95
    return None


def _cv_text(cv_data: Dict[str, Any]) -> str:
    parts = [cv_data.get('raw_text'), cv_data.get('location')]
    for key in ('experience', 'education'):
        for entry in cv_data.get(key) or []:
            if isinstance(entry, dict):
                parts.extend([entry.get('location'), entry.get('company'), entry.get('institution')])
    return ' '.join(p for p in parts if isinstance(p, str)).lower()


def _mentions_country(country: str, text: str) -> bool:
    return any(re.search(r'\b' + re.escape(term) + r'\b', text) for term in COUNTRY_MENTION_TERMS.get(country, ()))


def _detect_phone_country(cv_data: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
    """
    A phone country code alone is not evidence of nationality. Use it only for
    the home countries in PHONE_PREFIX_MAP, and only when the CV also mentions
    that country (address, schooling or employers).
    """
    phones = [cv_data.get('phone')] + list(cv_data.get('additional_phones') or [])
    text = None
    for phone in phones:
        e164 = normalize_phone_e164(_clean_text(phone))
        if not e164:
            continue
        digits = e164[1:]
        for prefix in sorted(PHONE_PREFIX_MAP.keys(), key=len, reverse=True):
            if digits.startswith(prefix):
                country = PHONE_PREFIX_MAP[prefix]
                text = _cv_text(cv_data) if text is None else text
                if _mentions_country(country, text):
                    return country, 'phone_country_code+cv_location', 0.8
                break
    return None


def _states_nationality(cv_data: Dict[str, Any], country: str) -> bool:
    """True when the raw text has a 'Nationality: <country>' style statement, or there is no text to check."""
    raw_text = cv_data.get('raw_text')
    if not isinstance(raw_text, str) or not raw_text.strip():
        return True
    words = [alias for alias, name in EXPLICIT_NATIONALITY_MAP.items() if name == country]
    label = r'(?:nationality|citizenship|citizen)\W{0,30}'
    return any(re.search(label + re.escape(word) + r'\b', raw_text, re.I) for word in words)


def infer_nationality_from_cv_data(cv_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float]:
    explicit = _normalize_explicit_nationality(cv_data.get('nationality'))
    # The LLM sometimes turns a Saudi/UAE address or phone into a Gulf nationality.
    if explicit in GCC_NATIONALITIES and not _states_nationality(cv_data, explicit):
        explicit = None
    if explicit:
        return explicit, 'explicit', 0.99

    for detector in (_detect_cnic_country, _detect_passport_country, _detect_phone_country):
        result = detector(cv_data)
        if result:
            return result

    return None, None, 0.0


def enhance_nationality_with_ai(cv_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kept for compatibility with existing call sites.
    Nationality is now resolved deterministically only.
    """
    inferred_nationality, source, confidence = infer_nationality_from_cv_data(cv_data)
    if not inferred_nationality:
        # Drop an unverified value (e.g. a Gulf nationality the CV never states).
        if _clean_text(cv_data.get('nationality')):
            cv_data['nationality'] = None
        return cv_data

    cv_data['nationality'] = inferred_nationality
    cv_data['nationality_inferred_from'] = source
    _ensure_confidence_dict(cv_data)['nationality'] = confidence
    return cv_data


def main():
    test_cv = {
        'nationality': None,
        'cnic': '12345-1234567-1',
        'passport': None,
        'phone': '+92 300 1234567',
        'extraction_confidence': {'nationality': 0.2},
    }

    enhanced = enhance_nationality_with_ai(test_cv)
    print(json.dumps(enhanced, indent=2))


if __name__ == '__main__':
    main()
