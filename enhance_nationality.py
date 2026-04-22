import json
import re
from typing import Any, Dict, Optional, Tuple


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

PHONE_PREFIX_MAP = {
    '92': 'Pakistan',
    '91': 'India',
    '880': 'Bangladesh',
    '977': 'Nepal',
    '94': 'Sri Lanka',
    '971': 'UAE',
    '966': 'Saudi Arabia',
    '974': 'Qatar',
    '965': 'Kuwait',
    '968': 'Oman',
    '973': 'Bahrain',
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


def _detect_phone_country(cv_data: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
    phone = _clean_text(cv_data.get('phone'))
    if not phone:
        return None

    digits = re.sub(r'\D', '', phone)
    digits = digits.lstrip('0')
    for prefix in sorted(PHONE_PREFIX_MAP.keys(), key=len, reverse=True):
        if digits.startswith(prefix):
            return PHONE_PREFIX_MAP[prefix], 'phone_country_code', 0.75
    return None


def infer_nationality_from_cv_data(cv_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float]:
    explicit = _normalize_explicit_nationality(cv_data.get('nationality'))
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
