"""
Deterministic clean-up of the fields the LLM extracts from a CV.

gpt-4o-mini reads the CV well but is unreliable on a few fields. An audit of 71
real CVs (2026-10-26) found it:
  - undercounting experience_years (jobs ending "Present"/"Continue" not counted),
  - filling country_of_interest with sentences, home-country cities or 'null',
  - copying several phone numbers into `phone`,
  - taking the email of an employer from a bundled experience certificate,
  - copying the "References on demand" line into previous_employment,
  - leaving position null despite a clear headline,
  - using an honorific ('Shah Ji') instead of the stated name.

Everything here is pure Python (no network, no LLM) so it can be unit-tested.
`clean_cv_fields` is the entry point; main.py calls it right after the LLM parse.
"""

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

PLACEHOLDER_STRINGS = {
    '', 'missing', 'null', 'none', 'nil', 'undefined', 'n/a', 'na', 'not provided',
    'not mentioned', 'not specified', 'not available', '-', '--',
}

SCALAR_TEXT_FIELDS = (
    'full_name', 'email', 'phone', 'location', 'nationality', 'position',
    'country_of_interest', 'linkedin_url', 'summary', 'professional_summary',
    'father_name', 'cnic', 'passport', 'date_of_birth', 'marital_status',
    'previous_employment', 'passport_expiry',
)
EXPERIENCE_TEXT_FIELDS = ('title', 'company', 'location', 'start_date', 'end_date', 'description', 'duration')
LIST_TEXT_FIELDS = ('skills', 'certifications', 'languages')


def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in PLACEHOLDER_STRINGS
    return False


def _clean_str(value: Any) -> Optional[str]:
    if not isinstance(value, str) or is_placeholder(value):
        return None
    return value.strip()


def strip_placeholder_values(parsed: Dict[str, Any]) -> None:
    """The LLM sometimes returns the string 'null' (or 'N/A') instead of JSON null."""
    for field in SCALAR_TEXT_FIELDS:
        if isinstance(parsed.get(field), str) and is_placeholder(parsed[field]):
            parsed[field] = None
    for field in LIST_TEXT_FIELDS:
        items = parsed.get(field)
        if isinstance(items, list):
            parsed[field] = [item for item in items if not (isinstance(item, str) and is_placeholder(item))]
    for entry in parsed.get('experience') or []:
        if isinstance(entry, dict):
            for field in EXPERIENCE_TEXT_FIELDS:
                if isinstance(entry.get(field), str) and is_placeholder(entry[field]):
                    entry[field] = None


# ---------------------------------------------------------------------------
# Experience entries that are internships or courses, not jobs
# ---------------------------------------------------------------------------

# Word boundaries matter: "intern" must not match "International" (e.g. "Redco
# International", "The International School"), which used to drop real jobs.
_INTERNSHIP_TITLE_RE = re.compile(
    r'\b(?:intern|internship|internee|trainee|apprentice|apprenticeship)\b'
    r'|\b(?:on[\s-]?(?:the[\s-]?)?job|industrial|summer)\s+training\b',
    re.I,
)
_COURSE_TITLE_RE = re.compile(
    r'\b(?:course|courses|coursework|certification|certificate|seminar|diploma|bootcamp)\b'
    r'|\btraining\s+program(?:me)?\b',
    re.I,
)


def is_internship_entry(entry: Dict[str, Any]) -> bool:
    # Only the title decides. Descriptions of real jobs often say "trained staff"
    # or "international assignment", and company names often contain "International".
    return bool(_INTERNSHIP_TITLE_RE.search(entry.get('title') or ''))


def is_course_entry(entry: Dict[str, Any]) -> bool:
    title = entry.get('title') or ''
    if not title.strip():
        title = entry.get('company') or ''
    return bool(_COURSE_TITLE_RE.search(title))


# ---------------------------------------------------------------------------
# Dates and experience years
# ---------------------------------------------------------------------------

PRESENT = 'present'

_PRESENT_RE = re.compile(
    r'\b(?:present|current(?:ly)?|continu\w*|till\s+(?:date|now|today)|to\s*date|up\s*to\s*date|'
    r'now|ongoing|on\s*going|running|still\s+working)\b',
    re.I,
)
_MONTHS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}
_MONTH_NAME_RE = re.compile(
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s,\-\'’]*((?:19|20)\d{2})\b',
    re.I,
)
_YMD_RE = re.compile(r'\b((?:19|20)\d{2})[-/.](\d{1,2})(?:[-/.]\d{1,2})?\b')
_DMY_RE = re.compile(r'\b(\d{1,2})[-/.](\d{1,2})[-/.]((?:19|20)\d{2})\b')
_MY_RE = re.compile(r'\b(\d{1,2})[-/.]((?:19|20)\d{2})\b')
_YEAR_RE = re.compile(r'\b((?:19|20)\d{2})\b')
_DURATION_RE = re.compile(
    r'(\d+(?:\.\d+)?)(?:\s*(?:-|to|–)\s*(\d+(?:\.\d+)?))?\s*\+?\s*(years?|yrs?|months?|mos?)\b',
    re.I,
)


def parse_cv_date(value: Any):
    """
    Parse a CV date string. Returns PRESENT for ongoing jobs, (year, month) where
    month is None for a bare year, or None when nothing date-like is found.
    Day-first is assumed for numeric dates (Pakistani convention).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if _PRESENT_RE.search(text):
        return PRESENT
    m = _YMD_RE.search(text)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = _DMY_RE.search(text)
    if m:
        first, second = int(m.group(1)), int(m.group(2))
        month = second if 1 <= second <= 12 else first
        if 1 <= month <= 12:
            return int(m.group(3)), month
    m = _MONTH_NAME_RE.search(text)
    if m:
        return int(m.group(2)), _MONTHS[m.group(1).lower()[:3]]
    m = _MY_RE.search(text)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(1)), None
    return None


def parse_duration_months(value: Any) -> Optional[float]:
    """'05 Months' -> 5, '2 Years' -> 24, '3-4 Years' -> 42, '1 year 6 months' -> 18."""
    if not isinstance(value, str):
        return None
    total = 0.0
    found = False
    for m in _DURATION_RE.finditer(value):
        low = float(m.group(1))
        high = float(m.group(2)) if m.group(2) else low
        amount = (low + high) / 2
        unit = m.group(3).lower()
        total += amount * 12 if unit.startswith('y') else amount
        found = True
    return total if found and total > 0 else None


def _month_index(year: int, month: Optional[int]) -> int:
    return year * 12 + (month or 1) - 1


def compute_experience_months(experience: Any, today: date, gcc_only: bool = False) -> Tuple[Optional[float], bool]:
    """
    Total months of work from the experience array.

    Overlapping jobs are merged (three parallel jobs since 2015 count once).
    Ongoing jobs count up to `today`. A job with a start date but no end date is
    treated as ongoing only when it is the most recent job, which is how the LLM
    usually renders "Present".

    Returns (months or None when no entry could be measured, complete) where
    `complete` is True when every job could be measured.
    """
    if not isinstance(experience, list):
        return None, False
    entries = [e for e in experience if isinstance(e, dict)]
    if gcc_only:
        entries = [e for e in entries if _is_gcc_location(e.get('location'))]
    if not entries:
        return None, False

    today_idx = _month_index(today.year, today.month)
    parsed_entries = []
    for entry in entries:
        start = parse_cv_date(entry.get('start_date'))
        end = parse_cv_date(entry.get('end_date'))
        if start == PRESENT:
            start = None
        parsed_entries.append((entry, start, end))
    starts = [_month_index(*s) for _, s, _ in parsed_entries if s]
    latest_start = max(starts) if starts else None

    intervals = []
    extra_months = 0.0
    measured = 0
    for entry, start, end in parsed_entries:
        if start:
            start_idx = _month_index(*start)
            if start_idx > today_idx:
                continue
            if end == PRESENT or (end is None and start_idx == latest_start):
                end_idx = today_idx
            elif end:
                end_idx = min(_month_index(*end), today_idx)
            else:
                continue
            if end_idx < start_idx:
                continue
            intervals.append((start_idx, end_idx))
            measured += 1
            continue
        duration = parse_duration_months(entry.get('duration')) or parse_duration_months(entry.get('end_date'))
        if duration:
            extra_months += duration
            measured += 1

    if not measured:
        return None, False

    total = extra_months
    intervals.sort()
    cur_start, cur_end = None, None
    for start_idx, end_idx in intervals:
        if cur_end is None or start_idx > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start_idx, end_idx
        else:
            cur_end = max(cur_end, end_idx)
    if cur_end is not None:
        total += cur_end - cur_start
    return total, measured == len(entries)


def _to_years(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        m = re.search(r'\d+(?:\.\d+)?', value)
        if not m:
            return None
        number = float(m.group(0))
    else:
        return None
    return number if 0 <= number <= 60 else None


def _round_years(years: float) -> int:
    # candidates.experience_years is an INTEGER column.
    return int(years + 0.5)


def resolve_experience_years(parsed: Dict[str, Any], today: date) -> None:
    """
    Dates beat the LLM's number: when every job has dates (or a stated duration)
    the computed total is used. When some jobs are undated the computed total is
    only a lower bound, so the larger of it and the LLM's number is kept. A total
    the CV states outright ("8 years of experience") wins if it is larger.
    """
    llm_years = _to_years(parsed.get('experience_years'))
    stated_years = _to_years(parsed.get('stated_experience_years'))
    months, complete = compute_experience_months(parsed.get('experience'), today)
    computed = months / 12 if months is not None else None

    years, source = llm_years, 'llm'
    if computed is not None and complete:
        years, source = computed, 'experience_dates'
    elif computed is not None and computed > (llm_years or 0):
        years, source = computed, 'experience_dates'
    if stated_years is not None and (years is None or stated_years > years):
        years, source = stated_years, 'stated'

    if years is None:
        return
    parsed['experience_years'] = _round_years(years)
    parsed['experience_years_source'] = source


GCC_COUNTRY_TERMS = (
    'saudi', 'ksa', 'uae', 'u.a.e', 'united arab emirates', 'emirates', 'dubai', 'abu dhabi',
    'sharjah', 'qatar', 'doha', 'kuwait', 'bahrain', 'oman', 'muscat', 'riyadh', 'jeddah',
    'dammam', 'gcc', 'gulf',
)


def _is_gcc_location(location: Any) -> bool:
    if not isinstance(location, str):
        return False
    lower = location.lower()
    return any(term in lower for term in GCC_COUNTRY_TERMS)


def resolve_gcc_years(parsed: Dict[str, Any], today: date) -> None:
    months, _ = compute_experience_months(parsed.get('experience'), today, gcc_only=True)
    if months:
        years = _round_years(months / 12)
        if years > 0:
            parsed['gcc_years'] = years


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

_PHONE_SPLIT_RE = re.compile(r'\s*(?:[,;/|\n\\]|\bor\b|&|\band\b)\s*', re.I)
_PK_MOBILE_RE = re.compile(r'(?:\+?\s*92|0092|0)?[\s-]*3\d{2}[\s-]*\d{7}')
# Country codes with the national number length that follows them.
_INTL_LENGTHS = {
    '92': (10,), '966': (9,), '971': (8, 9), '974': (8,), '965': (8,), '968': (8,),
    '973': (8,), '91': (10,), '880': (10,), '977': (10,), '94': (9,), '90': (10,),
}


def split_phone_numbers(raw: Any) -> List[str]:
    if isinstance(raw, list):
        raw = ', '.join(str(item) for item in raw if isinstance(item, (str, int)))
    if not isinstance(raw, str) or not raw.strip():
        return []
    numbers = []
    for part in _PHONE_SPLIT_RE.split(raw):
        part = part.strip()
        digits = re.sub(r'\D', '', part)
        if len(digits) < 7:
            continue
        if len(digits) > 15:
            # Two numbers separated only by spaces: pull out the mobile numbers.
            pieces = [p.strip() for p in _PK_MOBILE_RE.findall(part)]
            if pieces:
                numbers.extend(pieces)
                continue
        numbers.append(part)
    return numbers


def normalize_phone_e164(raw: Any) -> Optional[str]:
    """Return +<country><number> when the number is unambiguous, else None."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    digits = re.sub(r'\D', '', text)
    international = bool(re.match(r'^\(?\s*\+', text))
    if digits.startswith('00'):
        digits = digits[2:]
        international = True

    if international:
        if digits.startswith('920'):  # "+92 0300 ..." keeps the trunk zero
            digits = '92' + digits[3:]
        return '+' + digits if 8 <= len(digits) <= 15 else None

    if len(digits) == 11 and digits.startswith('03'):
        return '+92' + digits[1:]
    if len(digits) == 10 and digits.startswith('3'):
        return '+92' + digits
    for code, lengths in _INTL_LENGTHS.items():
        if digits.startswith(code) and len(digits) - len(code) in lengths:
            return '+' + digits
    return None


def clean_phone(parsed: Dict[str, Any]) -> None:
    """Keep the first number as `phone` (E.164 when possible); keep the rest aside."""
    numbers = split_phone_numbers(parsed.get('phone'))
    if not numbers:
        return
    normalized = []
    for number in numbers:
        value = normalize_phone_e164(number) or re.sub(r'\s+', ' ', number).strip()
        if value not in normalized:
            normalized.append(value)
    parsed['phone'] = normalized[0]
    if len(normalized) > 1:
        parsed['additional_phones'] = normalized[1:]


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+')
ROLE_EMAIL_LOCAL_PARTS = {
    'info', 'hr', 'hrd', 'admin', 'administrator', 'contact', 'contactus', 'career', 'careers',
    'job', 'jobs', 'recruitment', 'recruiting', 'hiring', 'office', 'support', 'sales',
    'enquiry', 'enquiries', 'inquiry', 'inquiries', 'noreply', 'no-reply', 'donotreply',
    'mail', 'email', 'hello', 'help', 'helpdesk', 'accounts', 'account', 'billing', 'finance',
    'webmaster', 'postmaster', 'reception', 'frontdesk', 'marketing', 'customercare',
    'customerservice', 'service', 'services', 'team', 'general', 'principal', 'registrar',
    'management', 'operations', 'procurement', 'purchase', 'booking', 'bookings',
    'reservations', 'feedback', 'query', 'media', 'press',
}
FREE_EMAIL_DOMAINS = {
    'gmail.com', 'googlemail.com', 'yahoo.com', 'ymail.com', 'hotmail.com', 'outlook.com',
    'live.com', 'msn.com', 'icloud.com', 'me.com', 'aol.com', 'proton.me', 'protonmail.com',
    'mail.com', 'gmx.com', 'yandex.com', 'zoho.com', 'indeedemail.com',
}
# Phrases that only appear in experience certificates / reference letters that are
# often scanned into the same PDF as the CV.
_CERTIFICATE_MARKER_RE = re.compile(
    r'to\s+whom\s+it\s+may\s+concern|this\s+is\s+to\s+certify|it\s+is\s+certified|'
    r'\b(?:experience|employment|service)\s+certificate\b|certificate\s+of\s+(?:employment|experience|service)|'
    r'\bwe\s+wish\b|\bhis\s+services\b|\bher\s+services\b|\bduring\s+(?:his|her|their)\s+(?:stay|tenure|employment)\b|'
    r'\bthroughout\s+(?:his|her|their)\s+tenure\b',
    re.I,
)
# Section headings that only appear in the CV itself.
_CV_MARKER_RE = re.compile(
    r'curriculum\s+vitae|\bresume\b|^\s*(?:career\s+)?objective\b|personal\s+(?:information|details|data|profile)|'
    r'^\s*contact(?:\s+(?:info|information|details))?\s*:?\s*$|^\s*(?:profile|summary|skills|education|experience|'
    r'work\s+experience|employment\s+history|languages)\s*:?\s*$',
    re.I | re.M,
)


def is_role_email(email: str) -> bool:
    local, _, domain = email.lower().partition('@')
    core = re.sub(r'\d+$', '', local)
    if core in ROLE_EMAIL_LOCAL_PARTS:
        return True
    first_token = re.split(r'[._-]', core)[0]
    return domain not in FREE_EMAIL_DOMAINS and first_token in ROLE_EMAIL_LOCAL_PARTS


def _in_certificate_context(raw_text: str, position: int) -> bool:
    """True when the nearest heading/phrase before `position` belongs to a certificate."""
    before = raw_text[:position]
    cert = [m.start() for m in _CERTIFICATE_MARKER_RE.finditer(before)]
    if not cert:
        return False
    cv = [m.start() for m in _CV_MARKER_RE.finditer(before)]
    return not cv or cert[-1] > cv[-1]


def _email_is_applicants(email: str, raw_text: Optional[str]) -> bool:
    if is_role_email(email):
        return False
    if not raw_text:
        return True
    positions = [m.start() for m in re.finditer(re.escape(email), raw_text, re.I)]
    if not positions:
        # Not in the text verbatim (OCR spacing the LLM repaired) - nothing to check against.
        return True
    return any(not _in_certificate_context(raw_text, pos) for pos in positions)


def clean_email(parsed: Dict[str, Any], raw_text: Optional[str]) -> None:
    email = _clean_str(parsed.get('email'))
    if email:
        match = _EMAIL_RE.search(email)
        email = match.group(0) if match else None
    if email and _email_is_applicants(email, raw_text):
        parsed['email'] = email
        return
    replacement = None
    for m in _EMAIL_RE.finditer(raw_text or ''):
        candidate = m.group(0)
        if not is_role_email(candidate) and not _in_certificate_context(raw_text, m.start()):
            replacement = candidate
            break
    parsed['email'] = replacement


# ---------------------------------------------------------------------------
# Country of interest
# ---------------------------------------------------------------------------

# alias -> canonical name; canonical names follow the values already stored
# (e.g. 'UAE', 'Saudi Arabia').
_COUNTRY_ALIASES = {
    'UAE': ['uae', 'u.a.e', 'united arab emirates', 'emirates', 'dubai', 'abu dhabi', 'abudhabi',
            'sharjah', 'ajman', 'ras al khaimah', 'fujairah', 'al ain'],
    'Saudi Arabia': ['saudi arabia', 'saudi', 'saudia', 'ksa', 'k.s.a', 'kingdom of saudi arabia', 'riyadh',
                     'jeddah', 'jiddah', 'dammam', 'al khobar', 'khobar', 'makkah', 'mecca', 'madinah',
                     'medina', 'jubail', 'yanbu', 'tabuk', 'taif', 'neom'],
    'Qatar': ['qatar', 'doha', 'lusail'],
    'Kuwait': ['kuwait'],
    'Bahrain': ['bahrain', 'manama'],
    'Oman': ['oman', 'muscat', 'salalah', 'sohar'],
    'Pakistan': ['pakistan', 'karachi', 'lahore', 'islamabad', 'rawalpindi', 'peshawar', 'faisalabad',
                 'multan', 'quetta', 'sialkot', 'gujranwala', 'abbottabad', 'sargodha', 'bahawalpur'],
    'India': ['india'],
    'Bangladesh': ['bangladesh'],
    'Nepal': ['nepal'],
    'Sri Lanka': ['sri lanka'],
    'Afghanistan': ['afghanistan'],
    'Iraq': ['iraq'],
    'Jordan': ['jordan'],
    'Lebanon': ['lebanon'],
    'Egypt': ['egypt'],
    'Turkey': ['turkey', 'turkiye', 'türkiye', 'istanbul'],
    'Iran': ['iran'],
    'Azerbaijan': ['azerbaijan'],
    'Georgia': ['georgia'],
    'Uzbekistan': ['uzbekistan'],
    'Kazakhstan': ['kazakhstan'],
    'Russia': ['russia'],
    'Belarus': ['belarus'],
    'Malaysia': ['malaysia', 'kuala lumpur'],
    'Singapore': ['singapore'],
    'Brunei': ['brunei'],
    'Maldives': ['maldives'],
    'Japan': ['japan'],
    'South Korea': ['south korea', 'korea'],
    'China': ['china'],
    'Hong Kong': ['hong kong'],
    'Australia': ['australia'],
    'New Zealand': ['new zealand'],
    'Canada': ['canada'],
    'USA': ['usa', 'u.s.a', 'united states', 'america'],
    'UK': ['united kingdom', 'u.k', 'england', 'britain', 'scotland', 'london'],
    'Ireland': ['ireland'],
    'Germany': ['germany'],
    'France': ['france'],
    'Italy': ['italy'],
    'Spain': ['spain'],
    'Portugal': ['portugal'],
    'Netherlands': ['netherlands', 'netherland', 'holland'],
    'Belgium': ['belgium'],
    'Luxembourg': ['luxembourg'],
    'Switzerland': ['switzerland'],
    'Austria': ['austria'],
    'Poland': ['poland'],
    'Romania': ['romania'],
    'Bulgaria': ['bulgaria'],
    'Croatia': ['croatia'],
    'Serbia': ['serbia'],
    'Bosnia and Herzegovina': ['bosnia'],
    'Montenegro': ['montenegro'],
    'North Macedonia': ['macedonia'],
    'Albania': ['albania'],
    'Hungary': ['hungary'],
    'Czech Republic': ['czech republic', 'czechia'],
    'Slovakia': ['slovakia'],
    'Slovenia': ['slovenia'],
    'Greece': ['greece'],
    'Cyprus': ['cyprus'],
    'Malta': ['malta'],
    'Sweden': ['sweden'],
    'Norway': ['norway'],
    'Denmark': ['denmark'],
    'Finland': ['finland'],
    'Iceland': ['iceland'],
    'Lithuania': ['lithuania'],
    'Latvia': ['latvia'],
    'Estonia': ['estonia'],
    'Libya': ['libya'],
    'Algeria': ['algeria'],
    'Morocco': ['morocco'],
    'South Africa': ['south africa'],
}
_GCC = {'UAE', 'Saudi Arabia', 'Qatar', 'Kuwait', 'Bahrain', 'Oman'}
_EUROPE = {
    'UK', 'Ireland', 'Germany', 'France', 'Italy', 'Spain', 'Portugal', 'Netherlands', 'Belgium',
    'Luxembourg', 'Switzerland', 'Austria', 'Poland', 'Romania', 'Bulgaria', 'Croatia', 'Serbia',
    'Bosnia and Herzegovina', 'Montenegro', 'North Macedonia', 'Albania', 'Hungary', 'Czech Republic',
    'Slovakia', 'Slovenia', 'Greece', 'Cyprus', 'Malta', 'Sweden', 'Norway', 'Denmark', 'Finland',
    'Iceland', 'Lithuania', 'Latvia', 'Estonia', 'Belarus', 'Russia',
}
# Region words are only kept when the CV names no country in that region.
_REGION_PATTERNS = (
    ('Europe', re.compile(r'\beurop(?:e|ean)\b|\bschengen\b', re.I), _EUROPE),
    ('GCC', re.compile(r'\bgulf\b|\bgcc\b|\bmiddle\s+east\b', re.I), _GCC),
)


def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias).replace(r'\ ', r'[\s-]+')
    return re.compile(r'(?<![A-Za-z])' + escaped + r'(?![A-Za-z])', re.I)


_COUNTRY_PATTERNS = [
    (canonical, _alias_pattern(alias))
    for canonical, aliases in _COUNTRY_ALIASES.items()
    for alias in aliases
]
# "UK" only in capitals; "uk" is too common inside words and handles.
_COUNTRY_PATTERNS.append(('UK', re.compile(r'(?<![A-Za-z])UK(?![A-Za-z])')))
_COUNTRY_FIELD_MAX = 100


def normalize_country_of_interest(value: Any) -> Optional[str]:
    """
    Turn the LLM's free text into country names ("UAE, Saudi Arabia, Qatar") or
    None when it names no country (e.g. 'international opportunities').
    """
    if isinstance(value, list):
        value = ', '.join(str(v) for v in value if isinstance(v, str))
    text = _clean_str(value)
    if not text:
        return None

    hits = []  # (position, name)
    for canonical, pattern in _COUNTRY_PATTERNS:
        m = pattern.search(text)
        if m:
            hits.append((m.start(), canonical))
    countries = set(name for _, name in hits)
    for region, pattern, members in _REGION_PATTERNS:
        m = pattern.search(text)
        if m and not (countries & members):
            hits.append((m.start(), region))

    ordered = []
    for _, name in sorted(hits):
        if name not in ordered:
            ordered.append(name)
    if not ordered:
        return None

    result = ordered[0]
    for name in ordered[1:]:
        if len(result) + 2 + len(name) > _COUNTRY_FIELD_MAX:
            break
        result = f'{result}, {name}'
    return result


# ---------------------------------------------------------------------------
# previous_employment
# ---------------------------------------------------------------------------

_REFERENCE_LINE_RE = re.compile(
    r'\breferences?\b.{0,60}\b(?:demand|request|requirement|required|asked)\b'
    r'|\b(?:available|furnished|provided)\s+(?:up)?on\s+(?:demand|request)\b',
    re.I | re.S,
)


def clean_previous_employment(parsed: Dict[str, Any]) -> None:
    """Drop the references line; the backend then builds the history from `experience`."""
    value = parsed.get('previous_employment')
    if isinstance(value, str) and _REFERENCE_LINE_RE.search(value):
        parsed['previous_employment'] = None


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

_SELF_EMPLOYED_RE = re.compile(
    r'\b(?:founder|owner|own\s+business|self[\s-]?employed|freelance\w*|proprietor|independent)\b', re.I
)
_DRIVER_VARIANTS = (
    'htv driver', 'heavy duty driver', 'heavy vehicle driver', 'light vehicle driver',
    'light duty driver', 'simple driver', 'driver', 'truck driver', 'bus driver', 'van driver',
)
_ROLE_SPLIT_RE = re.compile(r'\s*(?:/|\||,|;|&|\band\b|•)\s*', re.I)


def normalize_driver_position(position: Any) -> Any:
    """
    "HTV Driver" -> "Driver (HTV)". Only when driving is the candidate's first-listed
    role: "Electrician & Plumber / Mason / LTV Driver" is an electrician, not a driver.
    """
    if not isinstance(position, str) or not position.strip():
        return position
    first_role = _ROLE_SPLIT_RE.split(position.strip())[0].lower()
    if not any(variant in first_role for variant in _DRIVER_VARIANTS):
        return position
    if 'htv' in first_role or 'heavy' in first_role:
        return 'Driver (HTV)'
    if 'light' in first_role or 'ltv' in first_role:
        return 'Driver (Light Vehicle)'
    return 'Driver'


def pick_current_job_title(experience: Any, today: date) -> Optional[str]:
    """Current job first, then the most recent one; own-business titles only as a last resort."""
    if not isinstance(experience, list):
        return None
    jobs = [e for e in experience if isinstance(e, dict) and _clean_str(e.get('title'))]
    if not jobs:
        return None
    today_idx = _month_index(today.year, today.month)

    def recency(index_entry):
        index, entry = index_entry
        start = parse_cv_date(entry.get('start_date'))
        end = parse_cv_date(entry.get('end_date'))
        start_idx = _month_index(*start) if start and start != PRESENT else -1
        if end == PRESENT:
            end_idx = today_idx + 1
        elif end:
            end_idx = _month_index(*end)
        else:
            end_idx = start_idx
        # Undated entries keep CV order (CVs list the newest job first).
        return (end_idx, start_idx, -index)

    def self_employed(entry):
        return bool(_SELF_EMPLOYED_RE.search(f"{entry.get('title') or ''} {entry.get('company') or ''}"))

    ranked = sorted(enumerate(jobs), key=recency, reverse=True)
    employed = [entry for _, entry in ranked if not self_employed(entry)]
    best = (employed or [entry for _, entry in ranked])[0]
    return best['title'].strip()


def resolve_position(parsed: Dict[str, Any], today: date) -> None:
    if not _clean_str(parsed.get('position')):
        title = pick_current_job_title(parsed.get('experience'), today)
        parsed['position'] = title
        if title:
            parsed['position_source'] = 'current_job'
    parsed['position'] = normalize_driver_position(parsed.get('position'))


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------

HONORIFIC_TOKENS = {
    'ji', 'jee', 'sahib', 'sahab', 'saab', 'sb', 'bhai', 'bhaijan', 'bhaijaan', 'janab',
    'ustad', 'ustaad',
}
_NAME_STATEMENT_RES = (
    re.compile(r'\bmera\s+naam\s+([A-Za-z][A-Za-z.\s]{1,60}?)\s+(?:hai|hay|he|hy|h)\b', re.I),
    re.compile(r'\bmy\s+name\s+is\s+([A-Za-z][A-Za-z.\s]{1,60}?)(?=\s*(?:[,.;:!\n]|\band\b|\bi\b|$))', re.I),
    re.compile(r'^\s*(?:full\s+)?name\s*[:\-]\s*([A-Za-z][A-Za-z.\s]{1,60}?)\s*$', re.I | re.M),
)


def _name_tokens(name: str) -> List[str]:
    return [t for t in re.split(r'[\s.]+', name.lower()) if t]


def looks_like_honorific(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return any(token in HONORIFIC_TOKENS for token in _name_tokens(name))


def find_stated_name(raw_text: Optional[str]) -> Optional[str]:
    for pattern in _NAME_STATEMENT_RES:
        for m in pattern.finditer(raw_text or ''):
            name = re.sub(r'\s+', ' ', m.group(1)).strip(' .')
            tokens = _name_tokens(name)
            if 2 <= len(tokens) <= 5 and not looks_like_honorific(name):
                return name.title() if name.islower() else name
    return None


def resolve_full_name(parsed: Dict[str, Any], raw_text: Optional[str]) -> None:
    """'Shah Ji' is a nickname; prefer 'Mera naam Shakeel Nawaz hai' / 'My name is ...'."""
    if not looks_like_honorific(parsed.get('full_name')):
        return
    stated = find_stated_name(raw_text)
    if stated:
        parsed['name_used_in_cv'] = parsed.get('full_name')
        parsed['full_name'] = stated


# ---------------------------------------------------------------------------


def clean_cv_fields(parsed: Dict[str, Any], raw_text: Optional[str] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """Apply every deterministic fix to the LLM output (in place) and return it."""
    today = today or date.today()
    strip_placeholder_values(parsed)
    resolve_full_name(parsed, raw_text)
    clean_phone(parsed)
    clean_email(parsed, raw_text)
    parsed['country_of_interest'] = normalize_country_of_interest(parsed.get('country_of_interest'))
    clean_previous_employment(parsed)
    resolve_experience_years(parsed, today)
    resolve_gcc_years(parsed, today)
    resolve_position(parsed, today)
    return parsed
