import json
import os
import re
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Common country patterns and indicators
PAKISTAN_INDICATORS = {
    'universities': [
        'university of karachi', 'fast-nuces', 'comsats', 'iqra', 'lums',
        'punjab university', 'giki', 'air university', 'szabist', 'cecos',
        'bahria university', 'nust', 'pu lahore', 'bzu', 'khyber pakhtunkhwa',
        'sindh university', 'buitems', 'fccollege'
    ],
    'cities': ['karachi', 'lahore', 'islamabad', 'rawalpindi', 'peshawar', 'faisalabad', 'multan', 'quetta', 'hyderabad'],
    'companies': ['ptcl', 'ufone', 'zong', 'telenor', 'jazz', 'stc'],
    'languages': ['urdu'],
    'passport_prefix': ['pa', 'ab'],
}

INDIA_INDICATORS = {
    'universities': [
        'iit', 'delhi university', 'mumbai university', 'bangalore university',
        'anna university', 'university of hyderabad', 'amity university',
        'manipal', 'bits pilani', 'christ university', 'vit'
    ],
    'cities': ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'pune', 'kolkata', 'chennai', 'jaipur'],
    'languages': ['hindi', 'tamil', 'telugu', 'kannada'],
    'companies': ['infosys', 'tcs', 'wipro', 'accenture india', 'cognizant'],
}

GULF_COUNTRIES = {
    'UAE': ['dubai', 'abudhabi', 'sharjah'],
    'Saudi Arabia': ['riyadh', 'jeddah', 'dammam'],
    'Kuwait': ['kuwait city'],
    'Qatar': ['doha'],
    'Bahrain': ['manama'],
}

def infer_nationality_from_cv_data(cv_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float]:
    """
    Infer nationality from education, work experience, and other indicators.
    
    CRITICAL: Do NOT infer nationality from Gulf countries (UAE, KSA, Kuwait, etc.) alone.
    Someone working in Gulf is likely an EXPAT, not a Gulf national.
    Requires education, passport, or language confirmation.
    
    Returns:
        Tuple of (inferred_nationality, source, confidence_score)
    """
    
    # If nationality already explicitly stated, return it
    if cv_data.get('nationality') and cv_data.get('extraction_confidence', {}).get('nationality', 0) > 0.8:
        return cv_data['nationality'], 'explicit', cv_data.get('extraction_confidence', {}).get('nationality', 0.95)
    
    indicators = {
        'education_country': _detect_education_country(cv_data),
        'work_countries': _detect_work_countries(cv_data),
        'language_indicators': _detect_language_indicators(cv_data),
        'passport_country': _detect_passport_country(cv_data),
    }
    
    # Priority-based inference
    # 1. Education is most reliable indicator
    if indicators['education_country']:
        country, confidence = indicators['education_country']
        if country:
            return country, f'education ({country})', confidence
    
    # 2. Passport country code is very reliable
    if indicators['passport_country']:
        country, confidence = indicators['passport_country']
        if country:
            return country, f'passport_country_code', confidence
    
    # 3. Language indicators (can be reliable)
    if indicators['language_indicators']:
        country, confidence = indicators['language_indicators']
        if country:
            return country, f'language_skills', confidence
    
    # 4. Work countries - BUT CAREFUL WITH GULF COUNTRIES!
    # CRITICAL: Don't infer nationality from Gulf countries alone
    # Someone working in UAE/KSA is likely an expat, not a Gulf national
    work_countries = indicators['work_countries']
    if work_countries:
        country, confidence = work_countries[0]
        if country:
            # Check if it's a Gulf country
            gulf_countries = ['UAE', 'Saudi Arabia', 'Kuwait', 'Qatar', 'Bahrain', 'Oman', 'Gulf']
            
            if country in gulf_countries:
                # IMPORTANT: Don't infer Gulf nationality without other confirmation
                # This person is likely an expat working in the Gulf
                # Return None unless we have other indicators
                return None, None, 0.0
            else:
                # Non-Gulf country from work location is safe to infer
                return country, f'primary_work_location ({country})', confidence
    
    return None, None, 0.0


def _detect_education_country(cv_data: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Detect country from education information"""
    education_country = cv_data.get('primary_education_country')
    if education_country:
        return education_country, 0.75
    
    education = cv_data.get('education', '').lower() if cv_data.get('education') else ''
    
    for indicator in PAKISTAN_INDICATORS['universities']:
        if indicator in education:
            return 'Pakistan', 0.80
    
    for indicator in INDIA_INDICATORS['universities']:
        if indicator in education:
            return 'India', 0.75
    
    return None


def _detect_work_countries(cv_data: Dict[str, Any]) -> list:
    """Detect countries from work experience"""
    work_countries = []
    
    # Use extracted work countries if available
    if 'primary_work_countries' in cv_data:
        for country in cv_data.get('primary_work_countries', []):
            if country:
                work_countries.append((country, 0.65))
    
    employment = cv_data.get('previous_employment', '').lower() if cv_data.get('previous_employment') else ''
    
    # Check for Pakistan indicators
    pakistan_count = sum(1 for city in PAKISTAN_INDICATORS['cities'] if city in employment)
    if pakistan_count >= 2:
        work_countries.append(('Pakistan', 0.75))
    
    # Check for India indicators
    india_count = sum(1 for city in INDIA_INDICATORS['cities'] if city in employment)
    if india_count >= 2:
        work_countries.append(('India', 0.70))
    
    return work_countries


def _detect_language_indicators(cv_data: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Detect country from language skills"""
    languages = cv_data.get('languages', [])
    if not languages:
        return None
    
    languages_str = str(languages).lower()
    
    if 'urdu' in languages_str:
        return 'Pakistan', 0.70
    
    if 'hindi' in languages_str and 'urdu' not in languages_str:
        return 'India', 0.65
    
    if 'arabic' in languages_str:
        return 'Saudi Arabia', 0.60
    
    return None


def _detect_passport_country(cv_data: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Detect country from passport information"""
    passport = cv_data.get('passport_no', '').upper() if cv_data.get('passport_no') else ''
    
    if passport:
        if passport.startswith('PA') or passport.startswith('AB'):
            return 'Pakistan', 0.90
        elif passport.startswith('IN'):
            return 'India', 0.90
        elif passport.startswith('GB'):
            return 'United Kingdom', 0.90
    
    return None


def enhance_nationality_with_ai(cv_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use OpenAI to intelligently infer nationality from all available data.
    Only called when explicit nationality is not found.
    """
    
    if cv_data.get('nationality') and cv_data.get('extraction_confidence', {}).get('nationality', 0) > 0.8:
        return cv_data
    
    # Try rule-based detection first
    inferred_nationality, source, confidence = infer_nationality_from_cv_data(cv_data)
    
    if inferred_nationality and confidence > 0.6:
        cv_data['nationality'] = inferred_nationality
        cv_data['nationality_inferred_from'] = source
        if 'extraction_confidence' not in cv_data:
            cv_data['extraction_confidence'] = {}
        cv_data['extraction_confidence']['nationality'] = confidence
        return cv_data
    
    # Fall back to AI-based detection for edge cases
    prompt = f"""
    Based on the following CV information, infer the most likely nationality of the person.
    
    Education:
    {cv_data.get('education', 'Not provided')}
    
    Education Country: {cv_data.get('primary_education_country', 'Not detected')}
    
    Work Countries:
    {', '.join(cv_data.get('primary_work_countries', ['Not detected']))}
    
    Languages: {', '.join(cv_data.get('languages', ['Not listed']))}
    
    Passport: {cv_data.get('passport_no', 'Not provided')}
    
    Work History Summary:
    {cv_data.get('previous_employment', 'Not provided')}
    
    Return a JSON object with ONLY these fields:
    {{
        "nationality": "string (most likely country)",
        "inference_source": "string (what led to this conclusion)",
        "confidence": 0.0-1.0 (how confident are you, 0.5-0.9 range expected)
    }}
    
    Instructions:
    1. If person studied in Pakistan and worked in Pakistan, very likely Pakistani
    2. If person studied in Pakistan but worked in Gulf, likely Pakistani expat
    3. Urdu language is strong Pakistani indicator
    4. Multiple Pakistani cities in work history = strong indicator
    5. Return exact country name, not 'Pakistani' etc.
    6. Always provide reasoning in inference_source
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at inferring nationality from CV information. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        if result.get('nationality'):
            cv_data['nationality'] = result['nationality']
            cv_data['nationality_inferred_from'] = result.get('inference_source', 'AI analysis')
            if 'extraction_confidence' not in cv_data:
                cv_data['extraction_confidence'] = {}
            cv_data['extraction_confidence']['nationality'] = result.get('confidence', 0.65)
    
    except Exception as e:
        print(f"Error in AI-based nationality inference: {e}")
    
    return cv_data


def main():
    """Test the nationality inference"""
    test_cv = {
        "nationality": None,
        "primary_education_country": "Pakistan",
        "primary_work_countries": ["Pakistan", "Saudi Arabia"],
        "languages": ["Urdu", "English"],
        "education": "BS Computer Science from FAST-NUCES, Islamabad",
        "previous_employment": "Worked as Software Engineer at TCS Pakistan (2019-2021), then moved to Saudi Aramco (2021-2024)",
        "extraction_confidence": {"nationality": 0.2}
    }
    
    enhanced = enhance_nationality_with_ai(test_cv)
    print(json.dumps(enhanced, indent=2))


if __name__ == "__main__":
    main()
