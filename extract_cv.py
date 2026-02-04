import os
import json
import sys
from typing import Dict, Any
from openai import OpenAI
import PyPDF2
import docx
import requests
from io import BytesIO

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_text_from_pdf(file_path_or_url: str) -> str:
    """Extract text from PDF file or URL"""
    try:
        if file_path_or_url.startswith('http'):
            response = requests.get(file_path_or_url)
            pdf_file = BytesIO(response.content)
        else:
            pdf_file = open(file_path_or_url, 'rb')
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        
        if not file_path_or_url.startswith('http'):
            pdf_file.close()
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_path_or_url: str) -> str:
    """Extract text from DOCX file or URL"""
    try:
        if file_path_or_url.startswith('http'):
            response = requests.get(file_path_or_url)
            doc_file = BytesIO(response.content)
        else:
            doc_file = file_path_or_url
        
        doc = docx.Document(doc_file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")

def extract_cv_data(cv_url: str) -> Dict[str, Any]:
    """Extract structured data from CV using OpenAI GPT-4"""
    
    # Extract text based on file type
    if cv_url.lower().endswith('.pdf'):
        cv_text = extract_text_from_pdf(cv_url)
    elif cv_url.lower().endswith('.docx') or cv_url.lower().endswith('.doc'):
        cv_text = extract_text_from_docx(cv_url)
    else:
        raise Exception("Unsupported file format. Only PDF and DOCX are supported.")
    
    if not cv_text:
        raise Exception("No text could be extracted from the CV")
    
    # Create extraction prompt with enhanced nationality detection
    prompt = f"""
    Extract the following information from this CV/Resume. For each field, provide a confidence score (0.0 to 1.0) indicating how certain you are about the extracted value.
    
    Return ONLY a valid JSON object with this exact structure:
    {{
        "nationality": "string or null",
        "nationality_inferred_from": "string or null (source: explicit, education_location, work_location, language_skills, passport_country)",
        "primary_education_country": "string or null (country where primary education was obtained)",
        "primary_work_countries": ["array of countries where person has worked"],
        "position": "string or null (desired job position/title)",
        "experience_years": "number or null (total years of work experience)",
        "country_of_interest": "string or null (country they want to work in)",
        "skills": ["array of strings"],
        "languages": ["array of strings with proficiency levels"],
        "education": "string or null (highest education qualification)",
        "certifications": ["array of strings"],
        "previous_employment": "string or null (brief summary of work history)",
        "passport_expiry": "string or null (format: YYYY-MM-DD)",
        "professional_summary": "string or null (2-3 sentence summary)",
        "extraction_confidence": {{
            "nationality": 0.0-1.0,
            "position": 0.0-1.0,
            "experience_years": 0.0-1.0,
            "country_of_interest": 0.0-1.0,
            "skills": 0.0-1.0,
            "languages": 0.0-1.0,
            "education": 0.0-1.0,
            "certifications": 0.0-1.0,
            "previous_employment": 0.0-1.0,
            "professional_summary": 0.0-1.0
        }}
    }}
    
    CRITICAL NATIONALITY DETECTION RULES:
    
    1. EXPLICIT NATIONALITY:
       - If CV explicitly states "Nationality: Pakistan", "Pakistani National", use that with high confidence (0.95+)
       - Check for passport country codes (PA = Pakistan, IN = India, etc.)
    
    2. EDUCATION-BASED INFERENCE (0.7-0.8 confidence):
       - Identify all cities/universities mentioned in education section
       - If education is from Pakistan (University of Karachi, FAST-NUCES, Comsats Islamabad, etc.), likely Pakistani
       - Common Pakistani universities: BZU, PU, CECOS, IQRA, SZABIST, LUMS, GIKI, etc.
       - If education is from India, likely Indian; from UK/US/Australia, may have adopted that nationality
       - IMPORTANT: Primary education location is stronger indicator than work location
    
    3. WORK EXPERIENCE-BASED INFERENCE (0.6-0.7 confidence):
       - If person worked in Pakistan for majority of career, likely Pakistani
       - If person worked in Gulf (Saudi Arabia, UAE, Kuwait), they may be Pakistani/Indian expat
       - If ALL work experience is in Pakistan, very likely Pakistani (0.75+ confidence)
    
    4. LANGUAGE SKILLS:
       - Urdu language skills indicate likely Pakistani
       - Hindi indicates likely Indian
       - Arabic indicates likely Arab nationality
       - Multiple languages suggest expat background
    
    5. COMBINED APPROACH (when nationality not explicit):
       - Use this priority: Primary Education Country > All Work Countries > Languages > Secondary Education
       - If ambiguous, return the strongest indicator with accurate confidence score
       - Include "nationality_inferred_from" field to show reasoning
    
    Examples:
    - Person studied in Karachi, worked in Lahore: Pakistani (0.85)
    - Person studied in India, worked in Pakistan: Likely Indian (0.65)
    - Person studied in Pakistan, worked in Saudi Arabia: Likely Pakistani expat (0.75)
    
    Guidelines:
    - For missing information, use null
    - For empty arrays, use []
    - Confidence should reflect how explicitly the information is stated
    - Calculate experience_years from work history if not explicitly stated
    - Extract country_of_interest from objective/career goals
    - Keep professional_summary concise and factual
    - ALWAYS include reasoning for nationality inference in "nationality_inferred_from"
    - If uncertain about nationality, set it and provide lower confidence score with clear source
    
    CV Text:
    {cv_text}
    """
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert CV/Resume parser. Extract information accurately and provide confidence scores. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=2000
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        # Parse JSON
        extracted_data = json.loads(result_text)
        
        # Add metadata
        extracted_data['extraction_source'] = 'python-parser-v1'
        extracted_data['raw_text_length'] = len(cv_text)
        
        return extracted_data
        
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse OpenAI response as JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"OpenAI extraction failed: {str(e)}")

def main():
    """Main entry point for CV extraction"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "CV URL required as argument",
            "usage": "python extract_cv.py <cv_url>"
        }))
        sys.exit(1)
    
    cv_url = sys.argv[1]
    
    try:
        result = extract_cv_data(cv_url)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
