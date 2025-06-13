import os
import json
import re
import PyPDF2
import nltk
import spacy
import subprocess

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# Load or download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Define dataset paths using a relative path
BASE_DIR = os.path.join(os.path.dirname(__file__), "datasets")
SKILLS_FILE = os.path.join(BASE_DIR, "skills.json")
EDUCATION_FILE = os.path.join(BASE_DIR, "education.json")

# Function to load JSON data safely
def load_json(file_path, key):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {file_path} not found.")
        return {key: []}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load skills and education data
skills_set = set(load_json(SKILLS_FILE, "skills").get("skills", []))
education_keywords = set(load_json(EDUCATION_FILE, "education").get("education", []))

# Extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + " "
    except Exception as e:
        return {"error": f"PDF extraction failed: {str(e)}"}
    
    return text.strip() if text else {"error": "No text extracted"}

# Enhanced name extraction with better patterns and NER
def extract_name(text, first_n_chars=2000):
    # Increase the character limit to 2000 to capture names in longer headers
    first_portion = text[:first_n_chars]
    doc = nlp(first_portion)
    
    # Updated patterns to handle more name formats
    patterns = [
        # Name with optional title (e.g., "Dr. John Doe")
        r"(?i)(?:name\s*(?::|is|:|\-)?\s*)?(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Miss)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        # Name at the start of the document (e.g., "John Doe")
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        # Name with middle initial (e.g., "John M. Doe")
        r"(?i)(?:name\s*(?::|is|:|\-)?\s*)?(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Miss)?\s*([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)",
        # Name after a label (e.g., "Name: John Doe")
        r"(?i)name\s*(?::|is|:|\-)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        # Name in a header-like format (e.g., "John Doe | john.doe@email.com")
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[\|\-]\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, first_portion, re.MULTILINE)
        if matches:
            # Extract the name group (group 1 in all patterns)
            name = matches.group(1).strip()
            # Clean up any trailing punctuation or extra spaces
            name = re.sub(r'[^\w\s]', '', name).strip()
            return name
    
    # Enhanced spaCy NER with better filtering
    candidates = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Filter out entities that are unlikely to be names (e.g., single words, all uppercase)
            if len(ent.text.split()) >= 2 and not ent.text.isupper():
                candidates.append(ent.text)
    
    # If multiple candidates, prefer the one closest to the start of the document
    if candidates:
        # Clean up the name (remove extra spaces, punctuation, etc.)
        name = candidates[0].strip()
        name = re.sub(r'[^\w\s]', '', name).strip()
        return name
    
    # Fallback: Look for capitalized words in the first few lines
    lines = first_portion.split('\n')[:5]  # Check first 5 lines
    for line in lines:
        line = line.strip()
        # Look for two or more capitalized words (e.g., "John Doe")
        match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", line)
        if match:
            name = match.group(1).strip()
            # Ensure it's not part of a common non-name phrase (e.g., "Objective", "Education")
            if not any(keyword in name.lower() for keyword in ["objective", "education", "experience", "skills"]):
                return name
    
    return "Not Found"

# Enhanced email extraction with better regex and context
def extract_email(text):
    # Updated pattern to handle more email formats
    pattern = r"(?i)\b[A-Za-z0-9._%+-]+(?:\+[A-Za-z0-9._%+-]*)?@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    
    # First, look for emails near labels like "Email" or "Contact"
    email_section = re.search(r"(?:Email|E-mail|Contact)(?::|.)[^\n]*", text, re.IGNORECASE)
    if email_section:
        matches = re.findall(pattern, email_section.group(0))
        if matches:
            return matches[0].strip()
    
    # Fallback: Search the entire text
    matches = re.findall(pattern, text)
    if matches:
        # Clean up the email (remove surrounding whitespace)
        return matches[0].strip()
    
    # Additional fallback: Look for email-like patterns in common header sections
    header = text[:1000]  # Check first 1000 characters
    matches = re.findall(pattern, header)
    if matches:
        return matches[0].strip()
    
    return "Not Found"

# Enhanced phone number extraction with better prioritization and cleaning
def extract_contact_number(text):
    # Updated patterns to handle more formats
    patterns = [
        # Standard US format (e.g., (123) 456-7890, 123-456-7890)
        r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        # Continuous digit format (e.g., +1234567890)
        r"\b(?:\+\d{1,3})?\d{10,12}\b",
        # International format with separators (e.g., +44 123 456 7890)
        r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
        # With label (e.g., "Phone: (123) 456-7890")
        r"(?:Phone|Tel|Mobile|Contact)(?::|.)[^\n\d]*(\+?\d[\d\s\-\(\)\.]{8,15}\d)"
    ]
    
    # First, look for phone numbers near labels
    for pattern in patterns[-1:]:  # Check the "With label" pattern first
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Clean up the number (remove non-digits except +)
            cleaned_number = re.sub(r'[^\d+]', '', matches[0])
            # Ensure the number is at least 10 digits long (or 11 with country code)
            if len(cleaned_number) >= 10:
                return cleaned_number
    
    # Then try other patterns across the entire text
    for pattern in patterns[:-1]:
        matches = re.findall(pattern, text)
        if matches:
            # Clean up the number
            cleaned_number = re.sub(r'[^\d+]', '', matches[0])
            # Ensure the number is at least 10 digits long (or 11 with country code)
            if len(cleaned_number) >= 10:
                # Prefer numbers near the start of the document (likely in header)
                header = text[:1000]
                if re.search(pattern, header):
                    return cleaned_number
                return cleaned_number
    
    # Fallback: Look for any 10-12 digit sequence in the header
    header = text[:1000]
    match = re.search(r"\b(\+?\d{10,12})\b", header)
    if match:
        return match.group(1)
    
    return "Not Found"

# Enhanced skills extraction with better matching
def extract_skills(text):
    text_lower = text.lower()
    found_skills = set()
    
    # Exact match with word boundaries
    for skill in skills_set:
        skill_lower = skill.lower()
        # Find either standalone skill or skill as part of a list
        if re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
            found_skills.add(skill)
    
    # Check for skills in skills sections
    skill_sections = re.findall(r'(?:SKILLS|TECHNICAL SKILLS|EXPERTISE)(?:[^\n]*)([\s\S]*?)(?:EXPERIENCE|EDUCATION|PROJECTS|$)', text, re.IGNORECASE)
    if skill_sections:
        for section in skill_sections:
            # Split by common delimiters and check each item
            items = re.split(r'[,•|\n\-\/]', section)
            for item in items:
                item = item.strip().lower()
                for skill in skills_set:
                    if skill.lower() in item:
                        found_skills.add(skill)
    
    return list(found_skills) if found_skills else ["Not Found"]

# Enhanced education extraction
def extract_education(text):
    education_items = []
    
    # Extract education sections
    edu_sections = re.findall(r'(?:EDUCATION|ACADEMIC|QUALIFICATION)(?:[^\n]*)([\s\S]*?)(?:EXPERIENCE|SKILLS|PROJECTS|$)', text, re.IGNORECASE)
    
    if edu_sections:
        # Process each education section
        for section in edu_sections:
            # Split by common education separators like newlines or years
            items = re.split(r'(?:\n\s*\n|\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)', section)
            for item in items:
                if item.strip():
                    # Look for degree, institution, and date patterns in each item
                    degree_match = re.search(r'(Bachelor|Master|B\.Tech|M\.Tech|Intermediate|High School)', item, re.IGNORECASE)
                    college_match = re.search(r'(College|University|Institute|School)', item, re.IGNORECASE)
                    
                    if degree_match or college_match:
                        # Find year pattern
                        year_pattern = r'(20\d\d\s*-\s*20\d\d|20\d\d\s*-\s*Present|\d{4}\s*-\s*\d{4})'
                        year_match = re.search(year_pattern, item)
                        year = year_match.group(1) if year_match else ""
                        
                        # Clean the item text
                        cleaned_item = item.strip()
                        if year:
                            # Make sure the year is included in the cleaned item
                            if year not in cleaned_item:
                                cleaned_item = f"{cleaned_item} {year}"
                        
                        education_items.append(cleaned_item)
    
    # If no education items found from sections, try to extract education patterns from whole text
    if not education_items:
        # Common patterns for education entries
        education_patterns = [
            r"([^.\n]*College[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*University[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*School[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*Bachelor[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*B\.Tech[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*Intermediate[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)",
            r"([^.\n]*High School[^.\n]*\d{4}[^.\n]*\d{4}[^.\n]*)"
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                education_items.append(match.strip())
    
    # If still no education found, try splitting by years and looking for education keywords
    if not education_items:
        # Split text by year patterns and check each segment
        year_splits = re.split(r'(20\d\d\s*-\s*20\d\d|20\d\d\s*-\s*Present)', text)
        
        for i in range(1, len(year_splits), 2):
            if i+1 < len(year_splits):
                year = year_splits[i]
                content = year_splits[i+1]
                
                # Check if this segment contains education keywords
                if any(kw in content.lower() for kw in ["college", "university", "school", "bachelor", "b.tech", "intermediate"]):
                    # Find a reasonable chunk of text around education keywords
                    edu_match = re.search(r'([^.\n]{0,100}(college|university|school|bachelor|b\.tech|intermediate)[^.\n]{0,100})', content, re.IGNORECASE)
                    if edu_match:
                        education_items.append(f"{edu_match.group(1).strip()} {year}")
    
    # For each education item, try segmenting by institution/degree/year if it's too long
    structured_education = []
    for item in education_items:
        # If item contains multiple education details, split them
        parts = re.split(r'(?<=\d{4})', item)
        for part in parts:
            if len(part.strip()) > 10:  # Avoid empty parts
                structured_education.append(part.strip())
    
    if structured_education:
        return structured_education
    
    # If nothing else worked, try a different approach: look for lines with education keywords
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        if any(kw in line.lower() for kw in ["college", "university", "school", "bachelor", "b.tech", "intermediate"]):
            education_items.append(line)
    
    return education_items if education_items else ["Not Found"]

def extract_experience(text):
    experience_items = []
    
    # Identify experience sections using various section headers
    experience_sections = re.findall(
        r'(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|CAREER)(?:[^\n]*)([\s\S]*?)(?:EDUCATION|SKILLS|PROJECTS|$)', 
        text, 
        re.IGNORECASE
    )
    
    if experience_sections:
        for section in experience_sections:
            # Split experiences by typical separators (years or multiple job markers)
            job_splits = re.split(r'(?=\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)', section)
            
            for job_entry in job_splits:
                if not job_entry.strip():
                    continue
                
                # Structured experience extraction
                structured_experience = {
                    "Job Title": "Not Found",
                    "Company": "Not Found",
                    "Duration": "Not Found",
                    "Location": "Not Found",
                    "Key Responsibilities": []
                }
                
                # Job Title extraction (more robust)
                title_patterns = [
                    r'^([A-Za-z\s]+)(?=\s*at|\s*,|\s*@|$)',
                    r'([A-Za-z\s]+)\s*(?:at|@|,)\s*[A-Za-z\s]+',
                ]
                for pattern in title_patterns:
                    title_match = re.search(pattern, job_entry, re.MULTILINE)
                    if title_match:
                        structured_experience["Job Title"] = title_match.group(1).strip()
                        break
                
                # Company Name extraction
                company_patterns = [
                    r'(?:at|@|,)\s*([A-Za-z\s&]+)(?=\s*\d{4}|\s*-)',
                    r'[A-Za-z\s]+\s*(?:at|@|,)\s*([A-Za-z\s&]+)'
                ]
                for pattern in company_patterns:
                    company_match = re.search(pattern, job_entry, re.IGNORECASE)
                    if company_match:
                        structured_experience["Company"] = company_match.group(1).strip()
                        break
                
                # Location extraction (optional)
                location_match = re.search(r'([A-Za-z\s]+,\s*[A-Z]{2}|[A-Za-z\s]+)', job_entry)
                if location_match:
                    structured_experience["Location"] = location_match.group(1).strip()
                
                # Date/Duration extraction
                date_match = re.search(r'(\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)', job_entry)
                if date_match:
                    structured_experience["Duration"] = date_match.group(1).strip()
                
                # Responsibilities extraction (multiple methods)
                responsibility_patterns = [
                    r'•\s*(.+?)(?=\n•|\n\n|$)',  # Bullet point with •
                    r'[-]\s*(.+?)(?=\n[-]|\n\n|$)',  # Bullet point with -
                    r'^\s*[•-]\s*(.+?)$'  # Multiline bullet points
                ]
                
                responsibilities = []
                for pattern in responsibility_patterns:
                    resp_matches = re.findall(pattern, job_entry, re.MULTILINE | re.DOTALL)
                    cleaned_resps = [
                        resp.strip() for resp in resp_matches 
                        if resp.strip() and len(resp.strip()) > 10
                    ]
                    responsibilities.extend(cleaned_resps)
                
                # Remove duplicates while preserving order
                unique_responsibilities = []
                for resp in responsibilities:
                    if resp not in unique_responsibilities:
                        unique_responsibilities.append(resp)
                
                structured_experience["Key Responsibilities"] = unique_responsibilities
                
                # Add to experience items if meaningful information is found
                if any([
                    structured_experience["Job Title"] != "Not Found", 
                    structured_experience["Company"] != "Not Found", 
                    structured_experience["Key Responsibilities"]
                ]):
                    experience_items.append(structured_experience)
    
    # Fallback extraction method
    if not experience_items:
        # Alternative pattern matching across entire text
        job_patterns = [
            r'([A-Za-z\s]+)\s*(?:at|@)\s*([A-Za-z\s]+)\s*(\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)',
            r'([A-Za-z\s]+)\s*,\s*([A-Za-z\s]+)\s*(\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*Present)'
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                experience_items.append({
                    "Job Title": match[0].strip(),
                    "Company": match[1].strip(),
                    "Duration": match[2].strip(),
                    "Location": "Not Found",
                    "Key Responsibilities": []
                })
    
    return experience_items if experience_items else ["Not Found"]

# Updated extract_resume_details to preserve the structured Experience data
def extract_resume_details(file_path):
    text = extract_text_from_pdf(file_path)
    if isinstance(text, dict):  # Handle extraction errors
        return text
    
    # Extract education as a list of items
    education_list = extract_education(text)
    
    # Format education points with bullet points for display
    formatted_education = []
    for edu_item in education_list:
        formatted_edu = edu_item.strip()
        # Clean up any excessive spaces or formatting issues
        formatted_edu = re.sub(r'\s+', ' ', formatted_edu)
        formatted_education.append(formatted_edu)
    
    # Extract experience as a list of dictionaries
    experience_list = extract_experience(text)
    
    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Contact Number": extract_contact_number(text),
        "Skills": extract_skills(text),
        "Education": formatted_education,
        "Experience": experience_list,  # Keep as a list of dictionaries
        "Raw Text": text[:200] + "..."  # Show only first 200 chars
    }
