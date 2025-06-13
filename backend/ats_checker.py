import re
import PyPDF2
import os
import spacy
import textdistance
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import logging

# Set up logging for Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load spaCy model for NER and advanced text processing
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning("Installing spaCy model due to: %s", str(e))
    import subprocess
    subprocess.call("python -m spacy download en_core_web_sm", shell=True)
    nlp = spacy.load("en_core_web_sm")

# === Extract Text Functions ===
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        logger.debug("Successfully extracted text from PDF: %s", pdf_path)
        return text.lower()
    except Exception as e:
        logger.error("Error extracting text from PDF %s: %s", pdf_path, str(e))
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text.lower() for para in doc.paragraphs])
        logger.debug("Successfully extracted text from DOCX: %s", docx_path)
        return text
    except Exception as e:
        logger.error("Error extracting text from DOCX %s: %s", docx_path, str(e))
        return ""

# === Check ATS Compatibility ===
def check_ats_compatibility(resume_text, job_description=""):
    results = {
        'warnings': [],
        'recommendations': [],
        'score': 100,
        'details': {
            'structure': {},
            'formatting': {},
            'content': {},
            'keywords': {}
        }
    }

    # Check for important sections
    required_sections = [
        ('summary', r'summary|objective|profile'),
        ('experience', r'experience|work history|employment'),
        ('education', r'education|academic background|qualifications'),
        ('skills', r'skills|technical skills|competencies'),
        ('certifications', r'certifications|licenses|accreditations'),
        ('projects', r'projects|portfolio|achievements'),
        ('contact', r'contact|details|information')
    ]
    
    found_sections = []
    missing_sections = []
    for section, pattern in required_sections:
        if re.search(pattern, resume_text):
            found_sections.append(section)
        else:
            missing_sections.append(section)
    
    results['details']['structure']['found_sections'] = found_sections
    results['details']['structure']['missing_sections'] = missing_sections
    
    if missing_sections:
        warning = f"Missing sections: {', '.join(missing_sections)}"
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= min(5 * len(missing_sections), 20)  # Cap at 20 points penalty

    # Check section order
    ideal_order = ['contact', 'summary', 'experience', 'education', 'skills', 'projects', 'certifications']
    section_positions = {}
    for section in found_sections:
        for idx, (name, pattern) in enumerate(required_sections):
            if name == section:
                matches = list(re.finditer(pattern, resume_text))
                if matches:
                    section_positions[section] = matches[0].start()
    
    # Check if sections are in typical order
    if len(section_positions) > 1:
        ordered_sections = [k for k, v in sorted(section_positions.items(), key=lambda item: item[1])]
        results['details']['structure']['section_order'] = ordered_sections
        
        # Check if experience comes before education
        if 'experience' in ordered_sections and 'education' in ordered_sections:
            exp_idx = ordered_sections.index('experience')
            edu_idx = ordered_sections.index('education')
            if edu_idx < exp_idx:
                recommendation = "⚠️ Consider placing work experience before education (unless you're a recent graduate)."
                results['recommendations'].append(recommendation)
                logger.info(recommendation)
                results['score'] -= 3

    # Check contact information
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
    has_linkedin = bool(re.search(r'linkedin\.com', resume_text))
    
    results['details']['content']['contact_info'] = {
        'emails': emails,
        'phones': phones,
        'has_linkedin': has_linkedin
    }
    
    if not emails:
        warning = "❗ Missing email address."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 5
    
    if not phones:
        warning = "❗ Missing phone number."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 5
    
    if not has_linkedin:
        recommendation = "💡 Consider adding your LinkedIn profile."
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 2

    # Check date format consistency
    date_formats = {
        'month_year': re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}'),
        'mm_yyyy': re.compile(r'\d{2}/\d{4}'),
        'mm_dd_yyyy': re.compile(r'\d{2}/\d{2}/\d{4}'),
        'yyyy': re.compile(r'\b(19|20)\d{2}\b')
    }
    
    found_formats = {}
    for format_name, pattern in date_formats.items():
        dates = pattern.findall(resume_text)
        if dates:
            found_formats[format_name] = len(dates)
    
    results['details']['formatting']['date_formats'] = found_formats
    
    if len(found_formats) > 1:
        warning = "❗ Inconsistent date formats detected. Stick to one format."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 5
    elif len(found_formats) == 0:
        warning = "❗ No standard date formats detected."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 5

    # Check for action-oriented keywords
    action_keywords = [
        'managed', 'developed', 'achieved', 'improved', 'created',
        'led', 'implemented', 'analyzed', 'designed', 'collaborated',
        'coordinated', 'executed', 'facilitated', 'generated', 'increased',
        'negotiated', 'operated', 'performed', 'resolved', 'streamlined',
        'supervised', 'trained', 'upgraded', 'utilized', 'won'
    ]
    
    found_keywords = [kw for kw in action_keywords if kw in resume_text]
    missing_keywords = [kw for kw in action_keywords if kw not in resume_text]
    
    results['details']['content']['action_verbs'] = {
        'found': found_keywords,
        'missing': missing_keywords,
        'usage_ratio': len(found_keywords) / len(action_keywords)
    }
    
    if len(found_keywords) < 5:
        recommendation = f"💡 Add more action verbs. Consider: {', '.join(missing_keywords[:5])}"
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 7
    elif len(found_keywords) < 10:
        recommendation = f"💡 Consider adding more action verbs: {', '.join(missing_keywords[:3])}"
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 3

    # Check bullet point consistency
    bullet_patterns = [
        r'•\s', r'-\s', r'\*\s', r'✓\s', r'➢\s', r'○\s', r'►\s', r'∙\s', r'■\s', r'★\s'
    ]
    
    bullet_counts = {}
    for pattern in bullet_patterns:
        count = len(re.findall(pattern, resume_text))
        if count > 0:
            bullet_counts[pattern] = count
    
    results['details']['formatting']['bullet_types'] = bullet_counts
    
    if len(bullet_counts) > 1:
        warning = "❗ Inconsistent bullet point styles. Stick to one type."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 3

    # Check for keyword optimization using job description
    if job_description:
        similarity_score = calculate_tfidf_similarity(resume_text, job_description)
        results['details']['keywords']['job_description_similarity'] = round(similarity_score * 100, 2)
        
        if similarity_score < 0.35:
            recommendation = f"⚡️ Increase keyword alignment with job description. Similarity Score: {round(similarity_score * 100, 2)}%"
            results['recommendations'].append(recommendation)
            logger.info(recommendation)
            results['score'] -= 10
            
            # Extract key missing terms from job description
            missing_terms = extract_key_missing_terms(resume_text, job_description)
            if missing_terms:
                recommendation = f"💡 Consider adding these keywords from the job description: {', '.join(missing_terms[:5])}"
                results['recommendations'].append(recommendation)
                logger.info(recommendation)
                results['details']['keywords']['missing_job_keywords'] = missing_terms

    # Check for spacing issues
    double_spaces = len(re.findall(r'\s\s+', resume_text))
    if double_spaces > 5:
        warning = "❗ Multiple double spaces detected. Check formatting."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 3
        results['details']['formatting']['double_spaces'] = double_spaces

    # Check for forbidden elements (tables, columns, graphics)
    formatting_issues = []
    if re.search(r'table|column|graphic|image', resume_text):
        formatting_issues.append("tables/columns/graphics")
        warning = "❗ Avoid using tables, columns, or graphics."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 5
    
    # Check for hyperlinks
    hyperlinks = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', resume_text)
    if hyperlinks:
        formatting_issues.append("hyperlinks")
        recommendation = "⚠️ Avoid using hyperlinks – they may not parse correctly in ATS."
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 5
        results['details']['formatting']['hyperlinks'] = hyperlinks

    results['details']['formatting']['issues'] = formatting_issues

    # Check word count
    word_count = len(re.findall(r'\w+', resume_text))
    results['details']['content']['word_count'] = word_count
    
    if word_count < 300:
        warning = "⚠️ Resume might be too short (under 300 words)."
        results['warnings'].append(warning)
        logger.warning(warning)
        results['score'] -= 10
    elif word_count > 800:
        recommendation = "⚠️ Resume might be too long (over 800 words)."
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 5

    # Check for skill section quality
    if 'skills' in found_sections:
        skills_pattern = r'skills.*?(?:experience|education|projects|certifications|references|$)'
        skills_section = re.search(skills_pattern, resume_text, re.DOTALL | re.IGNORECASE)
        if skills_section:
            skills_text = skills_section.group(0)
            skill_words = extract_skills(skills_text)
            
            results['details']['content']['skills'] = skill_words
            
            if len(skill_words) < 5:
                recommendation = "💡 Add more specific skills to your skills section."
                results['recommendations'].append(recommendation)
                logger.info(recommendation)
                results['score'] -= 5
            
            # Check for skill categorization
            skill_categories = ['technical', 'soft', 'language', 'tool']
            has_categories = any(category in skills_text.lower() for category in skill_categories)
            
            if not has_categories and len(skill_words) > 10:
                recommendation = "💡 Consider categorizing your skills for better readability."
                results['recommendations'].append(recommendation)
                logger.info(recommendation)
                results['score'] -= 2

    # Check for personal pronouns
    pronouns = ['i', 'me', 'my', 'mine', 'myself']
    pronoun_count = sum(resume_text.lower().count(f" {p} ") for p in pronouns)
    results['details']['content']['pronoun_count'] = pronoun_count
    
    if pronoun_count > 5:
        recommendation = "⚠️ Avoid using personal pronouns (I, me, my) in your resume."
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 5

    # Check for file name recommendation (cannot check actual file name in this context)
    results['recommendations'].append("💡 Ensure your file name follows the format: FirstName_LastName_Resume.pdf")
    logger.info("Added recommendation for file name format.")

    # Check for quantifiable achievements
    quantifiable_pattern = r'\d+%|\$\d+|\d+ years|\d+ months|\d+ people|\d+ team|\d+ project|\d+ client'
    quantifiables = re.findall(quantifiable_pattern, resume_text)
    results['details']['content']['quantifiable_achievements'] = quantifiables
    
    if len(quantifiables) < 3:
        recommendation = "💡 Add more quantifiable achievements (%, $, numbers)."
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 5

    # Check for education details
    if 'education' in found_sections:
        degree_pattern = r'bachelor|master|ph\.?d|associate|diploma|certificate'
        has_degree = bool(re.search(degree_pattern, resume_text, re.IGNORECASE))
        
        if not has_degree:
            recommendation = "💡 Specify your degree type in the education section."
            results['recommendations'].append(recommendation)
            logger.info(recommendation)
            results['score'] -= 3

    # Check for acronyms with definitions
    acronyms = re.findall(r'\b[A-Z]{2,}\b', resume_text)
    defined_acronyms = re.findall(r'\([A-Z]{2,}\)', resume_text)
    
    if len(acronyms) > len(defined_acronyms) + 3:
        recommendation = "💡 Consider defining industry-specific acronyms."
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 2

    # Check for repeated words
    words = re.findall(r'\b\w+\b', resume_text.lower())
    word_counts = Counter(words)
    repeated_words = [word for word, count in word_counts.items() 
                     if count > 5 and word not in ['and', 'the', 'to', 'of', 'in', 'for', 'with', 'on', 'at']]
    
    if repeated_words:
        results['details']['content']['overused_words'] = repeated_words
        recommendation = f"💡 Avoid overusing these words: {', '.join(repeated_words[:3])}"
        results['recommendations'].append(recommendation)
        logger.info(recommendation)
        results['score'] -= 2

    # Final score capping
    results['score'] = max(0, min(results['score'], 100))

    logger.info("ATS compatibility check completed. Score: %d", results['score'])
    return results

# === Extract Skills from Text ===
def extract_skills(text):
    # Common skill keywords
    technical_skills = [
        'python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'node', 'aws',
        'azure', 'gcp', 'docker', 'kubernetes', 'linux', 'excel', 'powerpoint', 'word',
        'tableau', 'power bi', 'machine learning', 'data analysis', 'project management'
    ]
    
    found_skills = []
    for skill in technical_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            found_skills.append(skill)
    
    # Use NLP to extract potential skills
    doc = nlp(text)
    
    # Extract noun phrases that might be skills
    for chunk in doc.noun_chunks:
        if 2 <= len(chunk.text.split()) <= 4 and chunk.text.lower() not in found_skills:
            found_skills.append(chunk.text.lower())
    
    logger.debug("Extracted skills: %s", found_skills)
    return found_skills

# === Extract Missing Keywords ===
def extract_key_missing_terms(resume_text, job_description):
    # Extract important words from job description
    job_doc = nlp(job_description.lower())
    job_keywords = set()
    
    # Add nouns and proper nouns from job description
    for token in job_doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
            job_keywords.add(token.text)
    
    # Extract noun phrases (potential skill terms)
    for chunk in job_doc.noun_chunks:
        if 2 <= len(chunk.text.split()) <= 4:
            job_keywords.add(chunk.text)
    
    # Find missing keywords
    missing_keywords = []
    for keyword in job_keywords:
        if keyword not in resume_text.lower() and len(keyword) > 3:
            # Check if any similar term exists in resume
            has_similar = False
            for word in resume_text.lower().split():
                if word.strip() and len(word) > 3:
                    similarity = textdistance.levenshtein.normalized_similarity(keyword, word)
                    if similarity > 0.8:
                        has_similar = True
                        break
            
            if not has_similar:
                missing_keywords.append(keyword)
    
    logger.debug("Missing keywords from job description: %s", missing_keywords)
    return missing_keywords

# === TF-IDF Similarity for Keyword Optimization ===
def calculate_tfidf_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([resume_text, job_description.lower()])
        similarity_score = (vectors[0] * vectors[1].T).toarray()[0][0]
        logger.debug("TF-IDF similarity score: %.2f", similarity_score)
        return similarity_score
    except Exception as e:
        logger.error("Error calculating TF-IDF similarity: %s", str(e))
        return 0.0

# === Final Report Generation ===
def generate_report(analysis, output_file=None):
    report = []
    report.append("\n=== 📊 ATS Compatibility Report ===")
    
    # Score color based on value
    score = analysis['score']
    if score >= 80:
        score_color = "🟢"
    elif score >= 60:
        score_color = "🟡"
    else:
        score_color = "🔴"
    
    report.append(f"{score_color} Overall Score: {score}/100")

    if analysis['warnings']:
        report.append("\n⚠️ Warnings:")
        for warning in analysis['warnings']:
            report.append(f"- {warning}")

    if analysis['recommendations']:
        report.append("\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            report.append(f"- {rec}")

    # Add detailed section
    report.append("\n=== 🔍 Detailed Analysis ===")
    
    # Structure details
    structure = analysis['details']['structure']
    report.append("\n📑 Document Structure:")
    if structure.get('found_sections'):
        report.append(f"- Found sections: {', '.join(structure['found_sections'])}")
    if structure.get('missing_sections'):
        report.append(f"- Missing sections: {', '.join(structure['missing_sections'])}")
    if structure.get('section_order'):
        report.append(f"- Section order: {' → '.join(structure['section_order'])}")
    
    # Content details
    content = analysis['details']['content']
    report.append("\n📝 Content Analysis:")
    if 'word_count' in content:
        report.append(f"- Word count: {content['word_count']}")
    if 'pronoun_count' in content:
        report.append(f"- Personal pronoun usage: {content['pronoun_count']}")
    if 'quantifiable_achievements' in content:
        report.append(f"- Quantifiable achievements: {len(content['quantifiable_achievements'])}")
    if 'skills' in content:
        report.append(f"- Skills detected: {len(content['skills'])}")
        if len(content['skills']) > 0:
            report.append(f"  Sample skills: {', '.join(content['skills'][:5])}")
    if 'action_verbs' in content:
        report.append(f"- Action verb usage: {int(content['action_verbs']['usage_ratio'] * 100)}%")
    if 'overused_words' in content:
        report.append(f"- Overused words: {', '.join(content['overused_words'][:5])}")
    
    # Keywords details
    keywords = analysis['details']['keywords']
    if keywords:
        report.append("\n🔑 Keyword Analysis:")
        if 'job_description_similarity' in keywords:
            report.append(f"- Job description similarity: {keywords['job_description_similarity']}%")
        if 'missing_job_keywords' in keywords:
            report.append(f"- Missing keywords: {', '.join(keywords['missing_job_keywords'][:5])}")
    
    # Formatting details
    formatting = analysis['details']['formatting']
    report.append("\n🖋️ Formatting Analysis:")
    if 'bullet_types' in formatting:
        report.append(f"- Bullet point styles: {len(formatting['bullet_types'])}")
    if 'date_formats' in formatting:
        report.append(f"- Date format types: {len(formatting['date_formats'])}")
    if 'issues' in formatting:
        if formatting['issues']:
            report.append(f"- Formatting issues: {', '.join(formatting['issues'])}")
        else:
            report.append("- No major formatting issues detected")
    
    # Industry benchmark details (if available)
    if 'industry_benchmark' in analysis['details']:
        benchmark = analysis['details']['industry_benchmark']
        report.append("\n🏢 Industry Benchmarking:")
        report.append(f"- Detected Industry: {benchmark['industry'].capitalize()}")
        report.append(f"- Industry Relevance: {int(benchmark['industry_relevance'] * 100)}%")
        if benchmark['present_keywords']:
            report.append(f"- Present Keywords: {', '.join(benchmark['present_keywords'][:5])}")
        if benchmark['missing_keywords']:
            report.append(f"- Missing Keywords: {', '.join(benchmark['missing_keywords'][:5])}")
    
    if score >= 80:
        report.append("\n🎉 Status: ATS-Friendly Resume! Ready for submission.")
    elif score >= 60:
        report.append("\n🔄 Status: Resume needs minor improvements before submission.")
    else:
        report.append("\n⚠️ Status: Resume needs significant improvements before submission.")
    
    # Convert report to string
    report_str = "\n".join(report)
    
    # Log the report
    logger.info("ATS Report Generated:\n%s", report_str)
    
    # Save report to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_str)
            logger.info("Report saved to %s", output_file)
        except Exception as e:
            logger.error("Failed to save report to %s: %s", output_file, str(e))
    
    return report_str

# === Resume Benchmarking ===
def benchmark_against_industry(resume_text, industry=""):
    industry_keywords = {
        "software": ["python", "java", "javascript", "react", "node", "aws", "cloud", "agile", "scrum", "git"],
        "finance": ["financial", "accounting", "excel", "analysis", "investment", "banking", "portfolio", "forecasting"],
        "marketing": ["marketing", "social media", "campaign", "seo", "content", "analytics", "brand", "strategy"],
        "healthcare": ["patient", "clinical", "medical", "healthcare", "treatment", "diagnosis", "care", "hospital"],
        "data": ["data", "analysis", "statistics", "machine learning", "sql", "python", "r", "tableau", "visualization"]
    }
    
    if not industry or industry.lower() not in industry_keywords:
        # Auto-detect industry
        max_matches = 0
        detected_industry = ""
        for ind, keywords in industry_keywords.items():
            matches = sum(1 for kw in keywords if kw in resume_text.lower())
            if matches > max_matches:
                max_matches = matches
                detected_industry = ind
        
        industry = detected_industry if detected_industry else "general"
    
    # Get relevant keywords for the industry
    relevant_keywords = industry_keywords.get(industry.lower(), [])
    
    # Check how many industry keywords are present
    present_keywords = [kw for kw in relevant_keywords if kw in resume_text.lower()]
    
    benchmark_result = {
        "industry": industry,
        "industry_relevance": len(present_keywords) / len(relevant_keywords) if relevant_keywords else 0,
        "present_keywords": present_keywords,
        "missing_keywords": [kw for kw in relevant_keywords if kw not in resume_text.lower()]
    }
    
    logger.debug("Industry benchmark result: %s", benchmark_result)
    return benchmark_result

# === File Format Checker ===
def check_file_format(file_path):
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        format_issues = []
        
        if file_ext not in ['.pdf', '.docx']:
            issue = f"File format '{file_ext}' may not be ATS-friendly. Use .pdf or .docx"
            format_issues.append(issue)
            logger.warning(issue)
        
        if file_size > 5:
            issue = f"File size ({file_size:.2f}MB) exceeds 5MB, which may cause issues with some ATS"
            format_issues.append(issue)
            logger.warning(issue)
        
        if file_ext == '.pdf':
            # Check if PDF is searchable
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    first_page = reader.pages[0]
                    text = first_page.extract_text()
                    if not text or len(text) < 100:
                        issue = "PDF may not be searchable/machine-readable"
                        format_issues.append(issue)
                        logger.warning(issue)
            except Exception as e:
                issue = f"Error checking PDF: {str(e)}"
                format_issues.append(issue)
                logger.error(issue)
        
        result = {
            "format": file_ext,
            "size_mb": file_size,
            "issues": format_issues
        }
        
        logger.info("File format check result for %s: %s", file_path, result)
        return result
    except Exception as e:
        logger.error("Error in check_file_format for %s: %s", file_path, str(e))
        return {
            "format": "unknown",
            "size_mb": 0,
            "issues": [f"Error accessing file: {str(e)}"]
        }

# === Function for Server Use ===
def analyze_resume(file_path, job_description="", industry=""):
    """
    Analyze a resume file for ATS compatibility.
    Suitable for use in a server environment like Render.
    
    Args:
        file_path (str): Path to the resume file (PDF or DOCX)
        job_description (str): Job description text for keyword matching
        industry (str): Industry for benchmarking (optional)
    
    Returns:
        dict: Analysis results and report
    """
    logger.info("Starting ATS analysis for file: %s", file_path)
    
    # Check file format
    format_check = check_file_format(file_path)
    if format_check["issues"]:
        logger.warning("File format issues detected: %s", format_check["issues"])
    
    # Extract text from resume
    logger.info("Extracting text from resume...")
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        error_msg = "Unsupported file format. Please use PDF or DOCX."
        logger.error(error_msg)
        return {
            "error": error_msg,
            "format_check": format_check
        }
    
    if not text:
        error_msg = "Failed to extract text from the resume."
        logger.error(error_msg)
        return {
            "error": error_msg,
            "format_check": format_check
        }
    
    # Analyze the resume
    logger.info("Analyzing resume...")
    analysis = check_ats_compatibility(text, job_description)
    
    # Benchmark against industry
    if industry:
        benchmark = benchmark_against_industry(text, industry)
        analysis['details']['industry_benchmark'] = benchmark
        
        # Add industry-specific recommendations
        if benchmark['industry_relevance'] < 0.5:
            recommendation = f"💡 Add more {benchmark['industry']}-specific keywords: {', '.join(benchmark['missing_keywords'][:3])}"
            analysis['recommendations'].append(recommendation)
            logger.info(recommendation)
    
    # Generate report (without saving to file, as this is server-side)
    report = generate_report(analysis)
    
    return {
        "analysis": analysis,
        "report": report,
        "format_check": format_check
    }

# === Main Function for CLI Use ===
def main():
    print("🔍 ATS Resume Checker v2.1")
    print("==========================")
    file_path = input("📄 Enter resume file path (PDF/DOCX): ")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Check file format
    format_check = check_file_format(file_path)
    if format_check["issues"]:
        print("\n⚠️ File Format Issues:")
        for issue in format_check["issues"]:
            print(f"- {issue}")
        
        proceed = input("\nDo you want to continue with the analysis? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Get job description
    job_description = input("📝 Enter job description for keyword matching (optional): ")
    
    # Get industry for benchmarking
    industry = input("🏢 Enter industry for benchmarking (software/finance/marketing/healthcare/data) (optional): ")
    
    # Run analysis
    result = analyze_resume(file_path, job_description, industry)
    
    if "error" in result:
        print(result["error"])
        return
    
    # Print the report
    print(result["report"])
    
    # Optionally save the report
    save_report = input("Do you want to save the report to a file? (y/n): ")
    if save_report.lower() == 'y':
        output_file = os.path.splitext(file_path)[0] + "_ats_report.txt"
        with open(output_file, 'w') as f:
            f.write(result["report"])
        print(f"\nReport saved to {output_file}")

if __name__ == "__main__":
    main()
