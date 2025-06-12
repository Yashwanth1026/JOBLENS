import re
import PyPDF2
import os
import spacy
import textdistance
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Load spaCy model for NER and advanced text processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model... This might take a moment.")
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
        return text.lower()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([para.text.lower() for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
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
        results['warnings'].append(f"Missing sections: {', '.join(missing_sections)}")
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
                results['recommendations'].append("⚠️ Consider placing work experience before education (unless you're a recent graduate).")
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
        results['warnings'].append("❗ Missing email address.")
        results['score'] -= 5
    
    if not phones:
        results['warnings'].append("❗ Missing phone number.")
        results['score'] -= 5
    
    if not has_linkedin:
        results['recommendations'].append("💡 Consider adding your LinkedIn profile.")
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
        results['warnings'].append("❗ Inconsistent date formats detected. Stick to one format.")
        results['score'] -= 5
    elif len(found_formats) == 0:
        results['warnings'].append("❗ No standard date formats detected.")
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
        results['recommendations'].append(
            f"💡 Add more action verbs. Consider: {', '.join(missing_keywords[:5])}"
        )
        results['score'] -= 7
    elif len(found_keywords) < 10:
        results['recommendations'].append(
            f"💡 Consider adding more action verbs: {', '.join(missing_keywords[:3])}"
        )
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
        results['warnings'].append("❗ Inconsistent bullet point styles. Stick to one type.")
        results['score'] -= 3

    # Check for keyword optimization using job description
    if job_description:
        similarity_score = calculate_tfidf_similarity(resume_text, job_description)
        results['details']['keywords']['job_description_similarity'] = round(similarity_score * 100, 2)
        
        if similarity_score < 0.35:
            results['recommendations'].append(
                f"⚡️ Increase keyword alignment with job description. Similarity Score: {round(similarity_score * 100, 2)}%"
            )
            results['score'] -= 10
            
            # Extract key missing terms from job description
            missing_terms = extract_key_missing_terms(resume_text, job_description)
            if missing_terms:
                results['recommendations'].append(
                    f"💡 Consider adding these keywords from the job description: {', '.join(missing_terms[:5])}"
                )
                results['details']['keywords']['missing_job_keywords'] = missing_terms

    # Check for spacing issues
    double_spaces = len(re.findall(r'\s\s+', resume_text))
    if double_spaces > 5:
        results['warnings'].append("❗ Multiple double spaces detected. Check formatting.")
        results['score'] -= 3
        results['details']['formatting']['double_spaces'] = double_spaces

    # Check for forbidden elements (tables, columns, graphics)
    formatting_issues = []
    if re.search(r'table|column|graphic|image', resume_text):
        formatting_issues.append("tables/columns/graphics")
        results['warnings'].append("❗ Avoid using tables, columns, or graphics.")
        results['score'] -= 5
    
    # Check for hyperlinks
    hyperlinks = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', resume_text)
    if hyperlinks:
        formatting_issues.append("hyperlinks")
        results['recommendations'].append(
            "⚠️ Avoid using hyperlinks – they may not parse correctly in ATS."
        )
        results['score'] -= 5
        results['details']['formatting']['hyperlinks'] = hyperlinks

    results['details']['formatting']['issues'] = formatting_issues

    # Check word count
    word_count = len(re.findall(r'\w+', resume_text))
    results['details']['content']['word_count'] = word_count
    
    if word_count < 300:
        results['warnings'].append("⚠️ Resume might be too short (under 300 words).")
        results['score'] -= 10
    elif word_count > 800:
        results['recommendations'].append(
            "⚠️ Resume might be too long (over 800 words)."
        )
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
                results['recommendations'].append("💡 Add more specific skills to your skills section.")
                results['score'] -= 5
            
            # Check for skill categorization
            skill_categories = ['technical', 'soft', 'language', 'tool']
            has_categories = any(category in skills_text.lower() for category in skill_categories)
            
            if not has_categories and len(skill_words) > 10:
                results['recommendations'].append("💡 Consider categorizing your skills for better readability.")
                results['score'] -= 2

    # Check for personal pronouns
    pronouns = ['i', 'me', 'my', 'mine', 'myself']
    pronoun_count = sum(resume_text.lower().count(f" {p} ") for p in pronouns)
    results['details']['content']['pronoun_count'] = pronoun_count
    
    if pronoun_count > 5:
        results['recommendations'].append("⚠️ Avoid using personal pronouns (I, me, my) in your resume.")
        results['score'] -= 5

    # Check for file name
    # This would require access to the file name, which we don't have directly
    # But we can add a recommendation
    results['recommendations'].append("💡 Ensure your file name follows the format: FirstName_LastName_Resume.pdf")

    # Check for quantifiable achievements
    quantifiable_pattern = r'\d+%|\$\d+|\d+ years|\d+ months|\d+ people|\d+ team|\d+ project|\d+ client'
    quantifiables = re.findall(quantifiable_pattern, resume_text)
    results['details']['content']['quantifiable_achievements'] = quantifiables
    
    if len(quantifiables) < 3:
        results['recommendations'].append("💡 Add more quantifiable achievements (%, $, numbers).")
        results['score'] -= 5

    # Check for education details
    if 'education' in found_sections:
        degree_pattern = r'bachelor|master|ph\.?d|associate|diploma|certificate'
        has_degree = bool(re.search(degree_pattern, resume_text, re.IGNORECASE))
        
        if not has_degree:
            results['recommendations'].append("💡 Specify your degree type in the education section.")
            results['score'] -= 3

    # Check for acronyms with definitions
    acronyms = re.findall(r'\b[A-Z]{2,}\b', resume_text)
    defined_acronyms = re.findall(r'\([A-Z]{2,}\)', resume_text)
    
    if len(acronyms) > len(defined_acronyms) + 3:
        results['recommendations'].append("💡 Consider defining industry-specific acronyms.")
        results['score'] -= 2

    # Check for repeated words
    words = re.findall(r'\b\w+\b', resume_text.lower())
    word_counts = Counter(words)
    repeated_words = [word for word, count in word_counts.items() 
                     if count > 5 and word not in ['and', 'the', 'to', 'of', 'in', 'for', 'with', 'on', 'at']]
    
    if repeated_words:
        results['details']['content']['overused_words'] = repeated_words
        results['recommendations'].append(f"💡 Avoid overusing these words: {', '.join(repeated_words[:3])}")
        results['score'] -= 2

    # Final score capping
    results['score'] = max(0, min(results['score'], 100))

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
    
    return missing_keywords

# === TF-IDF Similarity for Keyword Optimization ===
def calculate_tfidf_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([resume_text, job_description.lower()])
        similarity_score = (vectors[0] * vectors[1].T).toarray()[0][0]
        return similarity_score
    except:
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
    
    if score >= 80:
        report.append("\n🎉 Status: ATS-Friendly Resume! Ready for submission.")
    elif score >= 60:
        report.append("\n🔄 Status: Resume needs minor improvements before submission.")
    else:
        report.append("\n⚠️ Status: Resume needs significant improvements before submission.")
    
    # Print report
    for line in report:
        print(line)
    
    # Save report to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            for line in report:
                f.write(line + "\n")
        print(f"\nReport saved to {output_file}")

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
    
    return {
        "industry": industry,
        "industry_relevance": len(present_keywords) / len(relevant_keywords) if relevant_keywords else 0,
        "present_keywords": present_keywords,
        "missing_keywords": [kw for kw in relevant_keywords if kw not in resume_text.lower()]
    }

# === File Format Checker ===
def check_file_format(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    
    format_issues = []
    
    if file_ext not in ['.pdf', '.docx']:
        format_issues.append(f"File format '{file_ext}' may not be ATS-friendly. Use .pdf or .docx")
    
    if file_size > 5:
        format_issues.append(f"File size ({file_size:.2f}MB) exceeds 5MB, which may cause issues with some ATS")
    
    if file_ext == '.pdf':
        # Check if PDF is searchable
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                first_page = reader.pages[0]
                text = first_page.extract_text()
                if not text or len(text) < 100:
                    format_issues.append("PDF may not be searchable/machine-readable")
        except Exception as e:
            format_issues.append(f"Error checking PDF: {e}")
    
    return {
        "format": file_ext,
        "size_mb": file_size,
        "issues": format_issues
    }

# === Main Function ===
def main():
    print("🔍 ATS Resume Checker v2.0")
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
    
    # Extract text from resume
    print("\nExtracting text from resume...")
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        print("❗️ Unsupported file format. Please use PDF or DOCX.")
        return
    
    if not text:
        print("❌ Failed to extract text from the resume.")
        return
    
    # Analyze the resume
    print("Analyzing resume...")
    analysis = check_ats_compatibility(text, job_description)
    
    # Benchmark against industry
    if industry:
        benchmark = benchmark_against_industry(text, industry)
        analysis['details']['industry_benchmark'] = benchmark
        
        # Add industry-specific recommendations
        if benchmark['industry_relevance'] < 0.5:
            analysis['recommendations'].append(
                f"💡 Add more {benchmark['industry']}-specific keywords: {', '.join(benchmark['missing_keywords'][:3])}"
            )

    # Generate and save report
    save_report = input("Do you want to save the report to a file? (y/n): ")
    output_file = None
    if save_report.lower() == 'y':
        output_file = os.path.splitext(file_path)[0] + "_ats_report.txt"
    
    generate_report(analysis, output_file)

