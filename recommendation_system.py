import pandas as pd

# Generate synthetic course data for multiple domains
courses = [
    {"id": 1, "title": "Python for Data Science", "description": "Learn Python programming and data analysis with Pandas, NumPy, and Matplotlib.", "domain": "IT"},
    {"id": 2, "title": "Web Development with Flask", "description": "Build web applications using Flask, Jinja2, and SQLAlchemy.", "domain": "IT"},
    {"id": 3, "title": "Machine Learning with Scikit-Learn", "description": "Understand and implement machine learning algorithms using Scikit-Learn.", "domain": "IT"},
    {"id": 4, "title": "Deep Learning with TensorFlow", "description": "Dive into deep learning with TensorFlow and Keras.", "domain": "IT"},
    {"id": 5, "title": "DevOps with Docker and Kubernetes", "description": "Learn DevOps principles and tools like Docker and Kubernetes.", "domain": "IT"},
    {"id": 6, "title": "Introduction to Business Management", "description": "Learn the fundamentals of business management, including planning, organizing, leading, and controlling.", "domain": "Business"},
    {"id": 7, "title": "Financial Accounting Basics", "description": "Understand the basics of financial accounting, including balance sheets, income statements, and cash flow.", "domain": "Business"},
    {"id": 8, "title": "Marketing Principles", "description": "Learn the principles of marketing, including market research, product development, and promotion strategies.", "domain": "Business"},
    {"id": 9, "title": "Human Resource Management", "description": "Understand the essentials of HR management, including recruitment, training, and performance management.", "domain": "Business"},
    {"id": 10, "title": "Graphic Design Fundamentals", "description": "Learn the basics of graphic design, including color theory, typography, and layout.", "domain": "Design"},
    {"id": 11, "title": "UI/UX Design Principles", "description": "Understand the principles of UI/UX design, including user research, wireframing, and prototyping.", "domain": "Design"},
    {"id": 12, "title": "Adobe Photoshop for Beginners", "description": "Learn how to use Adobe Photoshop for graphic design and photo editing.", "domain": "Design"}
]

# Generate synthetic job roles data for multiple domains
job_roles = [
    {"id": 1, "title": "Data Scientist", "required_skills": "Python, data analysis, machine learning, deep learning", "domain": "IT"},
    {"id": 2, "title": "Web Developer", "required_skills": "HTML, CSS, JavaScript, Flask, SQL", "domain": "IT"},
    {"id": 3, "title": "Machine Learning Engineer", "required_skills": "Python, machine learning, Scikit-Learn, TensorFlow", "domain": "IT"},
    {"id": 4, "title": "DevOps Engineer", "required_skills": "Docker, Kubernetes, CI/CD, cloud services", "domain": "IT"},
    {"id": 5, "title": "Business Analyst", "required_skills": "business management, financial analysis, market research", "domain": "Business"},
    {"id": 6, "title": "Marketing Manager", "required_skills": "marketing, promotion, market research, strategy", "domain": "Business"},
    {"id": 7, "title": "HR Manager", "required_skills": "HR management, recruitment, training, performance management", "domain": "Business"},
    {"id": 8, "title": "Graphic Designer", "required_skills": "graphic design, Photoshop, Illustrator, creativity", "domain": "Design"},
    {"id": 9, "title": "UI/UX Designer", "required_skills": "UI/UX design, user research, wireframing, prototyping", "domain": "Design"}
]

# Create DataFrames
courses_df = pd.DataFrame(courses)
job_roles_df = pd.DataFrame(job_roles)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorize course descriptions
tfidf_vectorizer = TfidfVectorizer()

def recommend_courses(job_role_id, top_n=3):
    job_role = job_roles_df[job_roles_df['id'] == job_role_id]
    job_role_skills = job_role['required_skills'].values[0]
    job_role_domain = job_role['domain'].values[0]
    
    # Filter courses by domain
    domain_courses = courses_df[courses_df['domain'] == job_role_domain]
    
    if domain_courses.empty:
        return pd.DataFrame(columns=['id', 'title', 'description', 'domain'])

    course_descriptions = domain_courses['description']
    tfidf_matrix = tfidf_vectorizer.fit_transform(course_descriptions)
    job_role_vec = tfidf_vectorizer.transform([job_role_skills])
    
    # Compute cosine similarity between job role and course descriptions
    cosine_similarities = cosine_similarity(job_role_vec, tfidf_matrix).flatten()
    
    # Get top N course indices
    top_course_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Get recommended courses
    recommended_courses = domain_courses.iloc[top_course_indices]
    return recommended_courses
