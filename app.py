from flask import Flask, jsonify, request, render_template
from recommendation_system import recommend_courses, job_roles_df

app = Flask(__name__)

@app.route('/')
def index():
    job_roles = job_roles_df.to_dict(orient='records')
    return render_template('index.html', job_roles=job_roles)

@app.route('/recommend', methods=['GET'])
def recommend():
    job_role_id = int(request.args.get('job_role_id'))
    top_n = int(request.args.get('top_n', 3))
    recommended_courses = recommend_courses(job_role_id, top_n)
    
    recommendations = recommended_courses.to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
