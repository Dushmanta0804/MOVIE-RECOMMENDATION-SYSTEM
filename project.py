from flask import Flask, request, url_for, render_template, redirect
import mysql.connector as sql
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# -------------------------
# MySQL connection (adjust creds if needed)
# -------------------------
conn = sql.connect(host='localhost', user='root', password='', database='mrs')
curs = conn.cursor()

# -------------------------
# Dataset & recommender setup
# -------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
df = pd.read_csv(CSV_PATH)

# ensure columns exist
if 'title' not in df.columns:
    raise ValueError("dataset.csv must contain a 'title' column")

df['genre'] = df['genre'].fillna('').astype(str)
df['overview'] = df['overview'].fillna('').astype(str)

# optional columns
if 'year' not in df.columns:
    df['year'] = "Unknown"
if 'duration' not in df.columns:
    df['duration'] = "Unknown"

df['combined'] = (df['genre'].astype(str) + " " + df['overview'].astype(str)).str.lower()

cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(df['combined'])
similarity = cosine_similarity(vectors)

def recommend_movies(movie_title, top_k=8):
    if not movie_title:
        return []
    title_lower = movie_title.strip().lower()
    titles_lower = df['title'].str.lower()
    if title_lower not in titles_lower.values:
        # try fuzzy substring match: pick closest title by substring
        # fallback: return empty; simplest approach
        return []
    idx = titles_lower[titles_lower == title_lower].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    results = []
    for i, score in distances[1: top_k+1]:
        row = df.iloc[i]
        results.append({
            "title": row.get('title', "Unknown"),
            "genre": row.get('genre', "Unknown"),
            "year": row.get('year', "Unknown"),
            "duration": row.get('duration', "Unknown"),
            "overview": row.get('overview', "")
        })
    return results

# -------------------------
# ROUTES
# -------------------------

# Public index page (rich UI). Users can search here and the form posts to /landing.
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


# login route (keeps your original logic but returns a login.html)
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('uname')
        userpass = request.form.get('userpass')
        try:
            query = "SELECT name, password FROM users WHERE name=%s AND password=%s;"
            curs.execute(query, [username, userpass])
            dt = curs.fetchone()
            if dt:
                # redirect to landing (movie UI). You can also redirect to a user dashboard.
                return redirect(url_for('landing'))
            else:
                return render_template('login.html', error="Invalid username or password")
        except Exception as e:
            return str(e)
    return render_template('login.html')


@app.route('/signup-page', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        username = request.form.get('uname')
        email = request.form.get('email')
        userpass = request.form.get('userpass')
        try:
            query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s);"
            curs.execute(query, [username, email, userpass])
            conn.commit()
            return redirect(url_for('login_page'))
        except Exception as e:
            return str(e)
    return render_template('login.html')


# Landing route that receives the search term and renders recommendations (Jinja)
@app.route('/landing', methods=['GET', 'POST'])
def landing():
    movies = []
    query = None
    if request.method == 'POST':
        query = request.form.get('movie_name')
        movies = recommend_movies(query, top_k=8)
    return render_template('landingpage.html', movies=movies, query=query)


if __name__ == "__main__":
    app.run(debug=True, port=2023)
