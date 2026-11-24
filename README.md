# Cover2Cover: Hybrid Book Recommendation System

**Live demo:** https://cover2cover-7jyqdxwpqvseup4b5npo8j.streamlit.app/

Cover2Cover recommends books by combining a popularity-based approach with collaborative filtering. Pick a book you like and get similar titles with covers and details.

## Features
- **Top 50 Books:** the most popular books in the dataset
- **Recommend:** select a book and get similar books from the collaborative filtering model
- Hybrid design: popularity for the top list, collaborative filtering for personalised suggestions
- Streamlit app with cover images and book details

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit

## Dataset
_Add the dataset name, source link and size (number of books, users and ratings)._

## How It Works
1. Rank books by rating count and average rating to build the popularity list.
2. Build a user-item matrix from the ratings.
3. Compute similarity between books and return the closest matches for the selected book.

The full analysis is documented in the notebook in this repository.

## Screenshots
![TopBooks](sample_2.png)
![Recommend_Book](sample.png)

## Run Locally
```bash
git clone https://github.com/nilvankar/Cover2Cover.git
cd Cover2Cover
pip install -r requirements.txt
streamlit run app.py
```

## Earlier Version
The first version used a React frontend, a Flask backend and MySQL login: [cover2cover_frontned](https://github.com/nilvankar/cover2cover_frontned).

## Author
Neel Vankar | [LinkedIn](https://linkedin.com/in/neel-vankar-7baa5624b) | [GitHub](https://github.com/nilvankar)
