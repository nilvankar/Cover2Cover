import json
import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
FALLBACK_FILE = DATA_DIR / "fallback_new_books.json"


@st.cache_resource
def load_recommendation_data():
    with (DATA_DIR / "popular.pkl").open("rb") as file:
        popular = pickle.load(file)
    with (DATA_DIR / "pt.pkl").open("rb") as file:
        pivot = pickle.load(file)
    with (DATA_DIR / "similarity.pkl").open("rb") as file:
        similarity = pickle.load(file)
    with (DATA_DIR / "books.pkl").open("rb") as file:
        books = pickle.load(file)
    return popular, pivot, similarity, books


def recommend_books(title, pivot, similarity, books, limit=5):
    if title not in pivot.index:
        return []

    book_index = np.where(pivot.index == title)[0][0]
    matches = sorted(
        enumerate(similarity[book_index]),
        key=lambda item: item[1],
        reverse=True,
    )[1 : limit + 1]

    recommendations = []
    for similar_index, score in matches:
        matching_books = books[books["Book-Title"] == pivot.index[similar_index]]
        if matching_books.empty:
            continue
        book = matching_books.drop_duplicates("Book-Title").iloc[0]
        recommendations.append(
            {
                "title": book["Book-Title"],
                "author": book["Book-Author"],
                "image_url": book["Image-URL-M"],
                "year": book["Year-Of-Publication"],
                "similarity_score": float(score),
            }
        )
    return recommendations


@st.cache_data(ttl=21600)
def fetch_new_books(api_key=None):
    params = {
        "q": "subject:fiction",
        "orderBy": "newest",
        "maxResults": 20,
    }
    if api_key:
        params["key"] = api_key

    try:
        for attempt in range(3):
            response = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=10)
            retryable = (
                response.status_code == 429
                or 500 <= response.status_code < 600
            )
            if retryable and attempt < 2:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Google Books returned an unexpected response.")
            items = data.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Google Books returned an invalid book list.")

            with FALLBACK_FILE.open("w", encoding="utf-8") as file:
                json.dump(items, file, ensure_ascii=False, indent=2)
            return {"items": items, "error": None}
    except (requests.RequestException, ValueError, OSError) as error:
        return {"items": [], "error": str(error)}

    return {"items": [], "error": "Google Books request failed."}


def load_fallback_new_books():
    with FALLBACK_FILE.open("r", encoding="utf-8") as file:
        items = json.load(file)
    if not isinstance(items, list):
        raise ValueError("Saved book results have an invalid format.")
    return items


def optional_google_api_key():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None
    return str(api_key).strip() or None


def show_cover(image_url):
    if isinstance(image_url, str) and image_url.strip():
        st.image(image_url, use_container_width=True)
    else:
        st.info("No cover image available.")


def display_text(value, fallback):
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def show_top_books(popular):
    st.title("Top 50 Books")
    st.caption("Popular books from the recommendation dataset.")
    for start in range(0, min(50, len(popular)), 4):
        columns = st.columns(4)
        for column, (_, book) in zip(columns, popular.iloc[start : start + 4].iterrows()):
            with column:
                show_cover(book.get("Image-URL-M"))
                st.markdown(f"**{display_text(book.get('Book-Title'), 'Untitled')}**")
                st.caption(display_text(book.get("Book-Author"), "Unknown author"))


def show_recommendations(pivot, similarity, books):
    st.title("Recommend")
    title = st.selectbox("Choose a book", options=pivot.index.tolist())
    if not st.button("Recommend", type="primary"):
        return

    results = recommend_books(title, pivot, similarity, books)
    if not results:
        st.warning("No similar books were found for that title.")
        return

    st.subheader(f"Books similar to {title}")
    columns = st.columns(4)
    for index, book in enumerate(results):
        with columns[index % len(columns)]:
            show_cover(book["image_url"])
            st.markdown(f"**{display_text(book['title'], 'Untitled')}**")
            st.caption(display_text(book["author"], "Unknown author"))
            st.caption(
                f"Year: {display_text(book['year'], 'Unknown')} · "
                f"Similarity: {book['similarity_score']:.2f}"
            )
            st.link_button(
                "View details",
                "https://www.google.com/search?q="
                + requests.utils.quote(f"{book['title']} {book['author']}"),
                use_container_width=True,
            )


def show_whats_new():
    st.title("What's New")
    st.caption("Recently published fiction from Google Books.")
    result = fetch_new_books(optional_google_api_key())
    new_books = result["items"]
    if result["error"]:
        st.warning(
            "Google Books is temporarily unavailable. "
            "Showing the most recently saved results."
        )
        try:
            new_books = load_fallback_new_books()
        except (OSError, ValueError):
            st.info("No saved book results are available yet.")
            return

    if not new_books:
        st.info("No new books were returned.")
        return

    for start in range(0, len(new_books), 4):
        columns = st.columns(4)
        for column, item in zip(columns, new_books[start : start + 4]):
            if not isinstance(item, dict):
                continue
            volume = item.get("volumeInfo") or {}
            if not isinstance(volume, dict):
                volume = {}
            image_links = volume.get("imageLinks") or {}
            if not isinstance(image_links, dict):
                image_links = {}
            authors = volume.get("authors")
            author_names = [
                author.strip()
                for author in authors or []
                if isinstance(author, str) and author.strip()
            ] if isinstance(authors, list) else []
            author_text = ", ".join(author_names) or "Unknown author"

            with column:
                show_cover(image_links.get("thumbnail"))
                st.markdown(f"**{display_text(volume.get('title'), 'Untitled')}**")
                st.caption(author_text)
                st.caption(
                    f"Published: {display_text(volume.get('publishedDate'), 'Unknown')}"
                )
                link = volume.get("infoLink")
                if isinstance(link, str) and link:
                    st.link_button("View book", link, use_container_width=True)


def main():
    st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
    popular, pivot, similarity, books = load_recommendation_data()

    page = st.sidebar.radio(
        "Pages",
        ["Top 50 Books", "Recommend", "What's New"],
    )
    if page == "Top 50 Books":
        show_top_books(popular)
    elif page == "Recommend":
        show_recommendations(pivot, similarity, books)
    else:
        show_whats_new()


if __name__ == "__main__":
    main()
