"""Streamlit interface for the shared note classifier."""

import hashlib
import json
from datetime import date
from urllib.parse import urlencode

import streamlit as st

from src.classifier import get_classifier

st.set_page_config(page_title="Note Classifier", layout="wide")
st.title("Note Classifier")
st.write("Enter one note per line to sort questions, tasks, and deadlines.")


@st.cache_resource
def load_classifier():
    return get_classifier()


try:
    classifier = load_classifier()
except (OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

notes = st.text_area("Notes (one per line)", height=240, key="notes_editor")
anchor = st.date_input("Reference date", value=date.today())
if st.button("Classify", type="primary"):
    lines = [line.strip() for line in notes.splitlines() if line.strip()]
    try:
        st.session_state["results"] = classifier.predict_batch(lines, anchor)
        st.session_state["classified_input"] = (notes, anchor)
    except ValueError as error:
        st.error(str(error))

results = st.session_state.get("results", [])
if "classified_input" in st.session_state and st.session_state["classified_input"] != (notes, anchor):
    st.info("Notes or reference date changed. Press Classify to update the results.")

if results:
    groups = {label: [] for label in ("question", "task", "deadline")}
    for index, result in enumerate(results):
        groups[result["pred"]].append((index, result))
    columns = st.columns(3)
    for column, (label, items) in zip(columns, groups.items()):
        with column:
            st.subheader({"question": "Questions", "task": "Tasks", "deadline": "Deadlines"}[label])
            if not items:
                st.caption("No notes in this category.")
            for index, item in items:
                if label == "question":
                    st.write(item["text"])
                    st.link_button("Search", "https://www.google.com/search?" + urlencode({"q": item["text"]}))
                elif label == "task":
                    identifier = hashlib.sha256(item["text"].encode()).hexdigest()[:16]
                    st.checkbox(item["text"], key=f"task_{identifier}_{index}")
                else:
                    st.write(item["text"])
                    st.caption(item["parsed_date"] or "Date unresolved")
                if item["needs_review"]:
                    st.caption("Review: " + item["disclaimer"])
    counts = " · ".join(f"{label}: {len(items)}" for label, items in groups.items())
    st.caption(f"{len(results)} notes · {counts}")
    st.download_button("Download results", data=json.dumps(results, indent=2),
                       file_name="classified_notes.json", mime="application/json")
else:
    st.info("Enter notes and press Classify.")
