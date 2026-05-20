import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import hmac
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Campus Voice AI", page_icon="🎓", layout="wide")

LIGHT_CSS = """
<style>
.main {
    background: #EDF0DA;
}
.hero {
    background: linear-gradient(135deg, #1C2541 0%, #3A506B 100%);
    padding: 28px;
    border-radius: 22px;
    color: #FFFFFF;
    box-shadow: 0 10px 28px rgba(28, 37, 65, 0.18);
    margin-bottom: 20px;
    border: 1px solid #A89B8C;
}
.hero h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    color: #FFFFFF;
}
.hero p {
    margin-top: 8px;
    font-size: 1rem;
    color: #EDF0DA;
}
.mini-card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(28, 37, 65, 0.10);
    border: 1px solid #A89B8C;
}
.stat-box {
    background: #FFFFFF;
    padding: 16px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(28, 37, 65, 0.10);
    border-left: 6px solid #3A506B;
    border: 1px solid #A89B8C;
}
.label {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-right: 6px;
    margin-top: 4px;
}
.high { background: #AA4465; color: #FFFFFF; }
.medium { background: #A89B8C; color: #1C2541; }
.low { background: #EDF0DA; color: #1C2541; border: 1px solid #A89B8C; }
.danger { background: #AA4465; color: #FFFFFF; }
.success { background: #3A506B; color: #FFFFFF; }
.note {
    background: #FFFFFF;
    border: 1px dashed #3A506B;
    padding: 14px;
    border-radius: 14px;
    color: #1C2541;
}
.section-title {
    font-size: 1.22rem;
    font-weight: 800;
    color: #1C2541;
    margin-bottom: 10px;
}
.small-text {
    color: #3A506B;
    font-size: 0.92rem;
}
.stDataFrame, div[data-testid="stDataFrame"] {
    background: #FFFFFF;
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

ABUSIVE_WORDS = {"abuse", "idiot", "stupid", "harass", "harassment", "threat", "kill", "suicide"}
EMERGENCY_WORDS = {"suicide", "kill myself", "threat", "attack", "rape", "harassment"}
CS_DEPARTMENTS = ["CS", "AI", "CY", "DS"]

USERS_FILE = "users.json"
COMPLAINTS_FILE = "complaints.json"
REPLIES_FILE = "replies.json"

def load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def init_storage():
    if not os.path.exists(USERS_FILE):
        save_json_file(USERS_FILE, {
            "students": {},
            "admin": {
                "username": "chairperson",
                "password": "admin123"
            }
        })
    if not os.path.exists(COMPLAINTS_FILE):
        save_json_file(COMPLAINTS_FILE, [])
    if not os.path.exists(REPLIES_FILE):
        save_json_file(REPLIES_FILE, {})

init_storage()

@st.cache_data
def load_training_data():
    df = pd.read_csv("complaints.csv")
    expected = ["text", "category", "sentiment", "priority", "department"]
    df = df[expected].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    df["department"] = df["department"].astype(str).str.upper().str.strip()
    df.loc[~df["department"].isin(CS_DEPARTMENTS), "department"] = "CS"
    return df

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def emergency_flag(text):
    t = clean_text(text)
    return any(word in t for word in EMERGENCY_WORDS) or any(word in t for word in ABUSIVE_WORDS)

def safe_split(X, y, test_size=0.25, random_state=42):
    vc = y.value_counts()
    if len(vc) < 2 or (vc < 2).any():
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def build_models(df):
    X = df["text"].map(clean_text)
    y_cat = df["category"]
    y_sent = df["sentiment"]
    y_pri = df["priority"]

    X_train, X_test, ycat_train, ycat_test = safe_split(X, y_cat)
    _, _, ysent_train, ysent_test = safe_split(X, y_sent)
    _, _, ypri_train, ypri_test = safe_split(X, y_pri)

    category_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    sentiment_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    priority_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))
    ])

    category_model.fit(X_train, ycat_train)
    sentiment_model.fit(X_train, ysent_train)
    priority_model.fit(X_train, ypri_train)

    def eval_model(model, X_test, y_test):
        pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "report": classification_report(y_test, pred, zero_division=0),
            "confusion": confusion_matrix(y_test, pred),
        }

    metrics = {
        "category": eval_model(category_model, X_test, ycat_test),
        "sentiment": eval_model(sentiment_model, X_test, ysent_test),
        "priority": eval_model(priority_model, X_test, ypri_test),
    }

    return category_model, sentiment_model, priority_model, metrics

def predict_all(text, category_model, sentiment_model, priority_model):
    txt = clean_text(text)
    category = category_model.predict([txt])[0]
    sentiment = sentiment_model.predict([txt])[0]
    priority = priority_model.predict([txt])[0]
    toxic = emergency_flag(txt)
    if toxic:
        priority = "High"
    summary = f"Complaint about {category} with {sentiment} sentiment and {priority} priority."
    return category, sentiment, priority, toxic, summary

def load_users():
    return load_json_file(USERS_FILE, {
        "students": {},
        "admin": {"username": "chairperson", "password": "admin123"}
    })

def save_users(data):
    save_json_file(USERS_FILE, data)

def load_complaints():
    return load_json_file(COMPLAINTS_FILE, [])

def save_complaints(data):
    save_json_file(COMPLAINTS_FILE, data)

def load_replies():
    return load_json_file(REPLIES_FILE, {})

def save_replies(data):
    save_json_file(REPLIES_FILE, data)

def ensure_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "department" not in st.session_state:
        st.session_state.department = "CS"
    if "anon_id" not in st.session_state:
        st.session_state.anon_id = f"ANON-{np.random.randint(1000,9999)}"

ensure_state()

df = load_training_data()
category_model, sentiment_model, priority_model, metrics = build_models(df)

def top_sidebar():
    with st.sidebar:
        st.markdown("## Campus Voice AI")
        if st.session_state.authenticated:
            st.write(f"User: **{st.session_state.username}**")
            st.write(f"Role: **{st.session_state.role}**")
            if st.button("Log out"):
                st.session_state.authenticated = False
                st.session_state.role = None
                st.session_state.username = ""
                st.rerun()
        else:
            st.write("Please sign up or log in.")

def auth_page():
    st.markdown(
        """
        <div class="hero">
            <h1>Campus Voice AI</h1>
            <p>Anonymous grievance and decision support system for Computer Science departments: CS, AI, CY, DS.</p>
            <p>Students submit complaints, the AI analyzes them, and the Chairperson can reply directly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Student Signup</div>', unsafe_allow_html=True)
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            username = st.text_input("Create Username")
            password = st.text_input("Create Password", type="password")
            department = st.selectbox("Department", CS_DEPARTMENTS)
            signup_btn = st.form_submit_button("Create Account")

        if signup_btn:
            users = load_users()
            if not full_name or not username or not password:
                st.error("Please fill all fields.")
            elif username in users["students"]:
                st.error("Username already exists.")
            else:
                users["students"][username] = {
                    "name": full_name,
                    "password": password,
                    "department": department
                }
                save_users(users)
                st.success("Student account created. Please log in.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Login</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            role = st.selectbox("Login as", ["Student", "Admin (Chairperson)"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Log in")

        if login_btn:
            users = load_users()
            if role == "Student":
                student = users["students"].get(username)
                if student and hmac.compare_digest(student["password"], password):
                    st.session_state.authenticated = True
                    st.session_state.role = "student"
                    st.session_state.username = username
                    st.session_state.department = student["department"]
                    st.rerun()
                else:
                    st.error("Invalid student credentials.")
            else:
                admin_user = users["admin"]["username"]
                admin_pass = users["admin"]["password"]
                if hmac.compare_digest(admin_user, username) and hmac.compare_digest(admin_pass, password):
                    st.session_state.authenticated = True
                    st.session_state.role = "admin"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        st.markdown('</div>', unsafe_allow_html=True)

def student_page():
    complaints = load_complaints()
    replies = load_replies()

    st.markdown(
        f"""
        <div class="hero">
            <h1>Welcome, {st.session_state.username}</h1>
            <p>Student Dashboard | Department: {st.session_state.department}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Submit Complaint", "My Complaints", "About"])

    with tab1:
        st.markdown('<div class="section-title">Submit a Complaint</div>', unsafe_allow_html=True)
        with st.form("complaint_form"):
            c1, c2 = st.columns(2)
            with c1:
                department = st.selectbox("Department", CS_DEPARTMENTS, index=CS_DEPARTMENTS.index(st.session_state.department))
            with c2:
                anonymous_id = st.text_input("Anonymous ID", value=st.session_state.anon_id)

            text = st.text_area("Complaint Text", height=180, placeholder="Example: The CS lab projector is not working.")
            submitted = st.form_submit_button("Analyze Complaint")

        if submitted:
            if not text.strip():
                st.error("Please write a complaint first.")
            else:
                category, sentiment, priority, toxic, summary = predict_all(text, category_model, sentiment_model, priority_model)
                new_id = len(complaints) + 1
                complaint = {
                    "id": new_id,
                    "student_username": st.session_state.username,
                    "anonymous_id": anonymous_id,
                    "department": department,
                    "text": text,
                    "category": category,
                    "sentiment": sentiment,
                    "priority": priority,
                    "toxic": toxic,
                    "summary": summary,
                    "status": "Pending",
                    "created_at": pd.Timestamp.now().isoformat()
                }
                complaints.append(complaint)
                save_complaints(complaints)

                st.success("Complaint submitted successfully.")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Category", category)
                p2.metric("Sentiment", sentiment)
                p3.metric("Priority", priority)
                p4.metric("Emergency", "Yes" if toxic else "No")

                if toxic:
                    st.error("Emergency keyword detected. Admin should review this immediately.")
                st.markdown(f'<div class="note">{summary}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-title">My Complaints</div>', unsafe_allow_html=True)
        mine = [c for c in complaints if c["student_username"] == st.session_state.username]
        if mine:
            mine_df = pd.DataFrame(mine)
            cols = ["id", "department", "text", "category", "sentiment", "priority", "status", "toxic"]
            st.dataframe(mine_df[cols], use_container_width=True)

            st.markdown("### Admin Replies")
            any_reply = False
            for c in mine:
                reply = replies.get(str(c["id"]), "").strip()
                if reply:
                    any_reply = True
                    st.markdown(
                        f"""
                        <div class="mini-card">
                            <strong>Complaint #{c["id"]}</strong><br>
                            <span class="small-text">{c["text"]}</span><br><br>
                            <strong>Admin Reply:</strong> <span style="color:#1C2541;">{reply}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            if not any_reply:
                st.info("No admin replies yet.")
        else:
            st.info("No complaints submitted yet.")

    with tab3:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
        st.write("This student portal allows anonymous complaints, AI analysis, and admin responses.")
        st.write("Supported departments: CS, AI, CY, DS.")
        st.markdown('</div>', unsafe_allow_html=True)

def admin_page():
    complaints = load_complaints()
    replies = load_replies()

    st.markdown(
        """
        <div class="hero">
            <h1>Chairperson Dashboard</h1>
            <p>Review complaints, reply to students, and monitor urgent cases.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    complaints_df = pd.DataFrame(complaints) if complaints else pd.DataFrame(columns=["id", "student_username", "anonymous_id", "department", "text", "category", "sentiment", "priority", "toxic", "summary", "status"])

    total = len(complaints_df)
    high_count = int((complaints_df["priority"] == "High").sum()) if total else 0
    medium_count = int((complaints_df["priority"] == "Medium").sum()) if total else 0
    low_count = int((complaints_df["priority"] == "Low").sum()) if total else 0
    emergency_count = int((complaints_df["toxic"] == True).sum()) if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="stat-box"><div class="section-title">Total Complaints</div><h2>{total}</h2></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat-box"><div class="section-title">High Priority</div><h2>{high_count}</h2></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat-box"><div class="section-title">Medium Priority</div><h2>{medium_count}</h2></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="stat-box"><div class="section-title">Emergency</div><h2>{emergency_count}</h2></div>', unsafe_allow_html=True)

    if total == 0:
        st.info("No complaints available yet.")
        return

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Complaint List</div>', unsafe_allow_html=True)
        display_df = complaints_df.copy()
        display_df["priority"] = display_df["priority"].map(
            lambda x: f"🔴 {x}" if x == "High" else (f"🟡 {x}" if x == "Medium" else f"🟢 {x}")
        )
        st.dataframe(
            display_df[["id", "student_username", "anonymous_id", "department", "text", "category", "sentiment", "priority", "status", "toxic"]],
            use_container_width=True
        )
        csv_bytes = pd.DataFrame(complaints).to_csv(index=False).encode("utf-8")
        st.download_button("Download complaints report", data=csv_bytes, file_name="complaints_report.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Reply to Complaint</div>', unsafe_allow_html=True)

        complaint_map = {
            f"#{c['id']} | {c['department']} | {c['priority']} | {c['text'][:45]}": c["id"]
            for c in complaints
        }

        selected_label = st.selectbox("Select a complaint", list(complaint_map.keys()))
        selected_id = complaint_map[selected_label]
        selected_complaint = next(c for c in complaints if c["id"] == selected_id)

        st.write(f"**Anonymous ID:** {selected_complaint['anonymous_id']}")
        st.write(f"**Student Username:** {selected_complaint['student_username']}")
        st.write(f"**Department:** {selected_complaint['department']}")
        st.write(f"**Complaint:** {selected_complaint['text']}")
        st.write(f"**Prediction:** {selected_complaint['summary']}")

        if selected_complaint["priority"] == "High":
            st.markdown('<div class="label high">High Priority</div>', unsafe_allow_html=True)
        elif selected_complaint["priority"] == "Medium":
            st.markdown('<div class="label medium">Medium Priority</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="label low">Low Priority</div>', unsafe_allow_html=True)

        if selected_complaint["toxic"]:
            st.markdown('<div class="label danger">Emergency / Risk</div>', unsafe_allow_html=True)

        reply = st.text_area("Write admin reply", key=f"admin_reply_{selected_id}", placeholder="Example: We are reviewing this issue and will update you soon.")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("Save Reply"):
                replies[str(selected_id)] = reply
                save_replies(replies)
                st.success("Reply saved successfully.")

        with c2:
            if st.button("Mark Resolved"):
                selected_complaint["status"] = "Resolved"
                save_complaints(complaints)
                st.success("Complaint marked as resolved.")

        saved_reply = replies.get(str(selected_id), "")
        if saved_reply:
            st.info(f"Saved Reply: {saved_reply}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mini-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cat_counts = complaints_df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax, palette="viridis")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)

    with c2:
        pri_counts = complaints_df["priority"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.pie(pri_counts.values, labels=pri_counts.index, autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2)

    pivot = pd.crosstab(complaints_df["department"], complaints_df["category"])
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax3)
    st.pyplot(fig3)

    st.markdown('</div>', unsafe_allow_html=True)

top_sidebar()

if not st.session_state.authenticated:
    auth_page()
else:
    if st.session_state.role == "student":
        student_page()
    elif st.session_state.role == "admin":
        admin_page()