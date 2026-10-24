import os
import sqlite3
import uuid
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import folium
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium
from ultralytics import YOLO
from folium.plugins import MarkerCluster

# ============================================================
# MargRakshak — Citizen + Administration prototype
# ============================================================

st.set_page_config(
    page_title="MargRakshak | Infrastructure Maintenance",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "infrastructure.db"
MODEL_PATH = ROOT / "best2.pt"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# The low threshold helps recover lower-confidence potholes.
# It may also add false positives; administrator verification is required.
CONFIDENCE_THRESHOLD = 0.10
NMS_IOU_THRESHOLD = 0.45
IMAGE_SIZE = 1024

CLASS_WEIGHTS = {
    "longitudinal crack": 0.60,
    "transverse crack": 0.60,
    "alligator crack": 0.90,
    "other corruption": 0.75,
    "pothole": 1.00,
}

DAMAGE_SYMBOLS = {
    "longitudinal crack": "╱",
    "transverse crack": "━",
    "alligator crack": "▦",
    "other corruption": "◇",
    "pothole": "●",
}

STATUS_ORDER = [
    "PENDING", "VERIFIED", "ASSIGNED", "IN_PROGRESS", "RESOLVED"
]
STATUS_LABELS = {
    "PENDING": ("Report received", "रिपोर्ट प्राप्त हुई"),
    "VERIFIED": ("Damage verified", "क्षति की पुष्टि हुई"),
    "ASSIGNED": ("Repair team assigned", "मरम्मत टीम नियुक्त"),
    "IN_PROGRESS": ("Repair in progress", "मरम्मत जारी"),
    "RESOLVED": ("Repair completed", "मरम्मत पूरी हुई"),
    "REJECTED": ("Report rejected", "रिपोर्ट अस्वीकार"),
    "UNASSESSED": ("Needs manual inspection", "जाँच की आवश्यकता"),
}

# Simple bilingual text dictionary. All primary navigation, forms and feedback
# are translated. Add translations here when introducing new visible text.
T = {
    "en": {
        "app_name": "RoadWatch India",
        "tagline": "Report road damage. Track repairs. Make roads safer.",
        "home": "Home",
        "report": "Report damage",
        "map": "Road map",
        "track": "Track a report",
        "admin": "Administration",
        "settings": "Settings",
        "language": "Language / भाषा",
        "english": "English",
        "hindi": "हिंदी",
        "home_title": "Your road maintenance dashboard",
        "home_intro": "Report a road problem or follow an existing report.",
        "total": "Total reports",
        "pending": "Awaiting verification",
        "active": "Active repairs",
        "resolved": "Repairs completed",
        "next_steps": "Your next steps",
        "report_title": "Report road damage",
        "report_help": "Upload a clear road photo and add the location. The AI result is a suggestion and must be checked.",
        "photo": "Road damage photo",
        "lat": "Latitude",
        "lon": "Longitude",
        "location_name": "Location description (optional)",
        "analyze": "Inspect image",
        "submit": "Submit report",
        "track_title": "Track your report",
        "report_number": "Report number",
        "track_button": "Track report",
        "admin_title": "Administration",
        "login": "Sign in",
        "username": "Username",
        "password": "Password",
        "logout": "Sign out",
        "save": "Save update",
        "status": "Status",
        "team": "Assigned repair team",
        "remarks": "Update / repair notes",
        "verify": "Verified by administrator",
        "filter": "Filter by status",
        "not_found": "Report not found. Check the report number and try again.",
        "no_detections": "No damage was detected above the current threshold. This does not prove the road is safe. Please review the image.",
        "model_missing": "AI model not loaded. Place best2.pt beside app.py to enable detection. You can still use the reporting and tracking features.",
        "people_stuck": "People often get stuck here",
        "stuck_report": "Use a clear photo, check the map coordinates, and avoid uploading faces or vehicle number plates where possible.",
        "stuck_admin": "AI results can be wrong. Confirm the image and location before approving a repair.",
        "stuck_track": "Use the report number shown after submission. It is not your Aadhaar number or another personal ID.",
        "risk": "Risk score",
        "rhs": "Road health score",
        "vsi": "Damage severity index",
        "coverage": "Image area covered",
        "detections": "AI detections",
        "condition": "Suggested condition",
        "action": "Suggested next action",
        "confidence": "Confidence threshold",
        "map_title": "Reported road damage",
        "map_help": "Markers show submitted reports. A marker is not a guarantee that the damage has been independently verified.",
        "checklist": "Field inspection checklist",
        "checklist_help": "Tick each item after a real inspection. Your checklist is saved.",
        "report_saved": "Report submitted. Save this report number:",
        "login_failed": "Sign-in failed. Check your credentials.",
        "saved": "Update saved.",
        "no_reports": "No reports yet.",
        "priority": "Priority",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "not_assessed": "Needs review",
        "next_action_citizen": "Next: keep your report number and check its status.",
        "next_action_admin": "Next: review pending reports and confirm uncertain AI results.",
    },
    "hi": {
        "app_name": "रोडवॉच इंडिया",
        "tagline": "सड़क की समस्या बताएँ। मरम्मत की स्थिति देखें। सड़कें सुरक्षित बनाएँ।",
        "home": "होम",
        "report": "क्षति की रिपोर्ट",
        "map": "सड़क का नक्शा",
        "track": "रिपोर्ट ट्रैक करें",
        "admin": "प्रशासन",
        "settings": "सेटिंग्स",
        "language": "Language / भाषा",
        "english": "English",
        "hindi": "हिंदी",
        "home_title": "आपका सड़क रखरखाव डैशबोर्ड",
        "home_intro": "सड़क की समस्या बताएँ या पहले से दी गई रिपोर्ट देखें।",
        "total": "कुल रिपोर्ट",
        "pending": "जाँच बाकी",
        "active": "चल रही मरम्मत",
        "resolved": "पूरी हुई मरम्मत",
        "next_steps": "आपके अगले कदम",
        "report_title": "सड़क की क्षति की रिपोर्ट करें",
        "report_help": "सड़क की साफ तस्वीर और स्थान दें। AI का परिणाम केवल सुझाव है; इसकी जाँच जरूरी है।",
        "photo": "सड़क की क्षति की तस्वीर",
        "lat": "अक्षांश (Latitude)",
        "lon": "देशांतर (Longitude)",
        "location_name": "स्थान का विवरण (वैकल्पिक)",
        "analyze": "तस्वीर जाँचें",
        "submit": "रिपोर्ट भेजें",
        "track_title": "अपनी रिपोर्ट ट्रैक करें",
        "report_number": "रिपोर्ट नंबर",
        "track_button": "रिपोर्ट देखें",
        "admin_title": "प्रशासन",
        "login": "साइन इन करें",
        "username": "यूज़रनेम",
        "password": "पासवर्ड",
        "logout": "साइन आउट",
        "save": "अपडेट सहेजें",
        "status": "स्थिति",
        "team": "नियुक्त मरम्मत टीम",
        "remarks": "अपडेट / मरम्मत नोट",
        "verify": "प्रशासन ने जाँच की",
        "filter": "स्थिति के अनुसार फ़िल्टर",
        "not_found": "रिपोर्ट नहीं मिली। नंबर जाँचकर फिर कोशिश करें।",
        "no_detections": "मौजूदा सीमा से ऊपर कोई क्षति नहीं मिली। इसका अर्थ यह नहीं कि सड़क सुरक्षित है। तस्वीर की जाँच करें।",
        "model_missing": "AI मॉडल लोड नहीं हुआ। पहचान चालू करने के लिए best2.pt को app.py के पास रखें। रिपोर्ट और ट्रैकिंग फिर भी इस्तेमाल कर सकते हैं।",
        "people_stuck": "यहाँ अक्सर परेशानी होती है",
        "stuck_report": "साफ तस्वीर लें, स्थान के निर्देशांक जाँचें और जहाँ संभव हो चेहरे या वाहन नंबर प्लेट शामिल न करें।",
        "stuck_admin": "AI गलत हो सकता है। मरम्मत मंजूर करने से पहले तस्वीर और स्थान की पुष्टि करें।",
        "stuck_track": "रिपोर्ट भेजने के बाद मिला रिपोर्ट नंबर इस्तेमाल करें। यह आधार नंबर या कोई निजी पहचान नंबर नहीं है।",
        "risk": "जोखिम स्कोर",
        "rhs": "सड़क स्वास्थ्य स्कोर",
        "vsi": "क्षति गंभीरता सूचकांक",
        "coverage": "तस्वीर में ढका क्षेत्र",
        "detections": "AI पहचान",
        "condition": "सुझाई गई स्थिति",
        "action": "सुझाया गया अगला कदम",
        "confidence": "पहचान विश्वास सीमा",
        "map_title": "रिपोर्ट की गई सड़क क्षति",
        "map_help": "निशान रिपोर्ट किए गए स्थान दिखाते हैं। निशान का अर्थ यह नहीं कि क्षति की स्वतंत्र पुष्टि हो गई है।",
        "checklist": "मैदानी जाँच सूची",
        "checklist_help": "वास्तविक जाँच के बाद हर बिंदु पर टिक करें। सूची सहेजी जाती है।",
        "report_saved": "रिपोर्ट भेज दी गई। यह रिपोर्ट नंबर सहेजें:",
        "login_failed": "साइन इन नहीं हुआ। जानकारी जाँचें।",
        "saved": "अपडेट सहेजा गया।",
        "no_reports": "अभी कोई रिपोर्ट नहीं है।",
        "priority": "प्राथमिकता",
        "high": "उच्च",
        "medium": "मध्यम",
        "low": "कम",
        "not_assessed": "जाँच आवश्यक",
        "next_action_citizen": "अगला कदम: रिपोर्ट नंबर सहेजें और स्थिति देखें।",
        "next_action_admin": "अगला कदम: लंबित रिपोर्ट देखें और AI परिणामों की पुष्टि करें।",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "last_report_number" not in st.session_state:
    st.session_state.last_report_number = ""

def tr(key):
    return T[st.session_state.lang].get(key, T["en"].get(key, key))

def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def short_id():
    return "RW-" + uuid.uuid4().hex[:8].upper()

# ============================================================
# DATABASE + SAFE MIGRATIONS
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn

def table_columns(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

def init_database():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timestamp TEXT NOT NULL,
            damage_type TEXT,
            detections INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0,
            vsi REAL DEFAULT 0,
            rhs REAL DEFAULT 100,
            coverage REAL DEFAULT 0,
            risk_score REAL DEFAULT 0,
            condition TEXT DEFAULT 'UNASSESSED',
            status TEXT DEFAULT 'PENDING',
            admin_verified INTEGER DEFAULT 0,
            admin_condition TEXT,
            assigned_team TEXT,
            admin_remarks TEXT,
            last_updated TEXT,
            location_name TEXT DEFAULT '',
            repair_notes TEXT DEFAULT '',
            resolved_at TEXT
        )
    """)
    # Add fields to an existing database without deleting old reports.
    additions = {
        "location_name": "TEXT DEFAULT ''",
        "repair_notes": "TEXT DEFAULT ''",
        "resolved_at": "TEXT",
        "admin_verified": "INTEGER DEFAULT 0",
        "admin_condition": "TEXT",
        "assigned_team": "TEXT",
        "admin_remarks": "TEXT",
        "last_updated": "TEXT",
    }
    cols = table_columns(conn, "reports")
    for col, sql_type in additions.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE reports ADD COLUMN {col} {sql_type}")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS report_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT DEFAULT '',
            changed_by TEXT DEFAULT 'system',
            created_at TEXT NOT NULL,
            FOREIGN KEY(report_id) REFERENCES reports(report_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspection_items (
            report_id TEXT NOT NULL,
            item TEXT NOT NULL,
            checked INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (report_id, item)
        )
    """)
    conn.commit()
    conn.close()

init_database()

def log_event(report_id, status, note="", changed_by="system"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO report_events(report_id,status,note,changed_by,created_at) VALUES(?,?,?,?,?)",
        (report_id, status, note, changed_by, now_text()),
    )
    conn.commit()
    conn.close()

def get_reports():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_report(report_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def save_upload(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    filename = f"{uuid.uuid4().hex}.jpg"
    path = UPLOAD_DIR / filename
    image.save(path, format="JPEG", quality=88, optimize=True)
    return path, image

# ============================================================
# YOLO MODEL AND DETECTION
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))

try:
    model = load_model()
    MODEL_LOADED = model is not None
    MODEL_ERROR = "" if MODEL_LOADED else f"Model file not found: {MODEL_PATH.name}"
except Exception as exc:
    model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(exc)

def run_detection(image):
    """Return normalized detections and an annotated image."""
    if model is None:
        return [], image.copy()

    results = model.predict(
        source=np.array(image),
        conf=CONFIDENCE_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )
    result = results[0]
    names = model.names
    detections = []
    annotated = image.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return detections, annotated

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        class_name = str(raw_name).strip().lower()
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
        })

    # Draw using OpenCV for light, predictable rendering.
    cv_image = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["class"].title()
        conf = det["confidence"]
        color = (170, 70, 25) if det["class"] == "pothole" else (30, 180, 160)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        y_label = max(22, y1 - 7)
        cv2.putText(cv_image, label, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    annotated = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    return detections, annotated

def calculate_risk(detections, image_width, image_height):
    """Risk heuristic for prioritizing review; not a failure probability."""
    if not detections:
        return {
            "risk_score": 0.0, "rhs": 100.0, "vsi": 0.0,
            "coverage": 0.0, "condition": "UNASSESSED",
            "action": "Manual inspection recommended",
            "response": "Review needed",
        }

    image_area = max(image_width * image_height, 1)
    total_burden = 0.0
    max_vsi = 0.0
    coverages = []
    for det in detections:
        class_name = str(det["class"]).strip().lower()
        confidence = float(det["confidence"])
        x1, y1, x2, y2 = det["box"]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        coverage = min((bw * bh) / image_area, 1.0)
        aspect = max(bw / bh, bh / bw)
        elongation = 1 + 0.15 * min(np.log(1 + aspect), 3)
        weight = CLASS_WEIGHTS.get(class_name, 0.75)
        burden = weight * confidence * np.sqrt(coverage) * elongation
        vsi = 100 * (1 - np.exp(-3 * burden))
        total_burden += burden
        max_vsi = max(max_vsi, vsi)
        coverages.append(coverage)

    rhs = float(100 * np.exp(-3 * total_burden))
    # Box areas can overlap; this is an approximate coverage, not a true mask union.
    coverage_pct = min(sum(coverages) * 100, 100.0)
    risk_score = float(np.clip(
        0.60 * (100 - rhs) + 0.30 * max_vsi + 0.10 * coverage_pct,
        0, 100
    ))

    # A detected pothole should never be labelled SAFE solely due to a low aggregate score.
    potholes = [d for d in detections if d["class"].lower() == "pothole"]
    max_pothole_conf = max((d["confidence"] for d in potholes), default=0.0)

    if max_pothole_conf >= 0.65 or risk_score >= 65:
        condition, action, response = "CRITICAL", "Urgent field inspection and repair planning", "24–72 hours"
    elif potholes or risk_score >= 40:
        condition, action, response = "AT RISK", "Verify damage and plan repair", "7–15 days"
    elif risk_score >= 20:
        condition, action, response = "WATCH", "Schedule field inspection", "15–30 days"
    else:
        condition, action, response = "WATCH", "Verify detected damage", "Inspection recommended"

    return {
        "risk_score": round(risk_score, 1),
        "rhs": round(rhs, 1),
        "vsi": round(max_vsi, 1),
        "coverage": round(coverage_pct, 2),
        "condition": condition,
        "action": action,
        "response": response,
    }

# ============================================================
# SHARED UI HELPERS
# ============================================================

def people_stuck(text):
    st.info(f"ℹ️ **{tr('people_stuck')}**\n\n{text}")

def show_metrics(risk):
    cols = st.columns(5)
    cols[0].metric(tr("risk"), risk["risk_score"])
    cols[1].metric(tr("rhs"), risk["rhs"])
    cols[2].metric(tr("vsi"), risk["vsi"])
    cols[3].metric(tr("coverage"), f"{risk['coverage']}%")
    cols[4].metric(tr("detections"), risk.get("detections", 0))

def status_timeline(report):
    status = report.get("status", "PENDING")
    conn = get_connection()
    events = conn.execute(
        "SELECT status,note,created_at FROM report_events WHERE report_id=? ORDER BY event_id",
        (report["report_id"],),
    ).fetchall()
    conn.close()

    if status == "REJECTED":
        st.error("✕ Rejected — " + (report.get("admin_remarks") or "Contact the relevant road authority for clarification."))
        return

    completed_index = STATUS_ORDER.index(status) if status in STATUS_ORDER else -1
    st.progress((completed_index + 1) / len(STATUS_ORDER) if completed_index >= 0 else 0.0)
    for idx, stage in enumerate(STATUS_ORDER):
        en, hi = STATUS_LABELS[stage]
        label = hi if st.session_state.lang == "hi" else en
        if idx <= completed_index:
            st.write(f"✓ **{label}**")
        else:
            st.write(f"○ {label}")
    if events:
        with st.expander("Status update history / स्थिति इतिहास"):
            for event in events:
                label = STATUS_LABELS.get(event["status"], (event["status"], event["status"]))[1 if st.session_state.lang == "hi" else 0]
                st.write(f"**{label}** · {event['created_at']}")
                if event["note"]:
                    st.caption(event["note"])

# ============================================================
# SIDEBAR / NAVIGATION
# ============================================================

# ============================================================
# TOP HEADER + HORIZONTAL NAVIGATION
# ============================================================

# Header: app name on the left, language selector on the right
header_left, header_right = st.columns([5, 1.5])

with header_left:
    st.markdown(
        """
        <div class="rw-brand">
            <div class="rw-logo">🛣️</div>
            <div>
                <h1>Marg<span>Rakshak</span></h1>
                <p>Safer Roads. Stronger India.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    language_choice = st.radio(
        "Language / भाषा",
        ["English", "हिंदी"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="top_language_choice",
    )
    st.session_state.lang = (
        "en" if language_choice == "English" else "hi"
    )

st.markdown('<div class="rw-nav-spacer"></div>', unsafe_allow_html=True)

# Horizontal navigation buttons
pages = [
    ("Home", "🏠", tr("home")),
    ("Report", "📷", tr("report")),
    ("Map", "🗺️", tr("map")),
    ("Track", "🔎", tr("track")),
    ("Admin", "🛡️", tr("admin")),
    ("Settings", "⚙️", tr("settings")),
]

nav_cols = st.columns([1, 1.35, 1, 1.25, 1.4, 1])

for col, (page_key, icon, label) in zip(nav_cols, pages):
    with col:
        is_active = st.session_state.page == page_key

        if st.button(
            f"{icon}  {label}",
            key=f"top_nav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if st.session_state.page != page_key:
                st.session_state.page = page_key
                st.rerun()

# Keep the header compact; the page content starts immediately below navigation.
st.markdown("""
<style>
/* ========================================
   MARGRAKSHAK — MODERN LIGHT THEME
   ======================================== */

:root {
    --primary: #0vi6d3 
    --primary-hover: #059669 
    --navy: #111827 (Rich Jet Black)
    --heading: #1f2937 (Dark Lead)
    --text: #4b5563 (Medium Gray)
    --muted: #9ca3af (Silver Gray)
    --background: #ffffff (Pure White)
    --surface: #f8fafc 
    --border: #e5e7eb (Soft Light Gray)
}

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--background) !important;
    color: var(--heading);
}

/* Remove excess space above the main content */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0.5rem !important;
}

.stAppViewBlockContainer,
.block-container {
    max-width: 1500px !important;
    padding: 0.5rem 2rem 2rem !important;
}

/* Streamlit header and toolbar */
[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
}

[data-testid="stToolbar"] {
    background: #ffffff !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Header branding */
.rw-brand {
    display: flex !important;
    align-items: center !important;
    gap: 14px;
    width: 100%;
    min-height: 70px;
    margin: 0 0 18px 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

.rw-logo {
    display: flex;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: #eaf1ff;
    color: var(--navy);
    font-size: 36px;
}

/* Keep the brand title from being clipped */
.rw-brand > div:last-child {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex: 1;
    min-width: 0;
    overflow: visible !important;
}

.rw-brand h1 {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    height: auto !important;
    overflow: visible !important;
    margin: 0 !important;
    padding: 0 !important;
    color: var(--navy) !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
    line-height: 1.25 !important;
    white-space: normal !important;
}

.rw-brand h1 span {
    color: var(--primary) !important;
}

.rw-brand p {
    margin: 4px 0 0 !important;
    padding: 0 !important;
    color: var(--muted) !important;
    font-size: 14px;
    line-height: 1.4 !important;
}

/* General headings and text */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: var(--heading);
}

.stApp p,
.stApp label,
.stApp [data-testid="stCaptionContainer"] {
    color: var(--text);
}

/* Top navigation buttons */
div.stButton > button {
    min-height: 46px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: #f0f5fd;
    color: var(--navy);
    font-weight: 600;
    transition: background 0.2s ease, border-color 0.2s ease;
}

div.stButton > button:hover {
    background: #e3ecfa;
    border-color: #b8cbed;
    color: var(--navy);
}

/* Primary and active buttons */
div.stButton > button[kind="primary"],
.stApp div[data-testid="stButton"] button[kind="primary"] {
    background: var(--primary);
    border: 1px solid var(--primary);
    color: #ffffff;
}

div.stButton > button[kind="primary"]:hover {
    background: var(--primary-hover);
    border-color: var(--primary-hover);
    color: #ffffff;
}

/* Dashboard cards */
.rw-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 16px rgba(23, 68, 119, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.rw-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(23, 68, 119, 0.10);
}

.rw-card h3 {
    margin-top: 8px;
    color: var(--heading);
    font-size: 19px;
}

.rw-card p {
    color: var(--muted);
    line-height: 1.6;
}

/* Inputs and dropdowns */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] > div {
    background-color: #ffffff;
    color: var(--heading);
    border-color: var(--border);
    border-radius: 9px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed #b8cbed;
    border-radius: 14px;
    padding: 12px;
}

/* Dataframes and tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 3px 12px rgba(23, 68, 119, 0.04);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--navy);
    font-weight: 750;
}

/* Alerts and notices */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f4f7fc;
    border-right: 1px solid var(--border);
}

/* Dividers */
hr {
    border-color: var(--border);
}

/* Responsive layout */
@media (max-width: 768px) {
    [data-testid="stAppViewBlockContainer"],
    .stAppViewBlockContainer,
    .block-container {
        padding: 0.5rem 1rem 1.5rem !important;
    }

    .rw-brand {
        gap: 10px;
        margin-bottom: 14px !important;
    }

    .rw-brand h1 {
        font-size: 24px !important;
        letter-spacing: -0.4px;
    }

    .rw-logo {
        width: 46px;
        height: 46px;
        font-size: 28px;
    }

    .rw-brand p {
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":
    st.title(f"🛣️ {tr('home_title')}")
    st.write(tr("home_intro"))
    reports = get_reports()

    total = len(reports)
    pending = int((reports["status"] == "PENDING").sum()) if total else 0
    active = int(reports["status"].isin(["ASSIGNED", "IN_PROGRESS"]).sum()) if total else 0
    resolved = int((reports["status"] == "RESOLVED").sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("total"), total)
    c2.metric(tr("pending"), pending)
    c3.metric(tr("active"), active)
    c4.metric(tr("resolved"), resolved)

    st.subheader(tr("next_steps"))
    citizen_col, admin_col = st.columns(2)
    with citizen_col:
        st.markdown("### 📷 " + tr("report"))
        st.write("Upload a photo, add coordinates, and get a report number.")
        if st.button(tr("report"), key="home_report", use_container_width=True):
            st.session_state.page = "Report"
            st.rerun()
        st.markdown("### 🔎 " + tr("track"))
        st.write("Follow the report from submission to repair.")
        if st.button(tr("track"), key="home_track", use_container_width=True):
            st.session_state.page = "Track"
            st.rerun()
    with admin_col:
        st.markdown("### 🗺️ " + tr("map"))
        st.write("View reports on the map and see their current status.")
        if st.button(tr("map"), key="home_map", use_container_width=True):
            st.session_state.page = "Map"
            st.rerun()
        st.markdown("### 🛡️ " + tr("admin"))
        st.write("Review AI results and update repair progress.")
        if st.button(tr("admin"), key="home_admin", use_container_width=True):
            st.session_state.page = "Admin"
            st.rerun()

    st.subheader("Recent reports / हाल की रिपोर्ट")
    if reports.empty:
        st.info(tr("no_reports"))
    else:
        display_cols = ["report_id", "timestamp", "damage_type", "condition", "status", "risk_score"]
        st.dataframe(reports[display_cols].head(8), use_container_width=True, hide_index=True)
    if not MODEL_LOADED:
        st.warning(tr("model_missing"))
    st.caption("Risk scores are prioritization aids, not probabilities of road failure.")

# ============================================================
# REPORT DAMAGE
# ============================================================

elif st.session_state.page == "Report":
    st.title("📷 " + tr("report_title"))
    st.write(tr("report_help"))
    if not MODEL_LOADED:
        st.warning(tr("model_missing"))

    uploaded = st.file_uploader(
        tr("photo"),
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear photo. Avoid including faces or number plates when possible.",
    )
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input(tr("lat"), min_value=-90.0, max_value=90.0, value=18.5204, format="%.6f")
    with c2:
        lon = st.number_input(tr("lon"), min_value=-180.0, max_value=180.0, value=73.8567, format="%.6f")
    location_name = st.text_input(tr("location_name"), max_chars=160)

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded road photo / अपलोड की गई तस्वीर", use_container_width=True)
            if st.button(tr("analyze"), type="primary", use_container_width=True):
                detections, annotated = run_detection(image)
                risk = calculate_risk(detections, image.width, image.height)
                risk["detections"] = len(detections)
                st.session_state.analysis = {
                    "image_bytes": uploaded.getvalue(),
                    "image_name": uploaded.name,
                    "lat": float(lat), "lon": float(lon),
                    "location_name": location_name,
                    "detections": detections,
                    "risk": risk,
                    "annotated": annotated,
                    "created_at": now_text(),
                }
        except Exception as exc:
            st.error(f"Could not open or inspect this image: {exc}")

    analysis = st.session_state.get("analysis")
    if analysis and uploaded is not None:
        st.subheader("AI inspection result / AI जाँच परिणाम")
        st.image(analysis["annotated"], caption="AI detections / AI पहचान", use_container_width=True)
        show_metrics(analysis["risk"])
        st.write(f"**{tr('condition')}:** {analysis['risk']['condition']}")
        st.write(f"**{tr('action')}:** {analysis['risk']['action']}")
        if not analysis["detections"]:
            st.warning(tr("no_detections"))
        else:
            det_df = pd.DataFrame([
                {"Damage type": d["class"].title(), "Confidence": round(d["confidence"], 3),
                 "Box": str(d["box"])}
                for d in analysis["detections"]
            ])
            st.dataframe(det_df, use_container_width=True, hide_index=True)

        people_stuck(tr("stuck_report"))
        if st.button(tr("submit"), type="primary", use_container_width=True):
            # Reuse the original uploaded image bytes and store a compact JPEG.
            try:
                from io import BytesIO
                source_image = Image.open(BytesIO(analysis["image_bytes"])).convert("RGB")
                filename = f"{uuid.uuid4().hex}.jpg"
                image_path = UPLOAD_DIR / filename
                source_image.save(image_path, "JPEG", quality=88, optimize=True)
                report_id = short_id()
                risk = analysis["risk"]
                dets = analysis["detections"]
                damage_types = sorted(set(d["class"] for d in dets))
                damage_type = ", ".join(damage_types) if damage_types else "No AI detection — manual review"
                avg_conf = float(np.mean([d["confidence"] for d in dets])) if dets else 0.0

                conn = get_connection()
                conn.execute("""
                    INSERT INTO reports (
                        report_id,image_path,latitude,longitude,timestamp,damage_type,
                        detections,confidence,vsi,rhs,coverage,risk_score,condition,status,
                        admin_verified,admin_condition,assigned_team,admin_remarks,last_updated,
                        location_name,repair_notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    report_id, str(image_path), analysis["lat"], analysis["lon"],
                    now_text(), damage_type, len(dets), avg_conf, risk["vsi"], risk["rhs"],
                    risk["coverage"], risk["risk_score"], risk["condition"], "PENDING",
                    0, None, "", "", now_text(), analysis["location_name"], "",
                ))
                conn.commit()
                conn.close()
                log_event(report_id, "PENDING", "Report submitted by citizen", "citizen")
                st.session_state.last_report_number = report_id
                st.session_state.pop("analysis", None)
                st.success(f"{tr('report_saved')} **{report_id}**")
                st.code(report_id)
                st.caption("Save this number to track your report later.")
            except Exception as exc:
                st.error(f"Could not save report: {exc}")

# ============================================================
# MAP
# ============================================================

elif st.session_state.page == "Map":
    st.title("🗺️ " + tr("map_title"))
    st.write(tr("map_help"))
    reports = get_reports()
    if reports.empty:
        st.info(tr("no_reports"))
    else:
        status_values = sorted(reports["status"].dropna().unique().tolist())
        chosen = st.multiselect(tr("filter"), status_values, default=status_values)
        shown = reports[reports["status"].isin(chosen)]
        center = [18.5204, 73.8567]
        if not shown.empty:
            center = [float(shown["latitude"].mean()), float(shown["longitude"].mean())]
        fmap = folium.Map(location=center, zoom_start=6, control_scale=True, tiles="OpenStreetMap")
        colors = {
            "PENDING": "orange", "VERIFIED": "blue", "ASSIGNED": "purple",
            "IN_PROGRESS": "red", "RESOLVED": "green", "REJECTED": "gray",
        }
        cluster = MarkerCluster().add_to(fmap)

        for _, r in shown.iterrows():
            try:
                lat = float(r["latitude"])
                lon = float(r["longitude"])

                # Skip missing or invalid coordinates
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue

                popup = (
                    f"<b>{r['report_id']}</b><br>"
                    f"Location: {r.get('location_name', '')}<br>"
                    f"Damage: {str(r['damage_type'])[:120]}<br>"
                    f"Condition: {r['condition']}<br>"
                    f"Status: {r['status']}<br>"
                    f"Risk score: {r['risk_score']}"
                )

                folium.Marker(
                    [lat, lon],
                    tooltip=f"{r['report_id']} — {r['status']}",
                    popup=folium.Popup(popup, max_width=300),
                    icon=folium.Icon(
                        color=colors.get(r["status"], "blue"),
                        icon="warning-sign",
                    ),
                ).add_to(cluster)
            except (TypeError, ValueError):
                continue
        st_folium(
            fmap,
            use_container_width=True,
            height=500,
            returned_objects=[]
        )

        st.caption(
            "Nearby reports are grouped together. Zoom in to view individual damage locations."
        )
        
# ============================================================
# TRACK A REPORT
# ============================================================

elif st.session_state.page == "Track":
    st.title("🔎 " + tr("track_title"))
    report_id = st.text_input(
        tr("report_number"),
        value=st.session_state.get("last_report_number", ""),
        placeholder="RW-1A2B3C4D",
    ).strip().upper()
    if st.button(tr("track_button"), type="primary", use_container_width=True):
        st.session_state.tracked_report = report_id

    tracked = st.session_state.get("tracked_report", "")
    if tracked:
        report = get_report(tracked)
        if not report:
            st.warning(tr("not_found"))
        else:
            st.subheader(f"Report {report['report_id']}")
            st.write(f"**Damage:** {report.get('damage_type') or 'Needs review'}")
            st.write(f"**Status:** {report.get('status')}")
            st.write(f"**Submitted:** {report.get('timestamp')}")
            st.write(f"**Last updated:** {report.get('last_updated') or report.get('timestamp')}")
            status_timeline(report)
            if report.get("assigned_team"):
                st.write(f"**{tr('team')}:** {report['assigned_team']}")
            if report.get("admin_remarks"):
                st.info(f"**Update from administration:** {report['admin_remarks']}")
            people_stuck(tr("stuck_track"))

# ============================================================
# ADMINISTRATION
# ============================================================

elif st.session_state.page == "Admin":
    st.title("🛡️ " + tr("admin_title"))
    people_stuck(tr("stuck_admin"))

    # Secrets are preferred. The fallback is only for local prototype use.
    try:
        admin_user = st.secrets["ADMIN_USERNAME"]
        admin_pass = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        admin_user = os.environ.get("ADMIN_USERNAME", "admin")
        admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")

    if not st.session_state.admin_logged_in:
        with st.form("admin_login_form"):
            username = st.text_input(tr("username"))
            password = st.text_input(tr("password"), type="password")
            login = st.form_submit_button(tr("login"), use_container_width=True)
        if login:
            if username == admin_user and password == admin_pass:
                st.session_state.admin_logged_in = True
                st.success("Signed in / साइन इन हो गया")
                st.rerun()
            else:
                st.error(tr("login_failed"))
        st.warning("Prototype login: configure ADMIN_USERNAME and ADMIN_PASSWORD in Streamlit secrets or environment variables before sharing this app.")
    else:
        if st.button(tr("logout")):
            st.session_state.admin_logged_in = False
            st.rerun()

        reports = get_reports()
        if reports.empty:
            st.info(tr("no_reports"))
        else:
            status_options = sorted(set(reports["status"].dropna().tolist()))
            status_filter = st.multiselect(tr("filter"), status_options, default=status_options)
            filtered = reports[reports["status"].isin(status_filter)]
            st.dataframe(
                filtered[["report_id", "timestamp", "location_name", "damage_type", "condition", "status", "risk_score"]],
                use_container_width=True, hide_index=True
            )
            report_ids = filtered["report_id"].tolist()
            if report_ids:
                selected_id = st.selectbox("Select report / रिपोर्ट चुनें", report_ids)
                report = get_report(selected_id)
                if report:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.write(f"**Location:** {report.get('location_name') or 'Not provided'}")
                        st.write(f"**Coordinates:** {report['latitude']}, {report['longitude']}")
                        st.write(f"**AI damage labels:** {report.get('damage_type')}")
                        st.write(f"**AI condition:** {report.get('condition')}")
                        st.write(f"**Risk score:** {report.get('risk_score')} / 100")
                        image_path = Path(report["image_path"])
                        if image_path.exists():
                            st.image(str(image_path), caption="Submitted road photo", use_container_width=True)
                    with c2:
                        if report.get("image_path") and Path(report["image_path"]).exists() and MODEL_LOADED:
                            if st.button("Re-run AI inspection / फिर से AI जाँचें"):
                                try:
                                    im = Image.open(report["image_path"]).convert("RGB")
                                    dets, annotated = run_detection(im)
                                    new_risk = calculate_risk(dets, im.width, im.height)
                                    new_risk["detections"] = len(dets)
                                    st.image(annotated, caption="Re-run results", use_container_width=True)
                                    show_metrics(new_risk)
                                    st.caption("Re-run results are not saved until an administrator explicitly updates the case.")
                                except Exception as exc:
                                    st.error(f"Could not rerun model: {exc}")

                        with st.form(f"admin_update_{selected_id}"):
                            new_status = st.selectbox(
                                tr("status"),
                                STATUS_ORDER + ["REJECTED"],
                                index=(STATUS_ORDER + ["REJECTED"]).index(report["status"])
                                if report["status"] in STATUS_ORDER + ["REJECTED"] else 0,
                            )
                            verified = st.checkbox(tr("verify"), value=bool(report.get("admin_verified", 0)))
                            admin_condition = st.selectbox(
                                "Verified condition / पुष्टि की गई स्थिति",
                                ["UNASSESSED", "WATCH", "AT RISK", "CRITICAL", "EMERGENCY", "SAFE"],
                                index=["UNASSESSED", "WATCH", "AT RISK", "CRITICAL", "EMERGENCY", "SAFE"].index(
                                    report.get("admin_condition") if report.get("admin_condition") in
                                    ["UNASSESSED", "WATCH", "AT RISK", "CRITICAL", "EMERGENCY", "SAFE"]
                                    else "UNASSESSED"
                                ),
                            )
                            assigned_team = st.text_input(tr("team"), value=report.get("assigned_team") or "")
                            admin_remarks = st.text_area(tr("remarks"), value=report.get("admin_remarks") or "")
                            repair_notes = st.text_area("Repair work notes / मरम्मत का विवरण", value=report.get("repair_notes") or "")
                            save = st.form_submit_button(tr("save"), use_container_width=True)
                        if save:
                            old_status = report["status"]
                            resolved_at = now_text() if new_status == "RESOLVED" else report.get("resolved_at")
                            conn = get_connection()
                            conn.execute("""
                                UPDATE reports
                                SET status=?, admin_verified=?, admin_condition=?,
                                    assigned_team=?, admin_remarks=?, repair_notes=?,
                                    last_updated=?, resolved_at=?
                                WHERE report_id=?
                            """, (
                                new_status, int(verified), admin_condition, assigned_team,
                                admin_remarks, repair_notes, now_text(), resolved_at, selected_id,
                            ))
                            conn.commit()
                            conn.close()
                            if old_status != new_status or admin_remarks:
                                log_event(selected_id, new_status, admin_remarks or repair_notes, "admin")
                            st.success(tr("saved"))
                            st.rerun()

                    # Persistent inspection checklist
                    st.subheader(tr("checklist"))
                    st.caption(tr("checklist_help"))
                    checklist_items = [
                        "Confirm reported location",
                        "Verify damage in the image",
                        "Check immediate safety concern",
                        "Record required repair",
                        "Confirm repair team",
                        "Upload / verify completion photo",
                    ]
                    conn = get_connection()
                    existing = {
                        row["item"]: bool(row["checked"])
                        for row in conn.execute(
                            "SELECT item,checked FROM inspection_items WHERE report_id=?",
                            (selected_id,),
                        ).fetchall()
                    }
                    conn.close()
                    for item in checklist_items:
                        checked = st.checkbox(
                            item,
                            value=existing.get(item, False),
                            key=f"inspection_{selected_id}_{item}",
                        )
                        conn = get_connection()
                        conn.execute("""
                            INSERT INTO inspection_items(report_id,item,checked,updated_at)
                            VALUES(?,?,?,?)
                            ON CONFLICT(report_id,item) DO UPDATE SET
                                checked=excluded.checked, updated_at=excluded.updated_at
                        """, (selected_id, item, int(checked), now_text()))
                        conn.commit()
                        conn.close()
                    if report.get("image_path") and Path(report["image_path"]).exists():
                        st.download_button(
                            "Download submitted photo / तस्वीर डाउनलोड करें",
                            data=Path(report["image_path"]).read_bytes(),
                            file_name=f"{selected_id}.jpg",
                            mime="image/jpeg",
                        )

        # Administrators can remove completed cases and their associated logs.
        st.divider()
        st.subheader("🧹 Clean up completed reports / पूरे हुए कार्य हटाएँ")
        st.caption(
            "Only RESOLVED reports can be deleted here. Deletion permanently removes the report, "
            "its status history, checklist, and uploaded photo. Consider exporting/backing up records first."
        )
        all_reports_for_cleanup = get_reports()
        if all_reports_for_cleanup.empty or not (all_reports_for_cleanup["status"] == "RESOLVED").any():
            st.info("No resolved reports are available to delete.")
        else:
            resolved_reports = all_reports_for_cleanup[all_reports_for_cleanup["status"] == "RESOLVED"]
            resolved_ids = resolved_reports["report_id"].tolist()
            resolved_labels = {
                row["report_id"]: f"{row['report_id']} — {row.get('location_name') or 'Location not provided'} — {row.get('timestamp', '')}"
                for _, row in resolved_reports.iterrows()
            }
            delete_id = st.selectbox(
                "Select a completed report to delete",
                resolved_ids,
                format_func=lambda rid: resolved_labels.get(rid, rid),
                key="delete_resolved_report_id",
            )
            confirm_delete = st.checkbox(
                "I understand this permanently deletes the selected report and its logs.",
                key="confirm_delete_resolved_report",
            )
            if st.button(
                "🗑️ Delete completed report",
                type="primary",
                disabled=not confirm_delete,
                key="delete_resolved_report_button",
            ):
                report_to_delete = get_report(delete_id)
                conn = get_connection()
                conn.execute("DELETE FROM report_events WHERE report_id = ?", (delete_id,))
                conn.execute("DELETE FROM inspection_items WHERE report_id = ?", (delete_id,))
                conn.execute("DELETE FROM reports WHERE report_id = ? AND status = 'RESOLVED'", (delete_id,))
                conn.commit()
                conn.close()
                if report_to_delete and report_to_delete.get("image_path"):
                    try:
                        photo_path = Path(report_to_delete["image_path"]).resolve()
                        # Only remove files inside this app's upload directory.
                        upload_root = UPLOAD_DIR.resolve()
                        if photo_path.is_relative_to(upload_root) and photo_path.is_file():
                            photo_path.unlink()
                    except OSError:
                        st.warning("Report was deleted, but its uploaded photo could not be removed.")
                st.success(f"Completed report {delete_id} deleted.")
                st.rerun()

# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":
    st.title("⚙️ " + tr("settings"))
    st.write("Language / भाषा: use the selector in the sidebar.")
    st.write("AI confidence threshold / AI पहचान सीमा")
    st.caption(
        f"Current threshold: {CONFIDENCE_THRESHOLD:.2f}. "
        "A lower threshold may find more potholes but can also add false detections."
    )
    st.write("Image size / तस्वीर का आकार")
    st.caption(f"{IMAGE_SIZE}px. Higher image sizes may improve small-object detail but use more memory.")
    st.subheader("Privacy and accuracy / गोपनीयता और सटीकता")
    st.write(
        "AI predictions are suggestions, not proof that a road is safe or unsafe. "
        "An administrator should verify the location and damage before making a repair decision."
    )
    st.write(
        "This prototype stores report photos and coordinates locally in the data folder. "
        "Restrict access to this folder before deploying the app publicly."
    )
    if st.button("Clear this browser session's temporary analysis / अस्थायी विश्लेषण हटाएँ"):
        st.session_state.pop("analysis", None)
        st.success("Temporary analysis cleared / अस्थायी विश्लेषण हटाया गया")

st.divider()
st.caption(
    "RoadWatch India prototype · AI risk scores are heuristic estimates for prioritization, "
    "not engineering certification or a probability of road failure."
)
