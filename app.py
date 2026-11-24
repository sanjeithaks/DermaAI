
import json
import urllib.request
import urllib.error
import sqlite3
from pathlib import Path
from datetime import datetime
from io import BytesIO
from html import escape

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PI_CAMERA_URL = "http://10.34.66.20:8081"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "analysis_history.db"

STAGE1_PATH = MODEL_DIR / "stage1_final_best.keras"
STAGE2_OLD_PATH = MODEL_DIR / "stage2_old_mobilenetv2.keras"
STAGE2_FOCAL_PATH = MODEL_DIR / "stage2_exp1_focal_loss.keras"
CONFIG_PATH = MODEL_DIR / "stage2_ensemble_config.json"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

STAGE1_THRESHOLD = 0.5


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DermaAI",
    page_icon="🩺",
    layout="wide"
)


# =====================================================
# 3. UI STYLING
# =====================================================

st.markdown("""
<style>
:root {
    --navy: #17343B;
    --teal: #087F78;
    --teal-dark: #123C46;
    --teal-border: #285A63;
    --muted: #587078;
    --canvas: #EDF3F5;
    --line: #D7E2E5;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
    color: var(--navy);
}

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--canvas);
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1440px;
    padding: 2rem 2.5rem 3rem;
}

@media (max-width: 760px) {
    [data-testid="stAppViewContainer"] .main .block-container {
        padding: 1.25rem 1rem 2rem;
    }
}

/* Dark teal sidebar */
[data-testid="stSidebar"] {
    background: var(--teal-dark) !important;
    border-right: 1px solid #214D56;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C6D9DC !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.35rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 10px;
    padding: 0.45rem 0.65rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.09);
}

/* Main headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--navy) !important;
    letter-spacing: -0.02em;
}

.main-title {
    color: var(--navy);
    font-size: clamp(1.8rem, 3vw, 2.45rem);
    font-weight: 750;
    letter-spacing: -0.045em;
    line-height: 1.15;
    margin: 0 0 0.45rem;
}

.sub-title {
    color: var(--muted);
    font-size: 1rem;
    margin: 0 0 1.4rem;
}

.eyebrow {
    color: #A9D8D4;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 0.65rem;
    text-transform: uppercase;
}

.hero-card {
    background: linear-gradient(120deg, #123C46 0%, #19545D 100%);
    border: 1px solid #28616A;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(18, 60, 70, 0.12);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    margin: 1rem 0 1.3rem;
    min-height: 218px;
    overflow: hidden;
    padding: 1.65rem 1.8rem;
    position: relative;
    animation: dashboard-enter 560ms cubic-bezier(0.2, 0.75, 0.25, 1) both;
}

.hero-copy {
    flex: 1;
    position: relative;
    z-index: 1;
}

.hero-art {
    flex: 0 0 245px;
    max-width: 32%;
    position: relative;
}

.hero-art svg {
    display: block;
    filter: drop-shadow(0 12px 18px rgba(4, 25, 30, 0.2));
    height: auto;
    width: 100%;
    animation: dashboard-float 5s ease-in-out 650ms infinite;
}

.hero-art-label {
    background: rgba(224, 250, 246, 0.12);
    border: 1px solid rgba(224, 250, 246, 0.22);
    border-radius: 999px;
    bottom: 0.1rem;
    color: #DDF7F3;
    font-size: 0.68rem;
    font-weight: 650;
    left: 50%;
    padding: 0.35rem 0.7rem;
    position: absolute;
    transform: translateX(-50%);
    white-space: nowrap;
}

.hero-card h2 {
    color: #FFFFFF !important;
    font-size: clamp(1.4rem, 2vw, 1.8rem);
    margin: 0 0 0.5rem;
}

.hero-card p {
    color: #D5E7E8;
    line-height: 1.65;
    margin: 0;
    max-width: 760px;
}

/* Metric cards - dark boxes */
.metric-card {
    background: var(--teal-dark);
    border: 1px solid var(--teal-border);
    border-radius: 15px;
    box-shadow: 0 7px 18px rgba(18, 60, 70, 0.09);
    min-height: 118px;
    overflow: hidden;
    padding: 20px 22px;
    position: relative;
}

.dashboard-metric-card {
    animation: dashboard-enter 480ms cubic-bezier(0.2, 0.75, 0.25, 1) both;
    transition: box-shadow 180ms ease, transform 180ms ease;
}

.dashboard-metric-card:hover {
    box-shadow: 0 12px 24px rgba(18, 60, 70, 0.16);
    transform: translateY(-3px);
}

.dashboard-metric-total {
    animation-delay: 100ms;
}

.dashboard-metric-lesion {
    animation-delay: 180ms;
}

.dashboard-metric-reports {
    animation-delay: 260ms;
}

.metric-card::after {
    border: 15px solid rgba(101, 207, 193, 0.08);
    border-radius: 50%;
    content: "";
    height: 108px;
    pointer-events: none;
    position: absolute;
    right: -48px;
    top: -50px;
    width: 108px;
}

.metric-label,
.metric-value {
    position: relative;
    z-index: 1;
}

.metric-label {
    color: #C6D9DC !important;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.015em;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: clamp(1.55rem, 2.3vw, 2rem);
    font-weight: 700;
    letter-spacing: -0.035em;
    margin-top: 10px;
    overflow-wrap: anywhere;
}

.dashboard-insight-card {
    align-items: center;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 7px 20px rgba(25, 55, 65, 0.05);
    display: flex;
    justify-content: space-between;
    gap: 1.5rem;
    margin: 1rem 0 1.25rem;
    padding: 1.15rem 1.4rem;
    animation: dashboard-enter 520ms cubic-bezier(0.2, 0.75, 0.25, 1) 320ms both;
    transition: box-shadow 180ms ease, transform 180ms ease;
}

.dashboard-insight-card:hover {
    box-shadow: 0 12px 26px rgba(25, 55, 65, 0.09);
    transform: translateY(-2px);
}

.dashboard-insight-copy h3 {
    color: var(--navy) !important;
    font-size: 1rem;
    margin: 0 0 0.35rem;
}

.dashboard-insight-copy p {
    color: var(--muted) !important;
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 0 0 0.8rem;
}

.dashboard-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem 1rem;
}

.dashboard-legend span {
    align-items: center;
    color: var(--muted) !important;
    display: inline-flex;
    font-size: 0.75rem;
    gap: 0.38rem;
}

.dashboard-legend strong {
    color: var(--navy) !important;
}

.dashboard-legend-dot {
    border-radius: 50%;
    height: 9px;
    width: 9px;
}

.dashboard-legend-dot.normal {
    background: #21B589;
}

.dashboard-legend-dot.lesion {
    background: #EC7184;
}

.dashboard-donut {
    align-items: center;
    background: conic-gradient(
        #21B589 0 var(--normal-share),
        #EC7184 var(--normal-share) calc(var(--normal-share) + var(--lesion-share)),
        #EAF0F2 calc(var(--normal-share) + var(--lesion-share)) 100%
    );
    border-radius: 50%;
    display: flex;
    flex: 0 0 94px;
    height: 94px;
    justify-content: center;
    position: relative;
    width: 94px;
    animation: dashboard-pop 650ms cubic-bezier(0.2, 0.75, 0.25, 1) 440ms both;
}

.dashboard-donut::before {
    background: #FFFFFF;
    border-radius: 50%;
    content: "";
    inset: 11px;
    position: absolute;
}

.dashboard-donut span {
    color: var(--navy) !important;
    font-size: 1.25rem;
    font-weight: 750;
    position: relative;
    z-index: 1;
}

.dashboard-cta {
    transition: box-shadow 180ms ease, transform 180ms ease;
}

.dashboard-cta:hover {
    box-shadow: 0 12px 24px rgba(18, 60, 70, 0.15);
    transform: translateY(-2px);
}

@keyframes dashboard-enter {
    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes dashboard-float {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-5px);
    }
}

@keyframes dashboard-pop {
    from {
        opacity: 0;
        transform: scale(0.88);
    }

    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* Content cards - dark boxes */
.content-card {
    background: var(--teal-dark);
    border: 1px solid var(--teal-border);
    border-radius: 15px;
    box-shadow: 0 7px 18px rgba(18, 60, 70, 0.08);
    margin-bottom: 1rem;
    padding: 1.35rem 1.5rem;
}

.content-card h3,
.content-card p,
.content-card b {
    color: #FFFFFF !important;
}

.content-card h3 {
    margin-top: 0;
}

.content-card p {
    line-height: 1.65;
    margin-bottom: 0.45rem;
}

.soft-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(25, 55, 65, 0.045);
    padding: 1.1rem 1.25rem;
}

.soft-card p {
    color: var(--muted);
    line-height: 1.6;
}

.soft-card strong {
    color: var(--navy) !important;
    display: block;
    font-weight: 700;
}

.details-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(25, 55, 65, 0.045);
    padding: 0.75rem 1.1rem;
}

.detail-item {
    border-bottom: 1px solid #E8EEF0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    padding: 0.8rem 0;
}

.detail-item:last-child {
    border-bottom: 0;
}

.detail-label {
    color: var(--muted) !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.detail-value {
    color: var(--navy) !important;
    font-size: 0.98rem;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.history-page-head {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
}

.history-page-badge {
    background: #F1EAFE;
    border: 1px solid #E3D7FC;
    border-radius: 999px;
    color: #7041A6 !important;
    display: inline-block;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.3rem 0.65rem;
    vertical-align: middle;
}

.history-count {
    color: var(--muted) !important;
    font-size: 0.88rem;
}

.history-count b {
    color: var(--navy) !important;
}

.history-overview {
    display: grid;
    gap: 0.85rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin: 0.2rem 0 1.1rem;
}

.history-visual-card {
    align-items: center;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 6px 16px rgba(25, 55, 65, 0.045);
    display: flex;
    gap: 0.8rem;
    min-width: 0;
    overflow: hidden;
    padding: 1rem 1.1rem;
    position: relative;
}

.history-visual-card::after {
    background: rgba(8, 127, 120, 0.055);
    border-radius: 50%;
    content: "";
    height: 82px;
    position: absolute;
    right: -24px;
    top: -30px;
    width: 82px;
}

.history-visual-icon {
    align-items: center;
    background: #E8F5F4;
    border-radius: 12px;
    display: flex;
    flex: 0 0 42px;
    font-size: 1.2rem;
    height: 42px;
    justify-content: center;
    position: relative;
    z-index: 1;
}

.history-visual-copy {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    min-width: 0;
    position: relative;
    z-index: 1;
}

.history-visual-label {
    color: var(--muted) !important;
    font-size: 0.76rem;
    font-weight: 600;
}

.history-visual-value {
    color: var(--navy) !important;
    font-size: 1.55rem;
    font-weight: 750;
    letter-spacing: -0.035em;
    line-height: 1.1;
}

.history-mix-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 6px 16px rgba(25, 55, 65, 0.045);
    grid-column: 1 / -1;
    padding: 1rem 1.2rem;
}

.history-mix-head,
.history-mix-legend {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    justify-content: space-between;
}

.history-mix-head strong {
    color: var(--navy) !important;
    font-size: 0.9rem;
}

.history-mix-head span,
.history-mix-legend span {
    color: var(--muted) !important;
    font-size: 0.76rem;
}

.history-mix-track {
    background: #EAF0F2;
    border-radius: 999px;
    display: flex;
    height: 11px;
    margin: 0.8rem 0 0.65rem;
    overflow: hidden;
    width: 100%;
}

.history-mix-normal {
    background: #21B589;
    height: 100%;
}

.history-mix-lesion {
    background: #EC7184;
    height: 100%;
}

.history-mix-legend span {
    align-items: center;
    display: inline-flex;
    gap: 0.4rem;
}

.history-legend-dot {
    border-radius: 50%;
    display: inline-block;
    height: 8px;
    width: 8px;
}

.history-legend-dot.normal {
    background: #21B589;
}

.history-legend-dot.lesion {
    background: #EC7184;
}

.history-heading {
    background: #F5F8FA;
    border-bottom: 1px solid var(--line);
    color: #60758C !important;
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.055em;
    min-height: 2.7rem;
    padding: 0.8rem 0.35rem;
    text-transform: uppercase;
}

.history-cell {
    background: #FFFFFF;
    border-bottom: 1px solid #E8EEF0;
    color: var(--navy) !important;
    display: flex;
    flex-direction: column;
    font-size: 0.88rem;
    justify-content: center;
    min-height: 3.8rem;
    overflow-wrap: anywhere;
    padding: 0.65rem 0.35rem;
}

.history-cell strong {
    color: var(--navy) !important;
    font-weight: 650;
}

.history-cell small {
    color: #8193A7 !important;
    font-size: 0.74rem;
    margin-top: 0.15rem;
}

.history-status {
    border: 1px solid transparent;
    border-radius: 999px;
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.025em;
    padding: 0.3rem 0.45rem;
    text-align: center;
    white-space: nowrap;
}

.history-status.normal {
    background: #E9FBF4;
    border-color: #B5F1D9;
    color: #087A58 !important;
}

.history-status.lesion {
    background: #FFF0F2;
    border-color: #FFD5DC;
    color: #C53151 !important;
}

.history-empty {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 7px 20px rgba(25, 55, 65, 0.045);
    margin-top: 1.2rem;
    padding: 2.1rem 1.5rem;
    text-align: center;
}

.history-empty h3 {
    color: var(--navy) !important;
    margin: 0.3rem 0 0.5rem;
}

.history-empty p {
    color: var(--muted) !important;
    margin: 0;
}

.history-detail-card {
    background: var(--teal-dark);
    border: 1px solid var(--teal-border);
    border-radius: 14px;
    color: #FFFFFF !important;
    margin: 1rem 0;
    padding: 1.1rem 1.3rem;
}

.history-detail-card * {
    color: #FFFFFF !important;
}

.history-mobile-card {
    display: none;
}

.history-mobile-action-marker {
    display: none;
}

div[data-testid="stHorizontalBlock"]:has(.history-mobile-action-marker) {
    display: none !important;
}

@media (max-width: 760px) {
    .history-overview {
        grid-template-columns: 1fr;
    }

    .history-mix-card {
        grid-column: auto;
    }

    .history-mobile-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 5px 14px rgba(25, 55, 65, 0.045);
        display: block;
        margin: 0.65rem 0 0.35rem;
        padding: 1rem;
    }

    .history-mobile-top {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: space-between;
    }

    .history-mobile-id,
    .history-mobile-file,
    .history-mobile-category {
        color: var(--navy) !important;
        font-weight: 700;
    }

    .history-mobile-id {
        font-size: 0.9rem;
    }

    .history-mobile-file {
        font-size: 0.96rem;
        margin: 0.8rem 0;
        overflow-wrap: anywhere;
    }

    .history-mobile-meta {
        border-top: 1px solid #E8EEF0;
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        padding-top: 0.75rem;
    }

    .history-mobile-meta-item {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        min-width: 0;
    }

    .history-mobile-meta-label {
        color: var(--muted) !important;
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }

    .history-mobile-meta-value {
        color: var(--navy) !important;
        font-size: 0.82rem;
        overflow-wrap: anywhere;
    }

    div[data-testid="stHorizontalBlock"]:has(.history-heading),
    div[data-testid="stHorizontalBlock"]:has(.history-cell) {
        display: none !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.history-mobile-action-marker) {
        display: flex !important;
        gap: 0.65rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.history-mobile-action-marker) > [data-testid="column"] {
        flex: 1 1 0% !important;
        min-width: 0 !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.history-mobile-action-marker) .stButton > button,
    div[data-testid="stHorizontalBlock"]:has(.history-mobile-action-marker) .stDownloadButton > button {
        min-height: 2.8rem;
        width: 100%;
    }
}

/* Buttons */
.stButton > button {
    background: var(--teal) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--teal) !important;
    border-radius: 10px;
    font-weight: 600;
    min-height: 2.7rem;
    transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
}

.stButton > button:hover {
    background: #06645F !important;
    color: #FFFFFF !important;
    border-color: #06645F !important;
    box-shadow: 0 5px 12px rgba(8, 127, 120, 0.18);
    transform: translateY(-1px);
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--teal) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--teal) !important;
    border-radius: 10px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background: #06645F !important;
    border-color: #06645F !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #9CB9BE;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(25, 55, 65, 0.04);
    padding: 0.5rem 0.75rem;
}

[data-testid="stFileUploader"] * {
    color: var(--navy) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F9FCFC;
    border: 0;
}

/* Inputs */
input, textarea {
    color: var(--navy) !important;
    background: #FFFFFF !important;
}

/* Select boxes */
[data-baseweb="select"] * {
    color: var(--navy) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(25, 55, 65, 0.035);
    margin-bottom: 0.55rem;
}

[data-testid="stExpander"] * {
    color: var(--navy) !important;
}

/* Captions */
.stCaption, small {
    color: var(--muted) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--teal-dark) !important;
    border: 1px solid var(--teal-border);
    border-radius: 12px;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] [data-baseweb="notification"] {
    background-color: transparent !important;
    color: #FFFFFF !important;
}

/* Links */
a {
    color: var(--teal) !important;
}

hr {
    border-color: var(--line) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

@media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    div[data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100%;
        min-width: 100% !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.history-mobile-action-marker) > [data-testid="column"] {
        flex: 1 1 0% !important;
        min-width: 0 !important;
    }

    [data-testid="stImage"] img {
        max-height: 58vh;
        object-fit: contain;
    }

    [data-testid="stDataFrame"] {
        max-width: 100%;
        overflow-x: auto;
    }

    .main-title {
        font-size: 1.8rem;
    }

    .sub-title {
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .hero-card {
        align-items: flex-start;
        gap: 0.6rem;
        padding: 1.25rem;
    }

    .hero-art {
        flex-basis: 110px;
        max-width: 34%;
    }

    .hero-art-label {
        display: none;
    }

    .metric-card {
        min-height: 100px;
        padding: 16px;
    }

    .dashboard-insight-card {
        gap: 0.8rem;
        padding: 1rem;
    }

    .dashboard-donut {
        flex-basis: 72px;
        height: 72px;
        width: 72px;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.8rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    .hero-card,
    .hero-art svg,
    .dashboard-metric-card,
    .dashboard-insight-card,
    .dashboard-donut,
    .dashboard-cta {
        animation: none !important;
        transition: none !important;
        transform: none !important;
    }

    .stButton > button {
        transition: none !important;
        transform: none !important;
    }
}
</style>
""", unsafe_allow_html=True)




# =====================================================
# 4. LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    stage1 = tf.keras.models.load_model(
        STAGE1_PATH,
        compile=False
    )

    stage2_old = tf.keras.models.load_model(
        STAGE2_OLD_PATH,
        compile=False
    )

    stage2_focal = tf.keras.models.load_model(
        STAGE2_FOCAL_PATH,
        compile=False
    )

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return stage1, stage2_old, stage2_focal, config


try:
    stage1_model, stage2_old_model, stage2_focal_model, config = load_models()

except Exception as error:
    st.error(f"Could not load models: {error}")
    st.stop()


# =====================================================
# 5. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_batch = np.expand_dims(image_array, axis=0)

    return image_batch


# =====================================================
# 6. GRAD-CAM
# =====================================================

def generate_gradcam(model, image_batch, class_index):

    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image_batch)

        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + 1e-8),
        heatmap
    )

    return heatmap.numpy()


def create_overlay(image, heatmap):

    image = image.convert("RGB").resize((224, 224))

    original = np.array(image)

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    return overlay


# =====================================================
# 7. PREDICTION FUNCTION
# =====================================================

def predict_image(image):

    image_batch = preprocess_image(image)

    # Stage 1: Normal vs Lesion
    stage1_prediction = stage1_model.predict(
        image_batch,
        verbose=0
    )

    lesion_probability = float(stage1_prediction[0][0])

    if lesion_probability < STAGE1_THRESHOLD:

        return {
            "stage1": "Normal",
            "lesion_probability": lesion_probability,
            "stage2": None,
            "confidence": None,
            "heatmap": None,
            "overlay": None,
            "probabilities": None
        }

    # Stage 2: Seven-class classification
    old_prediction = stage2_old_model.predict(
        image_batch,
        verbose=0
    )[0]

    focal_prediction = stage2_focal_model.predict(
        image_batch,
        verbose=0
    )[0]

    old_weight = float(config["model_1_weight"])
    focal_weight = float(config["model_2_weight"])

    final_prediction = (
        old_weight * old_prediction
        + focal_weight * focal_prediction
    )

    predicted_index = int(np.argmax(final_prediction))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(final_prediction[predicted_index])

    # Grad-CAM
    heatmap_old = generate_gradcam(
        stage2_old_model,
        image_batch,
        predicted_index
    )

    heatmap_focal = generate_gradcam(
        stage2_focal_model,
        image_batch,
        predicted_index
    )

    combined_heatmap = (
        old_weight * heatmap_old
        + focal_weight * heatmap_focal
    )

    maximum = np.max(combined_heatmap)

    if maximum > 0:
        combined_heatmap = combined_heatmap / maximum

    overlay = create_overlay(
        image,
        combined_heatmap
    )

    return {
        "stage1": "Lesion",
        "lesion_probability": lesion_probability,
        "stage2": predicted_class,
        "confidence": confidence,
        "heatmap": combined_heatmap,
        "overlay": overlay,
        "probabilities": final_prediction
    }


# =====================================================
# 8. DATABASE AND HISTORY
# =====================================================

def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            filename TEXT,
            stage1 TEXT,
            lesion_probability REAL,
            stage2 TEXT,
            confidence REAL
        )
    """)

    connection.commit()
    connection.close()


def save_history(filename, result):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO history (
            date,
            filename,
            stage1,
            lesion_probability,
            stage2,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        filename,
        result["stage1"],
        result["lesion_probability"],
        result["stage2"] or "Not classified",
        result["confidence"] if result["confidence"] is not None else 0
    ))

    connection.commit()
    connection.close()


def get_history():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT * FROM history ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records


def clear_history():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("DELETE FROM history")

    connection.commit()
    connection.close()


def navigate_to_analysis():

    st.session_state["analysis_result"] = None
    st.session_state["analysis_image"] = None
    st.session_state["page_navigation"] = "New Analysis"


initialize_database()


# =====================================================
# 9. PDF REPORT GENERATION
# =====================================================

def generate_pdf(record):

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    # Header
    pdf.setFillColorRGB(0.85, 0.92, 0.93)

    pdf.rect(
        0,
        height - 100,
        width,
        100,
        fill=1,
        stroke=0
    )

    pdf.setFillColorRGB(0.07, 0.24, 0.28)

    pdf.setFont("Helvetica-Bold", 24)

    pdf.drawString(
        50,
        height - 60,
        "DermaAI"
    )

    pdf.setFont("Helvetica-Bold", 17)

    pdf.drawString(
        50,
        height - 140,
        "Skin Image Analysis Report"
    )

    pdf.setFont("Helvetica", 11)

    details = [
        f"Report ID: {record[0]}",
        f"Date: {record[1]}",
        f"Image Filename: {record[2]}",
        f"Stage 1 Result: {record[3]}",
        f"Lesion Probability: {record[4] * 100:.2f}%",
        f"Stage 2 Category: {record[5]}",
        f"Stage 2 Confidence: {record[6] * 100:.2f}%"
    ]

    y = height - 190

    for detail in details:

        pdf.drawString(50, y, detail)

        y -= 30

    pdf.setFont("Helvetica-Bold", 12)

    pdf.drawString(
        50,
        y - 20,
        "Important Medical Disclaimer"
    )

    pdf.setFont("Helvetica", 10)

    disclaimer = [
        "This report is generated by an experimental educational AI prototype.",
        "It is not a medical diagnosis or a substitute for professional advice.",
        "Please consult a qualified dermatologist for medical evaluation."
    ]

    y -= 50

    for line in disclaimer:

        pdf.drawString(50, y, line)

        y -= 20

    pdf.save()

    buffer.seek(0)

    return buffer.getvalue()


# =====================================================
# 10. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("# 🩺 DermaAI")

    st.caption("AI skin screening prototype")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "New Analysis",
            "Analysis History",
            "About the Model"
        ],
        format_func=lambda item: {
            "Dashboard": "⌂  Dashboard",
            "New Analysis": "＋  New Analysis",
            "Analysis History": "◷  Analysis History",
            "About the Model": "ⓘ  About the Model"
        }[item],
        label_visibility="collapsed",
        key="page_navigation"
    )

    st.divider()

    st.caption("An educational screening prototype")


# =====================================================
# 11. DASHBOARD
# =====================================================

if page == "Dashboard":

    history = get_history()

    total_analyses = len(history)

    reports_available = len(history)
    lesion_flags = sum(1 for record in history if record[3] == "Lesion")
    normal_scans = sum(1 for record in history if record[3] == "Normal")
    normal_share = normal_scans / total_analyses * 100 if total_analyses else 0
    lesion_share = lesion_flags / total_analyses * 100 if total_analyses else 0

    st.markdown(
        '<div class="main-title">Welcome to DermaAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">A clear overview of your saved screening activity</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-card">
        <div class="hero-copy">
            <div class="eyebrow">Screening workspace</div>
            <h2>Understand your analysis at a glance</h2>
            <p>
            Review previous image screenings or start a new one to explore the
            model's classification, confidence scores and Grad-CAM visualization.
            </p>
        </div>
        <div class="hero-art" aria-hidden="true">
            <svg viewBox="0 0 280 220" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="lens-glass" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0" stop-color="#D9FFFA" stop-opacity=".95"/>
                        <stop offset="1" stop-color="#66C8C0" stop-opacity=".62"/>
                    </linearGradient>
                    <linearGradient id="scope-handle" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0" stop-color="#F2FCFA"/>
                        <stop offset="1" stop-color="#75D5C9"/>
                    </linearGradient>
                </defs>
                <circle cx="143" cy="105" r="94" fill="#54C2B8" opacity=".14"/>
                <circle cx="143" cy="105" r="73" fill="none" stroke="#B8F1E9" stroke-opacity=".22" stroke-width="1.5"/>
                <circle cx="143" cy="105" r="57" fill="none" stroke="#B8F1E9" stroke-opacity=".16" stroke-width="1.5" stroke-dasharray="4 7"/>
                <path d="M68 125c18-20 27-16 43-7 15 9 23 4 33-5 12-10 22-8 39 5" fill="none" stroke="#9CE3D9" stroke-opacity=".65" stroke-width="2"/>
                <path d="M74 139c16-12 26-10 39-2 14 9 26 6 36-2 12-10 22-10 34-3" fill="none" stroke="#9CE3D9" stroke-opacity=".38" stroke-width="2"/>
                <path d="M174 139l42 43" stroke="url(#scope-handle)" stroke-width="18" stroke-linecap="round"/>
                <path d="M174 139l42 43" stroke="#E9FFFC" stroke-opacity=".7" stroke-width="4" stroke-linecap="round"/>
                <circle cx="139" cy="103" r="48" fill="#E4FAF6" fill-opacity=".12" stroke="#D8FFF8" stroke-width="8"/>
                <circle cx="139" cy="103" r="37" fill="url(#lens-glass)" fill-opacity=".8"/>
                <circle cx="139" cy="103" r="22" fill="#C77A80" fill-opacity=".8"/>
                <circle cx="139" cy="103" r="12" fill="#8C4657" fill-opacity=".9"/>
                <path d="M117 84c10-8 21-10 32-6" fill="none" stroke="#FFFFFF" stroke-opacity=".8" stroke-width="4" stroke-linecap="round"/>
                <circle cx="212" cy="52" r="5" fill="#8AE3D8"/>
                <circle cx="77" cy="72" r="3" fill="#F4D98A"/>
            </svg>
            <div class="hero-art-label">AI-assisted screening</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card dashboard-metric-card dashboard-metric-total">
            <div class="metric-label">📊 &nbsp; Total screenings</div>
            <div class="metric-value">{total_analyses}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card dashboard-metric-card dashboard-metric-lesion">
            <div class="metric-label">🔎 &nbsp; Model flagged as lesion</div>
            <div class="metric-value">{lesion_flags}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card dashboard-metric-card dashboard-metric-reports">
            <div class="metric-label">📄 &nbsp; Reports available</div>
            <div class="metric-value">{reports_available}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dashboard-insight-card">
        <div class="dashboard-insight-copy">
            <h3>Screening overview</h3>
            <p>A visual summary of results saved in your local analysis history.</p>
            <div class="dashboard-legend">
                <span><i class="dashboard-legend-dot normal"></i>Normal <strong>{normal_scans}</strong></span>
                <span><i class="dashboard-legend-dot lesion"></i>Lesion flagged <strong>{lesion_flags}</strong></span>
            </div>
        </div>
        <div class="dashboard-donut"
             style="--normal-share: {normal_share:.1f}%; --lesion-share: {lesion_share:.1f}%"
             role="img"
             aria-label="Screening distribution: {normal_scans} normal, {lesion_flags} lesion flagged">
            <span>{total_analyses}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="content-card dashboard-cta">
        <h3>✨ &nbsp; Start a new analysis</h3>
        <p>
        Upload a dermatoscopic image to run the two-stage screening model.
        Results include a class prediction, confidence scores and an
        explainability view.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.button(
        "＋  Start a new analysis",
        use_container_width=True,
        on_click=navigate_to_analysis
    )

    st.markdown("### 🕘 &nbsp; Recent analyses")

    if history:

        recent_records = [
            {
                "Date": record[1],
                "Image": record[2],
                "Screening result": record[3],
                "Category": CLASS_LABELS.get(record[5], record[5])
            }
            for record in history[:5]
        ]
        st.dataframe(
            recent_records,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No previous analyses yet. Your saved screenings will appear here.")


# =====================================================
# 12. NEW ANALYSIS
# =====================================================

elif page == "New Analysis":

    st.markdown(
        '<div class="main-title">New Skin Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Upload a clear skin image to begin your screening</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="soft-card">
        <strong>📤 &nbsp; Image upload</strong>
        <p>Choose a JPG or PNG image. For best results, use a well-lit,
        focused image with the area of interest clearly visible.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")


    st.markdown("### 📷 Raspberry Pi camera")

    if st.button("📷 Capture & Analyse from Raspberry Pi",
                 use_container_width=True):

        try:
            with st.spinner("Capturing image and checking skin..."):

                request = urllib.request.Request(
                    f"{PI_CAMERA_URL}/capture-analyze",
                    data=b"",
                    method="POST"
                )

                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))

                if not payload.get("ok"):
                    raise RuntimeError("Pi capture failed.")

                capture = payload.get("capture", {})
                gate = payload.get("analysis", {})
                quality = gate.get("quality", {})
                status = gate.get("status", "")

                if status != "skin_detected" or not quality.get("ok", False):

                    st.session_state["analysis_result"] = None
                    st.session_state["analysis_image"] = None

                    st.warning(
                        "Cancer prediction skipped. The Pi did not approve "
                        "this image as good-quality skin."
                    )

                    st.write("Pi result:", status)

                    reasons = quality.get("reasons", [])
                    if reasons:
                        st.write("Image-quality issues:", ", ".join(reasons))

                else:

                    image_url = PI_CAMERA_URL + capture["url"]

                    image_request = urllib.request.Request(image_url)

                    with urllib.request.urlopen(
                        image_request, timeout=60
                    ) as response:
                        image_bytes = response.read()

                    captured_image = Image.open(
                        BytesIO(image_bytes)
                    ).convert("RGB")

                    with st.spinner(
                        "Skin detected. Running cancer prediction..."
                    ):
                        result = predict_image(captured_image)

                    save_history(capture["file"], result)

                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_image"] = captured_image.copy()

                    st.success("Camera capture and analysis completed!")

                    st.caption(
                        "Pi skin-detection score: "
                        + str(gate.get("skin", {}).get("p_skin", "N/A"))
                        + ". This is not a cancer probability."
                    )

        except urllib.error.URLError as error:
            st.error(
                "Cannot connect to Raspberry Pi. Check that both devices "
                "are connected to the same network. " + str(error)
            )

        except Exception as error:
            st.error("Camera analysis failed: " + str(error))


    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1.15, 0.85], gap="large")

        with col1:

            st.markdown("### 🖼️ &nbsp; Image preview")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### Image details")

            st.markdown(f"""
            <div class="details-card">
                <div class="detail-item">
                    <span class="detail-label">File name</span>
                    <span class="detail-value">{escape(uploaded_file.name)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Image dimensions</span>
                    <span class="detail-value">{image.width} × {image.height} pixels</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">File format</span>
                    <span class="detail-value">{escape(uploaded_file.type or "Image")}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔬  Analyze image", use_container_width=True):

            with st.spinner("AI is analyzing the image..."):

                try:

                    result = predict_image(image)

                    save_history(
                        uploaded_file.name,
                        result
                    )

                    st.session_state["analysis_result"] = result

                    st.session_state["analysis_image"] = image.copy()

                    st.success("Analysis completed successfully!")

                except Exception as error:

                    st.error(f"Analysis failed: {error}")

    # Display results
    result = st.session_state.get("analysis_result")

    if result is not None:

        st.divider()

        st.markdown("## 📋 &nbsp; Analysis results")

        if result["stage1"] == "Normal":

            result_col, confidence_col = st.columns(2)

            with result_col:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">INITIAL SCREENING</div>
                    <div class="metric-value">Normal</div>
                </div>
                """, unsafe_allow_html=True)

            with confidence_col:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">LESION PROBABILITY</div>
                    <div class="metric-value">{result['lesion_probability'] * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.info(
                "The first-stage model classified this image as Normal. "
                "This screening result does not rule out a skin condition."
            )

        else:

            class_code = result["stage2"]

            result_col, confidence_col, lesion_col = st.columns(3)

            with result_col:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PREDICTED CATEGORY</div>
                    <div class="metric-value">{CLASS_LABELS.get(class_code, class_code)}</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Class code: {class_code.upper()}")

            with confidence_col:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MODEL CONFIDENCE</div>
                    <div class="metric-value">{result['confidence'] * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with lesion_col:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">LESION PROBABILITY</div>
                    <div class="metric-value">{result['lesion_probability'] * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Classification probabilities")

            probability_data = {
                CLASS_LABELS[name]: float(result["probabilities"][i])
                for i, name in enumerate(CLASS_NAMES)
            }

            st.bar_chart(probability_data)

            st.markdown("### 🔥 &nbsp; Grad-CAM visualization")
            st.caption(
                "The overlay highlights image regions that contributed to the model output."
            )

            original_col, gradcam_col = st.columns(2)

            with original_col:

                st.markdown("**Original image**")

                st.image(
                    st.session_state["analysis_image"],
                    use_container_width=True
                )

            with gradcam_col:

                st.markdown("**Model attention overlay**")

                st.image(
                    result["overlay"],
                    use_container_width=True
                )

        st.warning(
            "Educational prototype only — this result is not a medical "
            "diagnosis. Please consult a qualified dermatologist."
        )


# =====================================================
# 13. ANALYSIS HISTORY
# =====================================================

elif page == "Analysis History":

    title_col, action_col = st.columns([4, 1.25])

    with title_col:

        st.markdown("""
        <div class="history-page-head">
            <div class="main-title">Analysis History</div>
            <span class="history-page-badge">SCREENING RECORDS</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-title">Search, filter, and review saved image screenings and reports.</div>',
            unsafe_allow_html=True
        )

    with action_col:

        st.write("")
        st.button(
            "＋  New analysis",
            use_container_width=True,
            on_click=navigate_to_analysis,
            key="history_new_analysis"
        )

    history = get_history()

    if not history:

        st.markdown("""
        <div class="history-empty">
            <div style="font-size: 2rem;">🗂️</div>
            <h3>No screening records yet</h3>
            <p>Completed analyses and downloadable reports will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.button(
            "Start your first analysis",
            use_container_width=True,
            on_click=navigate_to_analysis,
            key="history_first_analysis"
        )

    else:

        normal_scans = sum(1 for record in history if record[3] == "Normal")
        lesion_scans = sum(1 for record in history if record[3] == "Lesion")
        normal_percent = normal_scans / len(history) * 100
        lesion_percent = lesion_scans / len(history) * 100

        st.markdown(f"""
        <div class="history-overview">
            <div class="history-visual-card">
                <div class="history-visual-icon">📋</div>
                <div class="history-visual-copy">
                    <span class="history-visual-label">Total scans</span>
                    <span class="history-visual-value">{len(history)}</span>
                </div>
            </div>
            <div class="history-visual-card">
                <div class="history-visual-icon">🟢</div>
                <div class="history-visual-copy">
                    <span class="history-visual-label">Model result: normal</span>
                    <span class="history-visual-value">{normal_scans}</span>
                </div>
            </div>
            <div class="history-visual-card">
                <div class="history-visual-icon">🔎</div>
                <div class="history-visual-copy">
                    <span class="history-visual-label">Model flagged as lesion</span>
                    <span class="history-visual-value">{lesion_scans}</span>
                </div>
            </div>
            <div class="history-mix-card">
                <div class="history-mix-head">
                    <strong>Screening distribution</strong>
                    <span>Share of saved scans</span>
                </div>
                <div class="history-mix-track" role="img" aria-label="Screening distribution">
                    <div class="history-mix-normal" style="width: {normal_percent:.1f}%"></div>
                    <div class="history-mix-lesion" style="width: {lesion_percent:.1f}%"></div>
                </div>
                <div class="history-mix-legend">
                    <span><i class="history-legend-dot normal"></i>Normal · {normal_scans}</span>
                    <span><i class="history-legend-dot lesion"></i>Lesion flagged · {lesion_scans}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        filter_col, status_col, category_col, sort_col = st.columns(
            [3.7, 1.35, 1.65, 1.3],
            gap="small"
        )

        with filter_col:
            search_term = st.text_input(
                "Search",
                placeholder="Search by scan ID, image, or category...",
                key="history_search"
            ).strip().lower()

        with status_col:
            status_filter = st.selectbox(
                "Status",
                ["All statuses", "Normal", "Lesion flagged"],
                key="history_status_filter"
            )

        with category_col:
            category_filter = st.selectbox(
                "Category",
                ["All categories", "Not classified"] + list(CLASS_LABELS.values()),
                key="history_category_filter"
            )

        with sort_col:
            sort_order = st.selectbox(
                "Sort by",
                ["Newest first", "Oldest first"],
                key="history_sort_order"
            )

        filtered_history = []

        for record in history:

            if status_filter == "Normal" and record[3] != "Normal":
                continue

            if status_filter == "Lesion flagged" and record[3] != "Lesion":
                continue

            if category_filter == "Not classified" and record[5] != "Not classified":
                continue

            if category_filter in CLASS_LABELS.values():
                selected_code = next(
                    code for code, label in CLASS_LABELS.items()
                    if label == category_filter
                )
                if record[5] != selected_code:
                    continue

            category_label = CLASS_LABELS.get(record[5], record[5])
            searchable_text = " ".join((
                str(record[0]),
                record[1],
                record[2],
                record[3],
                record[5],
                category_label
            )).lower()

            if search_term and search_term not in searchable_text:
                continue

            filtered_history.append(record)

        filtered_history.sort(
            key=lambda record: record[0],
            reverse=sort_order == "Newest first"
        )

        st.markdown(
            f'<div class="history-count">Showing <b>{len(filtered_history)}</b> of {len(history)} total scans</div>',
            unsafe_allow_html=True
        )
        st.write("")

        column_widths = [1.05, 1.75, 1.25, 1.35, 2.0, 0.9, 0.52, 0.52]
        header_labels = [
            "Scan ID",
            "Image file",
            "Scan date",
            "Screening",
            "Predicted category",
            "Confidence",
            "View",
            "PDF"
        ]
        header_columns = st.columns(column_widths, gap="small")

        for column, label in zip(header_columns, header_labels):
            with column:
                st.markdown(
                    f'<div class="history-heading">{label}</div>',
                    unsafe_allow_html=True
                )

        for record in filtered_history:

            scan_id = f"SCN-{record[0]:04d}"
            filename = escape(record[2])
            date_parts = record[1].split(" ", 1)
            scan_date = escape(date_parts[0])
            scan_time = escape(date_parts[1]) if len(date_parts) > 1 else ""
            status_class = "normal" if record[3] == "Normal" else "lesion"
            status_label = "NORMAL" if record[3] == "Normal" else "LESION FLAGGED"
            category_label = CLASS_LABELS.get(record[5], record[5])
            category_code = record[5].upper() if record[5] != "Not classified" else ""
            confidence_label = (
                f"{record[6] * 100:.1f}%"
                if record[5] != "Not classified"
                else "—"
            )
            pdf_bytes = generate_pdf(record)

            row_columns = st.columns(column_widths, gap="small")

            with row_columns[0]:
                st.markdown(
                    f'<div class="history-cell"><strong>{scan_id}</strong></div>',
                    unsafe_allow_html=True
                )

            with row_columns[1]:
                st.markdown(
                    f'<div class="history-cell"><strong>{filename}</strong></div>',
                    unsafe_allow_html=True
                )

            with row_columns[2]:
                st.markdown(
                    f'<div class="history-cell"><strong>{scan_date}</strong><small>{scan_time}</small></div>',
                    unsafe_allow_html=True
                )

            with row_columns[3]:
                st.markdown(
                    f'<div class="history-cell"><span class="history-status {status_class}">{status_label}</span></div>',
                    unsafe_allow_html=True
                )

            with row_columns[4]:
                st.markdown(
                    f'<div class="history-cell"><strong>{escape(category_label)}</strong><small>{escape(category_code)}</small></div>',
                    unsafe_allow_html=True
                )

            with row_columns[5]:
                st.markdown(
                    f'<div class="history-cell"><strong>{confidence_label}</strong></div>',
                    unsafe_allow_html=True
                )

            with row_columns[6]:
                if st.button(
                    "◉",
                    key=f"history_view_{record[0]}",
                    help=f"View details for {scan_id}"
                ):
                    st.session_state["selected_history_id"] = record[0]

            with row_columns[7]:
                st.download_button(
                    "↓",
                    data=pdf_bytes,
                    file_name=f"DermaAI_Report_{record[0]}.pdf",
                    mime="application/pdf",
                    key=f"history_pdf_{record[0]}",
                    help=f"Download PDF report for {scan_id}"
                )

            st.markdown(f"""
            <div class="history-mobile-card">
                <div class="history-mobile-top">
                    <strong class="history-mobile-id">{scan_id}</strong>
                    <span class="history-status {status_class}">{status_label}</span>
                </div>
                <div class="history-mobile-file">{filename}</div>
                <div class="history-mobile-meta">
                    <div class="history-mobile-meta-item">
                        <span class="history-mobile-meta-label">Scan date</span>
                        <span class="history-mobile-meta-value">{scan_date} {scan_time}</span>
                    </div>
                    <div class="history-mobile-meta-item">
                        <span class="history-mobile-meta-label">Category</span>
                        <span class="history-mobile-meta-value">{escape(category_label)} {escape(category_code)}</span>
                    </div>
                    <div class="history-mobile-meta-item">
                        <span class="history-mobile-meta-label">Confidence</span>
                        <span class="history-mobile-meta-value">{confidence_label}</span>
                    </div>
                    <div class="history-mobile-meta-item">
                        <span class="history-mobile-meta-label">Lesion probability</span>
                        <span class="history-mobile-meta-value">{record[4] * 100:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            mobile_action_columns = st.columns(2, gap="small")

            with mobile_action_columns[0]:
                st.markdown(
                    '<span class="history-mobile-action-marker"></span>',
                    unsafe_allow_html=True
                )
                if st.button(
                    "👁  View details",
                    use_container_width=True,
                    key=f"history_mobile_view_{record[0]}"
                ):
                    st.session_state["selected_history_id"] = record[0]

            with mobile_action_columns[1]:
                st.download_button(
                    "↓  Download PDF",
                    data=pdf_bytes,
                    file_name=f"DermaAI_Report_{record[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"history_mobile_pdf_{record[0]}"
                )

        if not filtered_history:
            st.info("No scans match these search and filter settings.")

        selected_id = st.session_state.get("selected_history_id")
        selected_record = next(
            (record for record in history if record[0] == selected_id),
            None
        )

        if selected_record is not None:
            selected_category = CLASS_LABELS.get(
                selected_record[5],
                selected_record[5]
            )
            selected_confidence = (
                f"{selected_record[6] * 100:.2f}%"
                if selected_record[5] != "Not classified"
                else "Not available"
            )
            st.markdown(f"""
            <div class="history-detail-card">
                <strong>Scan details · SCN-{selected_record[0]:04d}</strong>
                <p>Image: {escape(selected_record[2])} &nbsp; · &nbsp; Date: {escape(selected_record[1])}</p>
                <p>Screening: {escape(selected_record[3])} &nbsp; · &nbsp; Classification: {escape(selected_category)}</p>
                <p>Lesion probability: {selected_record[4] * 100:.2f}% &nbsp; · &nbsp; Model confidence: {escape(selected_confidence)}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Close scan details", key="history_close_details"):
                st.session_state.pop("selected_history_id", None)
                st.rerun()

        st.divider()

        if st.button("🗑  Clear all history", key="history_clear_all"):

            clear_history()
            st.session_state.pop("selected_history_id", None)
            st.success("Analysis history cleared.")
            st.rerun()


# =====================================================
# 14. ABOUT THE MODEL
# =====================================================

elif page == "About the Model":

    st.markdown(
        '<div class="main-title">About the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">A transparent overview of this educational screening prototype</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="content-card">
        <h3>🧠 &nbsp; Two-stage image screening</h3>
        <p>The prototype uses a sequential classification workflow. An image is first screened as normal or lesion; images classified as a lesion are then passed to a seven-category classifier.</p>
        <p><b>Architecture:</b> MobileNetV2 ensemble &nbsp; · &nbsp; <b>Explainability:</b> Grad-CAM</p>
        <p><b>Training dataset:</b> HAM10000</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model performance")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">VALIDATION ACCURACY</div>
            <div class="metric-value">74.39%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MACRO F1 SCORE</div>
            <div class="metric-value">53.29%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Seven classification categories")

    category_columns = st.columns(3)

    for index, (code, label) in enumerate(CLASS_LABELS.items()):

        with category_columns[index % len(category_columns)]:
            st.markdown(f"""
            <div class="content-card">
                <div class="eyebrow">{code.upper()}</div>
                <p><b>{label}</b></p>
            </div>
            """, unsafe_allow_html=True)

    st.warning(
        "This is an experimental educational prototype, not a clinically "
        "validated medical device. It must not be used to diagnose skin "
        "cancer or guide treatment. Please consult a qualified dermatologist."
    )
