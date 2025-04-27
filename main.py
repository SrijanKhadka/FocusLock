import sys
import os
import sqlite3
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import re
import uuid
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTableWidget, QTableWidgetItem,
    QMessageBox, QTabWidget, QFileDialog, QInputDialog, QCheckBox,
    QDateTimeEdit, QSpinBox, QGridLayout, QProgressBar, QComboBox,
    QScrollArea, QStatusBar, QToolButton, QSizePolicy, QFrame,
    QGraphicsDropShadowEffect, QListWidget
)
from PyQt5.QtCore import QTimer, QDateTime, Qt, QThread, pyqtSignal, QSize, QUrl
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap, QIcon
import pyautogui
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
from PyQt5.QtWebEngineWidgets import QWebEngineView
# --- Lockdown imports ---
import ctypes
from ctypes import wintypes
import psutil
import threading

# Fix for missing LRESULT in some Python versions
if not hasattr(wintypes, 'LRESULT'):
    wintypes.LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize YOLOv5s for phone detection
yolo_model = YOLO('yolov5s.pt')

# Ensure screenshots directory exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Database setup with connection pooling
def get_db_connection():
    conn = sqlite3.connect("database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, name TEXT, email TEXT, password TEXT, screenshot_folder TEXT)''')
            try:
                c.execute("ALTER TABLE users ADD COLUMN screenshot_folder TEXT")
            except sqlite3.OperationalError:
                pass
            c.execute('''CREATE TABLE IF NOT EXISTS focus_scores
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, score REAL, timestamp TEXT, details TEXT, session_code TEXT)''')
            c.execute("PRAGMA table_info(focus_scores)")
            columns = [col[1] for col in c.fetchall()]
            if "session_code" not in columns:
                c.execute("ALTER TABLE focus_scores ADD COLUMN session_code TEXT")
            c.execute('''CREATE TABLE IF NOT EXISTS exam_sessions
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, session_code TEXT, start_time TEXT, duration INTEGER, status TEXT, teacher_id INTEGER, allowed_website TEXT, allow_retakes INTEGER DEFAULT 0)''')
            c.execute("PRAGMA table_info(exam_sessions)")
            columns = [col[1] for col in c.fetchall()]
            if "allowed_website" not in columns:
                c.execute("ALTER TABLE exam_sessions ADD COLUMN allowed_website TEXT")
            if "allow_retakes" not in columns:
                c.execute("ALTER TABLE exam_sessions ADD COLUMN allow_retakes INTEGER DEFAULT 0")
            # New: exam_attempts table
            c.execute('''CREATE TABLE IF NOT EXISTS exam_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                session_code TEXT,
                attempt_time TEXT,
                allowed_retake INTEGER DEFAULT 0
            )''')
            conn.commit()
    except sqlite3.Error as e:
        print(f"[ERROR] Database initialization failed: {e}")

init_db()

# Euclidean Distance
def euclidean(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Eye Openness (Eye Aspect Ratio)
def eye_openness(eye_top, eye_bottom, eye_left, eye_right):
    vertical_openness = euclidean(eye_top, eye_bottom)
    horizontal_openness = euclidean(eye_left, eye_right)
    return vertical_openness / horizontal_openness if horizontal_openness > 0 else 0.5

# Iris Position Ratio
def iris_position_ratio(iris_center, eye_left, eye_right, eye_top, eye_bottom):
    total_width = euclidean(eye_left, eye_right)
    total_height = euclidean(eye_top, eye_bottom)
    iris_to_left = euclidean(iris_center, eye_left)
    iris_to_top = euclidean(iris_center, eye_top)
    horizontal_ratio = iris_to_left / total_width if total_width > 0 else 0.5
    vertical_ratio = iris_to_top / total_height if total_height > 0 else 0.5
    return horizontal_ratio, vertical_ratio

# Head Tilt Ratio (Left/Right)
def head_tilt_ratio(left_temple, right_temple, nose_tip):
    left_to_nose = euclidean(left_temple, nose_tip)
    right_to_nose = euclidean(right_temple, nose_tip)
    return left_to_nose / right_to_nose if right_to_nose > 0 else 1.0

# Head Down Ratio (Up/Down)
def head_down_ratio(nose_tip, chin, eye_level):
    eye_to_nose = euclidean(eye_level, nose_tip)
    chin_to_nose = euclidean(nose_tip, chin)
    return eye_to_nose / chin_to_nose if chin_to_nose > 0 else 1.0

# Updated ALLOWED_WINDOWS to include "blackboard" and "terminal"
ALLOWED_WINDOWS = ["focusscore", "blackboard", "terminal"]

class MonitorThread(QThread):
    update_signal = pyqtSignal(float, str, str, bool)
    error_signal = pyqtSignal(str)

    def __init__(self, cap, current_user, session_code, session_start_time, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.current_user = current_user
        self.session_code = session_code
        self.session_start_time = session_start_time
        self.monitoring = False
        self.phone_alerts = 0
        self.head_turn_events = 0
        self.time_on_screen = 0
        self.total_time = 0
        self.cheating_detected = False
        # --- Event-based distraction counter ---
        self.last_window_focused = True
        self.window_switch_events = 0
        self.eyes_offscreen_counter = 0
        self.eyes_closed_counter = 0

    def run(self):
        try:
            self.monitoring = True
            # Persistent focus score for the session
            self.session_focus_score = 100.0
            last_penalty_time = 0
            last_recovery_time = 0
            recovery_interval = 5  # seconds without cheating before recovery
            penalty_log = []
            while self.monitoring:
                if not self.cap or not self.cap.isOpened():
                    self.error_signal.emit("Webcam not available")
                    break
                ret, frame = self.cap.read()
                if not ret:
                    self.error_signal.emit("Failed to read frame")
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                h, w, _ = frame.shape
                phone_detected = False
                results_yolo = yolo_model(frame)[0]
                for box in results_yolo.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = yolo_model.names[cls_id]
                    if label == 'cell phone' and conf > 0.5:
                        phone_detected = True
                        self.phone_alerts += 1
                        break
                focus, details = self.parent().get_focus_score(results, w, h, phone_detected)
                status = details.split(" | ")[0]
                debug_info = details.split(" | ")[1] if " | " in details else ""
                active_window = pyautogui.getActiveWindow().title.lower()
                window_focused = any(allowed_window in active_window for allowed_window in ALLOWED_WINDOWS)
                cheating_event = False
                penalty_reason = None
                penalty_amount = 0
                screenshot_needed = False
                # Cheating events and penalties
                if phone_detected:
                    cheating_event = True
                    penalty_reason = "Phone Detected"
                    penalty_amount = 30
                    screenshot_needed = True
                elif not window_focused:
                    cheating_event = True
                    penalty_reason = "Non-exam window active"
                    penalty_amount = 20
                    screenshot_needed = True
                elif "Head Turn Detected" in status:
                    cheating_event = True
                    penalty_reason = "Head Turn Detected"
                    penalty_amount = 15
                    screenshot_needed = True
                elif "Eyes Closed" in status:
                    cheating_event = True
                    penalty_reason = "Eyes Closed"
                    penalty_amount = 10
                    screenshot_needed = True
                elif "Eyes Off Screen" in status:
                    self.eyes_offscreen_counter += 1
                    if self.eyes_offscreen_counter >= 2:  # Take screenshot if offscreen for 2+ frames
                        cheating_event = True
                        penalty_reason = "Eyes Off Screen"
                        penalty_amount = 10
                        screenshot_needed = True
                else:
                    self.eyes_offscreen_counter = 0
                # Apply penalty if cheating event
                now = time.time()
                if cheating_event:
                    self.session_focus_score = max(0, self.session_focus_score - penalty_amount)
                    penalty_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Penalty: -{penalty_amount} ({penalty_reason}) -> {self.session_focus_score:.1f}")
                    last_penalty_time = now
                    if screenshot_needed:
                        try:
                            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', self.current_user['name'].lower())
                            folder = os.path.join("screenshots", safe_name)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_path = os.path.join(folder, f"screenshot_{self.session_code}_{timestamp_str}.png")
                            # Take a screenshot of the desktop, not the webcam
                            screenshot = pyautogui.screenshot()
                            screenshot.save(file_path)
                            print(f"[DEBUG] Desktop screenshot taken: {file_path}")
                        except Exception as e:
                            print(f"[ERROR] Could not save desktop screenshot: {e}")
                # Recovery if no cheating for recovery_interval
                elif now - last_penalty_time > recovery_interval and self.session_focus_score < 100:
                    self.session_focus_score = min(100, self.session_focus_score + 1)
                    penalty_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Recovery: +1 (Focused) -> {self.session_focus_score:.1f}")
                    last_recovery_time = now
                # Report details
                log_tail = penalty_log[-3:] if len(penalty_log) > 3 else penalty_log
                details_report = details + " | " + "; ".join(log_tail)
                self.update_signal.emit(self.session_focus_score, status, details_report, phone_detected)
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO focus_scores (student_id, score, timestamp, details, session_code) VALUES (?, ?, ?, ?, ?)",
                              (self.current_user["id"], self.session_focus_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), details_report, self.session_code))
                    conn.commit()
                time.sleep(0.5)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.monitoring = False

    def stop(self):
        self.monitoring = False
        self.wait()

class FocusScoreApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusScore")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)  # Ensure minimum size for responsiveness
        self.debug_label = QLabel("Debug: N/A")
        self.cheating_label = QLabel("Cheating Warning: None")
        self.cheating_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.role_selection_widget = QWidget()
        self.setup_role_selection_ui()
        self.role_selection_tab_index = self.tabs.addTab(self.role_selection_widget, "Welcome")
        self.teacher_login_widget = QWidget()
        self.setup_teacher_login_ui()
        self.teacher_login_tab_index = self.tabs.addTab(self.teacher_login_widget, "Teacher Login")
        self.student_login_widget = QWidget()
        self.setup_student_login_ui()
        self.student_login_tab_index = self.tabs.addTab(self.student_login_widget, "Student Login")
        self.student_widget = QWidget()
        self.setup_student_ui()
        self.student_tab_index = self.tabs.addTab(self.student_widget, "Student Portal")
        # Add Exam tab (hidden by default)
        self.exam_widget = QWidget()
        self.setup_exam_ui()
        self.exam_tab_index = self.tabs.addTab(self.exam_widget, "📝 Exam")
        self.tabs.setTabVisible(self.exam_tab_index, False)
        # Add Website tab (hidden by default)
        self.website_tab = QWidget()
        self.setup_website_tab_ui()
        self.website_tab_index = self.tabs.addTab(self.website_tab, "🌐 Exam Website")
        self.tabs.setTabVisible(self.website_tab_index, False)
        self.register_widget = QWidget()
        self.setup_register_ui()
        self.register_tab_index = self.tabs.addTab(self.register_widget, "Register")
        self.teacher_widget = QWidget()
        self.setup_teacher_ui()
        self.teacher_tab_index = self.tabs.addTab(self.teacher_widget, "Teacher Portal")
        self.tabs.setTabVisible(self.teacher_tab_index, False)
        self.tabs.setTabVisible(self.student_tab_index, False)
        self.tabs.setCurrentIndex(self.role_selection_tab_index)
        self.current_user = None
        self.cap = None
        self.monitoring = False
        self.monitoring_thread = None
        self.focus_score = 100.0
        self.session_scores = []
        self.session_timestamps = []
        self.session_start_time = None
        self.time_on_screen = 0
        self.total_time = 0
        self.last_screenshot_time = 0
        self.phone_detected = False
        self.gaze_counter = 0
        self.cheating_detected = False
        self.blink_counter = 0
        self.current_session_code = None
        self.phone_alerts = 0
        self.head_turn_events = 0
        self.student_email_text = ""
        self.student_password_text = ""
        self.teacher_email_text = ""
        self.teacher_password_text = ""
        self.allowed_website = None  # URL set by teacher for online exam
        self.allowed_websites = []   # List of allowed websites
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #212121;
                color: #E0E0E0;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #424242;
                background: #212121;
            }
            QTabBar::tab {
                background: #424242;
                color: #E0E0E0;
                padding: 12px 24px;
                border: none;
                font-weight: bold;
                min-width: 120px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #1976D2;
                color: #FFFFFF;
            }
            QTabBar {
                background: #212121;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
            }
            QLineEdit, QDateTimeEdit, QSpinBox, QComboBox {
                background: #424242;
                color: #E0E0E0;
                border: 1px solid #616161;
                padding: 8px;
                border-radius: 4px;
                min-width: 200px;
                max-width: 300px;
            }
            QPushButton {
                background: #1976D2;
                color: #FFFFFF;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: #1565C0;
            }
            QPushButton:disabled {
                background: #616161;
                color: #B0B0B0;
            }
            QTableWidget {
                background: #fff;
                color: #23272A;
                border: 1px solid #616161;
                gridline-color: #616161;
                alternate-background-color: #f0f0f0;
            }
            QTableWidget QTableCornerButton::section {
                background: #fff;
            }
            QHeaderView::section {
                background: #1976D2;
                color: #FFFFFF;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background: #1976D2;
                color: #fff;
            }
            QProgressBar {
                background: #424242;
                color: #E0E0E0;
                border: 1px solid #616161;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #1976D2;
                border-radius: 4px;
            }
            QCheckBox {
                color: #E0E0E0;
            }
            QGroupBox, QFrame[modernSection="true"] {
                border: 2px solid #1976D2;
                border-radius: 8px;
                margin-top: 12px;
                margin-bottom: 12px;
                padding: 16px;
                background: #23272A;
            }
        """)

    def setup_role_selection_ui(self):
        layout = QVBoxLayout()
        self.role_selection_widget.setLayout(layout)
        # Add theme toggle
        theme_toggle = QPushButton("Toggle Dark/Light Theme")
        theme_toggle.setCheckable(True)
        theme_toggle.setChecked(True)
        theme_toggle.clicked.connect(self.toggle_theme)
        welcome_label = QLabel("Welcome to FocusScore")
        welcome_label.setFont(QFont("Arial", 28, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        subtitle_label = QLabel("Choose your role to get started")
        subtitle_label.setFont(QFont("Arial", 16))
        subtitle_label.setAlignment(Qt.AlignCenter)
        button_layout = QHBoxLayout()
        teacher_button = QPushButton("Teacher")
        teacher_button.clicked.connect(self.show_teacher_login)
        student_button = QPushButton("Student")
        student_button.clicked.connect(self.show_student_login)
        register_button = QPushButton("Register")
        register_button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.register_tab_index))
        button_layout.addStretch()
        button_layout.addWidget(teacher_button)
        button_layout.addWidget(student_button)
        button_layout.addWidget(register_button)
        button_layout.addStretch()
        layout.addStretch(2)
        layout.addWidget(theme_toggle)
        layout.addWidget(welcome_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(40)
        layout.addLayout(button_layout)
        layout.addStretch(3)

    def setup_teacher_login_ui(self):
        layout = QVBoxLayout()
        self.teacher_login_widget.setLayout(layout)
        title_label = QLabel("Teacher Login")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.teacher_email_input = QLineEdit()
        self.teacher_email_input.setPlaceholderText("Email")
        self.teacher_email_input.textChanged.connect(self.on_teacher_email_changed)
        self.teacher_password_input = QLineEdit()
        self.teacher_password_input.setPlaceholderText("Password")
        self.teacher_password_input.setEchoMode(QLineEdit.Password)
        self.teacher_password_input.textChanged.connect(self.on_teacher_password_changed)
        self.teacher_show_password = QCheckBox("Show Password")
        self.teacher_show_password.stateChanged.connect(self.toggle_teacher_password_visibility)
        login_button = QPushButton("Login")
        login_button.clicked.connect(self.teacher_login)
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.role_selection_tab_index))
        back_button.setStyleSheet("background: #616161;")
        form_layout = QVBoxLayout()
        form_layout.addWidget(self.teacher_email_input)
        form_layout.addWidget(self.teacher_password_input)
        form_layout.addWidget(self.teacher_show_password)
        form_layout.setAlignment(Qt.AlignCenter)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(login_button)
        button_layout.addWidget(back_button)
        button_layout.addStretch()
        layout.addStretch(2)
        layout.addWidget(title_label)
        layout.addSpacing(30)
        layout.addLayout(form_layout)
        layout.addSpacing(20)
        layout.addLayout(button_layout)
        layout.addStretch(3)

    def setup_student_login_ui(self):
        layout = QVBoxLayout()
        self.student_login_widget.setLayout(layout)
        title_label = QLabel("Student Login")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.student_email_input = QLineEdit()
        self.student_email_input.setPlaceholderText("Email")
        self.student_email_input.textChanged.connect(self.on_student_email_changed)
        self.student_password_input = QLineEdit()
        self.student_password_input.setPlaceholderText("Password")
        self.student_password_input.setEchoMode(QLineEdit.Password)
        self.student_password_input.textChanged.connect(self.on_student_password_changed)
        self.student_show_password = QCheckBox("Show Password")
        self.student_show_password.stateChanged.connect(self.toggle_student_password_visibility)
        login_button = QPushButton("Login")
        login_button.clicked.connect(self.student_login)
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.role_selection_tab_index))
        back_button.setStyleSheet("background: #616161;")
        form_layout = QVBoxLayout()
        form_layout.addWidget(self.student_email_input)
        form_layout.addWidget(self.student_password_input)
        form_layout.addWidget(self.student_show_password)
        form_layout.setAlignment(Qt.AlignCenter)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(login_button)
        button_layout.addWidget(back_button)
        button_layout.addStretch()
        layout.addStretch(2)
        layout.addWidget(title_label)
        layout.addSpacing(30)
        layout.addLayout(form_layout)
        layout.addSpacing(20)
        layout.addLayout(button_layout)
        layout.addStretch(3)

    def setup_register_ui(self):
        layout = QVBoxLayout()
        self.register_widget.setLayout(layout)
        title_label = QLabel("Register")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.register_student_id_input = QLineEdit()
        self.register_student_id_input.setPlaceholderText("Student ID (required, unique for students)")
        self.register_name_input = QLineEdit()
        self.register_name_input.setPlaceholderText("Name")
        self.register_email_input = QLineEdit()
        self.register_email_input.setPlaceholderText("Email")
        self.register_password_input = QLineEdit()
        self.register_password_input.setPlaceholderText("Password")
        self.register_password_input.setEchoMode(QLineEdit.Password)
        self.register_role_input = QComboBox()
        self.register_role_input.addItems(["Student", "Teacher"])
        register_button = QPushButton("Register")
        register_button.clicked.connect(self.register)
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.role_selection_tab_index))
        back_button.setStyleSheet("background: #616161;")
        form_layout = QVBoxLayout()
        form_layout.addWidget(self.register_student_id_input)
        form_layout.addWidget(self.register_name_input)
        form_layout.addWidget(self.register_email_input)
        form_layout.addWidget(self.register_password_input)
        form_layout.addWidget(self.register_role_input)
        form_layout.setAlignment(Qt.AlignCenter)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(register_button)
        button_layout.addWidget(back_button)
        button_layout.addStretch()
        layout.addStretch(2)
        layout.addWidget(title_label)
        layout.addSpacing(30)
        layout.addLayout(form_layout)
        layout.addSpacing(20)
        layout.addLayout(button_layout)
        layout.addStretch(3)

    def setup_teacher_ui(self):
        # --- Main Scroll Area and Shadow Effect ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_widget = QWidget()
        scroll_area.setWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(18)
        shadow.setColor(QColor(30, 30, 30, 180))
        shadow.setOffset(0, 4)
        main_widget.setGraphicsEffect(shadow)

        # --- Session Management Section ---
        session_label = QLabel("Manage Exam Sessions")
        session_label.setFont(QFont("Arial", 18, QFont.Bold))
        session_grid = QGridLayout()
        self.start_time_input = QDateTimeEdit()
        self.start_time_input.setCalendarPopup(True)
        self.start_time_input.setDateTime(QDateTime.currentDateTime())
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 360)
        self.duration_input.setValue(60)
        schedule_button = QPushButton("Schedule")
        schedule_button.clicked.connect(self.schedule_session)
        schedule_button.setStyleSheet("background: #F9A825;")
        open_now_button = QPushButton("Open Now")
        open_now_button.clicked.connect(self.open_session_now)
        open_now_button.setStyleSheet("background: #4CAF50;")
        # --- Add retake checkbox to session creation ---
        self.allow_retakes_checkbox = QCheckBox("Allow Retakes for this Session")
        session_grid.addWidget(QLabel("Start Time:"), 0, 0)
        session_grid.addWidget(self.start_time_input, 0, 1)
        session_grid.addWidget(QLabel("Duration (min):"), 0, 2)
        session_grid.addWidget(self.duration_input, 0, 3)
        session_grid.addWidget(schedule_button, 0, 4)
        session_grid.addWidget(open_now_button, 0, 5)
        session_grid.addWidget(self.allow_retakes_checkbox, 0, 6)

        self.session_table = QTableWidget()
        self.session_table.setColumnCount(6)
        self.session_table.setHorizontalHeaderLabels(["Session Code", "Start Time", "Duration", "Status", "Allowed Website", "Actions"])
        self.session_table.horizontalHeader().setStretchLastSection(True)
        self.session_table.setColumnWidth(0, 150)
        self.session_table.setColumnWidth(1, 200)
        self.session_table.setColumnWidth(2, 100)
        self.session_table.setColumnWidth(3, 150)
        self.session_table.setColumnWidth(4, 250)
        self.session_table.setColumnWidth(5, 200)
        self.session_table.setAlternatingRowColors(True)

        # --- Search Bar Above Student Table ---
        search_layout = QHBoxLayout()
        self.student_search_input = QLineEdit()
        self.student_search_input.setPlaceholderText("Search students by name or email...")
        self.student_search_input.textChanged.connect(self.filter_student_table)
        search_icon = QToolButton()
        search_icon.setIcon(QIcon.fromTheme("edit-find"))
        search_icon.setEnabled(False)
        search_layout.addWidget(search_icon)
        search_layout.addWidget(self.student_search_input)
        search_layout.addStretch()

        # --- Collapsible Add Student Form ---
        self.add_student_frame = QFrame()
        self.add_student_frame.setFrameShape(QFrame.StyledPanel)
        self.add_student_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        add_student_layout = QVBoxLayout(self.add_student_frame)
        add_student_label = QLabel("Add Student")
        add_student_label.setFont(QFont("Arial", 18, QFont.Bold))
        add_student_grid = QGridLayout()
        self.student_id_input = QLineEdit()
        self.student_id_input.setPlaceholderText("Student ID (required, unique)")
        self.student_name_input = QLineEdit()
        self.student_name_input.setPlaceholderText("Student Name")
        self.student_email_input = QLineEdit()
        self.student_email_input.setPlaceholderText("Student Email")
        self.student_password_input = QLineEdit()
        self.student_password_input.setPlaceholderText("Student Password")
        add_student_grid.addWidget(QLabel("Student ID:"), 0, 0)
        add_student_grid.addWidget(self.student_id_input, 0, 1)
        add_student_grid.addWidget(QLabel("Name:"), 0, 2)
        add_student_grid.addWidget(self.student_name_input, 0, 3)
        add_student_grid.addWidget(QLabel("Email:"), 0, 4)
        add_student_grid.addWidget(self.student_email_input, 0, 5)
        add_student_grid.addWidget(QLabel("Password:"), 0, 6)
        add_student_grid.addWidget(self.student_password_input, 0, 7)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_student)
        add_student_layout.addWidget(add_student_label)
        add_student_layout.addLayout(add_student_grid)
        self.add_student_frame.setVisible(True)
        # Toggle button for collapsible form
        self.toggle_add_student_btn = QPushButton("Hide Add Student Form")
        self.toggle_add_student_btn.setCheckable(True)
        self.toggle_add_student_btn.setChecked(True)
        self.toggle_add_student_btn.clicked.connect(self.toggle_add_student_form)
        self.toggle_add_student_btn.setStyleSheet("background: #1976D2; color: #fff;")

        # --- Student List Table ---
        student_list_label = QLabel("Students")
        student_list_label.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(student_list_label)
        self.student_table = QTableWidget()
        self.student_table.setColumnCount(5)
        self.student_table.setHorizontalHeaderLabels(["ID", "Name", "Email", "Focus Score", "Actions"])
        self.student_table.cellDoubleClicked.connect(self.view_student_scores)
        self.student_table.horizontalHeader().setStretchLastSection(True)
        self.student_table.setColumnWidth(0, 50)
        self.student_table.setColumnWidth(1, 150)
        self.student_table.setColumnWidth(2, 250)
        self.student_table.setColumnWidth(3, 100)
        self.student_table.setColumnWidth(4, 300)
        self.student_table.setAlternatingRowColors(True)
        # Ensure 10 rows are visible by default
        row_height = self.student_table.verticalHeader().defaultSectionSize()
        self.student_table.setMinimumHeight(row_height * 10 + self.student_table.horizontalHeader().height())
        self.student_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_layout.addWidget(self.student_table)

        # --- Buttons (Refresh, Logout) ---
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_students_and_sessions)
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)
        button_layout.addStretch()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.logout_button)

        # --- Progress Bar ---
        self.progress_bar.setVisible(False)

        # --- Add widgets to main layout ---
        main_layout.addWidget(session_label)
        main_layout.addLayout(session_grid)
        main_layout.addWidget(self.session_table)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.toggle_add_student_btn)
        main_layout.addWidget(self.add_student_frame)
        main_layout.addSpacing(20)
        main_layout.addWidget(student_list_label)
        main_layout.addLayout(search_layout)
        main_layout.addWidget(self.student_table)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(button_layout)
        main_layout.addStretch()

        # --- Status Bar ---
        self.teacher_status_bar = QStatusBar()
        self.teacher_status_bar.setStyleSheet("background: #23272A; color: #E0E0E0; font-size: 13px;")
        self.teacher_status_bar.showMessage("Welcome to the Teacher Dashboard.")
        layout = QVBoxLayout(self.teacher_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(scroll_area)
        layout.addWidget(self.teacher_status_bar)

        # --- Add website field for online exam ---
        website_layout = QHBoxLayout()
        self.website_input = QLineEdit()
        self.website_input.setPlaceholderText("Enter allowed exam website (e.g. https://exam.com)")
        set_website_button = QPushButton("Set Allowed Website")
        set_website_button.clicked.connect(self.set_allowed_website)
        website_layout.addWidget(self.website_input)
        website_layout.addWidget(set_website_button)
        # Add retake checkbox after layout is defined
        self.allow_retakes_checkbox = QCheckBox("Allow Retakes for this Session")
        website_layout.addWidget(self.allow_retakes_checkbox)
        main_layout.insertLayout(2, website_layout)  # Insert after session table

        # --- Add website list widget ---
        self.website_list_widget = QListWidget()
        self.website_list_widget.setFixedHeight(80)
        remove_website_button = QPushButton("Remove Selected Website")
        remove_website_button.setStyleSheet("background: #D32F2F; color: #fff; font-size: 13px; border-radius: 4px; padding: 6px 12px;")
        remove_website_button.clicked.connect(self.remove_selected_website)
        website_list_layout = QVBoxLayout()
        website_list_layout.addWidget(QLabel("Allowed Websites:"))
        website_list_layout.addWidget(self.website_list_widget)
        website_list_layout.addWidget(remove_website_button)
        main_layout.insertLayout(3, website_list_layout)

    def toggle_add_student_form(self):
        visible = self.toggle_add_student_btn.isChecked()
        self.add_student_frame.setVisible(visible)
        self.toggle_add_student_btn.setText("Hide Add Student Form" if visible else "Show Add Student Form")

    def show_teacher_login(self):
        self.tabs.setCurrentIndex(self.teacher_login_tab_index)

    def show_student_login(self):
        self.tabs.setCurrentIndex(self.student_login_tab_index)

    def teacher_login(self):
        email = self.teacher_email_text.strip().lower()
        password = self.teacher_password_text.strip()
        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter email and password")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE email = ? AND password = ? AND role = 'teacher'",
                          (email, password))
                user = c.fetchone()
                if user:
                    screenshot_folder = user['screenshot_folder'] or os.path.join("screenshots", re.sub(r'[^a-zA-Z0-9]', '_', user['name'].lower()))
                    if not os.path.exists(screenshot_folder):
                        os.makedirs(screenshot_folder)
                    c.execute("UPDATE users SET screenshot_folder = ? WHERE id = ?", (screenshot_folder, user['id']))
                    conn.commit()
                    self.current_user = {"id": user['id'], "role": user['role'], "name": user['name'], "screenshot_folder": screenshot_folder}
                    self.teacher_email_input.clear()
                    self.teacher_password_input.clear()
                    self.teacher_show_password.setChecked(False)
                    self.teacher_email_text = ""
                    self.teacher_password_text = ""
                    self.tabs.setTabVisible(self.teacher_tab_index, True)
                    self.tabs.setTabVisible(self.student_tab_index, False)
                    self.tabs.setCurrentIndex(self.teacher_tab_index)
                    self.load_students_and_sessions()
                else:
                    QMessageBox.warning(self, "Error",  "Invalid credentials")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def student_login(self):
        email = self.student_email_text.strip().lower()
        password = self.student_password_text.strip()
        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter email and password")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE email = ? AND password = ? AND role = 'student'",
                          (email, password))
                user = c.fetchone()
                if user:
                    screenshot_folder = user['screenshot_folder'] or os.path.join("screenshots", re.sub(r'[^a-zA-Z0-9]', '_', user['name'].lower()))
                    if not os.path.exists(screenshot_folder):
                        os.makedirs(screenshot_folder)
                    c.execute("UPDATE users SET screenshot_folder = ? WHERE id = ?", (screenshot_folder, user['id']))
                    conn.commit()
                    self.current_user = {"id": user['id'], "role": user['role'], "name": user['name'], "screenshot_folder": screenshot_folder}
                    self.student_email_input.clear()
                    self.student_password_input.clear()
                    self.student_show_password.setChecked(False)
                    self.student_email_text = ""
                    self.student_password_text = ""
                    self.tabs.setTabVisible(self.student_tab_index, True)
                    self.tabs.setTabVisible(self.teacher_tab_index, False)
                    self.tabs.setCurrentIndex(self.student_tab_index)
                else:
                    QMessageBox.warning(self, "Error", "Invalid credentials")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def register(self):
        student_id = self.register_student_id_input.text().strip()
        name = self.register_name_input.text().strip()
        email = self.register_email_input.text().strip().lower()
        password = self.register_password_input.text().strip()
        role = self.register_role_input.currentText().lower()
        if not name or not email or not password or (role == "student" and not student_id):
            QMessageBox.warning(self, "Error", "Please fill all fields (Student ID required for students)")
            return
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
            QMessageBox.warning(self, "Error", "Invalid email address")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                if role == "student":
                    c.execute("SELECT * FROM users WHERE student_id = ?", (student_id,))
                    if c.fetchone():
                        QMessageBox.warning(self, "Error", "Student ID already registered")
                        return
                c.execute("SELECT * FROM users WHERE email = ?", (email,))
                if c.fetchone():
                    QMessageBox.warning(self, "Error", "Email already registered")
                    return
                screenshot_folder = os.path.join("screenshots", re.sub(r'[^a-zA-Z0-9]', '_', name.lower())) if role == "student" else None
                if screenshot_folder and not os.path.exists(screenshot_folder):
                    os.makedirs(screenshot_folder)
                if role == "student":
                    c.execute("INSERT INTO users (student_id, role, name, email, password, screenshot_folder) VALUES (?, ?, ?, ?, ?, ?)",
                              (student_id, role, name, email, password, screenshot_folder))
                else:
                    c.execute("INSERT INTO users (role, name, email, password, screenshot_folder) VALUES (?, ?, ?, ?, ?)",
                              (role, name, email, password, screenshot_folder))
                conn.commit()
                QMessageBox.information(self, "Success", "Registration successful")
                self.register_student_id_input.clear()
                self.register_name_input.clear()
                self.register_email_input.clear()
                self.register_password_input.clear()
                self.tabs.setCurrentIndex(self.role_selection_tab_index)
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def add_student(self):
        student_id = self.student_id_input.text().strip()
        name = self.student_name_input.text().strip()
        email = self.student_email_input.text().strip().lower()
        password = self.student_password_input.text().strip()
        if not student_id or not name or not email or not password:
            QMessageBox.warning(self, "Error", "Please fill all fields (Student ID required)")
            return
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
            QMessageBox.warning(self, "Error", "Invalid email address")
            return
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE student_id = ?", (student_id,))
                if c.fetchone():
                    QMessageBox.warning(self, "Error", "Student ID already registered")
                    return
                c.execute("SELECT * FROM users WHERE email = ?", (email,))
                if c.fetchone():
                    QMessageBox.warning(self, "Error", "Email already registered")
                    return
                screenshot_folder = os.path.join("screenshots", re.sub(r'[^a-zA-Z0-9]', '_', name.lower()))
                if not os.path.exists(screenshot_folder):
                    os.makedirs(screenshot_folder)
                c.execute("INSERT INTO users (student_id, role, name, email, password, screenshot_folder) VALUES (?, ?, ?, ?, ?, ?)",
                          (student_id, "student", name, email, password, screenshot_folder))
                conn.commit()
                self.load_students_and_sessions()
                QMessageBox.information(self, "Success", "Student added successfully")
                self.student_id_input.clear()
                self.student_name_input.clear()
                self.student_email_input.clear()
                self.student_password_input.clear()
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

    def change_student_password(self, student_id, email):
        new_password, ok = QInputDialog.getText(self, "Change Password", f"Enter new password for {email}:", QLineEdit.Normal)
        if ok and new_password:
            new_password = new_password.strip()
            if not new_password:
                QMessageBox.warning(self, "Error", "Password cannot be empty")
                return
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("UPDATE users SET password = ? WHERE id = ?", (new_password, student_id))
                    conn.commit()
                    QMessageBox.information(self, "Success", "Password updated successfully")
                    self.load_students_and_sessions()
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Error", f"Database error: {e}")

    def delete_student(self, student_id, student_name=None):
        name_str = f"{student_name}" if student_name else "this student"
        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete {name_str}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("DELETE FROM users WHERE id = ?", (student_id,))
                    c.execute("DELETE FROM focus_scores WHERE student_id = ?", (student_id,))
                    conn.commit()
                    self.load_students_and_sessions()
                    QMessageBox.information(self, "Success", "Student deleted successfully")
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Error", f"Database error: {e}")

    def generate_session_code(self):
        return str(uuid.uuid4())[:8].upper()

    def schedule_session(self):
        start_time = self.start_time_input.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        duration = self.duration_input.value()
        allowed_website = self.website_input.text().strip()
        if not allowed_website:
            QMessageBox.warning(self, "Error", "Allowed website is required for the session.")
            return
        if not (allowed_website.startswith("http://") or allowed_website.startswith("https://")):
            allowed_website = "https://" + allowed_website
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        if start_time < current_time:
            QMessageBox.warning(self, "Error", "Start time cannot be in the past")
            return
        session_code = self.generate_session_code()
        allow_retakes = 1 if self.allow_retakes_checkbox.isChecked() else 0
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("INSERT INTO exam_sessions (session_code, start_time, duration, status, teacher_id, allowed_website, allow_retakes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (session_code, start_time, duration, "Scheduled", self.current_user["id"], allowed_website, allow_retakes))
                conn.commit()
                QMessageBox.information(self, "Success", f"Session scheduled successfully! Session Code: {session_code}")
                self.load_students_and_sessions()
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

    def open_session_now(self):
        start_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        duration = self.duration_input.value()
        allowed_website = self.website_input.text().strip()
        if not allowed_website:
            QMessageBox.warning(self, "Error", "Allowed website is required for the session.")
            return
        if not (allowed_website.startswith("http://") or allowed_website.startswith("https://")):
            allowed_website = "https://" + allowed_website
        session_code = self.generate_session_code()
        allow_retakes = 1 if self.allow_retakes_checkbox.isChecked() else 0
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("INSERT INTO exam_sessions (session_code, start_time, duration, status, teacher_id, allowed_website, allow_retakes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (session_code, start_time, duration, "Active", self.current_user["id"], allowed_website, allow_retakes))
                conn.commit()
                QMessageBox.information(self, "Success", f"Session opened successfully! Session Code: {session_code}")
                self.load_students_and_sessions()
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

    def close_session(self, session_code):
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("UPDATE exam_sessions SET status = 'Closed' WHERE session_code = ?", (session_code,))
                conn.commit()
                self.load_students_and_sessions()
                QMessageBox.information(self, "Success", "Session closed successfully")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

    def load_students_and_sessions(self):
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT id, name, email FROM users WHERE role = 'student'")
                students = c.fetchall()
                self.student_table.setRowCount(len(students))
                email_count = {}
                for _, _, email in students:
                    email_count[email] = email_count.get(email, 0) + 1
                for row, (id, name, email) in enumerate(students):
                    c.execute("SELECT score FROM focus_scores WHERE student_id = ? ORDER BY timestamp DESC LIMIT 1", (id,))
                    score = c.fetchone()
                    score = str(score['score']) if score else "N/A"
                    self.student_table.setItem(row, 0, QTableWidgetItem(str(id)))
                    self.student_table.setItem(row, 1, QTableWidgetItem(name))
                    self.student_table.setItem(row, 2, QTableWidgetItem(email))
                    self.student_table.setItem(row, 3, QTableWidgetItem(score))
                    if email_count[email] > 1:
                        for col in range(4):
                            if self.student_table.item(row, col):
                                self.student_table.item(row, col).setBackground(QColor(255, 204, 204))
                c.execute("SELECT session_code, start_time, duration, status, allowed_website FROM exam_sessions WHERE teacher_id = ? ORDER BY start_time DESC",
                          (self.current_user["id"],))
                sessions = c.fetchall()
                self.session_table.setRowCount(len(sessions))
                current_time = QDateTime.currentDateTime()
                for row, (session_code, start_time, duration, status, allowed_website) in enumerate(sessions):
                    start_dt = QDateTime.fromString(start_time, "yyyy-MM-dd HH:mm:ss")
                    end_dt = start_dt.addSecs(duration * 60)
                    if status == "Scheduled" and current_time >= start_dt:
                        status = "Active"
                        c.execute("UPDATE exam_sessions SET status = 'Active' WHERE session_code = ?", (session_code,))
                    if status == "Active" and current_time > end_dt:
                        status = "Closed"
                        c.execute("UPDATE exam_sessions SET status = 'Closed' WHERE session_code = ?", (session_code,))
                    status_text = status
                    if status == "Scheduled":
                        status_text = f"⏳ {status}"
                    elif status == "Active":
                        status_text = f"▶ {status}"
                    elif status == "Closed":
                        status_text = f"❌ {status}"
                    self.session_table.setItem(row, 0, QTableWidgetItem(session_code))
                    self.session_table.setItem(row, 1, QTableWidgetItem(start_time))
                    self.session_table.setItem(row, 2, QTableWidgetItem(str(duration)))
                    self.session_table.setItem(row, 3, QTableWidgetItem(status_text))
                    self.session_table.setItem(row, 4, QTableWidgetItem(allowed_website or ""))
                    # Highlight row green if session is active
                    if status == "Active":
                        for col in range(6):
                            item = self.session_table.item(row, col)
                            if item:
                                item.setBackground(QColor(46, 204, 113))
                    # Actions column
                    action_layout = QHBoxLayout()
                    action_layout.setContentsMargins(0, 0, 0, 0)
                    action_layout.setSpacing(8)
                    action_widget = QWidget()
                    action_widget.setLayout(action_layout)
                    if status == "Active":
                        close_button = QPushButton("Close ✖️")
                        close_button.setMinimumWidth(100)
                        close_button.setStyleSheet("background: #D32F2F; color: #fff; padding: 5px 10px; font-size: 14px; border-radius: 6px;")
                        close_button.clicked.connect(lambda _, sc=session_code: self.close_session(sc))
                        action_layout.addWidget(close_button)
                    action_layout.addStretch()
                    self.session_table.setCellWidget(row, 5, action_widget)
                conn.commit()
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

    def validate_session(self):
        session_code = self.session_code_input.text().strip().upper()
        if not session_code:
            QMessageBox.warning(self, "Error", "Please enter a session code")
            return
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT start_time, duration, status, allowed_website, allow_retakes FROM exam_sessions WHERE session_code = ?", (session_code,))
                session = c.fetchone()
                if not session:
                    QMessageBox.warning(self, "Error", "Invalid session code")
                    return
                start_time, duration, status, allowed_website, allow_retakes = session
                # Check for previous attempt
                c.execute("SELECT * FROM exam_attempts WHERE student_id = ? AND session_code = ?", (self.current_user["id"], session_code))
                attempt = c.fetchone()
                if attempt and not (allow_retakes or attempt['allowed_retake']):
                    QMessageBox.warning(self, "Error", "You have already attempted this exam. Retakes are not allowed unless permitted by your teacher.")
                    return
                # If not attempted, log attempt
                if not attempt:
                    c.execute("INSERT INTO exam_attempts (student_id, session_code, attempt_time, allowed_retake) VALUES (?, ?, datetime('now'), 0)",
                              (self.current_user["id"], session_code))
                    conn.commit()
                start_dt = QDateTime.fromString(start_time, "yyyy-MM-dd HH:mm:ss")
                end_dt = start_dt.addSecs(duration * 60)
                current_time = QDateTime.currentDateTime()
                if status == "Scheduled" and current_time >= start_dt:
                    status = "Active"
                    c.execute("UPDATE exam_sessions SET status = 'Active' WHERE session_code = ?", (session_code,))
                    conn.commit()
                if status == "Active" and current_time > end_dt:
                    status = "Closed"
                    c.execute("UPDATE exam_sessions SET status = 'Closed' WHERE session_code = ?", (session_code,))
                    conn.commit()
                if status != "Active":
                    QMessageBox.warning(self, "Error", "This session is not active")
                    return
                self.current_session_code = session_code
                self.current_allowed_website = allowed_website
                QMessageBox.information(self, "Success", "Session validated! You can now start the exam.")
                self.session_code_input.setEnabled(False)
                self.validate_session_button.setEnabled(False)
                self.start_button.setEnabled(True)
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

    def calculate_grade(self, score):
        if score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D"
        else:
            return "F"

    def generate_focus_chart(self, focus_scores, timestamps):
        if not focus_scores or not timestamps:
            return None
        start_time = timestamps[0]
        relative_timestamps = [(t - start_time) for t in timestamps]
        smoothed_scores = pd.Series(focus_scores).rolling(window=10, min_periods=1).mean()
        plt.figure(figsize=(8, 4))
        plt.plot(relative_timestamps, smoothed_scores, color="#1976D2", linewidth=2, label="Focus Score")
        plt.axhline(50, color='#D32F2F', linestyle='--', label='Distraction Threshold')
        plt.axhline(80, color='#4CAF50', linestyle='--', label='High Focus')
        plt.title("Focus Score Trend", fontsize=12, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=10)
        plt.ylabel("Focus Score", fontsize=10)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, format="png", bbox_inches="tight")
        plt.close()
        return temp_file.name

    def view_student_scores(self, row, col):
        if col == 4:
            return
        student_id_text = self.student_table.item(row, 0).text()
        try:
            student_id = int(student_id_text)
        except ValueError:
            QMessageBox.warning(self, "Error", f"Invalid student ID: {student_id_text}")
            return
        student_name = self.student_table.item(row, 1).text()
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT score, timestamp, details, session_code FROM focus_scores WHERE student_id = ? ORDER BY timestamp DESC",
                          (student_id,))
                scores = c.fetchall()
                if not scores:
                    QMessageBox.information(self, "Info", "No focus scores available for this student")
                    return
                focus_scores = [score for score, _, _, _ in scores]
                timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() for _, ts, _, _ in scores]
                average_score = sum(focus_scores) / len(focus_scores) if focus_scores else 0
                grade = self.calculate_grade(average_score)
                chart_path = self.generate_focus_chart(focus_scores, timestamps)
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", f"focus_report_student_{student_id}.pdf", "PDF Files (*.pdf)")
                if not file_path:
                    if chart_path:
                        os.remove(chart_path)
                    return
                c = canvas.Canvas(file_path, pagesize=letter)
                width, height = letter
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "FocusScore - Student Report")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 70, f"Student: {student_name} (ID: {student_id})")
                c.drawString(50, height - 90, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, height - 120, "Summary")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 140, f"Average Focus Score: {average_score:.1f}")
                c.drawString(50, height - 160, f"Grade: {grade}")
                c.drawString(50, height - 180, f"Total Sessions: {len(set([s[3] for s in scores]))}")
                if chart_path:
                    c.drawString(50, height - 210, "Focus Score Trend:")
                    img = ImageReader(chart_path)
                    c.drawImage(img, 50, height - 410, width=400, height=200)
                    os.remove(chart_path)
                c.setFont("Helvetica-Bold", 14)
                y = height - 440
                c.drawString(50, y, "Detailed Scores")
                y -= 20
                c.setFont("Helvetica", 10)
                for score, timestamp, details, session_code in scores:
                    c.drawString(50, y, f"Score: {score:.1f} | Time: {timestamp} | Details: {details} | Session: {session_code}")
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50
                        c.setFont("Helvetica", 10)
                c.save()
                QMessageBox.information(self, "Success", f"Report saved to: {file_path}")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")

    def download_student_report(self):
        if not self.current_user or self.current_user["role"] != "student":
            QMessageBox.warning(self, "Error", "Login as a student to download report")
            return
        if not self.current_session_code:
            QMessageBox.warning(self, "Error", "No active session to generate report for")
            return
        student_id = self.current_user["id"]
        student_name = self.current_user["name"]
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT score, timestamp, details FROM focus_scores WHERE student_id = ? AND session_code = ? ORDER BY timestamp",
                          (student_id, self.current_session_code))
                scores_data = c.fetchall()
                if not scores_data:
                    QMessageBox.information(self, "Info", "No focus scores available for this session")
                    return
                c.execute("SELECT start_time, duration FROM exam_sessions WHERE session_code = ?",
                          (self.current_session_code,))
                session = c.fetchone()
                if not session:
                    QMessageBox.warning(self, "Error", "Session data not found")
                    return
                start_time, duration = session
                start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                focus_scores = [score for score, _, _ in scores_data]
                timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() for _, ts, _ in scores_data]
                average_score = sum(focus_scores) / len(focus_scores) if focus_scores else 0
                grade = self.calculate_grade(average_score)
                final_score = self.session_scores[-1] if self.session_scores else average_score
                final_grade = self.calculate_grade(final_score)
                distractions = 0
                phone_alerts = self.phone_alerts
                head_turn_events = self.head_turn_events
                for _, _, details in scores_data:
                    if "Non-exam window active" in details:
                        distractions += 1
                total_distractions = distractions + phone_alerts + head_turn_events
                chart_path = self.generate_focus_chart(focus_scores, timestamps)
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", f"study_session_report_{self.current_session_code}.pdf", "PDF Files (*.pdf)")
                if not file_path:
                    if chart_path:
                        os.remove(chart_path)
                    return
                c = canvas.Canvas(file_path, pagesize=letter)
                width, height = letter
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "FocusScore - Session Report")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 70, f"Student: {student_name} (ID: {student_id})")
                c.drawString(50, height - 90, f"Session Code: {self.current_session_code}")
                c.drawString(50, height - 110, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, height - 140, "Summary")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 160, f"Session Duration: {duration} minutes")
                c.drawString(50, height - 180, f"Average Focus Score: {average_score:.1f}")
                c.drawString(50, height - 200, f"Average Grade: {grade}")
                c.drawString(50, height - 220, f"Final Focus Score: {final_score:.1f}")
                c.drawString(50, height - 240, f"Final Grade: {final_grade}")
                c.drawString(50, height - 260, f"Total Distractions: {total_distractions}")
                c.drawString(50, height - 280, f"Phone Alerts: {phone_alerts}")
                c.drawString(50, height - 300, f"Head Turn Events: {head_turn_events}")
                if chart_path:
                    c.drawString(50, height - 330, "Focus Score Trend:")
                    img = ImageReader(chart_path)
                    c.drawImage(img, 50, height - 530, width=400, height=200)
                    os.remove(chart_path)
                c.save()
                QMessageBox.information(self, "Success", f"Report saved to: {file_path}")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")

    def logout(self):
        print("[DEBUG] logout called")
        from PyQt5.QtCore import QTimer
        # If student is monitoring, stop monitoring first (defer to event loop)
        if self.current_user and self.current_user.get("role") == "student" and self.monitoring:
            print("[DEBUG] logout: scheduling stop_monitoring via QTimer.singleShot...")
            QTimer.singleShot(0, self.stop_monitoring)
        self.current_user = None
        self.current_session_code = None
        self.session_code_input.setEnabled(True)
        self.validate_session_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.session_code_input.clear()
        # Hide all student-related tabs after a short delay to ensure UI remains responsive
        def do_tab_reset():
            print("[DEBUG] logout: hiding all student/teacher tabs and returning to welcome tab...")
            self.tabs.setTabVisible(self.student_tab_index, False)
            self.tabs.setTabVisible(self.exam_tab_index, False)
            self.tabs.setTabVisible(self.website_tab_index, False)
            self.tabs.setTabVisible(self.teacher_tab_index, False)
            self.tabs.setCurrentIndex(self.role_selection_tab_index)
            print("[DEBUG] logout finished, all student tabs hidden and returned to role selection")
        QTimer.singleShot(0, do_tab_reset)

    def get_focus_score(self, results, w, h, phone_detected):
        if not hasattr(self, 'offscreen_counter'):
            self.offscreen_counter = 0
        if not results.multi_face_landmarks:
            self.offscreen_counter = 0
            return 0, "No face detected"
        if len(results.multi_face_landmarks) > 1:
            self.offscreen_counter = 0
            return 0, "Multiple faces detected"
        face = results.multi_face_landmarks[0]
        landmarks = face.landmark
        def get_point(idx):
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)
        eye_top = get_point(159)
        eye_bottom = get_point(145)
        eye_left = get_point(33)
        eye_right = get_point(133)
        iris_center = get_point(468)
        nose_tip = get_point(1)
        left_temple = get_point(234)
        right_temple = get_point(454)
        chin = get_point(152)
        eye_level = get_point(151)
        eye_aspect_ratio = eye_openness(eye_top, eye_bottom, eye_left, eye_right)
        iris_horizontal, iris_vertical = iris_position_ratio(iris_center, eye_left, eye_right, eye_top, eye_bottom)
        head_tilt_value = head_tilt_ratio(left_temple, right_temple, nose_tip)
        head_down_value = head_down_ratio(nose_tip, chin, eye_level)
        focus = 100.0
        status = "Focused"
        debug_info = []
        head_turn_detected = False
        # --- Penalties ---
        if eye_aspect_ratio < 0.2:
            self.blink_counter += 1
            if self.blink_counter >= 3:
                self.offscreen_counter = 0
                focus = 0
                status = "Eyes Closed"
                debug_info.append("Penalty: Eyes closed (-100)")
                return focus, status + " | " + ", ".join(debug_info)
        else:
            self.blink_counter = 0
        debug_info.append(f"EAR: {eye_aspect_ratio:.2f}")
        # --- Gaze direction penalty (stricter, cumulative for offscreen) ---
        gaze_penalty = 0
        gaze_threshold = 0.04
        center = 0.5
        horizontal_deviation = abs(iris_horizontal - center)
        vertical_deviation = abs(iris_vertical - center)
        # Hard penalty for eyes off screen
        if iris_horizontal < 0.2 or iris_horizontal > 0.8 or iris_vertical < 0.2 or iris_vertical > 0.8:
            self.offscreen_counter += 1
            focus = max(0, 100 - 10 * self.offscreen_counter)
            status = "Eyes Off Screen"
            debug_info.append(f"Penalty: Eyes off screen (-{10 * self.offscreen_counter})")
        else:
            if self.offscreen_counter > 0:
                debug_info.append(f"Eyes returned to screen after {self.offscreen_counter} frames off screen.")
            self.offscreen_counter = 0
            if horizontal_deviation > gaze_threshold:
                gaze_penalty += min(50, 150 * (horizontal_deviation - gaze_threshold))
            if vertical_deviation > gaze_threshold:
                gaze_penalty += min(30, 100 * (vertical_deviation - gaze_threshold))
            if gaze_penalty > 0:
                focus -= gaze_penalty
                debug_info.append(f"Penalty: Gaze deviation (-{gaze_penalty:.1f})")
        debug_info.append(f"Iris H: {iris_horizontal:.2f}, V: {iris_vertical:.2f}")
        # --- Mild head turn/tilt penalty (proportional) ---
        mild_head_penalty = 0
        if 1.2 < head_tilt_value < 1.8:
            mild_head_penalty += (head_tilt_value - 1.2) * 10
        elif 0.2 < head_tilt_value < 0.8:
            mild_head_penalty += (0.8 - head_tilt_value) * 10
        if 0.75 < head_down_value < 1.3:
            mild_head_penalty += (head_down_value - 0.75) * 10
        if mild_head_penalty > 0:
            focus -= mild_head_penalty
            debug_info.append(f"Penalty: Mild head pose (-{mild_head_penalty:.1f})")
        # --- Major head turn/tilt penalty ---
        if head_tilt_value >= 1.8:
            penalty = (head_tilt_value - 1.8) * 50
            focus -= penalty
            head_turn_detected = True
            debug_info.append(f"Penalty: Major head tilt (-{penalty:.1f})")
        elif head_tilt_value <= 0.2:
            penalty = (0.2 - head_tilt_value) * 50
            focus -= penalty
            head_turn_detected = True
            debug_info.append(f"Penalty: Major head tilt (-{penalty:.1f})")
        if head_down_value >= 1.3:
            penalty = (head_down_value - 1.3) * 50
            focus -= penalty
            head_turn_detected = True
            debug_info.append(f"Penalty: Major head down (-{penalty:.1f})")
        elif head_down_value <= 0.75:
            penalty = (0.75 - head_down_value) * 50
            focus -= penalty
            head_turn_detected = True
            debug_info.append(f"Penalty: Major head up (-{penalty:.1f})")
        if phone_detected:
            self.offscreen_counter = 0
            focus = 0
            status = "Phone Detected"
            debug_info.append("Penalty: Phone detected (-100)")
            return max(0, focus), status + " | " + ", ".join(debug_info)
        if head_turn_detected:
            status = "Head Turn Detected"
        # If only minor distractions, don't drop below 70
        if focus < 70 and not head_turn_detected and not phone_detected and status != "Eyes Off Screen":
            focus = 70
        if focus < 0:
            focus = 0
        if focus > 100:
            focus = 100
        return focus, status + " | " + ", ".join(debug_info)

    def start_monitoring(self):
        if not self.current_user or self.current_user["role"] != "student":
            QMessageBox.warning(self, "Error", "Login as a student to start exam")
            return
        if not self.current_session_code:
            QMessageBox.warning(self, "Error", "Validate a session code first")
            return
        # Show and switch to Exam and Website tabs
        self.tabs.setTabVisible(self.exam_tab_index, True)
        self.tabs.setTabVisible(self.website_tab_index, True)
        self.tabs.setCurrentIndex(self.exam_tab_index)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot access webcam. Please check your camera device.")
            return
        self.monitoring = True
        self.focus_score = 100.0
        self.session_scores = []
        self.session_timestamps = []
        self.session_start_time = time.time()
        self.time_on_screen = 0
        self.total_time = 0
        self.last_screenshot_time = 0
        self.phone_detected = False
        self.gaze_counter = 0
        self.cheating_detected = False
        self.blink_counter = 0
        self.phone_alerts = 0
        self.head_turn_events = 0
        # Only update label text, do not recreate labels
        self.score_label.setText("Focus Score: 100.0")
        self.phone_label.setText("Phone Status: None")
        self.final_score_label.setText("Final Focus Score: N/A")
        self.final_score_label.setVisible(False)
        self.cheating_label.setText("Cheating Warning: None")
        self.cheating_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        self.debug_label.setText("Debug: N/A")
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        allowed_website = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT allowed_website FROM exam_sessions WHERE session_code = ?", (self.current_session_code,))
                row = c.fetchone()
                if row:
                    allowed_website = row[0]
        except Exception as e:
            allowed_website = None
        if allowed_website:
            if not (allowed_website.startswith("http://") or allowed_website.startswith("https://")):
                allowed_website = "https://" + allowed_website
            print(f"[DEBUG] Loading allowed website: {allowed_website}")
            self.website_tab_web_view.setVisible(True)
            self.website_tab_web_view.load(QUrl(allowed_website))
            self.website_tab_web_view.show()
            self.website_tab_web_view.raise_()
            print(f"[DEBUG] Student browser loading: {allowed_website}")
        else:
            print("[DEBUG] No allowed website set for this session.")
            self.website_tab_web_view.setVisible(False)
        self.monitoring_thread = MonitorThread(self.cap, self.current_user, self.current_session_code, self.session_start_time, self)
        self.monitoring_thread.update_signal.connect(self.update_ui_from_thread)
        self.monitoring_thread.error_signal.connect(self.handle_monitoring_error)
        self.monitoring_thread.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(50)  # Slower update rate to prevent freezing
        # --- Lockdown logic for FocusScoreApp ---
        self.enable_kiosk_mode()
        self.install_keyboard_hook()
        self.start_watchdog()
        # --- Reset and show website tab ---
        if self.website_tab_web_view:
            self.website_tab_web_view.setVisible(True)
            self.tabs.setTabVisible(self.website_tab_index, True)
            self.website_tab_web_view.load(QUrl("about:blank"))  # Clear first
            # Now load the allowed website
            allowed_website = self.current_allowed_website or None
            if not allowed_website:
                # Try to get from DB if not set
                try:
                    with get_db_connection() as conn:
                        c = conn.cursor()
                        c.execute("SELECT allowed_website FROM exam_sessions WHERE session_code = ?", (self.current_session_code,))
                        row = c.fetchone()
                        if row:
                            allowed_website = row[0]
                except Exception:
                    allowed_website = None
            if allowed_website:
                if not (allowed_website.startswith("http://") or allowed_website.startswith("https://")):
                    allowed_website = "https://" + allowed_website
                self.website_tab_web_view.load(QUrl(allowed_website))
        # ... rest of your code ...

    def update_ui_from_thread(self, focus_score, status, debug_info, phone_detected):
        print(f"[DEBUG] update_ui_from_thread: focus_score={focus_score}, status={status}, phone_detected={phone_detected}")
        # --- Smoothing: average last 5 scores ---
        self.session_scores.append(focus_score)
        if len(self.session_scores) > 5:
            smoothed_score = sum(self.session_scores[-5:]) / 5
        else:
            smoothed_score = sum(self.session_scores) / len(self.session_scores)
        self.focus_score = smoothed_score
        self.phone_detected = phone_detected
        self.session_timestamps.append(time.time())
        self.score_label.setText(f"Focus Score: {self.focus_score:.1f}")
        self.phone_label.setText("Phone Status: " + ("Phone Detected" if phone_detected else "None"))
        self.debug_label.setText(f"Debug: {debug_info}")
        if phone_detected or status != "Focused":
            self.cheating_detected = True
            self.cheating_label.setText(f"Cheating Warning: {status}")
            self.cheating_label.setStyleSheet("color: #D32F2F; font-weight: bold; font-size: 14px;")
        elif self.cheating_detected:
            self.cheating_label.setText("Cheating Warning: Suspicious Activity")
            self.cheating_label.setStyleSheet("color: #D32F2F; font-weight: bold; font-size: 14px;")
        else:
            self.cheating_label.setText("Cheating Warning: None")
            self.cheating_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        if "Head Turn Detected" in status:
            self.head_turn_events += 1

    def handle_monitoring_error(self, error_message):
        print(f"[ERROR] Monitoring thread error: {error_message}")
        self.stop_monitoring()
        QMessageBox.critical(self, "Error", f"Monitoring error: {error_message}")

    def stop_monitoring(self):
        print("[DEBUG] stop_monitoring called")
        if getattr(self, '_already_stopped', False):
            print("[DEBUG] stop_monitoring: already stopped, returning early")
            return
        self._already_stopped = True
        try:
            self.monitoring = False
            # Stop and delete QTimer
            if hasattr(self, "timer") and self.timer:
                try:
                    self.timer.stop()
                    self.timer.deleteLater()
                    print("[DEBUG] QTimer stopped and deleted")
                except Exception as e:
                    print(f"[ERROR] Error stopping/deleting QTimer: {e}")
                self.timer = None
            # Release webcam BEFORE waiting for thread
            if self.cap:
                try:
                    self.cap.release()
                    print("[DEBUG] Webcam released (before thread join)")
                except Exception as e:
                    print(f"[ERROR] Error releasing webcam: {e}")
                self.cap = None
            # Stop and wait for monitoring thread
            if self.monitoring_thread:
                try:
                    self.monitoring_thread.update_signal.disconnect()
                except Exception:
                    pass
                try:
                    self.monitoring_thread.error_signal.disconnect()
                except Exception:
                    pass
                try:
                    self.monitoring_thread.stop()
                    print("[DEBUG] Called stop() on monitoring thread")
                    finished = self.monitoring_thread.wait(3000)
                    if not finished:
                        print("[ERROR] Monitoring thread did not finish in time, terminating...")
                        try:
                            self.monitoring_thread.terminate()
                            self.monitoring_thread.wait(1000)
                            print("[DEBUG] Monitoring thread forcibly terminated")
                        except Exception as e:
                            print(f"[ERROR] Could not forcibly terminate monitoring thread: {e}")
                    self.monitoring_thread.deleteLater()
                    print("[DEBUG] Monitoring thread stopped and deleted")
                except Exception as e:
                    print(f"[ERROR] Error stopping/deleting monitoring thread: {e}")
                self.monitoring_thread = None
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[ERROR] Error destroying OpenCV windows: {e}")
            # Reset UI
            self.video_label.setText("Webcam Feed")
            self.score_label.setText("Focus Score: 100.0")
            self.phone_label.setText("Phone Status: None")
            self.cheating_label.setText("Cheating Warning: None")
            self.cheating_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
            self.debug_label.setText("Debug: N/A")
            if self.session_scores:
                final_score = self.session_scores[-1]
                self.final_score_label.setText(f"Final Focus Score: {final_score:.1f} (Grade: {self.calculate_grade(final_score)})")
                self.final_score_label.setVisible(True)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            if self.website_tab_web_view:
                self.website_tab_web_view.load(QUrl("about:blank"))
                self.website_tab_web_view.setVisible(False)
            self.tabs.setTabVisible(self.website_tab_index, False)
            # Do NOT hide the exam tab here; keep it visible for results/download
            self.disable_kiosk_mode()
            self.uninstall_keyboard_hook()
            self.stop_watchdog()
            print("[DEBUG] stop_monitoring finished, browser tab hidden, exam/results tab still visible")
            # Auto-save report for student
            if self.current_user and self.current_user.get("role") == "student" and self.current_session_code:
                try:
                    self.auto_save_student_report()
                except Exception as e:
                    print(f"[ERROR] Auto-save report failed: {e}")
        except Exception as e:
            print(f"[ERROR] Exception in stop_monitoring: {e}")
        finally:
            self._already_stopped = False

    def auto_save_student_report(self):
        # Save a PDF report for the current session in screenshots/<student_name>/focus_report_<session_code>_<timestamp>.pdf
        student_id = self.current_user["id"]
        student_name = self.current_user["name"]
        session_code = self.current_session_code
        if not session_code:
            return
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT score, timestamp, details FROM focus_scores WHERE student_id = ? AND session_code = ? ORDER BY timestamp",
                          (student_id, session_code))
                scores_data = c.fetchall()
                if not scores_data:
                    print("[DEBUG] No focus scores available for this session, skipping report.")
                    return
                c.execute("SELECT start_time, duration FROM exam_sessions WHERE session_code = ?",
                          (session_code,))
                session = c.fetchone()
                if not session:
                    print("[DEBUG] Session data not found, skipping report.")
                    return
                start_time, duration = session
                start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                focus_scores = [score for score, _, _ in scores_data]
                timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() for _, ts, _ in scores_data]
                average_score = sum(focus_scores) / len(focus_scores) if focus_scores else 0
                grade = self.calculate_grade(average_score)
                final_score = self.session_scores[-1] if self.session_scores else average_score
                final_grade = self.calculate_grade(final_score)
                distractions = 0
                phone_alerts = self.phone_alerts
                head_turn_events = self.head_turn_events
                for _, _, details in scores_data:
                    if "Non-exam window active" in details:
                        distractions += 1
                total_distractions = distractions + phone_alerts + head_turn_events
                chart_path = self.generate_focus_chart(focus_scores, timestamps)
                # --- Save to screenshots/<student_name>/focus_report_<session_code>_<timestamp>.pdf ---
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', student_name.lower())
                folder = os.path.join("screenshots", safe_name)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(folder, f"focus_report_{session_code}_{timestamp_str}.pdf")
                c = canvas.Canvas(file_path, pagesize=letter)
                width, height = letter
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "FocusScore - Session Report")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 70, f"Student: {student_name} (ID: {student_id})")
                c.drawString(50, height - 90, f"Session Code: {session_code}")
                c.drawString(50, height - 110, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, height - 140, "Summary")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 160, f"Session Duration: {duration} minutes")
                c.drawString(50, height - 180, f"Average Focus Score: {average_score:.1f}")
                c.drawString(50, height - 200, f"Average Grade: {grade}")
                c.drawString(50, height - 220, f"Final Focus Score: {final_score:.1f}")
                c.drawString(50, height - 240, f"Final Grade: {final_grade}")
                c.drawString(50, height - 260, f"Total Distractions: {total_distractions}")
                c.drawString(50, height - 280, f"Phone Alerts: {phone_alerts}")
                c.drawString(50, height - 300, f"Head Turn Events: {head_turn_events}")
                if chart_path:
                    c.drawString(50, height - 330, "Focus Score Trend:")
                    img = ImageReader(chart_path)
                    c.drawImage(img, 50, height - 530, width=400, height=200)
                    os.remove(chart_path)
                c.save()
                print(f"[DEBUG] Auto-saved report to: {file_path}")
        except Exception as e:
            print(f"[ERROR] Exception in auto_save_student_report: {e}")

    def update_video(self):
        if not self.monitoring or not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.handle_monitoring_error("Webcam frame not available. Please check your camera device.")
            return
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = face_mesh.process(rgb_frame)
        except Exception as e:
            self.handle_monitoring_error(f"MediaPipe error: {e}")
            return
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(rgb_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
        h, w, _ = rgb_frame.shape
        bytes_per_line = 3 * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_student_email_changed(self, text):
        self.student_email_text = text

    def on_student_password_changed(self, text):
        self.student_password_text = text

    def on_teacher_email_changed(self, text):
        self.teacher_email_text = text

    def on_teacher_password_changed(self, text):
        self.teacher_password_text = text

    def toggle_student_password_visibility(self, state):
        self.student_password_input.setEchoMode(QLineEdit.Normal if state else QLineEdit.Password)

    def toggle_teacher_password_visibility(self, state):
        self.teacher_password_input.setEchoMode(QLineEdit.Normal if state else QLineEdit.Password)

    def closeEvent(self, event):
        self.stop_monitoring()
        face_mesh.close()
        event.accept()

    def view_student_screenshots(self, student_id, student_name):
        # Placeholder for screenshot dialog logic
        QMessageBox.information(self, "Screenshots", f"Show screenshots for {student_name} (ID: {student_id}) here.")

    def filter_student_table(self, text):
        text = text.strip().lower()
        for row in range(self.student_table.rowCount()):
            name_item = self.student_table.item(row, 1)
            email_item = self.student_table.item(row, 2)
            name = name_item.text().lower() if name_item else ""
            email = email_item.text().lower() if email_item else ""
            match = text in name or text in email
            self.student_table.setRowHidden(row, not match)

    def set_allowed_website(self):
        url = self.website_input.text().strip()
        if url:
            self.allowed_website = url
            if url not in self.allowed_websites:
                self.allowed_websites.append(url)
                self.website_list_widget.addItem(url)
            self.teacher_status_bar.showMessage(f"Allowed website set: {url}")
            QMessageBox.information(self, "Allowed Website", f"Allowed website set to: {url}")
        else:
            self.allowed_website = None
            self.teacher_status_bar.showMessage("Allowed website cleared.")
            QMessageBox.information(self, "Allowed Website", "Allowed website cleared.")

    def remove_selected_website(self):
        selected_items = self.website_list_widget.selectedItems()
        for item in selected_items:
            url = item.text()
            self.allowed_websites.remove(url)
            self.website_list_widget.takeItem(self.website_list_widget.row(item))
        if self.allowed_website in [item.text() for item in selected_items]:
            self.allowed_website = self.allowed_websites[-1] if self.allowed_websites else None

    def setup_student_ui(self):
        layout = QVBoxLayout()
        self.student_widget.setLayout(layout)
        title_label = QLabel("Exam Portal")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.session_code_input = QLineEdit()
        self.session_code_input.setPlaceholderText("Enter Session Code")
        self.validate_session_button = QPushButton("Validate")
        self.validate_session_button.clicked.connect(self.validate_session)
        input_layout = QHBoxLayout()
        input_layout.addStretch()
        input_layout.addWidget(self.session_code_input)
        input_layout.addWidget(self.validate_session_button)
        input_layout.addStretch()
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addLayout(input_layout)
        layout.addSpacing(20)
        # Only Start Exam button here
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.start_button = QPushButton("Start Exam")
        self.start_button.clicked.connect(self.start_monitoring)
        button_layout.addWidget(self.start_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addStretch()

    def setup_exam_ui(self):
        layout = QVBoxLayout()
        self.exam_widget.setLayout(layout)
        # Video label
        self.video_label = QLabel("Webcam Feed")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid #616161; border-radius: 4px; background: #424242;")
        video_layout = QHBoxLayout()
        video_layout.addStretch()
        video_layout.addWidget(self.video_label)
        video_layout.addStretch()
        # Remove web_label and web_view from Exam tab
        # Labels
        self.score_label = QLabel("Focus Score: 100.0")
        self.phone_label = QLabel("Phone Status: None")
        self.final_score_label = QLabel("Final Focus Score: N/A")
        self.final_score_label.setVisible(False)
        self.cheating_label = QLabel("Cheating Warning: None")
        self.cheating_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        self.debug_label = QLabel("Debug: N/A")
        layout.addLayout(video_layout)
        layout.addWidget(self.score_label)
        layout.addWidget(self.phone_label)
        layout.addWidget(self.cheating_label)
        layout.addWidget(self.debug_label)
        layout.addWidget(self.final_score_label)
        layout.addWidget(self.progress_bar)
        # Stop/Download/Logout buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.stop_button = QPushButton("Stop Exam")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        download_button = QPushButton("Download Report")
        download_button.clicked.connect(self.download_student_report)
        logout_button = QPushButton("Logout")
        logout_button.clicked.connect(self.logout)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(download_button)
        button_layout.addWidget(logout_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addStretch()

    def setup_website_tab_ui(self):
        layout = QVBoxLayout()
        self.website_tab.setLayout(layout)
        self.website_tab_web_view = QWebEngineView()
        self.website_tab_web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.website_tab_web_view.setMinimumSize(800, 600)
        layout.addWidget(self.website_tab_web_view)

    def toggle_theme(self, checked):
        if checked:
            self.apply_styles()  # Dark theme
        else:
            self.setStyleSheet("")  # Light/default theme

    # --- Lockdown logic for FocusScoreApp ---
    def enable_kiosk_mode(self):
        self._old_flags = self.windowFlags()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.showFullScreen()

    def disable_kiosk_mode(self):
        if hasattr(self, '_old_flags'):
            self.setWindowFlags(self._old_flags)
        self.showNormal()

    def install_keyboard_hook(self):
        # Windows only
        WH_KEYBOARD_LL = 13
        WM_KEYDOWN = 0x0100
        VK_TAB = 0x09
        VK_MENU = 0x12  # Alt
        VK_LWIN = 0x5B
        VK_RWIN = 0x5C
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        KBDLLHOOKSTRUCT = wintypes.LPARAM
        self._keyboard_hook = None
        @ctypes.WINFUNCTYPE(wintypes.LRESULT, wintypes.INT, wintypes.WPARAM, wintypes.LPARAM)
        def low_level_keyboard_proc(nCode, wParam, lParam):
            if nCode == 0:
                kbd = ctypes.cast(lParam, ctypes.POINTER(ctypes.c_ulong * 6)).contents
                vk_code = kbd[0]
                if wParam == WM_KEYDOWN and vk_code in (VK_TAB, VK_MENU, VK_LWIN, VK_RWIN):
                    return 1
            return user32.CallNextHookEx(None, nCode, wParam, lParam)
        self._keyboard_hook_proc = low_level_keyboard_proc
        self._user32 = user32
        self._keyboard_hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, low_level_keyboard_proc, None, 0)

    def uninstall_keyboard_hook(self):
        if hasattr(self, '_keyboard_hook') and self._keyboard_hook:
            self._user32.UnhookWindowsHookEx(self._keyboard_hook)
            self._keyboard_hook = None

    def start_watchdog(self):
        self._watchdog_running = True
        def watchdog():
            forbidden = {'zoom.exe','teams.exe','skype.exe','anydesk.exe'}
            while self._watchdog_running:
                for p in psutil.process_iter(['name','pid']):
                    try:
                        if p.info['name'] and p.info['name'].lower() in forbidden:
                            p.kill()
                            print(f"[WATCHDOG] Killed forbidden process: {p.info['name']}")
                    except Exception:
                        pass
                time.sleep(2)
        self._watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self):
        self._watchdog_running = False

    def allow_student_retake(self, student_id):
        # Allow retake for the current session for this student
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO exam_attempts (student_id, session_code, attempt_time, allowed_retake) VALUES (?, ?, datetime('now'), 1)",
                          (student_id, self.current_session_code))
                conn.commit()
                QMessageBox.information(self, "Retake Allowed", "Student can now retake the exam for this session.")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FocusScoreApp()
    window.show()
    sys.exit(app.exec_())