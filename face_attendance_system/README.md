# 🎯 Face Recognition Attendance System

A professional, production-grade Face Recognition Attendance System built with Flask, OpenCV, and the face_recognition library. Automatically detects and recognizes employees via facial recognition to mark attendance — with liveness detection, email notifications, dark mode, and comprehensive admin controls.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Setup](#-environment-setup)
- [Database Setup](#-database-setup)
- [How to Run](#-how-to-run)
- [Usage Guide](#-usage-guide)
- [Admin Functionality](#-admin-functionality)
- [API Documentation](#-api-documentation)
- [Performance](#-performance)
- [Security](#-security)
- [Limitations](#-limitations)
- [Privacy Considerations](#-privacy-considerations)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## ✨ Features

### 🔐 Core Features
| Feature | Description |
|---------|-------------|
| **User Registration** | Register employees with name, ID, email, phone, department, role |
| **Face Detection** | HOG/CNN-based face detection with OpenCV |
| **Face Recognition** | 128-dimensional face embeddings with dlib |
| **Automatic Attendance** | One-click face scan marks check-in automatically |
| **Attendance Records** | Full history with date range filtering |
| **Camera Integration** | Live webcam feed with real-time recognition |
| **Duplicate Prevention** | Configurable time window prevents re-check-in |

### 📊 Dashboard & Reports
| Feature | Description |
|---------|-------------|
| **Real-time Dashboard** | Live stats: present, absent, late, attendance rate |
| **Quick Actions** | One-click access to register, enroll, scan, view records |
| **Daily Reports** | Chart.js bar + doughnut charts with 7-day trends |
| **CSV Export** | Download filtered attendance as CSV |
| **Excel Export** | Styled .xlsx with color-coded status columns |
| **Advanced Filtering** | Filter by employee, status, department, date range |
| **Search** | Search by name, employee ID, or department |
| **Sortable Columns** | Click headers to sort any table |

### 🛡️ Security & Anti-Spoofing
| Feature | Description |
|---------|-------------|
| **Liveness Detection** | 5-check anti-spoofing: texture, brightness, edges, color, variance |
| **Password Hashing** | Werkzeug pbkdf2:sha256 — never plain text |
| **Session Management** | Secure sessions with regeneration on login |
| **Role-Based Access** | Admin-only user management, employee attendance |
| **Rate Limiting** | Login: 10/min, API: 100/min, Uploads: 20/min |
| **Security Headers** | X-Content-Type-Options, X-Frame-Options, X-XSS-Protection |
| **Input Validation** | Server-side validation for all user inputs |

### 🎨 UI/UX
| Feature | Description |
|---------|-------------|
| **Dark Mode** | Full dark theme with toggle, persistent via localStorage |
| **Responsive Design** | Works on desktop, laptop, tablet, and mobile |
| **Loading States** | Overlay spinner + button loading animations |
| **Empty States** | Friendly messages when no data exists |
| **Confirmation Modals** | Safety dialogs before destructive actions |
| **Flash Notifications** | Auto-dismissing alerts for user feedback |
| **Table Sorting** | Client-side sortable columns |
| **Accessibility** | Focus-visible outlines, semantic HTML, ARIA labels |

### 🔔 Notifications
| Feature | Description |
|---------|-------------|
| **Attendance Confirmation** | Email on successful check-in |
| **Late Arrival Alert** | Alert to user + admins when checked in late |
| **Daily Summary** | Automated daily attendance report to admins |
| **Admin Notifications** | System event alerts to configured admin emails |

### ⚡ Performance & Monitoring
| Feature | Description |
|---------|-------------|
| **Performance Monitor** | FPS, detection time, recognition time, DB latency |
| **Frame Skipping** | Configurable interval for live stream processing |
| **Health Status** | Excellent / Good / Fair / Poor auto-classification |
| **System Status Page** | Full dashboard of system health and configuration |
| **Recognition Cooldown** | 60s cooldown prevents rapid re-recognition |
| **User Name Cache** | DB queries minimized during live recognition |

### 🧠 Decision Engine (PHASE X01)
| Feature | Description |
|---------|-------------|
| **Multi-Signal Fusion** | Combines similarity, liveness, quality, margin, and detection signals |
| **Weighted Scoring** | Configurable weights for each signal (similarity 35%, liveness 25%, etc.) |
| **Uncertainty Classification** | HIGH_CONFIDENCE / LOW_CONFIDENCE / UNCERTAIN states |
| **Hard Constraints** | Minimum thresholds for quality, similarity, liveness, margin |
| **Signal Conflict Detection** | Detects contradictory signals (high similarity + low liveness) |
| **Decision Types** | ACCEPT / REJECT / REVIEW with full explanations |
| **Candidate Margin Analysis** | Measures gap between top and second-best match |
| **Margin Signal** | Normalized margin score fed into decision engine |
| **Real-time Evaluation** | API endpoint for testing decisions with custom signals |

### 📜 Attendance Policy Engine
| Feature | Description |
|---------|-------------|
| **Present/Late Classification** | Configurable late-after time with grace period |
| **Check-in Window** | Allowed hours for check-in (e.g., 06:00–23:59) |
| **Duplicate Suppression** | Time-window based duplicate prevention |
| **Minimum Checkout Duration** | Prevent accidental immediate checkout |
| **Absent Identification** | Automatic absent user detection |
| **Centralized Rules** | All rules in `AttendancePolicy` service, not hardcoded |

---

## 🏗️ Architecture

```
                    ┌──────────┐
                    │   USER   │
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │   RESPONSIVE UI     │
              │  (Bootstrap 5 + JS) │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   FLASK (4 BP)      │
              │  main|user|att|api  │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
   ┌─────▼─────┐  ┌──────▼──────┐  ┌────▼────┐
   │User Service│  │Attendance   │  │ Camera  │
   │            │  │  Service    │  │ Service │
   └─────┬─────┘  └──────┬──────┘  └────┬────┘
         │               │               │
         │        ┌──────▼──────┐        │
         │        │ Attendance  │        │
         │        │  Policy     │        │
         │        └──────┬──────┘        │
         │               │               │
   ┌─────▼─────┐  ┌──────▼──────┐  ┌────▼────────┐
   │ Security  │  │  Database   │  │ Recognition │
   │ Middleware │  │  (SQLite)   │  │  Pipeline   │
   └───────────┘  └─────────────┘  └──────┬──────┘
                                          │
                              ┌────────────┼────────────┐
                              │            │            │
                        ┌─────▼────┐ ┌─────▼────┐ ┌────▼────┐
                        │ Detection│ │ Liveness │ │Matching │
                        │          │ │ (Anti-   │ │ Engine  │
                        │          │ │ Spoof)   │ │         │
                        └──────────┘ └──────────┘ └────┬────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │   DECISION ENGINE     │
                                           │  (Multi-Signal Fusion)│
                                           │                       │
                                           │  ┌─────────────────┐  │
                                           │  │ Candidate Margin │  │
                                           │  │   Analyzer       │  │
                                           │  └─────────────────┘  │
                                           │  ┌─────────────────┐  │
                                           │  │  Weighted Score  │  │
                                           │  │  + Constraints   │  │
                                           │  └─────────────────┘  │
                                           │  ACCEPT/REJECT/REVIEW │
                                           └───────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.8+, Flask 3.0 |
| **Database** | SQLite + SQLAlchemy ORM |
| **Face Detection** | OpenCV, face_recognition (dlib) |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5.3 |
| **Charts** | Chart.js 4.4 |
| **Icons** | Bootstrap Icons 1.11 |
| **Authentication** | Werkzeug (pbkdf2:sha256) |
| **Export** | CSV (stdlib), Excel (openpyxl) |
| **Email** | Python smtplib (env-configured) |

---

## 📁 Project Structure

```
face_attendance_system/
├── app/
│   ├── __init__.py              # App factory
│   ├── database.py              # SQLAlchemy setup
│   ├── models/                  # ORM models
│   │   ├── user.py              # User model (password hashing)
│   │   ├── attendance.py        # Attendance records
│   │   └── face_data.py         # Face encoding references
│   ├── services/                # Business logic
│   │   ├── user_service.py      # User CRUD operations
│   │   ├── attendance_service.py # Attendance recording
│   │   ├── attendance_policy.py # Centralized rules engine
│   │   ├── decision_engine.py   # Multi-signal decision engine (NEW)
│   │   ├── candidate_margin.py  # Candidate margin analysis (NEW)
│   │   ├── recognition_pipeline.py # Full recognition flow
│   │   ├── liveness_service.py  # Anti-spoofing detection
│   │   ├── notification_service.py # Email notifications
│   │   ├── camera_service.py    # Camera + recognition
│   │   ├── camera_manager.py    # Pure camera management
│   │   ├── enrollment_service.py # Face enrollment
│   │   └── performance_monitor.py # Metrics tracking
│   ├── vision/                  # Computer vision modules
│   │   ├── frame_validator.py   # Frame quality checks
│   │   ├── face_detector.py     # Face detection
│   │   ├── face_aligner.py      # Face alignment
│   │   ├── face_encoder.py      # Face embedding
│   │   └── face_recognizer.py   # Identity matching
│   ├── routes/                  # Flask blueprints
│   │   ├── main.py              # Dashboard, login, logout
│   │   ├── user_routes.py       # User CRUD (admin-only)
│   │   ├── attendance_routes.py # Attendance views + export
│   │   └── api_routes.py        # REST API endpoints
│   ├── utils/                   # Utilities
│   │   ├── helpers.py           # Common helper functions
│   │   ├── decorators.py        # Auth + role decorators
│   │   ├── security.py          # Rate limiting, validation
│   │   ├── error_handlers.py    # Custom error pages
│   │   └── logger.py            # Structured logging
│   ├── templates/               # Jinja2 HTML templates
│   │   ├── base.html            # Base layout + dark mode
│   │   ├── dashboard.html       # Main dashboard
│   │   ├── login.html           # Login page
│   │   ├── camera.html          # Live camera feed
│   │   ├── reports.html         # Charts & analytics
│   │   ├── status.html          # System health monitor
│   │   ├── errors/              # Error pages (400-503)
│   │   ├── users/               # User management pages
│   │   └── attendance/          # Attendance pages
│   └── static/                  # Static assets
│       ├── css/style.css        # Full stylesheet + dark mode
│       └── js/main.js           # Client-side utilities
├── config/
│   └── settings.py              # Configuration (env-based)
├── data/                        # Runtime data
│   ├── attendance.db            # SQLite database
│   ├── face_data/               # Face encodings
│   ├── uploads/                 # Uploaded images
│   └── logs/                    # Application logs
├── tests/                       # Test suite (223 tests)
│   ├── conftest.py              # Pytest fixtures
│   ├── test_comprehensive.py    # Full feature tests
│   ├── test_edge_cases.py       # Edge case tests
│   ├── test_routes.py           # Route integration tests
│   ├── test_user_service.py     # User service tests
│   ├── test_attendance_service.py # Attendance service tests
│   ├── test_decision_engine.py  # Decision engine tests (NEW)
│   └── test_candidate_margin.py # Candidate margin tests (NEW)
├── requirements.txt             # Python dependencies
├── run.py                       # Application entry point
├── seed_data.py                 # Demo data seeder
└── README.md                    # This file
```

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **CMake** (required for dlib compilation)
- **C++ compiler** (Visual Studio on Windows, build-essential on Linux)
- **Webcam** (for live attendance)

### Setup

```bash
# 1. Navigate to project directory
cd face_attendance_system

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Set environment variables for notifications
# See Environment Setup section below

# 6. Seed database with demo users
python seed_data.py

# 7. Run the application
python run.py
```

### Access Application

Open browser: **http://127.0.0.1:5000**

### Demo Credentials

| Employee ID | Password | Role |
|-------------|----------|------|
| EMP001 | 12345678 | employee |
| EMP002 | 12345678 | employee |
| EMP003 | 12345678 | employee |
| EMP004 | 12345678 | manager |
| EMP005 | 12345678 | employee |
| EMP006 | 12345678 | admin |
| EMP007 | 12345678 | employee |
| EMP008 | 12345678 | employee |

---

## ⚙️ Environment Setup

All configuration is done through environment variables. **Never hardcode credentials.**

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | dev-key (insecure) | Flask secret key for sessions |
| `DATABASE_URL` | sqlite:///data/attendance.db | Database connection string |

### Face Recognition

| Variable | Default | Description |
|----------|---------|-------------|
| `FACE_DETECTION_MODEL` | hog | Detection model: `hog` (CPU) or `cnn` (GPU) |
| `FACE_RECOGNITION_TOLERANCE` | 0.5 | Matching tolerance (lower = stricter) |
| `FACE_RECOGNITION_MODEL` | small | Embedding model: `small` or `large` |

### Attendance

| Variable | Default | Description |
|----------|---------|-------------|
| `ATTENDANCE_WINDOW_MINUTES` | 60 | Duplicate prevention window |
| `ATTENDANCE_LATE_AFTER` | 09:15 | Time after which check-in is "late" |
| `ATTENDANCE_GRACE_PERIOD_SECONDS` | 30 | Grace period before marking late |

### Camera

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | 0 | Camera device index |
| `CAMERA_WIDTH` | 640 | Capture width |
| `CAMERA_HEIGHT` | 480 | Capture height |
| `CAMERA_FPS` | 30 | Target FPS |

### Decision Engine (Advanced Multi-Signal)

| Variable | Default | Description |
|----------|---------|-------------|
| `DECISION_MIN_FACE_SIMILARITY` | 0.5 | Minimum face similarity to consider |
| `DECISION_MIN_LIVENESS_SCORE` | 0.6 | Minimum liveness score to accept |
| `DECISION_MIN_FACE_QUALITY` | 0.4 | Minimum face quality to process |
| `DECISION_MIN_CANDIDATE_MARGIN` | 0.1 | Minimum margin between top candidates |
| `DECISION_HIGH_CONFIDENCE_THRESHOLD` | 0.85 | Combined score for HIGH_CONFIDENCE |
| `DECISION_LOW_CONFIDENCE_THRESHOLD` | 0.60 | Combined score for LOW_CONFIDENCE |
| `DECISION_WEIGHT_SIMILARITY` | 0.35 | Weight for face similarity signal |
| `DECISION_WEIGHT_LIVENESS` | 0.25 | Weight for liveness score |
| `DECISION_WEIGHT_QUALITY` | 0.15 | Weight for face quality |
| `DECISION_WEIGHT_MARGIN` | 0.15 | Weight for candidate margin |
| `DECISION_WEIGHT_DETECTION` | 0.10 | Weight for detection confidence |
| `DECISION_REQUIRE_LIVENESS` | true | Whether liveness check is mandatory |
| `DECISION_LIVENESS_MIN_CHECKS` | 3 | Minimum liveness checks to pass |
| `CANDIDATE_MARGIN_CLEAR_THRESHOLD` | 0.15 | Minimum margin to consider "clear" |

### Notifications (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOTIFICATIONS_ENABLED` | false | Enable email notifications |
| `MAIL_SERVER` | localhost | SMTP server |
| `MAIL_PORT` | 587 | SMTP port |
| `MAIL_USE_TLS` | true | Use TLS encryption |
| `MAIL_USERNAME` | | SMTP username |
| `MAIL_PASSWORD` | | SMTP password |
| `MAIL_DEFAULT_SENDER` | attendance@company.com | Sender email |
| `ADMIN_NOTIFICATION_EMAILS` | | Comma-separated admin emails |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (none) | When set, API mutations require X-API-Key header |

### Example `.env` file

```bash
# Copy this to .env and fill in your values
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=sqlite:///data/attendance.db

# Face Recognition
FACE_DETECTION_MODEL=hog
FACE_RECOGNITION_TOLERANCE=0.5

# Attendance
ATTENDANCE_LATE_AFTER=09:15
ATTENDANCE_WINDOW_MINUTES=60

# Camera
CAMERA_INDEX=0

# Notifications
NOTIFICATIONS_ENABLED=false
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
ADMIN_NOTIFICATION_EMAILS=admin@company.com
```

---

## 🗄️ Database Setup

The database is automatically created on first run via `db.create_all()` in the app factory.

### Models

**Users Table**
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| employee_id | String(50) | Unique employee identifier |
| first_name | String(100) | First name |
| last_name | String(100) | Last name |
| email | String(150) | Unique email |
| phone | String(20) | Phone (optional) |
| department | String(100) | Department |
| role | String(50) | employee / manager / admin |
| is_active | Boolean | Active status |
| password_hash | String(256) | Hashed password |
| face_registered | Boolean | Face enrolled? |
| face_data_path | String(500) | Encoding file path |
| created_at | DateTime | Creation time |
| updated_at | DateTime | Last update |

**Attendance Table**
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| user_id | Integer | FK → users.id |
| attendance_date | Date | Attendance date |
| check_in_time | DateTime | Check-in timestamp |
| check_out_time | DateTime | Check-out timestamp |
| status | String(20) | present / late / checked_out / absent |
| confidence_score | Float | Recognition confidence (0–1) |
| face_image_path | String(500) | Captured face image |
| camera_source | String(100) | Camera identifier |
| created_at | DateTime | Record creation |

---

## 🚀 How to Run

```bash
# Development mode (default)
python run.py

# Or with environment variable
FLASK_ENV=development python run.py
```

The server starts at **http://127.0.0.1:5000**.

---

## 📖 Usage Guide

### 1. Register a User

1. Log in as admin (EMP006 / 12345678)
2. Click **"Register User"** on dashboard or go to `/users/add`
3. Fill in: Employee ID, name, email, department, role
4. Click **"Create User"**

### 2. Enroll a Face

1. Go to Users → click face icon on unregistered user
2. Upload 3–5 clear face photos (different angles)
3. Click **"Register Face"**
4. System creates face encoding from images

### 3. Use Attendance

1. Click **"Start Attendance"** on dashboard or go to `/camera`
2. Allow camera access when prompted
3. Position face in the camera frame
4. System detects → validates liveness → recognizes → marks attendance
5. Status appears on screen with confidence score

### 4. View Attendance

1. Go to **Attendance** → Today's records
2. Or **History** for date range filtering
3. Use filters: employee, status, department, search
4. Sort by clicking column headers

### 5. Export Reports

1. Go to **History** page
2. Apply desired filters
3. Click **Export** dropdown → CSV or Excel
4. File downloads with current filter applied

---

## 👨‍💼 Admin Functionality

Admin users (role = `admin`) can:

- ✅ Register new users
- ✅ Edit user profiles
- ✅ Deactivate / reactivate users
- ✅ Permanently delete users
- ✅ Register face data for users
- ✅ Delete face data
- ✅ Manual check-in / check-out
- ✅ View all attendance records
- ✅ Export attendance reports
- ✅ Configure frame skip interval
- ✅ Send daily summary emails
- ✅ Access system status dashboard

**Non-admin users** can only:
- ✅ View their own attendance
- ✅ Use camera for face recognition
- ✅ View reports

---

## 📡 API Documentation

### Health & Status

```
GET  /api/health              → System health check
GET  /api/system/status       → Component status (camera, engine, DB)
GET  /api/performance         → Performance metrics
GET  /api/pipeline            → Recognition pipeline info
GET  /api/liveness            → Liveness detection config
GET  /api/policy              → Attendance policy config
GET  /api/notifications       → Notification config
```

### Users

```
GET  /api/users               → List all users
POST /api/users               → Create user (requires API key)
GET  /api/users/<id>          → Get user details
PUT  /api/users/<id>          → Update user (requires API key)
```

### Attendance

```
GET  /api/attendance          → Get attendance records
POST /api/attendance/mark     → Mark attendance (requires API key)
POST /api/attendance/check-out → Check out (requires API key)
GET  /api/attendance/stats    → Get statistics
GET  /api/attendance/report   → Daily report
POST /api/notifications/daily-summary → Send daily summary
```

### Recognition

```
POST /api/recognize           → Recognize face from uploaded image
```

### Frame Skip Configuration

```
POST /api/performance/frame-skip  → Set frame skip interval (1-30)
Body: { "interval": 2 }
```

### Decision Engine (PHASE X01)

```
GET  /api/decision-engine/policy   → Get current decision policy config
PUT  /api/decision-engine/policy   → Update decision policy (requires API key)
GET  /api/decision-engine/status   → Get decision engine + margin analyzer status
POST /api/decision-engine/evaluate → Evaluate decision with custom signals
PUT  /api/candidate-margin/threshold → Update margin threshold (requires API key)
```

**Evaluate endpoint example:**
```json
POST /api/decision-engine/evaluate
{
  "face_similarity": 0.92,
  "liveness_score": 0.85,
  "face_quality": 0.80,
  "candidate_margin": 0.35,
  "detection_confidence": 0.90,
  "liveness_checks_passed": 4,
  "liveness_checks_total": 5,
  "user_id": 1
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "decision": "accept",
    "uncertainty_state": "low_confidence",
    "combined_score": 0.797,
    "explanation": {
      "decision_summary": "ACCEPT",
      "primary_reason": "Identity confirmed with acceptable confidence",
      "signal_assessment": "Strong face similarity (0.92); High liveness confidence (0.85)"
    }
  }
}
```

---

## ⚡ Performance

### Optimizations

| Optimization | Description |
|--------------|-------------|
| Frame Resizing | Frames resized to 640px width for faster detection |
| Cached Encodings | Face encodings loaded once, reused |
| User Name Cache | DB queries minimized during live recognition |
| Recognition Cooldown | 60s cooldown prevents rapid re-recognition |
| Efficient Numpy | Vectorized distance calculations |
| Frame Skipping | Process every Nth frame in live stream |
| DB Latency Tracking | Monitors database query performance |

### Health Status Levels

| Status | FPS | Avg Processing | Description |
|--------|-----|----------------|-------------|
| Excellent | ≥ 15 | < 100ms | Optimal performance |
| Good | ≥ 10 | < 200ms | Performing well |
| Fair | ≥ 5 | < 500ms | Adequate performance |
| Poor | < 5 | ≥ 500ms | Degraded performance |

---

## 🔒 Security

### Password Security
- ✅ Passwords **never stored in plain text**
- ✅ Werkzeug `generate_password_hash()` with salt
- ✅ `pbkdf2:sha256` hashing algorithm
- ✅ Password verification via `check_password_hash()`

### Input Validation

| Field | Rules |
|-------|-------|
| Employee ID | 2–50 chars, alphanumeric + hyphens/underscores |
| Email | Valid format, max 150 chars |
| Name | 1–100 chars, letters/spaces/hyphens |
| Phone | Optional, 7–20 chars |
| Role | Must be: employee, manager, admin |

### Security Headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`

### Rate Limiting

| Endpoint | Limit |
|----------|-------|
| Login | 10 requests / 60 seconds |
| API (general) | 100 requests / 60 seconds |
| File uploads | 20 requests / 60 seconds |

### Session Security
- ✅ Session regeneration on login (prevents fixation)
- ✅ HTTPOnly cookies
- ✅ SameSite=Lax
- ✅ 8-hour session lifetime
- ✅ Login-protected routes with role checking

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_edge_cases.py -v

# Run with coverage
python -m pytest tests/ --cov=app
```

### Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_comprehensive.py | 38 | User registration, face, recognition, attendance, API, UI, security |
| test_edge_cases.py | 64 | Camera, detection, recognition, attendance, UI, security, policy, API |
| test_routes.py | 17 | Route integration, auth, admin access |
| test_user_service.py | 14 | User CRUD operations |
| test_attendance_service.py | 11 | Attendance marking, checkout, records, stats |
| test_decision_engine.py | 28 | Decision signals, policy, engine logic, edge cases |
| test_candidate_margin.py | 27 | Margin analysis, signal normalization, edge cases, thresholds |
| **Total** | **223** | **All features tested** |

---

## ⚠️ Limitations

### Liveness Detection
- Basic texture/brightness/edge analysis — not a full anti-spoofing system
- Cannot defeat high-quality printed photos
- Cannot detect video replay attacks on high-resolution screens
- Cannot detect 3D masks
- For production, consider a dedicated liveness model (e.g., FaceLivenessDetection)

### Face Recognition
- Requires good lighting for reliable detection
- Accuracy depends on face registration quality
- HOG model (CPU) is slower than CNN model (GPU)
- Recognition tolerance may need tuning per environment

### Camera
- Requires webcam access (browser permission)
- Server-side camera requires physical camera connected
- Performance varies by hardware

### Database
- SQLite is single-writer — not suitable for high-concurrency production
- For production, migrate to PostgreSQL or MySQL

---

## 🔒 Privacy Considerations

- **Face data** is stored as mathematical encodings, not images (after processing)
- **Face images** used for encoding are saved temporarily and can be deleted
- **Passwords** are irreversibly hashed
- **API responses** never expose password hashes or sensitive data
- **Logs** sanitize sensitive fields (passwords, tokens, biometrics)
- **Data deletion** is supported (user deactivation + hard delete)
- **No third-party services** — all processing is local
- **No telemetry** — no data leaves the server

---

## 🚀 Future Improvements

### High Priority
- [ ] PostgreSQL/MySQL support for production
- [ ] REST API rate limiting with Redis
- [ ] PDF report export
- [ ] Multi-camera support
- [ ] Batch face enrollment

### Medium Priority
- [ ] WebSocket for real-time attendance updates
- [ ] Advanced liveness detection (3D depth, blink detection)
- [ ] GPU-accelerated recognition (CUDA)
- [ ] LDAP/Active Directory integration
- [ ] Two-factor authentication

### Low Priority
- [ ] Multi-language support (i18n)
- [ ] Mobile app (React Native / Flutter)
- [ ] Cloud deployment (Docker + AWS/GCP)
- [ ] Analytics dashboard with trends
- [ ] Scheduled reports via email

---

## 📄 License

Educational project. Feel free to use and modify.

### PHASE X01 Changelog
- Added `decision_engine.py` — Multi-signal decision engine with weighted scoring
- Added `candidate_margin.py` — Candidate margin analyzer for identity clarity
- Added 55 new tests (test_decision_engine.py + test_candidate_margin.py)
- Added API endpoints for decision engine configuration and evaluation
- Added decision engine configuration in `settings.py`
- Updated recognition pipeline to use decision engine
- Updated README with decision engine documentation

---

*Built with ❤️ using Flask, OpenCV, and face_recognition*
