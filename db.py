"""Data and auth layer for the University Finder.

Everything that touches SQLite lives here, with no Streamlit code, so app.py can
just call these functions and show the result. The database file is
data/users.db and is created automatically on first use.

Every function returns a small tuple instead of raising, so the app never shows
a traceback: (True/False, message) for actions, (data, error) for reads.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "users.db")

# Password hashing settings.
PBKDF2_ITERATIONS = 260_000
SALT_BYTES = 16

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 8

GENERIC_LOGIN_ERROR = "Wrong username or password"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE COLLATE NOCASE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_universities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    institution TEXT NOT NULL,
    country TEXT,
    rank_label TEXT,
    source TEXT,
    score REAL,
    note TEXT DEFAULT '',
    saved_at TEXT NOT NULL,
    UNIQUE(user_id, institution)
);
"""


# ---------------------------------------------------------------
# CONNECTION HELPERS
# ---------------------------------------------------------------
def _connect():
    """Open the database, creating the data folder if it does not exist yet."""
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create the tables on first run. Returns (ok, error_message)."""
    try:
        conn = _connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        return True, ""
    except (sqlite3.Error, OSError) as exc:
        return False, f"Could not open the user database ({exc})."


def _now():
    """Timestamp we store with users and saved rows (readable, not timezone-shifted)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hash_password(password, salt):
    """Salted PBKDF2-HMAC-SHA256 hash. The plain password only exists in memory."""
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return digest.hex()


# ---------------------------------------------------------------
# ACCOUNTS
# ---------------------------------------------------------------
def validate_username(username):
    """Return an error message for a bad username, or None if it is fine."""
    if not USERNAME_PATTERN.match(username or ""):
        return ("Username must be 3-30 characters and use only letters, digits "
                "and underscores.")
    return None


def validate_password(password):
    """Return an error message for a bad password, or None if it is fine."""
    if len(password or "") < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


def register_user(username, password):
    """Create an account. Returns (user_dict_or_None, error_message)."""
    username = (username or "").strip()
    password = password or ""

    error = validate_username(username) or validate_password(password)
    if error:
        return None, error

    salt = secrets.token_hex(SALT_BYTES)
    password_hash = _hash_password(password, salt)
    try:
        conn = _connect()
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, _now()),
            )
            conn.commit()
            return {"id": cursor.lastrowid, "username": username}, ""
        finally:
            conn.close()
    except sqlite3.IntegrityError:
        # The UNIQUE COLLATE NOCASE index also catches different capitalisation.
        return None, f"The username '{username}' is already taken. Please choose another one."
    except (sqlite3.Error, OSError) as exc:
        return None, f"Could not create the account ({exc})."


def authenticate(username, password):
    """Check a login. Returns (user_dict_or_None, error_message).

    A wrong username and a wrong password both return the same message, so the
    app never reveals which one was wrong.
    """
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return None, GENERIC_LOGIN_ERROR

    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        return None, f"Could not read the user database ({exc})."

    if row is None:
        return None, GENERIC_LOGIN_ERROR

    expected = row["password_hash"]
    actual = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected, actual):
        return None, GENERIC_LOGIN_ERROR

    return {"id": row["id"], "username": row["username"]}, ""


# ---------------------------------------------------------------
# SAVED UNIVERSITIES
# ---------------------------------------------------------------
def save_university(user_id, institution, country, rank_label, source, score, note=""):
    """Save one university for a user.

    Returns (status, message) where status is "saved", "duplicate" (the
    UNIQUE(user_id, institution) constraint already covers this university) or
    "error".
    """
    institution = (institution or "").strip()
    if not user_id or not institution:
        return "error", "A user and a university name are needed to save a university."

    try:
        numeric_score = float(score) if score is not None else None
    except (TypeError, ValueError):
        numeric_score = None

    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO saved_universities "
                "(user_id, institution, country, rank_label, source, score, note, saved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, institution, country, rank_label, source, numeric_score,
                 note or "", _now()),
            )
            conn.commit()
            return "saved", f"Saved {institution}."
        finally:
            conn.close()
    except sqlite3.IntegrityError:
        return "duplicate", f"{institution} is already saved."
    except (sqlite3.Error, OSError) as exc:
        return "error", f"Could not save {institution} ({exc})."


def list_saved(user_id):
    """Return (rows, error) for one user's saved universities, newest first."""
    if not user_id:
        return [], ""
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT institution, country, rank_label, source, score, note, saved_at "
                "FROM saved_universities WHERE user_id = ? "
                "ORDER BY saved_at DESC, institution",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows], ""
    except (sqlite3.Error, OSError) as exc:
        return [], f"Could not read the saved universities ({exc})."


def update_note(user_id, institution, note):
    """Replace the note of one saved university. Returns (ok, message)."""
    if not user_id or not institution:
        return False, "Pick a saved university first."
    try:
        conn = _connect()
        try:
            cursor = conn.execute(
                "UPDATE saved_universities SET note = ? "
                "WHERE user_id = ? AND institution = ?",
                (note or "", user_id, institution),
            )
            conn.commit()
        finally:
            conn.close()
        if cursor.rowcount == 0:
            return False, "That university is not in your saved list."
        return True, f"Note updated for {institution}."
    except (sqlite3.Error, OSError) as exc:
        return False, f"Could not update the note ({exc})."


def remove_saved(user_id, institution):
    """Remove one saved university. Returns (ok, message)."""
    if not user_id or not institution:
        return False, "Pick a saved university first."
    try:
        conn = _connect()
        try:
            cursor = conn.execute(
                "DELETE FROM saved_universities WHERE user_id = ? AND institution = ?",
                (user_id, institution),
            )
            conn.commit()
        finally:
            conn.close()
        if cursor.rowcount == 0:
            return False, "That university is not in your saved list."
        return True, f"Removed {institution}."
    except (sqlite3.Error, OSError) as exc:
        return False, f"Could not remove the university ({exc})."
