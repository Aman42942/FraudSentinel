"""
src/database.py  —  FraudSentinel Fraud Ledger (SQLite)
=========================================================
Persistent storage for all transactions, batch scans, and webhook config.
"""
import sqlite3, os, json
from datetime import datetime

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "fraudsentinel.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    c    = conn.cursor()

    # Transactions ledger
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          TEXT PRIMARY KEY,
            amount      REAL,
            tx_type     TEXT,
            model_used  TEXT,
            score       REAL,
            prediction  INTEGER,
            features    TEXT,
            timestamp   TEXT,
            source      TEXT DEFAULT 'live',
            status      TEXT DEFAULT 'PENDING',
            ip_address  TEXT,
            device_id   TEXT,
            location    TEXT
        )
    """)

    # Batch scan records
    c.execute("""
        CREATE TABLE IF NOT EXISTS batch_scans (
            scan_id     TEXT PRIMARY KEY,
            filename    TEXT,
            total       INTEGER,
            fraud_count INTEGER,
            model_used  TEXT,
            timestamp   TEXT
        )
    """)



    # Manual Rules Engine
    c.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            condition   TEXT,
            action      TEXT,
            active      INTEGER DEFAULT 1,
            created_at  TEXT
        )
    """)

    # Active model config
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # Default active model
    c.execute("INSERT OR IGNORE INTO settings VALUES ('active_model','XGBoost')")
    c.execute("INSERT OR IGNORE INTO settings VALUES ('auto_block_threshold','0.95')")

    # SaaS: API Keys
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            api_key      TEXT PRIMARY KEY,
            client_name  TEXT,
            usage_count  INTEGER DEFAULT 0,
            is_active    INTEGER DEFAULT 1,
            created_at   TEXT
        )
    """)

    # SaaS: Webhooks
    c.execute("""
        CREATE TABLE IF NOT EXISTS webhooks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT,
            secret      TEXT,
            is_active   INTEGER DEFAULT 1,
            created_at  TEXT
        )
    """)

    # Cyber: Velocity Logs (Rate Limiting)
    c.execute("""
        CREATE TABLE IF NOT EXISTS velocity_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address  TEXT,
            api_key     TEXT,
            timestamp   REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_tx(tx_id, amount, tx_type, model_used, score, prediction,
              features_dict=None, source="live", ip=None, device=None, loc=None):
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO transactions (
            id, amount, tx_type, model_used, score, prediction, 
            features, timestamp, source, status, ip_address, device_id, location
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        tx_id, amount, tx_type, model_used,
        round(score, 5), int(prediction),
        json.dumps(features_dict or {}),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source, 'PENDING', ip, device, loc
    ))
    conn.commit()
    conn.close()


def get_history(page=1, per_page=50, prediction=None, source=None):
    conn   = get_conn()
    offset = (page - 1) * per_page
    where  = []
    params = []
    if prediction is not None:
        where.append("prediction = ?"); params.append(int(prediction))
    if source is not None:
        where.append("source = ?"); params.append(source)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows  = conn.execute(
        f"SELECT * FROM transactions {where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    ).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM transactions {where_sql}", params
    ).fetchone()[0]
    conn.close()
    return [dict(r) for r in rows], total


def search_by_id(tx_id):
    conn = get_conn()
    row  = conn.execute(
        "SELECT * FROM transactions WHERE id LIKE ?", (f"%{tx_id}%",)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_db_stats():
    conn  = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud = conn.execute("SELECT COUNT(*) FROM transactions WHERE prediction=1").fetchone()[0]
    amt   = conn.execute("SELECT SUM(amount) FROM transactions").fetchone()[0] or 0
    today = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE timestamp LIKE ?",
        (datetime.now().strftime("%Y-%m-%d") + "%",)
    ).fetchone()[0]
    conn.close()
    return {"total": total, "fraud": fraud, "legit": total - fraud,
            "total_amount": round(amt, 2), "today": today,
            "fraud_rate": round(fraud / max(total, 1) * 100, 2)}


def insert_batch(scan_id, filename, total, fraud_count, model_used):
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO batch_scans VALUES (?,?,?,?,?,?)
    """, (scan_id, filename, total, fraud_count, model_used,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_active_model():
    conn  = get_conn()
    row   = conn.execute("SELECT value FROM settings WHERE key='active_model'").fetchone()
    conn.close()
    return row["value"] if row else "XGBoost"


def set_active_model(name):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO settings VALUES ('active_model',?)", (name,))
    conn.commit()
    conn.close()




# ── MANAGEMENT FUNCTIONS ──

def update_tx_status(tx_id, status):
    conn = get_conn()
    conn.execute("UPDATE transactions SET status=? WHERE id=?", (status, tx_id))
    conn.commit()
    conn.close()

def get_rules():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM rules WHERE active=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_rule(name, condition, action):
    conn = get_conn()
    conn.execute("INSERT INTO rules (name, condition, action, created_at) VALUES (?,?,?,?)",
                 (name, condition, action, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def delete_rule(rule_id):
    conn = get_conn()
    conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
    conn.commit()
    conn.close()

# ── SAAS & CYBER FUNCTIONS ──

def generate_api_key(client_name):
    import uuid
    api_key = f"fs_live_{uuid.uuid4().hex}"
    conn = get_conn()
    conn.execute("INSERT INTO api_keys (api_key, client_name, created_at) VALUES (?,?,?)",
                 (api_key, client_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return api_key

def get_api_keys():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM api_keys").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def validate_api_key(api_key):
    conn = get_conn()
    row = conn.execute("SELECT is_active FROM api_keys WHERE api_key=?", (api_key,)).fetchone()
    if row and row['is_active']:
        conn.execute("UPDATE api_keys SET usage_count = usage_count + 1 WHERE api_key=?", (api_key,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def revoke_api_key(api_key):
    conn = get_conn()
    conn.execute("UPDATE api_keys SET is_active=0 WHERE api_key=?", (api_key,))
    conn.commit()
    conn.close()

def get_webhooks():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM webhooks WHERE is_active=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def check_velocity(ip_address, api_key=None, window_seconds=60, max_requests=100):
    import time
    now = time.time()
    conn = get_conn()
    
    # Clean old logs
    conn.execute("DELETE FROM velocity_logs WHERE timestamp < ?", (now - window_seconds,))
    
    # Log current request
    conn.execute("INSERT INTO velocity_logs (ip_address, api_key, timestamp) VALUES (?,?,?)",
                 (ip_address, api_key, now))
    
    # Check count
    count = conn.execute("SELECT COUNT(*) FROM velocity_logs WHERE ip_address=? AND timestamp > ?",
                         (ip_address, now - window_seconds)).fetchone()[0]
    
    conn.commit()
    conn.close()
    return count <= max_requests


# Initialize on import
init_db()
