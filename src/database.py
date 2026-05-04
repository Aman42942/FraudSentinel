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
            source      TEXT DEFAULT 'live'
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

    # Webhook config
    c.execute("""
        CREATE TABLE IF NOT EXISTS webhooks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT,
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

    conn.commit()
    conn.close()


def insert_tx(tx_id, amount, tx_type, model_used, score, prediction,
              features_dict=None, source="live"):
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO transactions
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        tx_id, amount, tx_type, model_used,
        round(score, 5), int(prediction),
        json.dumps(features_dict or {}),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source,
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


def get_webhooks():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM webhooks WHERE active=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_webhook(url):
    conn = get_conn()
    conn.execute("INSERT INTO webhooks (url, created_at) VALUES (?,?)",
                 (url, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


# Initialize on import
init_db()
