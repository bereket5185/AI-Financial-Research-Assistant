import sqlite3
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "financial_ai.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a SQLite database connection.

    row_factory allows us to access database rows like dictionaries.
    """
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """
    Create all required database tables if they do not already exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            filepath TEXT,
            year INTEGER,
            file_size INTEGER,
            status TEXT DEFAULT 'processed',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # RESEARCH HISTORY
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS research_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT,
            search_query TEXT,
            agent_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE SET NULL
        )
        """
    )

    # --------------------------------------------------------
    # RESEARCH SOURCES
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS research_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_id INTEGER NOT NULL,
            document_id INTEGER,
            page INTEGER,
            chunk INTEGER,
            similarity REAL,
            source_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (research_id)
                REFERENCES research_history(id)
                ON DELETE CASCADE,

            FOREIGN KEY (document_id)
                REFERENCES documents(id)
                ON DELETE SET NULL
        )
        """
    )

    # --------------------------------------------------------
    # EXTRACTED METRICS
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS extracted_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_id INTEGER,
            document_id INTEGER,
            metric TEXT NOT NULL,
            value REAL,
            unit TEXT,
            period TEXT,
            page INTEGER,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (research_id)
                REFERENCES research_history(id)
                ON DELETE CASCADE,

            FOREIGN KEY (document_id)
                REFERENCES documents(id)
                ON DELETE SET NULL
        )
        """
    )


       # ========================================================
    # DATABASE MIGRATIONS
    # ========================================================

    # Check existing users columns
    cursor.execute("PRAGMA table_info(users)")
    user_columns = [row["name"] for row in cursor.fetchall()]

    # Add password column to older databases
    if "password" not in user_columns:
        cursor.execute(
            """
            ALTER TABLE users
            ADD COLUMN password TEXT
            """
        )

    # Add role column if an older database does not have it
    if "role" not in user_columns:
        cursor.execute(
            """
            ALTER TABLE users
            ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
            """
        )
    
    
    
    
    conn.commit()
    conn.close()

    print("Database initialized successfully!")
    print(f"Database location: {DATABASE_PATH}")


# ============================================================
# USER FUNCTIONS
# ============================================================

def create_user(username, password=None, role="user"):
    """
    Create a new user.

    Returns:
        int: Newly created user ID
        None: If username already exists
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users (username, password, role)
            VALUES (?, ?, ?)
            """,
            (username, password, role)
        )

        conn.commit()
        return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None

    finally:
        conn.close()


def get_user_by_username(username):
    """
    Get a user by username.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            LIMIT 1
            """,
            (username,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def get_user_by_id(user_id):
    """
    Get a user by ID.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE id = ?
            LIMIT 1
            """,
            (user_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def get_all_users():
    """
    Return all users.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            ORDER BY created_at DESC
            """
        )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def save_document(
    filename,
    filepath=None,
    year=None,
    file_size=None,
    status="processed"
):
    """
    Save a document to the database.

    If the filename already exists, the existing document ID
    is returned instead of creating a duplicate.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        # Check whether document already exists
        cursor.execute(
            """
            SELECT id
            FROM documents
            WHERE filename = ?
            LIMIT 1
            """,
            (filename,)
        )

        existing = cursor.fetchone()

        if existing:
            return existing["id"]

        cursor.execute(
            """
            INSERT INTO documents
            (
                filename,
                filepath,
                year,
                file_size,
                status
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                filepath,
                year,
                file_size,
                status
            )
        )

        conn.commit()
        return cursor.lastrowid

    finally:
        conn.close()


def get_document_by_id(document_id):
    """
    Get a document by its database ID.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE id = ?
            LIMIT 1
            """,
            (document_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def get_document_by_filename(filename):
    """
    Get a document by filename.

    This function is required by app.py when saving
    retrieved research sources.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE filename = ?
            LIMIT 1
            """,
            (filename,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def get_documents():
    """
    Return all documents.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM documents
            ORDER BY uploaded_at DESC
            """
        )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


def delete_document(document_id):
    """
    Delete a document.

    Related research sources and extracted metrics will have
    their document_id set to NULL because of ON DELETE SET NULL.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (document_id,)
        )

        conn.commit()
        return cursor.rowcount > 0

    finally:
        conn.close()


# ============================================================
# RESEARCH HISTORY FUNCTIONS
# ============================================================

def save_research(
    question,
    answer,
    search_query="",
    agent_type="RAG",
    user_id=None
):
    """
    Save a research question and answer.

    Returns:
        int: Newly created research ID
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO research_history
            (
                user_id,
                question,
                answer,
                search_query,
                agent_type
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question,
                answer,
                search_query,
                agent_type
            )
        )

        conn.commit()
        return cursor.lastrowid

    finally:
        conn.close()


def get_research_history(limit=50, user_id=None):
    """
    Get recent research history.

    If user_id is supplied, only that user's research is returned.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        if user_id is None:
            cursor.execute(
                """
                SELECT *
                FROM research_history
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,)
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM research_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit)
            )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


def get_research_by_id(research_id):
    """
    Get one research record by ID.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM research_history
            WHERE id = ?
            LIMIT 1
            """,
            (research_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def delete_research(research_id):
    """
    Delete a research record.

    Its sources and metrics are deleted automatically because
    those tables use ON DELETE CASCADE for research_id.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM research_history
            WHERE id = ?
            """,
            (research_id,)
        )

        conn.commit()
        return cursor.rowcount > 0

    finally:
        conn.close()


# ============================================================
# RESEARCH SOURCE FUNCTIONS
# ============================================================

def save_research_source(
    research_id,
    document_id=None,
    page=None,
    chunk=None,
    similarity=None,
    source_text=None
):
    """
    Save one retrieved source for a research result.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO research_sources
            (
                research_id,
                document_id,
                page,
                chunk,
                similarity,
                source_text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                research_id,
                document_id,
                page,
                chunk,
                similarity,
                source_text
            )
        )

        conn.commit()
        return cursor.lastrowid

    finally:
        conn.close()


def save_research_sources(research_id, sources):
    """
    Save multiple retrieved sources.

    Each source may contain keys such as:
        document_id
        page
        chunk
        similarity
        score
        source_text
        text
    """

    if not sources:
        return []

    saved_ids = []

    for source in sources:
        if not isinstance(source, dict):
            continue

        document_id = source.get("document_id")

        page = source.get("page")
        chunk = source.get("chunk")

        similarity = source.get("similarity")

        if similarity is None:
            similarity = source.get("score")

        source_text = source.get("source_text")

        if source_text is None:
            source_text = source.get("text")

        saved_id = save_research_source(
            research_id=research_id,
            document_id=document_id,
            page=page,
            chunk=chunk,
            similarity=similarity,
            source_text=source_text
        )

        saved_ids.append(saved_id)

    return saved_ids


def get_research_sources(research_id):
    """
    Get all sources belonging to a research result.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                rs.*,
                d.filename AS document_filename,
                d.year AS document_year
            FROM research_sources rs
            LEFT JOIN documents d
                ON rs.document_id = d.id
            WHERE rs.research_id = ?
            ORDER BY rs.similarity DESC
            """,
            (research_id,)
        )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


# ============================================================
# EXTRACTED METRIC FUNCTIONS
# ============================================================

def save_metric(
    metric,
    value,
    unit=None,
    period=None,
    page=None,
    source=None,
    research_id=None,
    document_id=None
):
    """
    Save one extracted financial metric.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO extracted_metrics
            (
                research_id,
                document_id,
                metric,
                value,
                unit,
                period,
                page,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                research_id,
                document_id,
                metric,
                value,
                unit,
                period,
                page,
                source
            )
        )

        conn.commit()
        return cursor.lastrowid

    finally:
        conn.close()


def save_metrics(metrics, research_id=None):
    """
    Save multiple financial metrics.

    Each metric dictionary can contain:
        metric
        name
        value
        numeric_value
        unit
        period
        page
        source
        document_id
    """

    if not metrics:
        return []

    saved_ids = []

    for item in metrics:
        if not isinstance(item, dict):
            continue

        metric_name = item.get("metric")

        if metric_name is None:
            metric_name = item.get("name")

        value = item.get("value")

        if value is None:
            value = item.get("numeric_value")

        unit = item.get("unit")
        period = item.get("period")
        page = item.get("page")
        source = item.get("source")
        document_id = item.get("document_id")

        if metric_name is None:
            continue

        saved_id = save_metric(
            metric=metric_name,
            value=value,
            unit=unit,
            period=period,
            page=page,
            source=source,
            research_id=research_id,
            document_id=document_id
        )

        saved_ids.append(saved_id)

    return saved_ids


def get_metrics_by_research(research_id):
    """
    Get financial metrics saved for a research result.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                em.*,
                d.filename AS document_filename,
                d.year AS document_year
            FROM extracted_metrics em
            LEFT JOIN documents d
                ON em.document_id = d.id
            WHERE em.research_id = ?
            ORDER BY em.created_at DESC
            """,
            (research_id,)
        )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


def get_metrics_by_document(document_id):
    """
    Get financial metrics associated with a document.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM extracted_metrics
            WHERE document_id = ?
            ORDER BY created_at DESC
            """,
            (document_id,)
        )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


# ============================================================
# DATABASE STATISTICS
# ============================================================

def get_database_statistics():
    """
    Return basic database statistics.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        statistics = {}

        cursor.execute("SELECT COUNT(*) AS count FROM users")
        statistics["users"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) AS count FROM documents")
        statistics["documents"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) AS count FROM research_history")
        statistics["research"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) AS count FROM research_sources")
        statistics["sources"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) AS count FROM extracted_metrics")
        statistics["metrics"] = cursor.fetchone()["count"]

        return statistics

    finally:
        conn.close()


def get_table_names():
    """
    Return all SQLite table names.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )

        return [row["name"] for row in cursor.fetchall()]

    finally:
        conn.close()


# ============================================================
# DATABASE TEST
# ============================================================

def test_database():
    """
    Test database initialization and basic database access.
    """

    initialize_database()

    print()
    print("Database location:")
    print(DATABASE_PATH)

    print()
    print("Tables:")

    tables = get_table_names()

    expected_tables = [
        "documents",
        "extracted_metrics",
        "research_history",
        "research_sources",
        "users"
    ]

    for table in expected_tables:
        if table in tables:
            print(f"  ✓ {table}")
        else:
            print(f"  ✗ {table}")

    print()
    print("Statistics:")

    stats = get_database_statistics()

    print(f"  users: {stats['users']}")
    print(f"  documents: {stats['documents']}")
    print(f"  research: {stats['research']}")
    print(f"  sources: {stats['sources']}")
    print(f"  metrics: {stats['metrics']}")

    print()
    print("Database test completed successfully.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    initialize_database()
    test_database()
