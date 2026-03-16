"""Tests for DuckDB multi-source connector.

Skipped entirely if DuckDB is not available.
"""

import pytest

duckdb = pytest.importorskip("duckdb")

from event_jepa_cube.duckdb_connector import DuckDBConnector  # noqa: E402, I001
from event_jepa_cube.duckdb_connector import _validate_identifier  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def connector():
    conn = DuckDBConnector(database=":memory:", embedding_dim=3)
    yield conn
    conn.close()


@pytest.fixture
def populated_connector(connector):
    """Connector with sample event data in the default schema."""
    conn = connector._ensure_open()
    conn.execute("""
        CREATE TABLE event_sequences (
            sequence_id VARCHAR,
            embedding FLOAT[],
            timestamp DOUBLE,
            modality VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO event_sequences VALUES
        ('seq1', [1.0, 0.0, 0.0], 1.0, 'text'),
        ('seq1', [0.0, 1.0, 0.0], 2.0, 'text'),
        ('seq1', [0.0, 0.0, 1.0], 3.0, 'text'),
        ('seq2', [0.5, 0.5, 0.0], 1.0, 'text'),
        ('seq2', [0.0, 0.5, 0.5], 2.0, 'text')
    """)
    return connector


@pytest.fixture
def entity_connector(connector):
    """Connector with sample entity data."""
    conn = connector._ensure_open()
    conn.execute("""
        CREATE TABLE entities (
            entity_id VARCHAR,
            modality VARCHAR,
            embedding FLOAT[],
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO entities VALUES
        ('e1', 'text', [1.0, 0.0, 0.0], 'electronics'),
        ('e1', 'image', [0.9, 0.1, 0.0], 'electronics'),
        ('e2', 'text', [0.8, 0.2, 0.0], 'electronics'),
        ('e3', 'text', [0.0, 0.0, 1.0], 'clothing')
    """)
    return connector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestValidateIdentifier:
    def test_valid_identifier(self):
        assert _validate_identifier("my_table") == '"my_table"'

    def test_valid_identifier_with_digits(self):
        assert _validate_identifier("table_2") == '"table_2"'

    def test_invalid_identifier_space(self):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            _validate_identifier("my table")

    def test_invalid_identifier_semicolon(self):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            _validate_identifier("table;DROP")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


class TestConnection:
    def test_connect_in_memory(self):
        conn = DuckDBConnector(database=":memory:", embedding_dim=3)
        assert conn._conn is not None
        conn.close()
        assert conn._conn is None

    def test_context_manager(self):
        with DuckDBConnector(database=":memory:", embedding_dim=3) as conn:
            assert conn._conn is not None
        assert conn._conn is None

    def test_ensure_open_raises_after_close(self):
        conn = DuckDBConnector(database=":memory:", embedding_dim=3)
        conn.close()
        with pytest.raises(RuntimeError, match="closed"):
            conn._ensure_open()


# ---------------------------------------------------------------------------
# Attach / Detach
# ---------------------------------------------------------------------------


class TestAttach:
    def test_attach_duckdb_file(self, connector, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        # Create a DuckDB file first
        tmp_conn = duckdb.connect(db_path)
        tmp_conn.execute("CREATE TABLE t (id INT)")
        tmp_conn.close()

        connector.attach("testdb", db_path, db_type="duckdb", read_only=True)
        assert "testdb" in connector.attached_databases

    def test_detach(self, connector, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        tmp_conn = duckdb.connect(db_path)
        tmp_conn.execute("CREATE TABLE t (id INT)")
        tmp_conn.close()

        connector.attach("testdb", db_path, db_type="duckdb")
        assert "testdb" in connector.attached_databases

        connector.detach("testdb")
        assert "testdb" not in connector.attached_databases

    def test_attached_databases_initially_empty(self, connector):
        assert connector.attached_databases == []


# ---------------------------------------------------------------------------
# Build Warehouse
# ---------------------------------------------------------------------------


class TestBuildWarehouse:
    def test_union_two_sources(self, connector, tmp_path):
        # Create two DuckDB files with the same table
        for name, vals in [("db1", "(1, [1.0,0.0], 1.0, 'text')"), ("db2", "(2, [0.0,1.0], 2.0, 'text')")]:
            path = str(tmp_path / f"{name}.duckdb")
            tmp = duckdb.connect(path)
            tmp.execute("""
                CREATE TABLE events (
                    sequence_id INT,
                    embedding FLOAT[],
                    timestamp DOUBLE,
                    modality VARCHAR
                )
            """)
            tmp.execute(f"INSERT INTO events VALUES {vals}")
            tmp.close()
            connector.attach(name, path, db_type="duckdb", read_only=True)

        counts = connector.build_warehouse(["events"])
        assert counts["events"] == 2

        # Verify sourcedb column exists
        rows = connector._ensure_open().execute("SELECT sourcedb FROM warehouse.events ORDER BY sourcedb").fetchall()
        sources = [r[0] for r in rows]
        assert "db1" in sources
        assert "db2" in sources

    def test_missing_table_returns_zero(self, connector, tmp_path):
        path = str(tmp_path / "db1.duckdb")
        tmp = duckdb.connect(path)
        tmp.execute("CREATE TABLE other (id INT)")
        tmp.close()
        connector.attach("db1", path, db_type="duckdb", read_only=True)

        counts = connector.build_warehouse(["nonexistent_table"])
        assert counts["nonexistent_table"] == 0

    def test_source_filter(self, connector, tmp_path):
        for name in ["db1", "db2"]:
            path = str(tmp_path / f"{name}.duckdb")
            tmp = duckdb.connect(path)
            tmp.execute("CREATE TABLE events (id INT)")
            tmp.execute("INSERT INTO events VALUES (1)")
            tmp.close()
            connector.attach(name, path, db_type="duckdb", read_only=True)

        counts = connector.build_warehouse(["events"], source_filter={"events": ["db1"]})
        assert counts["events"] == 1


# ---------------------------------------------------------------------------
# Load Sequences
# ---------------------------------------------------------------------------


class TestLoadSequences:
    def test_load_default_schema(self, populated_connector):
        seqs = populated_connector.load_sequences()
        assert set(seqs.keys()) == {"seq1", "seq2"}
        assert len(seqs["seq1"].embeddings) == 3
        assert len(seqs["seq2"].embeddings) == 2

    def test_load_custom_query(self, populated_connector):
        seqs = populated_connector.load_sequences(query="SELECT * FROM event_sequences WHERE sequence_id = 'seq1'")
        assert set(seqs.keys()) == {"seq1"}
        assert len(seqs["seq1"].embeddings) == 3

    def test_load_column_map(self, connector):
        conn = connector._ensure_open()
        conn.execute("""
            CREATE TABLE custom_events (
                sid VARCHAR,
                emb FLOAT[],
                ts DOUBLE,
                mod VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO custom_events VALUES
            ('s1', [1.0, 0.0], 1.0, 'audio'),
            ('s1', [0.0, 1.0], 2.0, 'audio')
        """)
        seqs = connector.load_sequences(
            "custom_events",
            column_map={
                "sequence_id": "sid",
                "embedding": "emb",
                "timestamp": "ts",
                "modality": "mod",
            },
        )
        assert "s1" in seqs
        assert seqs["s1"].modality == "audio"

    def test_load_empty_table(self, connector):
        conn = connector._ensure_open()
        conn.execute("""
            CREATE TABLE event_sequences (
                sequence_id VARCHAR,
                embedding FLOAT[],
                timestamp DOUBLE,
                modality VARCHAR
            )
        """)
        seqs = connector.load_sequences()
        assert seqs == {}

    def test_missing_columns_raises(self, connector):
        conn = connector._ensure_open()
        conn.execute("CREATE TABLE bad (x INT)")
        with pytest.raises(ValueError, match="Required columns not found"):
            connector.load_sequences("bad")


# ---------------------------------------------------------------------------
# Load Entities
# ---------------------------------------------------------------------------


class TestLoadEntities:
    def test_load_default_schema(self, entity_connector):
        entities = entity_connector.load_entities()
        assert set(entities.keys()) == {"e1", "e2", "e3"}
        assert set(entities["e1"].embeddings.keys()) == {"text", "image"}
        assert entities["e2"].hierarchy_info.get("category") == "electronics"

    def test_load_with_category(self, entity_connector):
        entities = entity_connector.load_entities()
        assert entities["e3"].hierarchy_info["category"] == "clothing"

    def test_load_empty_table(self, connector):
        conn = connector._ensure_open()
        conn.execute("""
            CREATE TABLE entities (
                entity_id VARCHAR,
                modality VARCHAR,
                embedding FLOAT[],
                category VARCHAR
            )
        """)
        entities = connector.load_entities()
        assert entities == {}


# ---------------------------------------------------------------------------
# Write Results
# ---------------------------------------------------------------------------


class TestWriteResults:
    def test_write_representations(self, connector):
        connector.write_representations({"seq1": [1.0, 2.0, 3.0]})
        rows = connector._ensure_open().execute("SELECT * FROM representations").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "seq1"
        assert list(rows[0][1]) == [1.0, 2.0, 3.0]

    def test_write_patterns(self, connector):
        connector.write_patterns({"seq1": [0, 2, 4]})
        rows = connector._ensure_open().execute("SELECT * FROM patterns").fetchall()
        assert len(rows) == 1
        assert list(rows[0][1]) == [0, 2, 4]

    def test_write_predictions(self, connector):
        connector.write_predictions({"seq1": [[1.0, 0.0], [0.0, 1.0]]})
        rows = connector._ensure_open().execute("SELECT * FROM predictions ORDER BY step").fetchall()
        assert len(rows) == 2
        assert rows[0][1] == 1  # step
        assert rows[1][1] == 2

    def test_write_relationships(self, connector):
        connector.write_relationships({"e1": ["e2", "e3"]})
        rows = connector._ensure_open().execute("SELECT * FROM relationships").fetchall()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_sequences_only(self, populated_connector):
        result = populated_connector.run_pipeline(write_results=False)
        assert set(result.keys()) == {"representations", "patterns", "predictions", "relationships"}
        assert "seq1" in result["representations"]
        assert "seq2" in result["representations"]
        assert result["relationships"] is None

    def test_with_entities(self, connector):
        conn = connector._ensure_open()
        # Create sequences
        conn.execute("""
            CREATE TABLE event_sequences (
                sequence_id VARCHAR, embedding FLOAT[], timestamp DOUBLE, modality VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO event_sequences VALUES
            ('s1', [1.0, 0.0, 0.0], 1.0, 'text'),
            ('s1', [0.0, 1.0, 0.0], 2.0, 'text')
        """)
        # Create entities
        conn.execute("""
            CREATE TABLE entities (
                entity_id VARCHAR, modality VARCHAR, embedding FLOAT[], category VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO entities VALUES
            ('e1', 'text', [1.0, 0.0, 0.0], 'cat1'),
            ('e2', 'text', [0.9, 0.1, 0.0], 'cat1')
        """)
        result = connector.run_pipeline(entities_table="entities", write_results=False)
        assert result["relationships"] is not None

    def test_write_results_creates_tables(self, populated_connector):
        populated_connector.run_pipeline(write_results=True)
        conn = populated_connector._ensure_open()
        # Verify tables exist and have data
        rep_count = conn.execute("SELECT COUNT(*) FROM representations").fetchone()[0]
        pat_count = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert rep_count > 0
        assert pat_count > 0
        assert pred_count > 0

    def test_no_write(self, populated_connector):
        populated_connector.run_pipeline(write_results=False)
        conn = populated_connector._ensure_open()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        assert "representations" not in tables
        assert "patterns" not in tables
        assert "predictions" not in tables


# ---------------------------------------------------------------------------
# End-to-end: run_from_sources
# ---------------------------------------------------------------------------


class TestRunFromSources:
    def test_end_to_end_two_duckdb_sources(self, connector, tmp_path):
        # Create two DuckDB files with event data
        for name, vals in [
            ("src1", "('s1', [1.0,0.0,0.0], 1.0, 'text'), ('s1', [0.0,1.0,0.0], 2.0, 'text')"),
            ("src2", "('s2', [0.5,0.5,0.0], 1.0, 'text'), ('s2', [0.0,0.5,0.5], 2.0, 'text')"),
        ]:
            path = str(tmp_path / f"{name}.duckdb")
            tmp = duckdb.connect(path)
            tmp.execute("""
                CREATE TABLE event_sequences (
                    sequence_id VARCHAR, embedding FLOAT[], timestamp DOUBLE, modality VARCHAR
                )
            """)
            tmp.execute(f"INSERT INTO event_sequences VALUES {vals}")
            tmp.close()

        result = connector.run_from_sources(
            sources=[
                {"name": "src1", "connection_string": str(tmp_path / "src1.duckdb")},
                {"name": "src2", "connection_string": str(tmp_path / "src2.duckdb")},
            ],
            tables=["event_sequences"],
            write_results=False,
        )

        assert "warehouse_row_counts" in result
        assert result["warehouse_row_counts"]["event_sequences"] == 4
        assert "s1" in result["representations"]
        assert "s2" in result["representations"]
