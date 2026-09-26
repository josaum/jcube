"""Tests for MaterializationResult -> Mycelia payload conversion."""

from event_jepa_cube.materializer import (
    EntityTimeline,
    MaterializationResult,
    result_to_mycelia_payloads,
)
from event_jepa_cube.sequence import EventSequence


def test_result_to_mycelia_payloads_builds_rich_rows():
    timeline = EntityTimeline(
        entity_id="internacao-1",
        entity_type="INTERNACAO",
        sequence=EventSequence(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            timestamps=[1.0, 2.0],
            modality="db_row",
        ),
        source_tables=["agg_tb_capta_internacao", "agg_tb_capta_auditoria"],
        event_count=2,
        time_span_days=1.25,
    )
    result = MaterializationResult(
        entity_type="INTERNACAO",
        tables_scanned=2,
        entities_found=1,
        total_events=2,
        embedding_dim=2,
        timelines={"internacao-1": timeline},
        representations={"internacao-1": [0.9, 0.1]},
        patterns={"internacao-1": [0]},
        predictions={"internacao-1": [[0.8, 0.2], [0.7, 0.3]]},
    )

    payload = result_to_mycelia_payloads(
        result,
        tenant_id="workspace",
        repo="mycelia-workspace",
        rev="43f5e23",
    )

    assert payload["scope"] == {
        "tenant_id": "workspace",
        "repo": "mycelia-workspace",
        "rev": "43f5e23",
    }
    assert payload["representations"]["internacao-1"] == [0.9, 0.1]
    assert payload["representation_metadata"]["internacao-1"]["entity_type"] == "INTERNACAO"
    assert payload["representation_metadata"]["internacao-1"]["salient_dimensions"] == [0]
    assert "entity:internacao-1" in payload["representation_relations"]["internacao-1"]
    assert "table:agg_tb_capta_internacao" in payload["representation_relations"]["internacao-1"]
    assert payload["representation_text"]["internacao-1"].startswith("representation entity=internacao-1")

    assert payload["predictions"]["internacao-1"] == [[0.8, 0.2], [0.7, 0.3]]
    assert payload["prediction_metadata"]["internacao-1_step_1"]["step"] == 1
    assert payload["prediction_metadata"]["internacao-1_step_2"]["prediction_dim"] == 2
    assert "step:1" in payload["prediction_relations"]["internacao-1_step_1"]
    assert payload["prediction_text"]["internacao-1_step_2"].startswith("prediction entity=internacao-1")


def test_result_to_mycelia_payloads_keeps_empty_scope_and_missing_outputs_stable():
    timeline = EntityTimeline(
        entity_id="patient-1",
        entity_type="PACIENTE",
        sequence=EventSequence(embeddings=[[1.0]], timestamps=[1.0], modality="db_row"),
        source_tables=["agg_tb_crm_paciente"],
        event_count=1,
        time_span_days=0.0,
    )
    result = MaterializationResult(
        entity_type="PACIENTE",
        tables_scanned=1,
        entities_found=1,
        total_events=1,
        embedding_dim=1,
        timelines={"patient-1": timeline},
    )

    payload = result_to_mycelia_payloads(result)

    assert payload["scope"] == {}
    assert payload["representations"] == {}
    assert payload["predictions"] == {}
    assert payload["representation_metadata"] == {}
    assert payload["prediction_metadata"] == {}
