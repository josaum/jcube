# Investigation: Nature of Data in `data/`

## Summary
`/Users/josaum/projects/jcube/data` is not generic model input. It is a three-stage healthcare graph-training dataset: a 6.26 GB DuckDB operational warehouse, a 5.38 MB catalog that describes 411 of its tables, and an 882 MB derived temporal edge list used by the JEPA pipeline. The current `jcube_graph.parquet` is older than the March 23, 2026 `source_db`-prefix fix and therefore merges instance IDs across hospital systems, which materially corrupts graph identity.

## Symptoms
- The user asked to truly understand the actual data under `data/`, not just pipeline comments.
- `event_jepa_cube/scale_pipeline.py` claims the data is a healthcare temporal knowledge graph with ontology and instance nodes.
- The current code prefixes instance IDs with `source_db/`, but sampled rows from the existing `jcube_graph.parquet` do not.

## Investigation Log

### Phase 1 - Initial Assessment
**Hypothesis:** The `data/` directory contains a source healthcare warehouse, a schema catalog, and a graph artifact derived from them for JEPA training.  
**Findings:** The active pipeline file explicitly describes that flow and distinguishes ontology nodes from instance nodes.  
**Evidence:** `event_jepa_cube/scale_pipeline.py:1`, `event_jepa_cube/scale_pipeline.py:14`, `event_jepa_cube/scale_pipeline.py:18`.  
**Conclusion:** The code intends a DB -> catalog-driven graph -> embedding-training pipeline.

### Phase 2 - Real Artifact Inspection
**Hypothesis:** The files on disk match the intended three-stage pipeline.  
**Findings:** The directory contains:
- `data/aggregated_fixed_union.db` at `6,256,078,848` bytes, mtime `2026-03-16T20:42:38`
- `data/ai_friendly_catalog.json` at `5,376,001` bytes, mtime `2026-03-22T16:19:32`
- `data/jcube_graph.parquet` at `882,443,164` bytes, mtime `2026-03-23T00:09:05`
- `data/weights/node_emb_epoch_2.pt` at `4,452,900,508` bytes, mtime `2026-03-23T00:10:37`

The catalog reports `total_tables=411`, `total_records=46,668,764`, and `healthcare_systems_count=46`. Its healthcare terminology map includes `TUSS`, `CID`, `INTERNACAO`, `GLOSA`, `UTI`, `LEITO`, `PACIENTE`, and related Brazilian healthcare terms.  
**Evidence:** Direct DuckDB / JSON / Parquet inspection via local Python venv on March 23, 2026.  
**Conclusion:** The files do represent a healthcare operational warehouse plus derived training artifacts.

### Phase 3 - Source Warehouse Characterization
**Hypothesis:** `aggregated_fixed_union.db` is a multi-system operational warehouse, not a single normalized application DB.  
**Findings:** The DuckDB file contains `417` tables. Table names and schemas span audit, CRM, billing, tickets, capta, beneficiary, and hospitalization domains. Examples:
- `agg_tb_tickets_x_setor_tixs` with `ID_CD_TICKETS`, `ID_CD_SETOR`, `ID_CD_LOGIN`, timestamps, and `source_db`
- `agg_tb_fatura_itens_fait` with invoice, budget, product, quantity, and value fields
- `agg_tb_capta_produtividade_medica_prme` with questionnaire, response, admission, login, and evolution identifiers

Meaningful `source_db` values are present in the warehouse, including `PASA`, `GHO-PETROBRAS`, `GHO-CASSI`, `GHO-CUIABA`, `GHO-REALGRANDEZA`, and `GOHOSP-CNU`.  
**Evidence:** Direct inspection of top-row-count catalog tables and SQL grouping on `source_db`.  
**Conclusion:** This is a federated warehouse that unions operational data from many healthcare systems or hospitals into common `agg_tb_*` tables with a `source_db` discriminator.

### Phase 4 - Catalog Coverage
**Hypothesis:** The catalog fully controls graph materialization and may omit real DB tables.  
**Findings:** The database has `417` tables, but the catalog covers only `411`. Six DB tables are absent from the catalog:
- `agg_tb_capta_paciente_ps`
- `agg_tb_crm_contatos_crco`
- `agg_tb_crm_pacientes_crpc`
- `agg_tb_crm_pessoas_crmp`
- `agg_tb_fatura_internacao_sem_gho_telecontatos_fist`
- `agg_tb_formulario_rah_completo_frco`

All six satisfy the structural graph rule used by the pipeline: they each have at least two `ID_CD_*` columns and at least one timestamp/date column.  
**Evidence:** The graph query builder iterates only `catalog["tables"]` in `event_jepa_cube/scale_pipeline.py:90` and filters for `len(id_cols) >= 2` and timestamps at `event_jepa_cube/scale_pipeline.py:102`. The six extra tables were profiled directly from DuckDB.  
**Conclusion:** The graph is catalog-driven, not schema-driven, and the current materialization necessarily ignores six edge-rich real tables.

### Phase 5 - Derived Graph Characterization
**Hypothesis:** `jcube_graph.parquet` is a temporal edge list, not feature tensors or tabular training rows.  
**Findings:** The Parquet schema is exactly:
- `subject_id STRING`
- `predicate STRING`
- `object_id STRING`
- `t_epoch DOUBLE`

The file contains `165,593,307` rows, `166` row groups, and `1,041` distinct predicates. Sample rows look like:
- `('ID_CD_ARQUIVO_1', 'HAS_HOSPITAL_VIA_agg_tb_arquivos_relatorio_auditoria_rah_arar', 'ID_CD_HOSPITAL_114', 1753956942.53)`
- `('ID_CD_ARQUIVO_2', 'HAS_HOSPITAL_VIA_agg_tb_arquivos_relatorio_auditoria_rah_arar', 'ID_CD_HOSPITAL_114', 1753956969.48)`

Top predicates include `HAS_SETOR_VIA_agg_tb_tickets_x_setor_tixs`, `HAS_LOGIN_VIA_agg_tb_tickets_x_setor_tixs`, `HAS_FATURA_VIA_agg_tb_fatura_itens_fait`, `HAS_INTERNACAO_VIA_agg_tb_fatura_itens_fait`, and `HAS_TIPO_PERGUNTA_VIA_agg_tb_capta_produtividade_medica_prme`.  
**Evidence:** Direct Parquet metadata and SQL queries over `read_parquet('data/jcube_graph.parquet')`.  
**Conclusion:** The graph artifact is a time-stamped relation graph synthesized from operational tables by turning `ID_CD_*` co-occurrence into `(subject, predicate, object, time)` edges.

### Phase 6 - Intended Semantics in Code
**Hypothesis:** The code intends a split between global ontology anchors and hospital-scoped operational entities.  
**Findings:** The pipeline explicitly separates:
- Ontology nodes such as CID, TUSS, medications, and categories, encoded once with Qwen and then frozen
- Instance nodes such as patients, admissions, invoices, beds, and hospitals, learned through `nn.Embedding`

The materializer prefixes instance IDs with `source_db/` while leaving ontology IDs global:
- local builder: `event_jepa_cube/scale_pipeline.py:122`, `event_jepa_cube/scale_pipeline.py:128`
- remote builder: `event_jepa_cube/scale_pipeline.py:395`, `event_jepa_cube/scale_pipeline.py:401`

Ontology extraction uses catalog heuristics: one `ID_CD_*` column, descriptive `NM_` / `DS_` columns, `row_count <= 50000`, at least three FK references, and entity type not in `INSTANCE_TYPES`.  
**Evidence:** `event_jepa_cube/scale_pipeline.py:5`, `event_jepa_cube/scale_pipeline.py:9`, `event_jepa_cube/scale_pipeline.py:122`, `event_jepa_cube/scale_pipeline.py:128`, `event_jepa_cube/scale_pipeline.py:455`, `event_jepa_cube/scale_pipeline.py:470`, `event_jepa_cube/scale_pipeline.py:474`, `event_jepa_cube/scale_pipeline.py:482`.  
**Conclusion:** The intended data model is:
- global ontology identities for shared medical meaning
- per-source instance identities for operational entities that would otherwise collide across hospitals

### Phase 7 - Artifact/Code Mismatch
**Hypothesis:** The graph artifact predates the `source_db` prefix logic now present in code.  
**Findings:** The current `jcube_graph.parquet` has zero node IDs containing `/`. Git history shows:
- `8036340` at March 23, 2026 00:02:33 -0300: initial full-scale pipeline commit
- `8a379fd` at March 23, 2026 01:16:48 -0300: `fix: scope instance node IDs by source_db to prevent cross-hospital collisions`

The existing parquet file timestamp is March 23, 2026 00:09:05, which is about 67 minutes before the prefix fix commit.  
**Evidence:** Git log and blame on `event_jepa_cube/scale_pipeline.py` around lines `122-129`, plus file mtime on `data/jcube_graph.parquet`.  
**Conclusion:** The graph artifact on disk was generated before the current source-scoping fix and therefore does not implement the identity semantics the code now expects.

### Phase 8 - Collision Verification
**Hypothesis:** Missing `source_db` prefixes cause real cross-hospital identity collisions, not a merely theoretical risk.  
**Findings:** The same raw IDs appear across multiple `source_db` values in source tables:
- `103,496` `ID_CD_FATURA` values collide across more than one `source_db` in `agg_tb_fatura_itens_fait`
- `72,968` `ID_CD_ORCAMENTO` values collide across more than one `source_db` in `agg_tb_fatura_itens_fait`
- `184,631` `ID_CD_INTERNACAO` values collide across more than one `source_db` in `agg_tb_capta_evo_status_caes`
- `13,864` `ID_CD_TICKETS` values collide across more than one `source_db` in `agg_tb_tickets_x_setor_tixs`
- `998` `ID_CD_LOGIN` values collide across more than one `source_db` in `agg_tb_login_log_llog`

Sample: `ID_CD_FATURA = 5` appears in `21` distinct source systems, including `PASA`, `GHO-PETROBRAS`, `GHO-CASSI`, `GHO-CUIABA`, and `GOHOSP-CNU`.  
**Evidence:** Direct SQL grouping on warehouse tables using `COUNT(DISTINCT source_db) > 1`.  
**Conclusion:** The current graph merges unrelated operational entities into shared node IDs. This is structural corruption of graph identity, not just a naming preference issue.

### Phase 9 - Downstream Consumer Risk
**Hypothesis:** Different consumers will react differently once the graph is regenerated with prefixed instance IDs.  
**Findings:**
- `event_jepa_cube/jcube_bridge.py:69` correctly parses both `GHO-BRADESCO/ID_CD_INTERNACAO_117926` and `ID_CD_CID_A419`.
- `event_jepa_cube/jcube_bridge.py:180` filters by `source_db`, so the current unprefixed graph makes hospital-scoped filtering ineffective.
- `event_jepa_cube/jcube_bridge.py:195` stores a whole batch with the `filter_tag` from the last parsed item in that batch.
- `event_jepa_cube/exploit_twin.py:102` extracts entity type by splitting directly on `_`, which breaks for prefixed instance IDs.
- `event_jepa_cube/exploit_twin.py:310` builds lookup keys like `ID_CD_INTERNACAO_{eid}`, which will not match prefixed graph IDs after regeneration.
- `event_jepa_cube/graph_loader.py:65` and `event_jepa_cube/graph_loader.py:121` operate directly on DuckDB columns and raw IDs, so they are not dependent on graph node string format.

**Evidence:** `event_jepa_cube/jcube_bridge.py:69`, `event_jepa_cube/jcube_bridge.py:180`, `event_jepa_cube/jcube_bridge.py:195`, `event_jepa_cube/exploit_twin.py:102`, `event_jepa_cube/exploit_twin.py:310`, `event_jepa_cube/graph_loader.py:65`, `event_jepa_cube/graph_loader.py:121`.  
**Conclusion:** Regenerating the graph with correct source-scoped instance IDs is necessary, but `exploit_twin.py` must be updated first or in the same change set. `jcube_bridge.py` already understands the new ID format but still has filtering semantics to revisit.

## Root Cause
The current graph artifact was generated from a federated healthcare warehouse before the March 23, 2026 `source_db`-scoping fix landed in `event_jepa_cube/scale_pipeline.py`. As a result, instance node IDs such as admissions, invoices, tickets, and logins are represented only as `ID_CD_*_<raw_id>` instead of `{source_db}/ID_CD_*_<raw_id>`. Because the source warehouse unions records from at least 46 healthcare systems and reuses the same raw IDs across them, the graph collapses unrelated hospital entities into single nodes. The catalog also omits six real DB tables that structurally qualify for edge generation, so the graph is both identity-corrupted and incomplete.

## Eliminated Hypotheses
- The graph artifact already includes `source_db`-prefixed instance IDs. Eliminated by direct scan: zero `/` in any node ID.
- The graph is built by introspecting the live DB schema. Eliminated by the catalog-only iteration in `scale_pipeline.py`.
- `graph_loader.py` depends on parquet node ID string format. Eliminated by direct code inspection; it queries DuckDB using raw DB IDs.

## Open Questions
- How many additional edges and predicates will the six missing catalog tables add once cataloged?
- Are there any ontology-like `ID_CD_*` entity types not currently covered correctly by `INSTANCE_TYPES`?
- The invalidation of `node_emb_epoch_2.pt` is a strong inference from the collision evidence and the forthcoming node-vocabulary change, but the report did not rerun training to empirically compare old versus regenerated embeddings.

## Recommendations
1. Update `data/ai_friendly_catalog.json` to include the six missing DB tables before regenerating graph artifacts.
2. Regenerate `data/jcube_graph.parquet` from the post-`8a379fd` materializer so instance IDs are scoped by `source_db`.
3. Update `event_jepa_cube/exploit_twin.py` to parse optional `source_db/` prefixes before splitting on `_`, and to resolve DB entity IDs against full graph node IDs instead of concatenating `ID_CD_*_` strings directly.
4. Revisit `event_jepa_cube/jcube_bridge.py` batching so `filter_tag` is not inherited from whichever node happened to be last in the current batch.
5. Treat `data/weights/node_emb_epoch_2.pt` as a pre-fix baseline artifact, not a trustworthy production embedding set, unless it is proven to have been trained on a regenerated source-scoped graph.

## Preventive Measures
- Add a validation step that fails materialization if any `INSTANCE_TYPES` node ID in the output graph lacks a `/`.
- Add a validation step that compares live DB table count with catalog table count and reports uncataloged edge-eligible tables.
- Persist graph metadata alongside weights: graph file hash, node count, predicate count, materializer git commit, and whether source-scoping was enabled.
