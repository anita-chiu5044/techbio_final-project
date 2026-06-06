# Full Pipeline User Flow

Date: 2026-06-06

Goal: start from user-uploaded full image(s), run all local modules, store results in the agent DB, and support QA/review.

## What The User Eventually Does

Clinical/reviewer-facing flow:

```text
1. Log in / choose review session
2. Upload one smear image or a folder of cell/ROI images
3. Backend runs YOLO -> MedSAM -> classifier
4. Agent opens the case summary
5. User asks QA questions or reviews flagged cells
6. User accepts/corrects/excludes/unclassifiable cells
7. Agent regenerates summary/report from reviewed DB state
```

The current repository does not yet provide a polished web/chat UI. The backend pipeline and QA tools are now connected through CLI scripts.

## Full Backend Pipeline Command

```bash
cd /home/yucheng/Desktop/techbio/techbio_final-project

python scripts/run_full_agent_pipeline.py \
  --input /path/to/user_uploaded_image_or_folder \
  --session-id case_001 \
  --user-id user_001 \
  --yolo-model best.pt \
  --medsam-config MedSAM3/configs/lisc_lora_config.yaml \
  --medsam3-dir MedSAM3 \
  --classifier-ckpt /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/ce_uniform/best.pth \
  --logit-adjustment \
  --output-root /home/yucheng/Desktop/techbio_pipeline_output/full_agent_sessions
```

Output folder:

```text
/home/yucheng/Desktop/techbio_pipeline_output/full_agent_sessions/case_001
```

Main outputs:

```text
01_yolo/detections.jsonl
01_yolo/images.jsonl
01_yolo/summary.json
02_medsam_input/
03_medsam_output/inference_summary.csv
03_medsam_output/**/**_mask.png
04_classifier/classifier_results.json
04_classifier/agent_pipeline_summary.json
04_classifier/agent_report.txt
cell_map.csv
ymca_agent.db
agent_pipeline_summary.json
agent_report.txt
```

## Stage Meaning

```text
YOLO:
  Detects coarse cells: RBC / WBC / Platelets.
  Only WBC detections become downstream candidates.

ROI builder:
  Converts every WBC bounding box into context-padded square TIFF input.
  For real user sessions use all WBC detections, not top-1 only.

MedSAM:
  Produces clean masked cell patches with transparent/white background.
  Writes inference_summary.csv with OK / FAIL / NO_DETECTION status.

Classifier:
  Uses ce_uniform/best.pth as placeholder.
  Command uses --logit-adjustment.
  Checkpoint can be replaced later without changing downstream DB contract.

Agent DB:
  Stores model_label separately from review_label.
  Does not overwrite raw model outputs during human review.

QA:
  Reads the same DB and calls AgentTools methods.
```

## QA Commands

After full pipeline finishes:

```bash
DB=/home/yucheng/Desktop/techbio_pipeline_output/full_agent_sessions/case_001/ymca_agent.db
CASE=case_001
```

Ask for case summary:

```bash
python scripts/qa_agent_cli.py --db "$DB" --case-id "$CASE" --question summary
```

Generate/report current draft:

```bash
python scripts/qa_agent_cli.py --db "$DB" --case-id "$CASE" --question report
```

List uncertain/review-needed cells:

```bash
python scripts/qa_agent_cli.py --db "$DB" --case-id "$CASE" --question uncertain
```

Inspect one cell:

```bash
python scripts/qa_agent_cli.py --db "$DB" --case-id "$CASE" --question "cell det_000001"
```

Accept model label after user review:

```bash
python scripts/qa_agent_cli.py \
  --db "$DB" \
  --case-id "$CASE" \
  --action accept \
  --cell-id det_000001 \
  --reviewer-id clinician_001
```

Correct label after user review:

```bash
python scripts/qa_agent_cli.py \
  --db "$DB" \
  --case-id "$CASE" \
  --action correct \
  --cell-id det_000001 \
  --label LYT \
  --reviewer-id clinician_001
```

Exclude unusable cell:

```bash
python scripts/qa_agent_cli.py \
  --db "$DB" \
  --case-id "$CASE" \
  --action exclude \
  --cell-id det_000001 \
  --note "overlap / poor segmentation"
```


## Conversational Human Review

The clinical user should not need to type low-level actions. The intended chat behavior is:

```text
User: 把 det_000001 改成 LYT
Agent tool call: update_cell_review(cell_id="det_000001", review_label="LYT", review_status="corrected")

User: 接受 det_000002
Agent tool call: update_cell_review(cell_id="det_000002", review_status="accepted_model_label")

User: det_000003 不要用，重疊太嚴重
Agent tool call: update_cell_review(cell_id="det_000003", review_status="excluded", note="重疊太嚴重")

User: det_000004 無法分類
Agent tool call: update_cell_review(cell_id="det_000004", review_status="unclassifiable")
```

The temporary CLI already supports explicit conversational shortcuts:

```bash
python scripts/qa_agent_cli.py   --db "$DB"   --case-id "$CASE"   --question "把 det_000001 改成 LYT"   --reviewer-id clinician_001
```

The DB keeps both fields:

```text
model_label = original classifier output, immutable by review
review_label = human correction, used by summarize_case when available
```

Verified smoke behavior:

```text
Input: 把 cell_00001_MYB_0001_mask 改成 LYT
Before: model_label=MYB, review_label=None, review_status=unreviewed
After:  model_label=MYB, review_label=LYT, review_status=corrected
Summary: hard_counts={"LYT": 1}, model_counts_raw={"MYB": 1}
```

## Resume / Debug

Run only early stages as a dry-run:

```bash
python scripts/run_full_agent_pipeline.py \
  --input /path/to/image.png \
  --session-id dryrun_demo \
  --dry-run \
  --limit 1 \
  --medsam-max-images 1
```

Resume from a stage if outputs already exist:

```bash
python scripts/run_full_agent_pipeline.py \
  --input /path/to/image_or_folder \
  --session-id case_001 \
  --start-at classifier
```

Stop after MedSAM if classifier should be run later:

```bash
python scripts/run_full_agent_pipeline.py \
  --input /path/to/image_or_folder \
  --session-id case_001 \
  --stop-after medsam
```

## Important Caveats

```text
1. MedSAM can be slow; use --medsam-max-images for smoke tests.
2. YOLO low confidence and high-overlap detections should remain review-required.
3. Rare / immature / high-risk classifier outputs should remain review-required.
4. Current QA CLI is deterministic. The future local Qwen agent should call the same DB tools.
5. This MVP is morphology-review support, not complete WBC differential, true blast percentage, or final diagnosis.
```
