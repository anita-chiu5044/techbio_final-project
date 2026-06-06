# Agent Pipeline Smoke Test

Date: 2026-06-06

Purpose: connect the current placeholder classifier checkpoint to the local agent database/reporting flow.

## Current Placeholder Classifier

Use the original stable baseline checkpoint until classifier research finishes:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/ce_uniform/best.pth
```

Classifier entrypoint:

```bash
python checkpoints_classifier/classifier_inference.py \
  --image /path/to/clean_patch_or_folder \
  --ckpt /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/ce_uniform/best.pth \
  --logit-adjustment \
  --topk 5 \
  --output classifier_results.json \
  --format json
```

## Agent Bridge Script

The bridge script runs classifier inference, ingests the JSON records, updates the agent DB, summarizes the case, and writes a report:

```bash
python scripts/run_classifier_agent_pipeline.py \
  --image /path/to/clean_patch_or_folder \
  --ckpt /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/ce_uniform/best.pth \
  --logit-adjustment \
  --db /home/yucheng/Desktop/techbio_pipeline_output/agent_pipeline_smoke/ymca_agent.db \
  --output-dir /home/yucheng/Desktop/techbio_pipeline_output/agent_pipeline_smoke \
  --case-id case_agent_pipeline_smoke \
  --conversation-id conv_agent_pipeline_smoke \
  --user-id local_user \
  --topk 5
```

The script performs:

```text
classifier_inference.py
  -> classifier JSON records
  -> AgentTools.apply_classifier_result()
  -> AgentTools.summarize_case()
  -> AgentTools.generate_case_report()
```

If a real YOLO/MedSAM cell already exists in DB, the bridge tries to match by `clean_patch_path`. For explicit mapping, pass:

```bash
--cell-map-csv cell_map.csv
```

where `cell_map.csv` has:

```csv
image,cell_id
/path/to/clean_patch.png,det_000001
```

If no DB cell exists, the bridge creates a minimal smoke-test cell with segmentation marked OK. This is only for integration smoke testing.

## Verified Smoke Run

Command tested:

```bash
python scripts/run_classifier_agent_pipeline.py \
  --image /home/yucheng/Desktop/techbio_pipeline_output/medsam_output/granulocyte_immature/MYB/MYB_0001_mask.png \
  --ckpt /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/ce_uniform/best.pth \
  --logit-adjustment \
  --db /tmp/ymca_agent_pipeline_smoke2.db \
  --output-dir /tmp/ymca_agent_pipeline_smoke2 \
  --case-id case_smoke2 \
  --conversation-id conv_smoke2 \
  --user-id user_smoke \
  --topk 5
```

Classifier output:

```text
MYB_0001_mask.png
#1 MYB 0.9629
#2 MMZ 0.0204
#3 BAS 0.0114
#4 PMO 0.0019
#5 PMB 0.0016
```

Agent output:

```text
record_count: 1
applied: 1
review_needed_count: 1
hard_counts: {}
model_counts_raw: {"MYB": 1}
```

`MYB` remains in the review queue by design because rare / immature / high-risk classes should be confirmed by the clinical user.

Generated files:

```text
/tmp/ymca_agent_pipeline_smoke2/classifier_results.json
/tmp/ymca_agent_pipeline_smoke2/agent_pipeline_summary.json
/tmp/ymca_agent_pipeline_smoke2/agent_report.txt
/tmp/ymca_agent_pipeline_smoke2.db
```

## Checkpoint Swap

When the classifier team provides a better checkpoint, only replace `--ckpt`. The downstream contract is unchanged as long as `classifier_inference.py` still emits:

```json
{
  "image": "...",
  "top1_class": "...",
  "top1_prob": 0.0,
  "predictions": [
    {"rank": 1, "class": "...", "probability": 0.0}
  ]
}
```
