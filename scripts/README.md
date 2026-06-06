# Scripts

Reusable repo-level scripts for integration and analysis.

## `analyze_pipeline_outputs.py`

Analyzes YOLO and MedSAM outputs and writes:

```text
docs/analysis/pipeline_output_analysis.md
docs/analysis/pipeline_output_analysis.json
```

Example:

```bash
python scripts/analyze_pipeline_outputs.py   --yolo-dir /home/yucheng/Desktop/techbio_pipeline_output/yolo   --medsam-dir /home/yucheng/Desktop/techbio_pipeline_output/medsam_output   --out-dir docs/analysis
```

The script reports YOLO confidence/bbox statistics, WBC overlap, MedSAM status by class, mask coverage, edge-touch ratio, and NO_DETECTION examples.
