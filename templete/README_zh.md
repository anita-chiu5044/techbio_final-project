# 報告撰寫指引工作區

本資料夾是固定放置「領域同仁已核准、可供 Qwen 撰寫報告或回答個案層級問題時遵守」的結構化指引。

請只放入結構化、專案核准後的內容。不要將完整原始 SOP、PDF、病歷、報告影本或其他敏感文件直接放進 prompt。

## 建議檔案

```text
report_template_zh.md
allowed_phrases_zh.md
prohibited_claims_zh.md
review_triggers_zh.yaml
critical_flags_zh.yaml
cell_abbreviation_canonical_map_zh.md
qc_review_template_zh.md
source_notes_zh.md
```

## 領域同仁建議提供的內容

```text
- 已去識別化的報告句型或報告模板
- 可接受的形態學描述與建議用語
- 禁止使用的診斷過度推論句型
- 需要緊急複核或升級處理的 trigger，例如疑似 APL 相關用語
- QC review checklist 的標準文字
- 細胞縮寫與正式名稱對照表
- 本院、台大醫院或地方實驗室 SOP 討論後的摘要筆記
```

## 使用規則

```text
若來源文件篇幅很長或含敏感資訊，請先由領域同仁摘要成下列結構化檔案之一，
原始來源文件應存放於受控位置，不直接提供給 Qwen 或放入 prompt。
```

## 關於「參照台大醫院格式」

本工作區可放置「參照台大醫院或本院臨床報告習慣」整理出的中文模板，但除非已有台大醫院公開文件或專案內部核准來源，否則不得宣稱任何內容為台大醫院官方格式。

建議標示方式：

```text
格式狀態：參照本院/台大臨床報告習慣整理，待領域同仁核准。
```
