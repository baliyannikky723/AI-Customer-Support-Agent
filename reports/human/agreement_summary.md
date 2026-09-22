# Human Evaluation & Inter-Rater Agreement Status

> [!IMPORTANT]
> **Status**: `HUMAN AGREEMENT NOT YET MEASURED`
> 
> In strict adherence to scientific integrity rules, **we do NOT fabricate human scores or synthetic agreement statistics**.
> The complete 50-example stratified human evaluation package has been generated and isolated.

## 1. Human Annotation Package Deliverables

- **Stratified Sample (50 cases)**: [`data/evaluation/human_eval_set.jsonl`](file:///e:/AI-Customer-Support-Agent/data/evaluation/human_eval_set.jsonl)
- **Annotation Template**: [`data/evaluation/human_annotations_template.jsonl`](file:///e:/AI-Customer-Support-Agent/data/evaluation/human_annotations_template.jsonl)
- **Reviewer Rubric & Guidelines**: [`reports/human/annotation_guidelines.md`](file:///e:/AI-Customer-Support-Agent/reports/human/annotation_guidelines.md)

## 2. Reproduction Steps for Importing Human Reviews

```bash
# 1. Complete annotations in data/evaluation/human_annotations_template.jsonl
# 2. Save file as data/evaluation/human_annotations.jsonl
# 3. Execute agreement calculation
python scripts/analyze_judge_human_agreement.py
```
