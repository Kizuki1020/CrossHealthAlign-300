# Model Outputs

## File Format

Each JSON file contains the raw outputs from one LLM evaluated on
CrossHealthAlign-300 under a unified protocol.

### Structure

```json
{
  "model": "model-identifier",
  "timestamp": "ISO 8601 datetime",
  "results": [
    {
      "item_id": "CAI-001",
      "pred_label": "EQUIVALENT|RELATED_NOT_EQUIVALENT|NO_JUSTIFIED_EQUIVALENT|CONTEXT_DEPENDENT",
      "pred_candidate": "c1|c2|c3|c4|NONE",
      "confidence": 0.95,
      "justification": "Brief reasoning for the classification decision"
    }
  ]
}
```

### Fields

- `pred_label`: The model's predicted relationship category. Maps to the four
  permitted output labels defined in the system prompt.
- `pred_candidate`: The selected candidate (c1-c4) or "NONE" for rejection.
- `confidence`: Self-reported confidence score (0-1 scale).
- `justification`: Natural-language explanation of the model's reasoning.

## Evaluation Protocol

All models received:
- Identical 300 items in identical order
- Same system prompt and few-shot examples (manuscript Methods; prompt files not included in this data-only deposit)
- Same output template
- Same candidate ordering (frozen before evaluation)

## Models

| File | Model | Provider |
|------|-------|----------|
| gpt-5.5.json | GPT-5.5 | OpenAI |
| claude-sonnet-4-6.json | Claude-Sonnet-4.6 | Anthropic |
| deepseek-v3.2.json | DeepSeek-V3.2 | DeepSeek |

## Note

A fourth model (Gemini-3.5-Flash) was evaluated but excluded from analysis
due to systematic parse errors (287/300 outputs unparseable). Its outputs
are not included in this package.

## Gold Labels

Ground-truth labels for scoring are in ../01_Benchmark/expert_labels.json.
