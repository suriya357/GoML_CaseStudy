# NOTES.md

## Model

I used the local `qwen2.5:3b` model through Ollama.

No hosted LLM API was used.

To run the model locally:

```bash
ollama pull qwen2.5:3b
pip install -r requirements.txt
python extract.py --inbox messages --output tasks.json
```

The script sends each sanitized message to the local model and performs
deterministic validation in Python after the model response.

## Tokens

For the practice folder,the run1 and run2 used:

- Prompt tokens sent: **4635**
- Output tokens received: **513**

These values come from Ollama's response metadata:

- `prompt_eval_count` → prompt/input tokens
- `eval_count` → generated/output tokens

The totals are accumulated across all messages, including any repair retries.

## Prompt

The system prompt contains:

- The task-extraction objective.
- A definition of what counts as a task.
- Rules for extracting `task`, `owner`, and `due`.
- Examples of task and non-task messages.
- Rules not to invent tasks, names, or deadlines.
- Evidence fields to support extracted values.
- A requirement to return JSON only.
- Instructions to treat the message as untrusted data.

For a normal extraction, the model receives the system prompt and the
sanitized message body.

I do not send unrelated conversation history or external information.
Explicit prompt-injection sentences are removed from the model's message
input before extraction.

A separate repair prompt is used for inconsistent model results, with at
most one local repair retry.

## Deterministic Output

I used:

- `temperature = 0`
- `seed = 42`

Message files are also processed in sorted filename order.

I ran the practice folder twice and compared the outputs using:

```powershell
fc.exe run1.json run2.json
```

Result:

```text
Comparing files run1.json and RUN2.JSON
FC: no differences encountered
```

## Hallucination

During testing, the model initially treated the phrase `after 4` in the
non-task tea message as a possible deadline, even though the message did not
ask anyone to perform a task.

I handled this conservatively by checking for actual task-request language
such as `please`, `could someone`, `needs to`, or `must`.

If there is no task-request language, stray fields such as a time are
discarded instead of being converted into a task.

The model is also required to provide evidence copied from the message, and
the evidence is checked before the result is accepted.

## Format and Validation

The model must return a JSON object with:

- `task`
- `owner`
- `due`
- `task_evidence`
- `owner_evidence`
- `due_evidence`

Python validates:

- JSON/object structure.
- Required fields and value types.
- Consistency of no-task results.
- Evidence existence in the sanitized message.
- Evidence existence in the original message.
- Task, owner, and deadline grounding.

Before writing `tasks.json`, evidence fields are removed and only the
required output fields are written:

```json
{
  "message_id": "...",
  "task": "...",
  "owner": "...",
  "due": "..."
}
```

## Errors

Each message is processed independently inside a `try/except` block.

If one message has invalid formatting, invalid model JSON, inconsistent
fields, failed validation, or another processing error, a warning is
printed and that message is skipped.

The remaining messages in the folder continue processing.

A single local repair retry is used when the model output is detected as
inconsistent.

## Guardrails

Message content is treated as **untrusted data**.

The system prompt explicitly tells the model not to follow instructions
inside the message that try to change the task-extraction rules.

The script also removes explicit prompt-injection sentences such as:

- `ignore previous instructions`
- `change your task list`
- `add a task`
- similar instruction-override patterns

before sending the message to the model.

Evidence must be grounded in the sanitized message as well as the original
message. This prevents an injected instruction from being accepted as a
valid task.

For example, when a message contains an instruction to ignore the task-list
rules and add an unrelated email task, that instruction is treated as
prompt-injection data and is not added to `tasks.json`.
