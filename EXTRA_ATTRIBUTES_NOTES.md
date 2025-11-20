## Extra Attributes & Prompt Alignment Notes

### Flexible Attribute Strategy
- Add a single `extra_attributes: Dict[str, str]` field on `PatientProfile` so novel facts live in one predictable map rather than spawning ad-hoc schema changes.
- Update the patient profile builder to coerce any JSON objects, bullet lists, or stray `key: value` lines into that dict. Unknown top-level keys should be swept into `extra_attributes` before validation so nothing is dropped.
- Teach the `patient_profile` prompt to mention `extra_attributes` explicitly: “If a detail doesn’t match the named fields, move it under `extra_attributes` with a short snake_case key.” This keeps the runtime schema stable while still capturing nuance.
- Guardrails:
  - Document acceptable key formats and optionally warn when brand-new keys appear so teams can decide whether to promote them to first-class fields later.
  - Keep downstream consumers (simulated patient, critique agent, frontend) generic—just render the map as “Additional attributes”—so no one hardcodes user-defined keys.
  - Periodically review collected keys; when the same one shows up frequently, graduate it into the typed schema and clear it from the extras map.

### Implementation Plan (completed)
1. `personas/models.py`: extend `PatientProfile` with `extra_attributes` + description about short snake_case keys.
2. `personas/patient_profile_builder.py`: normalize strings/lists, parse loose data into the dict, and move unexpected top-level keys into `extra_attributes` before `model_validate`.
3. `personas/prompts.yaml`: rewrite the patient-profile system prompt to mention the new field and describe how to use it.
4. UI: surface the dict in `frontend/src/App.tsx` with a simple list + styling so extra details are visible to clinicians.
5. Document the workflow (this file) so future editors know why we funnel miscellaneous data into a single map.

### Simulated Patient Prompt Bug
- Root cause: `SimulatedPatientAgent._build_system_prompt` only read the `header` key. We were editing `simulated_patient.system_prompt`, so changes never reached the agent.
- Fix: in `_build_system_prompt`, try `header`, fall back to `system_prompt`, then a baked-in default. Prompt edits now propagate without restarting anything beyond the API server.
