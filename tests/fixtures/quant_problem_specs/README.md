# QuantProblemSpec consumer fixtures

`vanilla_call.json` is a public-synthetic copy of the Haircut Engine QuantProblemSpec v0 fixture validated in `googa27/haircut-engine` for Project 5 issue #91.

Rules:

- The fixture is public synthetic only; no client, bank, RUT, token, private market, or Deloitte data belongs here.
- FEM tests consume this JSON as an external contract fixture, not as a Haircut Engine Python import.
- Changes must preserve measure, numeraire, units, valuation/vintage dates, boundary details, requested outputs, and schema version.
- If the upstream fixture changes incompatibly, update this copy and the adapter test in the same PR, or raise a compatibility issue in Project 5.
