# Contributing

Contributions should preserve reproducibility, patient-level split isolation, source provenance, and medical safety boundaries.

Before opening a pull request:

```bash
ruff check .
pytest
```

Do not commit model weights, adapters, ECG datasets, patient identifiers, credentials, generated outputs, or licensed assets. New dataset adapters must document the source license, citation, label ontology, and split strategy.
