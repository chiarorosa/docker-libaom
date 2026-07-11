# results/ — artefatos gerados (proveniência)

Este diretório guarda dataset, modelos e resultados do pipeline de poda de
particionamento AV1 guiada por ML. **A rastreabilidade completa (scripts ↔
dataset ↔ modelos ↔ resultados ↔ commits) está em
[`../docs/RASTREABILIDADE.md`](../docs/RASTREABILIDADE.md).**

## Canônico (atual, válido)
- `dataset_h9/` — dataset ground-truth cpu-used=0, 16 seqs × 4 QP, luma real +
  contexto de taxa-distorção. 135 GB. Não versionado (tamanho); ver manifest.csv.
- `models/surrogate_real/` — ConvNeXt professor (macro-F1 0,203). Reproduzível.
- `models/student_real/` — estudante destilado implantado (**versionado no git**).
- `models/gate2_final*.csv` — Gate 2 (H9a supera pixels).
- `benchmark/` — H7/H8 (curva 7–29 % speedup) e ablação de atribuição.

## Regras
- Só `models/student_real/` e alguns CSVs de sumário são versionados; payloads
  grandes (dataset, surrogate .pt) ficam fora do git — reproduzíveis pelos
  scripts + `docs/PROTOCOLO_avaliacao.md`.
- Partição treino/val/teste **congelada** — nunca usar sequências de teste
  (Jockey, RaceNight, RiverBank) em decisões de modelo/feature/limiar.
