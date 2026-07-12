# Fase 5 — Benchmark de tese (Gate 5) — Runbook

> **Execution shape:** Fase 5 é uma **campanha de compute**, não implementação de código. T1 (rebuild) é mecânico (subagente-suitable). T2 (piloto) e T3 (full) são **jobs detached de horas/dias**, conduzidos pelo controlador e monitorados ao longo do tempo (como o treino da Fase 3). T4 é análise curta.

**Goal:** Medir no encoder real, no teste held-out, se o H9a domina a variância em taxa BD a speedup casado (Gate 5).

**Base:** spec `docs/superpowers/specs/2026-07-11-fase5-benchmark-tese-design.md`.

## Global Constraints

- Rodar no container `av1_bench`; jobs longos **detached** (`nohup`), monitorados.
- Protocolo congelado (full): Jockey/RaceNight/RiverBank, ≥10 quadros, cq {20,32,43,55}, cpu-used=0, `--threads=1`.
- `src/aom_baseline` intocável; âncora = `libaom_perf_anchor`.
- Pontos operacionais/config **congelados no spec** antes de ver números de teste.
- Commits sem menção a IA/Claude/Co-Authored-By.

---

### T1 — Rebuild `libaom_perf` (36 features) + sanidade

O binário `libaom_perf/aomenc` é de 07-09 (pré-Fase-4). Recompilar para pegar o
`student_node_features` de 36 + header novo.

- [ ] **Rebuild:**
```bash
docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_perf \
  --target aomenc -j"$(nproc)" 2>&1 | tail -4'
```
Esperado: build OK; `aomenc` com mtime de hoje.

- [ ] **Sanidade (o estudante H9a realmente poda):** encodar 2 quadros de Jockey
com a âncora e com o `libaom_perf` (ml, τ_none agressivo) e confirmar que o tempo
do ml < âncora e o bitstream difere (poda ativa):
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
S=src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv &&
for e in "libaom_perf_anchor:" "libaom_perf:AV1_STUDENT_TAU_NONE=0.80"; do
  b=${e%%:*}; env=${e#*:}; t0=$(date +%s);
  env $env build/$b/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 \
    --end-usage=q --cq-level=32 -w 3840 -h 2160 --limit=2 -o /tmp/s_$b.ivf $S 2>/dev/null;
  echo "$b $(( $(date +%s)-t0 ))s $(stat -c%s /tmp/s_$b.ivf)B";
done'
```
Esperado: `libaom_perf` mais rápido que o anchor e tamanho de bitstream diferente
(poda H9a ativa). Se forem idênticos, o build não pegou o código novo — investigar.

---

### T2 — Piloto (detached, ~4 h) + gate do piloto

Valida o pipeline ponta a ponta + sinal direcional. Controlador lança detached e
monitora.

- [ ] **Lançar (Jockey, 2 quadros, cq {32,43}):**
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
S=src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv &&
nohup bash -c "
build/venv-ml/bin/python src/scripts/benchmark/h7h8_bench.py --seq $S \
  --frames 2 --cqs 32 43 --preset safe --out-dir results/benchmark/h9_pilot/curve &&
build/venv-ml/bin/python src/scripts/benchmark/ablation_attrib.py --seq $S \
  --frames 2 --cqs 32 43 --methods ml variance random --tau-none 0.90 0.80 \
  --out-dir results/benchmark/h9_pilot/ablation &&
build/venv-ml/bin/python src/scripts/benchmark/analyze_ablation.py \
  --dirs results/benchmark/h9_pilot/ablation
" > results/benchmark/h9_pilot.log 2>&1 &
echo "pilot pid $!"'
```

- [ ] **Monitorar** (`tail` do log; sem output incremental durante cada encode):
`docker exec av1_bench tail -20 /workspace/results/benchmark/h9_pilot.log`

- [x] **Gate do piloto (FEITO):** rodou ponta a ponta; sinal direcional forte
(ml ~0% BD @1,2–1,34× vs variância 2,4–3,7%). Achado: faixas de speedup não se
sobrepõem com grid estreito → grid largo (T2.5).

---

### T2.5 — Calibração na VALIDAÇÃO (HoneyBee) — congela o grid sem tocar em teste

```bash
docker exec av1_bench bash -lc 'cd /workspace &&
S=src/samples/HoneyBee_3840x2160_120fps_420_8bit_YUV_RAW.yuv &&
nohup bash -c "
build/venv-ml/bin/python src/scripts/benchmark/ablation_attrib.py --seq \$S \
  --frames 3 --cqs 32 43 --methods ml variance random \
  --tau-none 0.95 0.90 0.80 0.70 0.60 0.50 --out-dir results/benchmark/h9_calib &&
build/venv-ml/bin/python src/scripts/benchmark/analyze_ablation.py \
  --dirs results/benchmark/h9_calib && echo CALIB_DONE
" > results/benchmark/h9_calib.log 2>&1 &
echo "calib pid \$!"'
```
- [ ] **Gate da calibração:** as faixas de speedup de `ml` e `variância` se
sobrepõem (o `analyze_ablation` compara em ≥2 níveis de speedup casado). Se não,
alargar/deslocar o grid e repetir (ainda em validação). Congelar o grid resultante.

---

### T3 — Full (detached, ~18–24h) + monitoramento

Protocolo congelado. Por sequência (Jockey→RaceNight→RiverBank), retomável.
Ablação com o grid **congelado** na T2.5.

- [ ] **Lançar por sequência** (exemplo Jockey; repetir com RaceNight/RiverBank):
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
S=src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv && O=results/benchmark/h9_test/Jockey &&
nohup bash -c "
build/venv-ml/bin/python src/scripts/benchmark/h7h8_bench.py --seq $S --frames 10 \
  --cqs 20 32 43 55 --preset safe --out-dir $O/curve_safe &&
build/venv-ml/bin/python src/scripts/benchmark/h7h8_bench.py --seq $S --frames 10 \
  --cqs 20 32 43 55 --preset aggressive --out-dir $O/curve_aggr &&
build/venv-ml/bin/python src/scripts/benchmark/ablation_attrib.py --seq $S --frames 10 \
  --cqs 20 32 43 55 --methods ml variance random \
  --tau-none 0.95 0.90 0.80 0.70 0.60 0.50 --out-dir $O/ablation
" > results/benchmark/h9_test_Jockey.log 2>&1 &
echo "full Jockey pid $!"'
```
Cada seq ≈ 6–8 h (~34 s/quadro; 10 quadros); total ~18–24 h. Monitorar; ao concluir
uma seq, lançar a próxima (ou encadear). Retomável: uma seq incompleta é re-lançada.

---

### T4 — Análise + Gate 5 + docs

- [ ] **Veredito de atribuição em speedup casado** (todas as seqs):
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/benchmark/analyze_ablation.py --dirs \
  results/benchmark/h9_test/*/ablation'
```
- [ ] **Escrever `docs/RESULTADOS_fase5.md`:** tabelas da curva H9a (BD × speedup)
por seq + ablação em speedup casado + **veredito Gate 5** (H9a domina variância em
≥2/3 seqs? domina aleatório em todas?).
- [ ] **Atualizar** `ANDAMENTO_tese.md` (Fase 5 ✅/veredito) + memória; commit + push.
- [ ] Se Gate 5 falhar: reportar honestamente (recuo `PLANO_H9` §7: diagnóstico +
teto H9c).

---

## Notas
- Ablação `ablation_attrib.py` imprime "Jockey held-out" fixo (cosmético); a saída
  real vai por `--out-dir` por seq. Não bloqueia; generalizar só se incomodar.
- Estimativas de tempo pressupõem ~400–500 s/quadro (4K, cpu-used=0, 1 thread).
