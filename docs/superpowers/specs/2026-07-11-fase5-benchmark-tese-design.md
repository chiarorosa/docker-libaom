# Spec — Fase 5: benchmark de tese (Gate 5) — H9a no encoder real

**Data:** 2026-07-11. Branch `ml-partition-dev`. Sucede a Fase 4
(`2026-07-11-fase4-*`, commits `b3cd3c1`..`ecf436b`). Contexto:
`docs/PLANO_H9_contribuicao_tese.md` (Fase 5), `docs/PROTOCOLO_avaliacao.md`
(congelado), `docs/RASTREABILIDADE.md`, `docs/ANDAMENTO_tese.md` §4.

---

## 1. Objetivo

Medir, no **encoder real** e no conjunto de **teste held-out**, se a poda H9a
entrega ganho de tempo **atribuível ao modelo** — dominando a variância trivial
em taxa BD a speedup casado. Este é o **Gate 5**, o árbitro final da tese.

Executado em duas etapas: um **piloto** (config reduzida, valida o pipeline +
sinal preliminar) e o **full** (protocolo congelado, campanha detached de dias).

## 2. Fora de escopo (e por quê)

- **Braço pixels-24 na ablação.** A variância é o representante mais forte do teto
  de pixels (a ablação real anterior `2380e91` mostrou variância ≥ estudante de
  pixels). Bater a variância subsome bater pixels-24. A escada de atribuição fica
  completa entre as duas ablações: `2380e91` (pixels ≤ variância) + Fase 5
  (H9a > variância). Evita um 2º build + ~2× encodes.
- **Teto H8 (surrogate replay) no teste.** Já caracterizado (Jockey); não é o número
  implantado. Fase 5 mede o estudante H9a implantado.
- **Retreino/re-export.** Usa o `student_h9a` da Fase 3 (header de 36 features já
  implantado na Fase 4).

## 3. Restrições (Global Constraints)

- **Protocolo CONGELADO** (`PROTOCOLO_avaliacao.md`), inegociável no full:
  - Teste: **Jockey, RaceNight, RiverBank** (4K); **≥10 quadros**; QP cq {20,32,43,55}.
  - cpu-used=0; **single-thread** (`--threads=1`, como em `run_benchmark.encode`) —
    timing limpo, sem `--psnr`.
- **Editar no Windows, rodar no Docker** `av1_bench`. Campanha longa → **detached**
  (nohup no container), monitorada ao longo de dias.
- **`src/aom_baseline` intocável.** Âncora = `libaom_perf_anchor` (build do baseline).
- **Sem menção a IA/Claude/Co-Authored-By** nos commits.
- **Congelamento a priori:** os pontos operacionais e a config saem deste spec
  ANTES de ver qualquer número de teste (anti-cherry-picking).

## 4. Builds

- **`libaom_perf`** (Release, `-DPARTITION_ML_STUDENT=1`, `src/aom`) — **REBUILD
  obrigatório**: hoje tem o código pré-Fase-4; precisa recompilar para pegar as 36
  features + header novo. É o encoder de teste (ml/variância/aleatório via
  `AV1_STUDENT_BASELINE`; τ via env).
- **`libaom_perf_anchor`** (aom_baseline) — âncora de tempo/BD (já existe).
- **`libaom_ml_check/aomdec`** — decoder (já existe).

## 5. Pontos operacionais e ablação (congelados aqui)

**Curva H9a** — usar os **presets existentes** de `h7h8_bench.py` (evita inventar
τ e reusa pontos já validados no H7/H8): `--preset safe` (P0/P_rect/P_ref) e
`--preset aggressive` (A1/A2/A3). O ponto H8_surrogate é **auto-descartado** quando
não há arquivos de replay (o caso do teste), então a curva sai só com os pontos do
estudante implantado H9a. Total: 6 pontos operacionais mapeando a fronteira.

**Ablação de atribuição** (`ablation_attrib.py`, NONE-commit puro: split=inf,
rest=off): métodos **ml / variance / random** (`AV1_STUDENT_BASELINE`), sweep
`--tau-none` **alargado** `{0.95, 0.90, 0.80, 0.70, 0.60, 0.50}` (mesmo grid para os
três métodos). Motivo: métodos diferentes podam a taxas diferentes por τ, então um
grid estreito deixa suas faixas de speedup **sem sobreposição**, e a comparação a
speedup casado (`analyze_ablation.py`) não interpola. O grid largo garante
sobreposição. É uma correção **neutra ao método** (faixa de medição), não tuning de
operating point. Tudo no mesmo `libaom_perf` (variância e aleatório computados em C,
independentes das features; ml usa o header de 36).

## 6. Duas etapas antes do full (papéis separados, integridade preservada)

**Correção de custo (medida no rebuild):** o encode de benchmark roda a **~34
s/quadro** em 4K cpu-used=0 single-thread (não ~500 s — aquele número era da
extração de ground-truth com `LOG_PARTITION_DATA`). Logo o full inteiro é **~18–24 h**,
não dias.

**6.1 Piloto de plumbing (FEITO, Jockey/2fr).** Serviu **só** para validar o pipeline
ponta a ponta (rebuild → `h7h8_bench` → `ablation_attrib` → `analyze_ablation`).
Passou. Os números do Jockey são **preview direcional** e **não** informam config
(Jockey é teste). Achado: com grid estreito as faixas de speedup não se sobrepõem
(ml 1,2–1,34×, variância 1,85–2,12×) → motivou o grid largo (§5). Bônus: o H8 rodou
(replay antigo do Jockey existe) — é o teto de pixels, ignorado para o Gate 5.

**6.2 Calibração na VALIDAÇÃO (roda agora, antes do full).** Rodar a ablação com o
grid largo (§5) numa seq de **validação** (**HoneyBee**), poucos quadros (~3, cq
{32,43}) — a razão de speedup é ~independente do nº de quadros. Objetivo: confirmar
que as faixas de `ml`/`variância` se **sobrepõem** com o grid largo (e ajustá-lo se
não). Por ser **dado de validação**, esses números podem **legitimamente** informar
o congelamento do grid, **sem tocar em teste**. Saída: `results/benchmark/h9_calib/`.
Após esta etapa o grid está **congelado**; o full no teste é intocado.

**Gate do piloto (não é o Gate 5):** (i) o pipeline roda ponta a ponta sem erro
(rebuild → encode → decode → BD/speedup → `analyze_ablation.py`); (ii) sinal
**direcional** — a curva ml tem BD/speedup sãos e, na ablação, ml ≤ variância em
BD a speedup casado (mesmo ruidoso em 2 quadros). Se o pipeline quebrar ou o sinal
inverter grosseiramente, corrigir antes de comprometer dias no full.

## 7. Full (protocolo congelado; campanha detached)

Para cada seq ∈ {Jockey, RaceNight, RiverBank}: **10 quadros, cq {20,32,43,55}**,
âncora única + curva (3 pontos) + ablação (ml/variance/random × 3 τ). ~60–70
encodes/seq × ~70 min ≈ **~2,5–3 dias/seq**; total **~7–10 dias**. Lançado
detached, monitorado; resultados incrementais por seq.

## 8. Gate 5 (sucesso da tese)

Em **speedup casado** (interpolação, `analyze_ablation.py`), o **H9a domina a
variância** em taxa BD, por margem além do ruído, em **≥2 das 3** sequências de
teste; e domina o **aleatório** em todas. A curva H9a reporta a fronteira
taxa BD × speedup implantável.

Se falhar (variância ≥ H9a no tempo de parede), reportar honestamente: o oráculo
superestimou; a contribuição recai na caracterização do teto informacional +
o estudo H9c (conforme `PLANO_H9` §7).

## 9. Componentes e arquivos

| Item | Papel |
|---|---|
| `build_dataset`/scripts prontos | nenhum código novo esperado; usa `h7h8_bench.py`, `ablation_attrib.py`, `analyze_ablation.py`, `run_benchmark.py` como estão |
| `results/benchmark/h9_pilot/` | saída do piloto (runs.csv, summary.csv, curve.csv) |
| `results/benchmark/h9_test/<seq>/` | saída do full por sequência |
| `docs/RESULTADOS_fase5.md` (novo) | tabela final: curva H9a + ablação em speedup casado + veredito Gate 5 |

Nota: se algum script precisar de ajuste (ex.: `ablation_attrib.py` rotula a saída
"Jockey"; generalizar para a seq corrente), é uma edição pequena e delimitada,
tratada como tarefa no plano.

## 10. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Full quebra após dias de compute | Piloto valida o pipeline antes; full é retomável por seq (âncora+pontos por seq independentes) |
| Container/daemon cai a meio | Detached + retomável; monitorar; re-lançar a seq incompleta |
| Ruído de BD em poucos quadros | Full usa ≥10 quadros/3 seqs (protocolo); piloto é só direcional |
| Oráculo superestimou (Gate 5 aquém) | Resultado honesto previsto em `PLANO_H9` §7 (recuo para diagnóstico + H9c) |
| Speedup single-thread lento | Inerente à metodologia congelada; é o custo de um número limpo |

## 11. Entregáveis

1. Piloto validado (`results/benchmark/h9_pilot/`) + veredito do gate do piloto.
2. Full por sequência (`results/benchmark/h9_test/`): curva H9a + ablação.
3. `docs/RESULTADOS_fase5.md` com as tabelas + veredito do Gate 5.
4. `ANDAMENTO_tese.md`/memória atualizados.
