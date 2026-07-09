# Protocolo de avaliação — CONGELADO (Fase 0)

**Congelado em 2026-07-09, antes de qualquer resultado do H9.** Este documento
fixa, a priori, a partição de dados, as sequências de teste, os QPs, a contagem
de quadros, os binários e as métricas. Qualquer desvio deve ser justificado por
escrito em commit posterior. Objetivo: eliminar a objeção de *cherry-picking* —
os números da tese saem exclusivamente do conjunto de **teste**, nunca usado em
qualquer decisão de modelo, feature ou limiar.

---

## 1. Partição de sequências (16 UVG 4K, `src/samples/`)

**Regra:** partição **por sequência** (sem vazamento temporal/espacial). Nenhuma
sequência de teste aparece em treino ou validação.

| Conjunto | Sequências | Uso |
|---|---|---|
| **Teste (3)** | **Jockey, RaceNight, RiverBank** | Só os números finais da tese. Congelado. |
| **Validação (3)** | HoneyBee, FlowerPan, Lips | Seleção de modelo, features e limiares operacionais |
| **Treino (10)** | Beauty, Bosphorus, CityAlley, FlowerFocus, FlowerKids, ReadySetGo, ShakeNDry, SunBath, Twilight, YachtRide | Treino de substituto e estudante |

Justificativa da escolha de teste: Jockey (movimento rápido, mantido por
continuidade com H1–H8); RaceNight (baixa luz, movimento, grão); RiverBank
(textura natural detalhada, panorâmica) — cobrem movimento, ruído e textura.

## 2. Cobertura de codificação

- **QPs:** cq-level 20, 32, 43, 55 (`--end-usage=q`); base_qindex = 4·cq.
- **Quadros:** treino/validação ≥ 5 quadros por sequência (amostragem temporal
  via `--skip`); **teste ≥ 10 quadros** por sequência (taxa BD estável — corrige
  a ressalva de ruído de 2 quadros da ablação anterior).
- **Preset:** `--cpu-used=0` (ground truth por busca RD completa; único regime em
  que todas as classes de partição são exploradas — cf. `partition-dataset-pipeline`).
- **Threads:** `--threads=1` (medição de tempo determinística).

## 3. Binários

| Papel | Build | Origem |
|---|---|---|
| Teste | `libaom_perf` | `src/aom`, `-DPARTITION_ML_STUDENT=1`, Release |
| Âncora | `libaom_perf_anchor` | `src/aom_baseline` (v3.10.0 puro), Release |
| Decodificador/paridade | `libaom_ml_check` | `src/aom`, generic, flag ON |
| Extração | `libaom_logpart` | `src/aom`, `-DLOG_PARTITION_DATA=1` |

O `src/aom_baseline` permanece **intocado** como controle cego. A compilação
padrão de `src/aom` (flags desligadas) é verificada byte-a-byte idêntica ao
âncora antes de qualquer medição.

## 4. Métricas

- **Taxa BD** (Bjøntegaard) sobre os 4 QPs, PSNR-Y calculado **externamente**
  (decodifica-se o stream e compara-se com a fonte em numpy) — não depende de
  build com `CONFIG_INTERNAL_STATS`, e o tempo não é poluído por `--psnr`.
- **Speedup** = tempo âncora / tempo teste (mediana de repetições).
- **Comparação em speedup casado**: taxa BD interpolada num grid de speedups
  comuns (`analyze_ablation.py`), pois cada método/limiar mapeia para speedups
  distintos.

## 5. Escada de ablação de atribuição (obrigatória para o número da tese)

Mesma política de poda, mesma sequência de teste, trocando só a fonte do escore:

1. **aleatório** — piso;
2. **variância** — `exp(−var/V₀)`, o teto de pixels a superar;
3. **pixels-24** — o ML anterior (≈ variância);
4. **H9a** — +vizinhança+posição (livre);
5. **H9b** — +proxy de resíduo (barato);
6. **H9c** — +none_rdcost (teto, não-implantável).

**Referência SOTA:** `intra_cnn_based_part_prune` nativo do libaom, ligado por
speed feature, medido no mesmo protocolo.

## 6. Critério de sucesso (definido a priori)

- **Gate 2 (offline, decisivo):** H9a e/ou H9b superam a variância na simulação
  oráculo em **risco casado**, por margem clara, na **validação**. Se falhar, a
  tese reporta o diagnóstico + teto (H9c) e não há integração em C.
- **Gate 5 (tese):** no **teste** held-out, a curva do estudante H9 **domina a
  variância** em taxa BD a speedup casado, por margem além do ruído, em **≥ 2 das
  3** sequências de teste.

Números-alvo de referência (não são o critério, apenas expectativa): speedup na
faixa de 10–30 % a taxa BD < 1,5 %, com H9 estritamente abaixo da variância na
mesma faixa de speedup.
