# Arquitetura do pruner de particionamento implantado (H9a)

**O que roda dentro do libaom em tempo de codificação.** Referência técnica da
solução embarcada — distinta da metodologia (validação) e dos resultados. Código:
`src/aom/av1/encoder/partition_strategy.c`, sob `#if PARTITION_ML_STUDENT`.

---

## 1. O artefato implantado (não é o ConvNeXt)

A solução em runtime é uma **MLP pequena por tamanho de bloco** (o "estudante"),
**não** o ConvNeXt (esse foi apenas o professor da destilação e a análise de teto —
pesado demais para inferência por nó, nunca embarcado).

- **Topologia:** `36 → 64 → 32 → 3` (uma por tamanho de bloco: 64/32/16px), executada
  pela função nativa `av1_nn_predict`.
- **Pesos:** compilados em `av1/encoder/partition_student_weights.h` (gerado por
  `export_weights.py`), um `NN_CONFIG` por tamanho.
- **Saída:** softmax de 3 classes = `[P(NONE), P(SPLIT), P(REST)]`.

O bloco 8×8 **não é modelado** (terminal; a poda nativa já o trata).

## 2. Onde engancha e o fluxo por nó

A busca de partição do libaom (`av1_rd_pick_partition`) é recursiva no quadtree do
superbloco (64→32→16→8). Em **cada nó quadrado** (64/32/16), **antes** da busca RD
cara, entra o gancho `student_prune_partition(cm, x, part_state)`:

1. `student_node_features(cm, x, blk, feats)` → monta o **vetor de 36 features H9a**.
2. `av1_nn_predict(feats, nnconfig, 1, logits)` + `av1_nn_softmax` → as 3 probs.
3. **Política de 3 ações** (§4) → poda o conjunto de candidatos.
4. O encoder segue a busca RD normal sobre o conjunto **reduzido**.

**Custo:** as features vêm de dados **já residentes** no encoder naquele ponto (luma
do bloco, `xd->above/left_mbmi`, `dc_q`, posição) → extração ~zero; a inferência é um
matmul minúsculo (~microssegundos). A poda economiza ordens de magnitude mais tempo
do que consome.

## 3. As 36 features (H9a)

Vetor único, espelhado bit-a-bit entre Python (`features.node_features_h9a`, fonte de
verdade) e C (`student_node_features`):

- **Bloco A (0–23) — pixels:** variância, quadrantes, gradientes, perfis linha/coluna,
  bordas, DC, qindex, contexto hierárquico pai/irmãos, posição no superbloco.
- **Bloco B (24–31) — vizinhança de particionamento:** `has_above`/`has_left`,
  tamanhos log2 dos blocos acima/esquerda (`xd->above/left_mbmi`), `neigh_finer`,
  `neigh_aniso`. Contexto RD **ausente dos pixels**.
- **Bloco C (32–35) — quantização/posição:** `log1p(dc_q²/256)`, posição normalizada
  (row/col), profundidade (log2 do tamanho).

## 4. A política de 3 ações (o coração da poda)

`partition_strategy.c` (~linha 2032), com limiares por nível (`student_get_taus`):

```c
const StudentTaus *tau = student_get_taus(block_size_wide[blk->bsize]);
if      (probs[0] > tau->none)   av1_disable_all_splits(part_state);      // (a) NONE-commit
else if (probs[1] > tau->split)  av1_set_square_split_only(part_state);   // (b) força SPLIT
else if (probs[2] < tau->rest)   av1_disable_rect_partitions(part_state); // (c) rect-off
// senão: busca completa (nada podado)
```

- **(a) NONE-commit** — `P(NONE) > τ_none`: decide o nó como NONE e **corta a subárvore
  inteira** (sem recursão, sem busca de candidatos). É a **alavanca primária** de
  economia de tempo.
- **(b) força-SPLIT** — `P(SPLIT) > τ_split`: pula os candidatos de forma no nó e só
  recorre nos 4 filhos.
- **(c) rect-off** — `P(REST) < τ_rest`: mantém a busca NONE vs SPLIT, mas **desliga os
  candidatos retangulares/AB/4-way** (~8 dos 9). Poda fina, secundária.

**Segurança/validade:** a política **só REMOVE candidatos, nunca força um ilegal** → o
bitstream permanece válido; o decoder e o formato não mudam. É **puramente redução do
espaço de busca no encoder**. Verificado: com a flag desligada, o bitstream é
**byte-idêntico** ao `aom_baseline` (Gate 4).

## 5. Os limiares τ e o ponto de operação

`StudentTaus` (τ_none, τ_split, τ_rest) é **por nível** — `taus[3]` indexado
16/32/64px (`student_get_taus`). Configurados por variáveis de ambiente
(`AV1_STUDENT_TAU_{NONE,SPLIT,REST}` globais; `_16/_32/_64` por nível) para varrer nos
experimentos **sem recompilar**; **na versão implantada seriam constantes compiladas**
(ou uma pequena tabela).

**O "operating point" é literalmente o vetor de τ.** Baixar `τ_none` faz o encoder
aceitar a decisão de NONE do modelo com menos confiança → corta mais subárvores →
mais rápido, mais perda de BD. Subir `τ_none` faz o oposto. `τ_rest` regula quão
agressivamente as formas retangulares são descartadas.

## 6. Equilibrado vs agressivo — MESMA arquitetura, só muda τ

Os dois operating points da tese **não são dois modelos nem dois códigos**. Mesma MLP,
mesmas features, mesmo `av1_nn_predict`, mesma política de 3 ações. **A única diferença
são os valores de τ:**

| ponto | τ_none | τ_rest | comportamento | BD-rate / TS (médias teste) |
|---|---|---|---|---|
| **Equilibrado** (~P_rect) | 0,95 | 0,20 | commita NONE só com P(NONE) > 0,95 (muito confiante) | ~0,46% BD @ TS ~26,5% (1,36×) |
| **Agressivo** (~A3) | 0,60 | 0,40 | commita NONE com P(NONE) > 0,60 + mais rect-off | ~1,4–2% BD @ TS ~48–57% (2,0–2,3×) |

*(τ podem ser por nível; ex.: o ponto refinado P_ref usava τ_none 0,85/0,80/0,80 em
16/32/64px. Os τ exatos dos dois pontos são fixados a partir das curvas de teste.)*

## 7. Enquadramento: um "botão de aceleração aprendido"

Como a arquitetura é única e o ponto de operação é só o τ, a solução implantada é
**um modelo + uma política**, com um **knob de agressividade** (o τ) — análogo ao
`cpu-used` nativo do AV1. "Equilibrado" e "agressivo" são **dois presets do mesmo
knob**, não dois encoders. É por isso que a comparação natural (Fase 6) é o **knob de
aceleração aprendido** vs o **knob nativo** (`cpu-used=1/2`) do libaom.

---

**Referências de código:** `partition_strategy.c` — `student_prune_partition`
(gancho + política), `student_node_features` (features), `student_get_taus`
(limiares); `partition_student_weights.h` (pesos gerados). Fonte de verdade das
features: `src/scripts/partition_model/features.py::node_features_h9a`. Ver também
`docs/RESULTADOS_fase5.md` (números) e `docs/RASTREABILIDADE.md` (inventário).
