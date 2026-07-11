# Andamento da tese — status e próximo passo

**Documento vivo.** Onde a tese está, o que foi decidido, e o próximo passo
concreto. Atualizado em 2026-07-10 (branch `ml-partition-dev`). Índice de
artefatos: `RASTREABILIDADE.md`. Planos: `PLANO_hipoteses_experimentos.md`,
`PLANO_H9_contribuicao_tese.md`. Protocolo congelado: `PROTOCOLO_avaliacao.md`.

---

## 1. Arco da tese (o que estabelecemos, em ordem)

1. **Infraestrutura** — instrumentação C do libaom, extração de dataset, ConvNeXt
   substituto, destilação → estudante MLP embarcado (`av1_nn_predict`), harness de
   benchmark (taxa BD + tempo). *(H1–E, concluído.)*
2. **Exploração H1–H6** — sinal da luminância parecia limitado (teto ~13–18%).
   *(Depois invalidado — ver item 4.)*
3. **Pré-H7** — descoberta de que a alavanca não era o modelo, e sim o **espaço de
   ações da política** (poda de retangulares via P(REST)) e a **métrica** (custo,
   não nós). Implementado.
4. **Bug crítico da luma-em-branco** — o dataset guardava luma `float32 [0,1]` e os
   consumidores assumiam `uint8`, treinando os modelos **sobre imagem em branco**.
   Corrigido; toda a cadeia re-treinada sobre pixels reais. **Invalidou as
   conclusões H1–H6** (eram sobre entrada vazia).
5. **H7/H8 (dado limpo)** — curva de operação real: **~7–29% de speedup a 0,4–1,6%
   de taxa BD**; teto do substituto (H8) ≈ de graça (−0,11% BD).
6. **Ablação de atribuição** — resultado **negativo**: no lever NONE-commit, um
   limiar de variância trivial **empata/supera** o estudante de pixels. Ou seja,
   os pixels saturam na variância; o ganho não era atribuível ao ML.
7. **Pivô H9** — hipótese: **contexto de taxa-distorção barato** supera o teto de
   pixels. Protocolo congelado (Fase 0), instrumentação e re-extração (Fase 1),
   **Gate 2 offline PASSOU** (Fase 2): o contexto RD grátis (H9a) supera pixels
   ~50% relativo no lever NONE-commit em risco casado.

**Tese, em uma frase (estado atual):** demonstra-se que o particionamento
All-Intra satura, no domínio de pixels, numa estatística trivial (variância) — por
ablação rigorosa — e que **contexto de taxa-distorção barato (vizinhança + quant +
posição) é necessário e suficiente para superar esse teto**, produzindo uma poda
aprendida com ganho de tempo atribuível ao ML. Falta confirmar no encoder real
(Gate 5).

---

## 2. Status por fase

| Fase | Descrição | Estado | Evidência |
|---|---|---|---|
| Infra (H1–E) | instrumentação, dataset, surrogate, destilação, benchmark | ✅ | commits até `71097b1` |
| Pré-H7 | poda de rect + métrica de custo + τ por nível + contexto hierárquico | ✅ | `834639e`, `PREH7_analise_alavancas.md` |
| Bug luma-branco | diagnóstico + correção + guarda | ✅ | `cb9d407`, `63f299c` |
| H7/H8 | curva speedup/BD real + teto do substituto | ✅ | `172e669`, `benchmark/h7h8_real*` |
| Ablação atribuição | ml vs variância vs aleatório (resultado negativo) | ✅ | `2380e91`, `benchmark/ablation_matched.csv` |
| H9 Fase 0 | protocolo congelado (test/val/train, QPs, métricas) | ✅ | `aabbbee`, `PROTOCOLO_avaliacao.md` |
| H9 Fase 1 | instrumentação RD (B/C/E) + re-extração 16 seqs | ✅ | `a17c525`, `a6748c5`, `dataset_h9/` (64 pkl) |
| H9 Fase 2 | Gate 2 offline (sinal do contexto RD) | ✅ **PASSOU** | `8866757`, `models/gate2_final.csv` |
| **H9 Fase 3** | **retreinar surrogate + destilar estudante com features H9a** | ⏳ **PRÓXIMO** | — |
| H9 Fase 4 | sincronizar features B/C em C; paridade; no-op byte-idêntico | ⬜ | — |
| H9 Fase 5 | benchmark no teste held-out + ablação de atribuição + SOTA nativo | ⬜ | — |

Legenda: ✅ concluído · ⏳ próximo · ⬜ pendente.

---

## 3. Veredito do Gate 2 (base da decisão atual)

`models/gate2_final.csv` (10 seqs treino, 60k superblocos, ensemble). No lever
**NONE-commit** (relevante para tempo), redução de custo em risco casado de
SPLIT-lost {0,5/1/2}%:

| subset | 0,5% | 1% | 2% |
|---|---:|---:|---:|
| variância | 0 | 0 | 0 |
| pixels24 | 10,1 | 15,3 | 18,9 |
| **H9a (contexto RD grátis)** | **15,7** | **20,1** | **24,9** |
| H9c (teto, none_rdcost) | 33,0 | 33,0 | 39,7 |

**Decisão: cenário (a)** — seguir com o modelo **pré-busca H9a** (pixels +
vizinhança + quant + posição; SATD do bloco D não agrega, descartado). O teto H9c
(`none_rdcost`, pós-NONE) fica documentado como headroom para uma extensão futura
(poda pós-NONE aprendida).

**Ressalva carregada até o fim:** a simulação oráculo superestima o tempo real
(~5×: 35% custo → 7% wall-clock). O que vale é a **margem relativa** de ~50% do
H9a sobre pixels; o árbitro final é o **Gate 5** (benchmark de tempo real no
teste held-out).

---

## 4. PRÓXIMO PASSO (Fase 3 — imediato)

Retreinar a cadeia com o conjunto de features **H9a** (índices 0–35 de
`features.node_features_h9`), sobre `dataset_h9`, respeitando a partição
congelada.

1. **Substituto com ramo de contexto RD.** Decidir pela evidência: ou estender o
   ConvNeXt com um ramo lateral que injeta as features B/C, ou — dado que o ganho
   é tabular — um substituto tabular mais forte. Treinar em treino, selecionar em
   validação (`train.py`, adaptado para o vetor H9a). *Gate:* macro-F1/SPLIT-recall
   ≥ substituto pixels.
2. **Destilar o estudante H9a** (`distill.py`) e checar na simulação oráculo
   (`simulate_pruning.py`) que o estudante ≥ variância na validação.
3. **Exportar pesos** (`export_weights.py`) → `partition_student_weights.h`.

*Comandos de arranque:* ver `RASTREABILIDADE.md` §6 (mesma cadeia, com o vetor de
features H9a e `--dataset-dir results/dataset_h9`).

**Depois (Fase 4):** sincronizar as features B/C em C dentro de
`student_node_features` (`partition_strategy.c`) — leitura de `xd->above/left_mbmi`,
`dc_q`, posição; estender `check_feature_parity.py` para os novos índices;
verificar no-op byte-idêntico. **Fase 5:** benchmark no teste (Jockey/RaceNight/
RiverBank, ≥10 quadros) + ablação de atribuição (H9a vs pixels vs variância vs
aleatório) + comparação com o `intra_cnn_based_part_prune` nativo.

---

## 5. Decisões em aberto

- **Arquitetura do substituto H9a** (ConvNeXt+ramo RD vs tabular) — decidir na
  Fase 3 pela validação.
- **Publicação do dataset** — `docs/ZENODO_datasheet.md` pronto; conversão para
  `.npz` uint8 (`pkl_to_npz.py`) **não executada** (backup bruto bin+pkl indo para
  Google Drive primeiro).
- **Extensão pós-NONE (H9c)** — opcional, documentada como headroom; só se o
  Gate 5 do H9a pré-busca ficar aquém.

---

## 6. Riscos vivos

- Gate 5 pode não confirmar a margem do Gate 2 (oráculo superestima) → mitigação:
  a decisão final é sempre o tempo de parede; se a margem não sobreviver, a
  contribuição recai na caracterização do teto informacional + o estudo H9c.
- Custo de inferência das features B/C é ~zero (dados residentes), então não há
  risco de "a poda não se pagar" no regime H9a (grátis).
