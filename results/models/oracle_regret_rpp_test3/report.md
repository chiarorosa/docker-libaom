# A5 — crivo offline ponderado por *regret* (triagem de soluções)

**Data:** 2026-07-19  
**Split:** validação + teste held-out — Jockey, RaceNight, RiverBank (1816393 nós de decisão). Modelos treinados nas 10 restantes.
**Reprodução:** `python src/scripts/partition_model/oracle_regret.py`

Crivo de **triagem**, não predição do encoder (que é o árbitro final). Risco = *regret* ponderado por custo RD, normalizado pela RD total (`reg_frac` = % de sobrecarga RD), em vez de contagem de erros.

## 1. Ranking de triagem — `cost_red = 30%` casado

Menor `reg_frac` = menos custo RD desperdiçado por unidade de busca poupada = melhor candidato a avançar. `split_lost` (contagem, o critério antigo) ao lado, para contraste.

| # | solução | reg_frac % ↓ | reg_rel ↓ | split_lost % | cost_red atingível |
|--:|---|--:|--:|--:|---|
| 1 | GNN | 0.000 | 14.845 | 0.41 | 13–44% |
| 2 | H9c | 0.002 | 18.325 | 0.32 | 20–48% |
| 3 | GNN_causal | 0.005 | 73.807 | 3.36 | 6–42% |
| 4 | RPP_A_Bshuf_s1 | 0.006 | 93.288 | 5.44 | 3–44% |
| 5 | RPP_A_B_s0 | 0.006 | 84.298 | 3.19 | 4–42% |
| 6 | RPP_A_B_s2 | 0.006 | 78.150 | 3.33 | 4–42% |
| 7 | RPP_A_B_s1 | 0.006 | 74.621 | 3.22 | 4–42% |
| 8 | H9a | 0.006 | 75.244 | 3.39 | 4–42% |
| 9 | RPP_A_B_C_s1 | 0.006 | 77.825 | 3.22 | 5–41% |
| 10 | H9a_b1 | 0.007 | 92.257 | 2.01 | 7–43% |
| 11 | RPP_A_B_C_s0 | 0.007 | 80.907 | 2.91 | 4–42% |
| 12 | RPP_A_C_s2 | 0.007 | 97.727 | 5.59 | 3–44% |
| 13 | RPP_A_s1 | 0.007 | 102.529 | 5.24 | 3–44% |
| 14 | RPP_A_B_C_s2 | 0.007 | 83.865 | 3.07 | 4–42% |
| 15 | RPP_A_Bshuf_s2 | 0.008 | 96.614 | 5.44 | 3–43% |
| 16 | RPP_A_Bshuf_s0 | 0.009 | 114.370 | 5.57 | 2–43% |
| 17 | RPP_A_s0 | 0.009 | 105.183 | 5.29 | 3–44% |
| 18 | RPP_A_s2 | 0.009 | 105.063 | 4.82 | 3–43% |
| 19 | H9a_cw | 0.011 | 67.249 | 2.85 | 2–35% |
| 20 | RPP_A_C_s1 | 0.012 | 106.572 | 5.31 | 3–43% |
| 21 | pixels24 | 0.019 | 90.232 | 5.33 | 0–38% |
| 22 | convnext_ce_h9 | 0.022 | 205.152 | 5.64 | 0–37% |
| 23 | RPP_A_C_s0 | 0.027 | 126.184 | 5.12 | 4–44% |
| 24 | convnext_regret | 0.028 | 544.447 | 8.15 | 0–37% |
| 25 | convnext_ce_h9_f256 | 0.028 | 428.501 | 7.11 | 0–38% |
| 26 | variance | 0.030 | 385.155 | 13.26 | 2–82% |
| 27 | random | 0.512 | 1990.521 | 18.25 | 2–67% |
| — | convnext_ce | (não alcança 30%) | | | 0–26% |
| — | regret | (não alcança 30%) | | | 34–82% |

## 2. Fronteira `reg_frac %` por `cost_red` casado

| solução | 15% | 20% | 30% | 40% | 50% |
|---|--:|--:|--:|--:|--:|
| random | 0.247 | 0.333 | 0.512 | 0.700 | 0.904 |
| variance | 0.005 | 0.010 | 0.030 | 0.050 | 0.099 |
| pixels24 | 0.002 | 0.005 | 0.019 | — | — |
| convnext_ce | 0.004 | 0.007 | — | — | — |
| convnext_ce_h9 | 0.006 | 0.009 | 0.022 | — | — |
| convnext_ce_h9_f256 | 0.007 | 0.011 | 0.028 | — | — |
| convnext_regret | 0.007 | 0.010 | 0.028 | — | — |
| H9a | 0.000 | 0.001 | 0.006 | 0.024 | — |
| H9a_b1 | 0.000 | 0.001 | 0.007 | 0.025 | — |
| H9a_cw | 0.001 | 0.003 | 0.011 | — | — |
| H9c | — | — | 0.002 | 0.014 | — |
| regret | — | — | — | 0.088 | 0.131 |
| GNN | 0.000 | 0.000 | 0.000 | 0.001 | — |
| GNN_causal | 0.000 | 0.001 | 0.005 | 0.014 | — |
| RPP_A_s0 | 0.001 | 0.002 | 0.009 | 0.023 | — |
| RPP_A_s1 | 0.001 | 0.002 | 0.007 | 0.021 | — |
| RPP_A_s2 | 0.001 | 0.003 | 0.009 | 0.027 | — |
| RPP_A_B_s0 | 0.001 | 0.001 | 0.006 | 0.019 | — |
| RPP_A_B_s1 | 0.001 | 0.001 | 0.006 | 0.022 | — |
| RPP_A_B_s2 | 0.000 | 0.001 | 0.006 | 0.022 | — |
| RPP_A_Bshuf_s0 | 0.001 | 0.003 | 0.009 | 0.025 | — |
| RPP_A_Bshuf_s1 | 0.001 | 0.002 | 0.006 | 0.021 | — |
| RPP_A_Bshuf_s2 | 0.001 | 0.002 | 0.008 | 0.024 | — |
| RPP_A_C_s0 | 0.001 | 0.004 | 0.027 | 0.059 | — |
| RPP_A_C_s1 | 0.001 | 0.003 | 0.012 | 0.037 | — |
| RPP_A_C_s2 | 0.001 | 0.002 | 0.007 | 0.026 | — |
| RPP_A_B_C_s0 | 0.001 | 0.002 | 0.007 | 0.027 | — |
| RPP_A_B_C_s1 | 0.001 | 0.001 | 0.006 | 0.022 | — |
| RPP_A_B_C_s2 | 0.000 | 0.002 | 0.007 | 0.023 | — |

## 3. O crivo concorda com o encoder onde há chão real?

| par | vencedor real | crivo (reg_frac) diz | chão |
|---|---|---|---|
| GNN vs H9a | **H9a** | GNN (❌ diverge) | limpo |
| pixels24 vs variance | **variance** | pixels24 (❌ diverge) | contaminado |

> **GNN vs H9a (chão limpo):** RESULTADOS_approachB.md §5 (Jockey 5fr, replay fiel): H9a ~2x melhor BD em todo tau. Chão LIMPO (ambos modelos competentes).
> **pixels24 vs variância (chão contaminado):** ablation_matched.csv: variancia vence o 'ML'. Chão CONTAMINADO -- o 'ML' era o estudante de pixels fraco (macro-F1 0,203, CB-2); nao vale como verdade sobre um pruner competente.

## 4. Leitura para a tese

**O que o crivo faz bem — triagem por níveis.** Ele separa inequivocamente o piso (`random`), a heurística barata (`variância`) e a família aprendida, ordenando por custo RD real desperdiçado e não por contagem cega de erros. Para *filtrar candidatos obviamente inferiores* antes de gastar encode, é defensável e melhor que a contagem: a variância tem `split_lost` baixo em regimes onde seu `reg_frac` é alto (corta poucos true-SPLIT, mas caros) — só o *regret* expõe isso.

**O que o crivo NÃO faz — adjudicar entre modelos competitivos.** No único par com chão real limpo (GNN vs H9a), o crivo **diverge** do encoder: rankeia o GNN à frente, o encoder rebaixa o GNN ~2× em BD (`RESULTADOS_approachB.md`). Logo o crivo serve para eliminar perdedores, **não** para escolher o vencedor final — isso é do encoder.

**Achado que refina a explicação do Approach B.** O `approachB:118-121` atribui a derrota real do GNN a 'poucas podas confiantes **caras em RD**'. A medição **não sustenta** isso: as podas NONE do GNN são baratas por AMBOS os critérios (`split_lost` 0,02% e `reg_frac`≈0 a 30% de cost_red). A falha real do GNN, portanto, **não está na ação NONE-commit** medida aqui; sua causa não fica estabelecida por esta análise (candidatos não testados: vazamento de vizinhança na expressividade do grafo; descasamento cpu0↔cpu1; dano por outra ação da política). Registrar como pergunta aberta, não como causa provada.

**Limitação estrutural do crivo.** O *regret* usa rótulos RDO cpu-used=0 (a referência de treino); a implantação roda cpu-used≥1. O crivo mede qualidade de decisão contra o ótimo cpu0, não o custo cpu1 exato — outra razão para o encoder permanecer o árbitro final.

