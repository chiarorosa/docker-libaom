# A5 — crivo offline ponderado por *regret* (triagem de soluções)

**Data:** 2026-07-19  
**Split:** validação + teste held-out — HoneyBee, FlowerPan, Lips, Jockey, RaceNight, RiverBank (3808703 nós de decisão). Modelos treinados nas 10 restantes.
**Reprodução:** `python src/scripts/partition_model/oracle_regret.py`

Crivo de **triagem**, não predição do encoder (que é o árbitro final). Risco = *regret* ponderado por custo RD, normalizado pela RD total (`reg_frac` = % de sobrecarga RD), em vez de contagem de erros.

## 1. Ranking de triagem — `cost_red = 30%` casado

Menor `reg_frac` = menos custo RD desperdiçado por unidade de busca poupada = melhor candidato a avançar. `split_lost` (contagem, o critério antigo) ao lado, para contraste.

| # | solução | reg_frac % ↓ | reg_rel ↓ | split_lost % | cost_red atingível |
|--:|---|--:|--:|--:|---|
| 1 | GNN | 0.000 | 34.583 | 0.38 | 11–43% |
| 2 | H9c | 0.004 | 65.453 | 0.41 | 18–43% |
| 3 | GNN_causal | 0.005 | 156.404 | 2.66 | 6–40% |
| 4 | H9a | 0.007 | 169.835 | 2.59 | 5–40% |
| 5 | H9a_b1 | 0.007 | 185.737 | 1.63 | 7–41% |
| 6 | H9a_cw | 0.012 | 173.683 | 2.35 | 2–32% |
| 7 | pixels24 | 0.022 | 218.937 | 4.33 | 0–35% |
| 8 | convnext_regret | 0.038 | 1329.295 | 7.82 | 0–33% |
| 9 | variance | 0.072 | 1235.140 | 13.67 | 6–84% |
| 10 | random | 0.634 | 4829.858 | 18.07 | 2–67% |
| — | convnext_ce | (não alcança 30%) | | | 0–26% |
| — | regret | (não alcança 30%) | | | 34–83% |

## 2. Fronteira `reg_frac %` por `cost_red` casado

| solução | 15% | 20% | 30% | 40% | 50% |
|---|--:|--:|--:|--:|--:|
| random | 0.299 | 0.407 | 0.634 | 0.870 | 1.131 |
| variance | 0.027 | 0.042 | 0.072 | 0.103 | 0.159 |
| pixels24 | 0.002 | 0.006 | 0.022 | — | — |
| convnext_ce | 0.005 | 0.011 | — | — | — |
| convnext_regret | 0.008 | 0.014 | 0.038 | — | — |
| H9a | 0.000 | 0.001 | 0.007 | — | — |
| H9a_b1 | 0.000 | 0.001 | 0.007 | 0.030 | — |
| H9a_cw | 0.001 | 0.003 | 0.012 | — | — |
| H9c | — | 0.000 | 0.004 | 0.020 | — |
| regret | — | — | — | 0.085 | 0.131 |
| GNN | 0.000 | 0.000 | 0.000 | 0.001 | — |
| GNN_causal | 0.000 | 0.001 | 0.005 | 0.019 | — |

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

