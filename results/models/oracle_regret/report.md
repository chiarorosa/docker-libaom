# A5 — crivo offline ponderado por *regret* (triagem de soluções)

**Data:** 2026-07-19  
**Split:** validação + teste held-out — HoneyBee, FlowerPan, Lips, Jockey, RaceNight, RiverBank (792840 nós de decisão). Modelos treinados nas 10 restantes.
**Reprodução:** `python src/scripts/partition_model/oracle_regret.py`

Crivo de **triagem**, não predição do encoder (que é o árbitro final). Risco = *regret* ponderado por custo RD, normalizado pela RD total (`reg_frac` = % de sobrecarga RD), em vez de contagem de erros.

## 1. Ranking de triagem — `cost_red = 30%` casado

Menor `reg_frac` = menos custo RD desperdiçado por unidade de busca poupada = melhor candidato a avançar. `split_lost` (contagem, o critério antigo) ao lado, para contraste.

| # | solução | reg_frac % ↓ | reg_rel ↓ | split_lost % | cost_red atingível |
|--:|---|--:|--:|--:|---|
| 1 | GNN | 0.000 | 6.305 | 0.25 | 12–45% |
| 2 | H9c | 0.003 | 6.800 | 0.37 | 19–44% |
| 3 | GNN_causal | 0.004 | 21.887 | 2.40 | 7–42% |
| 4 | H9a | 0.006 | 24.810 | 2.16 | 5–42% |
| 5 | H9a_otimo | 0.009 | 24.718 | 1.99 | 2–35% |
| 6 | pixels24 | 0.015 | 30.352 | 4.20 | 0–38% |
| 7 | variance | 0.060 | 221.784 | 13.10 | 6–85% |
| 8 | random | 0.612 | 966.129 | 18.27 | 2–67% |
| — | regret | (não alcança 30%) | | | 34–83% |

## 2. Fronteira `reg_frac %` por `cost_red` casado

| solução | 15% | 20% | 30% | 40% | 50% |
|---|--:|--:|--:|--:|--:|
| random | 0.285 | 0.395 | 0.612 | 0.847 | 1.123 |
| variance | 0.023 | 0.035 | 0.060 | 0.084 | 0.128 |
| pixels24 | 0.002 | 0.005 | 0.015 | — | — |
| H9a | 0.000 | 0.001 | 0.006 | 0.023 | — |
| H9a_otimo | 0.000 | 0.003 | 0.009 | — | — |
| H9c | — | 0.000 | 0.003 | 0.015 | — |
| regret | — | — | — | 0.072 | 0.114 |
| GNN | 0.000 | 0.000 | 0.000 | 0.001 | — |
| GNN_causal | 0.000 | 0.001 | 0.004 | 0.016 | — |

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

