# A1 — Índice de evidências

Este documento cumpre uma função de aparato, não de exposição. Ele lista, para cada
afirmação numérica destinada aos capítulos de Metodologia e de Resultados, a
condição experimental exata em que o número foi medido, o documento e a seção que
o relatam pela primeira vez, o artefato em `results/` que o sustenta, o script que
o reproduz e uma verificação direta — feita nesta auditoria, por leitura do
sistema de arquivos — de que o artefato citado existe no repositório. Nenhum valor
aqui foi recalculado; todos são transcritos dos documentos-fonte listados em
`00_PLANO_capitulos.md §7`. A tabela de §3 resolve, com procedência, as duas
divergências numéricas identificadas entre documentos; a de §4 relaciona os
artefatos citados que não foram encontrados no repositório.

**Convenção de verificação.** A coluna **existe?** registra apenas a presença do
artefato no caminho indicado (`ls`/`Glob` executados em 2026-07-29), não a
correção do seu conteúdo — exceto nos casos assinalados com "conferido", em que o
valor foi lido diretamente do CSV citado. Célula em branco na coluna de script
significa que o documento-fonte não nomeia um script de reprodução para aquela
linha especificamente (reproduzível pela cadeia geral da seção).

---

## 1. Metodologia

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Tamanho do conjunto de dados | 26,98 M amostras; 10,07 M nós de decisão (16,91 M são folhas 8×8 constantes) | 16 seqs UVG 4K × 4 QP, `cpu-used=0`, 5 quadros/seq amostrados | `RASTREABILIDADE.md §3` | `results/dataset_h9/manifest.csv` | `build_dataset.py` + `rebuild_manifest_stats.py` | sim |
| Partição do dataset, sem vazamento | treino 15,76 M / 6,09 M decisão (10 seqs); validação 5,89 M / 2,07 M (HoneyBee, FlowerPan, Lips); teste 5,32 M / 1,91 M (Jockey, RaceNight, RiverBank) | mesma extração acima | `RASTREABILIDADE.md §3`; `PROTOCOLO_avaliacao.md` | `results/dataset_h9/manifest.csv` | — | sim |
| Correção de layout do manifesto | inflação de +0,68% em `num_samples` (registro 4116B sobre payload 4144B) | comparação `manifest.csv` vs `manifest.csv.bak-4116` | `RASTREABILIDADE.md §3` | `results/dataset_h9/manifest.csv.bak-4116` (existe, 21.043 bytes, não rastreado pelo git) | `rebuild_manifest_stats.py` | sim |
| Vetor de atributos | H9a = 36 (blocos A+B+C); H9c = 39 (A+B+C+E) | `features.py::node_features_h9{a,c}` | `SINTESE_resultados_metodologia.md §2.5` | `src/scripts/partition_model/features.py:196-220` (código-fonte) | — | sim |
| Custo de inferência isolado (ns/chamada) | CNN nativa 24.763/24.606; H9a 484/488; H9c 770/745 (2 execuções) | encode real, cpu-used=1, Tango cq32 / BoxingPractice cq43, 3 quadros | `RESULTADOS_microbench_pruner.md §2` | `results/benchmark/microbench/pruner_cost.csv` | `src/scripts/benchmark/microbench_pruner.py` | sim |
| Razão de custo de inferência isolada | MLP ~50× mais barato por chamada que a CNN nativa | idem acima | `RESULTADOS_microbench_pruner.md §3` | idem | idem | sim |
| Custo implantado do pruner (extração + inferência, sobre o tempo de encode) | CNN 0,16–0,21%; H9a 0,26–0,32% | mesmo encode, acumuladores em C | `RESULTADOS_microbench_pruner.md §6.2b` | idem (medição em log de execução, não em CSV versionado) | idem | script sim; log de saída não versionado |
| Extração domina a inferência do H9a | 3.596–3.883 ns/chamada (extração) vs 496–497 ns/chamada (inferência), 7,2–7,8× | idem | `RESULTADOS_microbench_pruner.md §6.2` | idem | idem | idem |
| Calibração da softmax implantada (ECE) | ECE top-label 0,0112; ECE NONE 0,0206; SPLIT 0,0050; REST 0,0179 | teste congelado, 1.816.393 nós de decisão | `RESULTADOS_calibracao.md §2.1` | `results/models/student_h9a/calibration/ece.csv` | `src/scripts/partition_model/calibration.py` | sim |
| Precisão no limiar implantado (τ=0,90) | precisão NONE 0,956 (cobertura 29,9%); SPLIT 0,965 (cobertura 4,1%) | idem | `RESULTADOS_calibracao.md §2.2` | `results/models/student_h9a/calibration/threshold_precision.csv` | idem | sim |
| Paridade C↔Python do H9d (round-trip) | `\|softmax(PyTorch) − softmax(C)\| = 1,35e-07`, 192 vetores aleatórios | exportação de pesos, offline | `RESULTADOS_H9d_etapa2_C.md §1.1` | não versionado como CSV; verificação de execução única | `src/scripts/partition_model/h9d_export_weights.py` | script sim; saída não versionada |
| Garantia de no-op do H9d | bitstream byte-idêntico a `ml_balanced` com H9d desligado (1.574.775 B, PSNR-Y 40,9720 dB) | BoxingPractice cq32, CTC | `RESULTADOS_H9d_CTC.md §1.1` | `results/benchmark/fase6/raw_results.csv` | `src/scripts/fase6/ctc_h9d.py --integrity` | sim |
| Custo de busca de AB/4-way (alavanca C1/C3) | 34,3% agregado do tempo local de busca (mín. 28,9% em RiverBank, máx. 41,3% em Jockey) | 3 seqs de teste, cpu-used=0, cq32, 2 quadros, 875.317 nós | `RESULTADOS_C1_custo_por_candidato.md §3` | `results/benchmark/partstats/part_timing_t1.csv`, `partstats_racenight/part_timing.csv`, `partstats_riverbank/part_timing.csv` | `src/scripts/benchmark/analyze_partstats.py` | sim |
| Resolução temporal medida (piso de ruído) | CV mediano 0,28% do tempo bruto; σ do TS pareado ±0,23 pp (`ml_balanced`), ±0,09 pp (`native_cpu1`); resolução ~0,46 pp / ~0,18 pp | 5 repetições intercaladas, Crosswalk, 4 CQ | `RESULTADOS_BLOCO7_E3_DEC_E2.md §3` | `results/benchmark/fase6_repeat/raw_results.csv` | `src/scripts/fase6/encode_repeat.py` + `report_e3_dec_e2.py` | sim |
| Duas definições de TS na tese | divergem até ~3 pp (ex.: `cpu-used=1` 32,59% canônica vs 30,42% síntese) | mesmos 8 seqs CTC | `INVENTARIO_solucoes.md §0`; `RESULTADOS_fase6_swap_h9c.md §7` | `results/benchmark/fase6_analysis/ts_definitions.csv` | `src/scripts/fase6/analyze_frontier.py` | sim, conferido — ver §3 |

---

## 2. Diagnóstico do domínio de pixels (Solução 1 / R1)

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Ablação de atribuição original (piloto) | ML 1,39% BD vs variância 0,76% BD @ speedup 1,3× (variância vence) | política casada NONE-commit, 1 seq (Jockey), 2 quadros | `INVENTARIO_solucoes.md §2.1`; `RASTREABILIDADE.md §5.2` | `results/benchmark/ablation_matched.csv` | `src/scripts/benchmark/h7h8_bench.py`, `ablation_attrib.py` | sim |
| Teto do substituto (H8, replay) | BD-rate −0,11%; TS 2,4–3,3%; speedup ~1,02× | replay das probs do ConvNeXt pelo gancho H8, teste (Jockey) | `INVENTARIO_solucoes.md §2`; `RASTREABILIDADE.md §5.1` | `results/benchmark/h7h8_real_summary.csv` | `src/scripts/benchmark/h7h8_bench.py` | sim |
| Hierarquia no crivo A5 (reg_frac @ cost_red 25%, menor é melhor) | variância 0,0573; convnext_regret 0,0219; convnext_ce 0,0207; pixels24 0,0121; H9a 0,0036 | 6 seqs held-out (val+teste), 792.840 nós | `INVENTARIO_solucoes.md §2.2`; `RESULTADOS_convnext_regret.md §2` | `results/models/oracle_regret/ranking.csv`, `oracle_regret_convnext/ranking.csv` | `src/scripts/partition_model/oracle_regret.py` | sim |
| ConvNeXt com alvo de perda de otimalidade (do inglês *regret*) — refutado | convnext_regret pior que convnext_ce em toda a faixa: 1,06× (25%) a 3,80× (5%) de `cost_red` | mesma vara held-out do A5 | `RESULTADOS_convnext_regret.md §2` | `results/models/oracle_regret_convnext/ranking.csv` | `train_surrogate_regret.py`; `oracle_regret.py` | sim |
| Capacidade não é o gargalo do ConvNeXt | `val_wloss` 0,9235 (fusion_dim 256) vs 0,9250 (128) — diferença de 0,16% | mesmo treino, dois checkpoints | `RESULTADOS_convnext_regret.md §1.1` | `results/models/surrogate_regret/metrics.csv` | `train_surrogate_regret.py` | sim |
| Seleção do checkpoint original (correção de registro) | mínimo de `val_loss` 1,7999 e máximo de macro-F1 0,2034 concordam na época 13 | 30 épocas completas | `RESULTADOS_convnext_regret.md §4` | `results/models/surrogate_real/metrics.csv` | `train.py` | sim |
| E5 — H9a vs variância a tempo casado (FlowerPan) | H9a vence por 4,6× (@~1,15×) e 1,85× (@~1,27×) | política casada NONE-commit, validação, 10 quadros, 4 CQ | `RESULTADOS_E5_ablacao_validacao.md §2` | `results/benchmark/e5_ablation/FlowerPan/curve.csv` | `src/scripts/benchmark/ablation_attrib.py` | sim |
| E5 — precipício da variância (Lips) | speedup salta de 1,006× (τ=0,99) para 3,563× (τ=0,97); BD de 0,019% para 6,58% | idem, Lips | `RESULTADOS_E5_ablacao_validacao.md §3` | `results/benchmark/e5_ablation/Lips/curve.csv` | idem | sim |
| E5 — veredito do portão estrito | atingido em 1 de 2 sequências de validação (não 2 de 3, por decisão de escopo da HoneyBee) | 144 codificações, 34 pontos | `RESULTADOS_E5_ablacao_validacao.md §4` | `results/benchmark/e5_ablation/{FlowerPan,Lips}/{curve,runs}.csv` | idem | sim |
| GNN não-causal fura o teto do oráculo | +28 pp de `cost_red` sobre o MLP independente (L0) | validação (HoneyBee/FlowerPan/Lips), features h9a (36) | `RESULTADOS_approachB.md §2` | `results/models/gnn_L2/gate_oracle.csv` | `train_gnn.py`; `gate1_gnn.py` | sim |
| GNN causal — ganho evapora | empata o MLP no ponto de 1% SPLIT-lost (51,1 vs 52,0 de `cost_red`) | arestas restritas a pai + irmãos anteriores | `RESULTADOS_approachB.md §3` | `results/models/gnn_L2_causal/gate_oracle.csv` | idem | sim |
| GNN pixel-only *deployable* recupera o ganho offline | recupera ~93–100% do limite não-causal; supera o H9a em +20–25 pp de `cost_red` | features pixel-only (28), arestas completas | `RESULTADOS_approachB.md §4` | `results/models/gnn_L2_pixel/gate_oracle.csv` | idem | sim |
| Benchmark real do GNN — refutado | H9a domina o GNN por ~2× em BD em toda a varredura de τ | replay fiel (gancho H8), Jockey, 5 quadros | `RESULTADOS_approachB.md §5` | `results/benchmark/gnn_frontier/frontier_Jockey.csv`, `results/benchmark/gnn_replay/gnn_replay_Jockey.csv` | `gnn_replay.py`; `src/scripts/benchmark/gnn_frontier_bench.py` | sim |
| H9a é majoritariamente um modelo de pixels | 24 dos 36 atributos do H9a são descritores de luma (bloco A); nenhum é grandeza de custo RD | leitura de código | `RESULTADOS_auditoria_dominio_pixels.md §2` | `src/scripts/partition_model/features.py:196-220` (código-fonte) | — | sim |
| Bloco D (SATD do bloco-fonte, "H9b") — nulo no Gate 2 | cost_red 46,9/50,5/53,4/57,8 vs H9a 47,0/49,7/52,6/57,3, aos limites de risco 0,5/1/2/3% | Gate 2, val+teste | `RASTREABILIDADE.md §5.3`; `RESULTADOS_auditoria_dominio_pixels.md §3` | `results/models/gate2_final.csv` | `src/scripts/partition_model/gate2_signal.py` | sim |
| Bloco D' (predizibilidade intra a partir dos vizinhos) — portão não passa | 16px: CE 0,5144 (H9a+D') vs 0,5149 (H9a), AUC 0,846 vs 0,847 (nulo, 15.855 nós); 32px positivo mas 8× menos dados | offline, mesma bancada do Gate 2 | `RESULTADOS_auditoria_dominio_pixels.md §6.1` | `results/models/gate_intra_pred.csv`, `gate_intra_pred_sweep.csv` | `src/scripts/partition_model/gate_intra_pred.py` | sim |
| Fechamento da família de pixels | cinco tentativas independentes negativas: ConvNeXt-CE, ConvNeXt-perda de otimalidade, GNN/Approach B, bloco D, bloco D' | consolidação | `RESULTADOS_auditoria_dominio_pixels.md §6.2`; `INVENTARIO_solucoes.md §7` | múltiplos, listados acima | múltiplos | sim |
| Contradição declarada e não resolvida | `pixels24` vs variância: A5/CB-1 discordam (crivo favorece pixels24; ablação de 2 quadros favorece variância); E5 arbitra H9a×variância, não este par | — | `INVENTARIO_solucoes.md §2.1`; `RESULTADOS_convnext_regret.md §5.1` | ver linhas acima | — | sim |

---

## 3. H9a — poda pré-busca (R2)

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Gate 2 — cost_red em risco casado | pixels24 44,3/45,3/47,9/50,5%; **H9a 47,0/49,7/52,6/57,3%**; H9c (teto) 57,4/57,4/63,4/63,4% (limites de SPLIT-lost 0,5/1/2/3%) | 10 seqs de treino, offline | `RASTREABILIDADE.md §5.3` | `results/models/gate2_final.csv`, `gate2_final_sweep.csv` | `src/scripts/partition_model/gate2_signal.py` | sim |
| Gate 2 — lever NONE-commit isolado | pixels24 10–19%; H9a 16–25% (≈+50% relativo sobre pixels) | idem | `RASTREABILIDADE.md §5.3`; `SINTESE §4` | `results/models/gate2_final_sweep.csv` (seleção por risco casado; `gate2_final.csv` não é a fonte destes valores) | idem | sim |
| Gate 3 — validação (H9a) | ~55–58% de redução de custo a SPLIT-lost ≤1% (oráculo), ~2× os pixels | HoneyBee/FlowerPan/Lips | `SINTESE_resultados_metodologia.md §4` | `results/models/student_h9a/oracle_sim_student.csv` (não `gate3_h9a.csv`, que não contém esta faixa) | `src/scripts/partition_model/simulate_pruning.py` | sim, conferido: linha `tau_none=0,95, tau_split=0,70, tau_rest=0,10` → `cost_reduction_pct=55,54`, `true_split_lost_pct=0,192` |
| Fase 5 — teste held-out (média 3 seqs) | P0 0,155%/19,05%/~1,24×; P_rect 0,464%/26,49%/~1,36×; P_ref 0,595%/29,53%/~1,42×; H8 0,12–0,53%/2,4–3,3%/~1,03× | Jockey/RaceNight/RiverBank, 10 quadros, cq{20,32,43,55}, cpu-used=0 | `SINTESE_resultados_metodologia.md §4`; `INVENTARIO_solucoes.md §3.1` | `results/benchmark/h9_test/{Jockey,RaceNight,RiverBank}/{curve_safe,curve_aggr,ablation}/summary.csv` | `src/scripts/benchmark/h7h8_bench.py`; `results/benchmark/fase5_final.py` | sim |
| Fase 5 — pilar 3, atribuição a política casada (razões ml/variância) | Jockey 11×; RaceNight 44×; RiverBank 94× | ablação NONE-commit, 3 seqs teste | `RESULTADOS_fase5.md §3.2` | `results/benchmark/h9_test/*/ablation/` | `src/scripts/benchmark/analyze_ablation.py` | sim |
| Fase 6 CTC — resultado final (8 seqs) | ml_balanced +0,568%/17,72%/1,223×; ml_aggr +1,403%/31,51%/1,530×; cpu1 +0,449%/32,59%/1,508×; cpu2 +0,536%/42,72%/1,788×; cpu3 +2,722%/67,94%/3,159× | CTC Classe A1, 15 quadros, cq{20,32,43,55}, cpu-used=0 | `RESULTADOS_fase6.md §3`; `INVENTARIO_solucoes.md §1,§3.2` | `results/benchmark/fase6/bdrate_average.csv` | `src/scripts/fase6/{encode_ctc.py,report_ctc.py}` | sim |
| Swap H9a (CNN nativa desligada) | cpu1 bal 0,915%/40,20%/1,694×, aggr 1,685%/51,82%/2,104×; cpu2 bal 1,030%/50,05%/2,046×, aggr 1,805%/60,97%/2,610×; cpu3 bal 3,866%/73,09%/3,754×, aggr 4,347%/77,30%/4,465× | `AV1_DISABLE_NATIVE_CNN=1`, 8 seqs, 4 CQ, 192 encodes | `RESULTADOS_fase6.md §4.1`; `INVENTARIO_solucoes.md §3.3` | `results/benchmark/fase6_swap/swap_average.csv` | `src/scripts/fase6/{encode_swap.py,report_swap.py}` | sim |
| Fronteira Pareto do swap (9 pontos) | apenas 1 ponto estritamente dominado (`cpu1 h9a_bal`, dominado por `cpu2 native`) | idem | `RESULTADOS_fase6.md §4.2` | `results/benchmark/fase6_swap/swap_average.csv` | idem | sim |
| C2 — contribuição marginal por nível (Jockey) | desligar nível 64 derruba BD de 1,41%→0,11% (perde ~92% do custo de qualidade) por −0,223× de speedup; nível 32 rende speedup grátis no RaceNight (+0,391×, BD melhora 0,009 pp) | leave-one-out por env-var, P_ref, 3 seqs teste, 5 quadros, 4 CQ | `RESULTADOS_C2_sweep_niveis.md §2-3` | `results/benchmark/c2_levels/raw.csv` | `src/scripts/benchmark/c2_level_sweep.py` | sim |
| C5 — densidade da fronteira de τ | gaps de speedup entre τ vizinhos ≤0,15× (máximo, RaceNight); maioria ~0,03× | sweep de 8 τ, 3 seqs teste, 5 quadros, 4 CQ | `RESULTADOS_C5_fronteira_tau.md §2` | `results/benchmark/c5_finetau/raw.csv` | `src/scripts/benchmark/c5_fine_tau.py` | sim |
| Custo de inferência isolado do H9a (referência cruzada) | ~50× mais barato que a CNN nativa por chamada | ver bloco de Metodologia | `SINTESE §4`; `RESULTADOS_microbench_pruner.md` | `results/benchmark/microbench/pruner_cost.csv` | `microbench_pruner.py` | sim |

---

## 4. H9c — refinamento pós-NONE (R3)

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Gate 3 — validação (H9c) | 61,2% de redução de custo a 0,20% de SPLIT-lost (τ_none=0,80, τ_split=0,70) | HoneyBee/FlowerPan/Lips, oráculo | `SINTESE_resultados_metodologia.md §5` | `results/models/student_h9c/gate3_h9c.csv` | `src/scripts/partition_model/simulate_pruning.py` | sim, **conferido diretamente**: linha `0.80,0.70,-1.00` → `cost_reduction_pct=61,208`, `true_split_lost_pct=0,2009` |
| O *confound* do H9a (Neon1224) | H9a@0,9 sozinho 0,312%/17,10%; H9a+H9c 0,270%/17,36%; H9c sozinho 0,037%/4,23% | isolamento por neutralização do H9a (τ=2/2/−1), cpu0 | `SINTESE §5`; `RESULTADOS_BLOCO7_E1_E4.md §2` | `results/benchmark/fase6/raw_results.csv` (linhas `h9ciso_tau90`) | `src/scripts/fase6/encode_h9c_iso.py`, `report_bloco7.py` | sim |
| Swap H9c — grade CTC completa (8 seqs) | cpu1 h9c95 0,414%/30,34%/1,469×; cpu2 h9c95 0,516%/40,73%/1,746×; cpu3 h9c90 3,397%/70,67%/3,474× | 192 encodes, mesmo anchor da Fase 6 | `RESULTADOS_fase6_swap_h9c.md §3` | `results/benchmark/fase6_swap_h9c/swap_average.csv` | `src/scripts/fase6/report_swap.py` | sim |
| Decomposição por regime de CQ | h9c_tau95_cpu1: cq20+32 0,065%/20,97% vs native_cpu1 0,153%/25,26% (H9c custa menos da metade do BD-rate) | idem, decomposto por CQ | `RESULTADOS_fase6_swap_h9c.md §4` | `results/benchmark/fase6_analysis/cq_decomposition.csv` | `src/scripts/fase6/analyze_frontier.py` | sim |
| Teste pareado (h9c_tau95_cpu2, cq20+32) | Δ BD −0,086 pp; p=0,015; melhor em 7/8 seqs | teste t bilateral pareado, n=8 | `RESULTADOS_fase6_swap_h9c.md §5` | `results/benchmark/fase6_analysis/paired_tests.csv` | idem | sim |
| Fronteira Pareto sobre 8 seqs | 13 de 17 configurações não dominadas; h9c_tau95_cpu1 (0,414%/30,34%) abaixo do 1º degrau nativo (0,449%) | idem | `RESULTADOS_fase6_swap_h9c.md §6` | `results/benchmark/fase6_analysis/pareto_frontier.csv` | idem | sim |
| E1 — H9c em 8/8 sequências | h9c_tau95 +0,160%/12,6%/1,15×; h9c_tau90 +0,171%/13,6%/1,16×, ambos menos de 1/3 do custo do H9a implantado (+0,568%) | 16 encodes, 8 seqs completas, cpu0 | `RESULTADOS_BLOCO7_E1_E4.md §1` | `results/benchmark/fase6/raw_results.csv` | `src/scripts/fase6/encode_h9c_cq20.py`, `report_bloco7.py` | sim |
| E4 — decomposição do confound generalizada | em média 64% (28–95% por seq) do TS atribuído ao H9c era, na verdade, do H9a rodando nos defaults | 4 seqs (Neon1224 + 3 novas), 12 encodes | `RESULTADOS_BLOCO7_E1_E4.md §2` | `results/benchmark/fase6/raw_results.csv` (linhas `h9ciso_*`) | `src/scripts/fase6/encode_h9c_iso.py` | sim |
| E3 — joelho de τ e τ45 | joelho em τ≈60–70 (preço salta de 0,013–0,042 para 0,107 pp/pp); τ45 dominado por `native_cpu1` (0,643%/21,4% vs 0,449%/32,6%, 8 seqs) | subconjunto casado 3 seqs + extensão 8 seqs, cpu0 | `RESULTADOS_BLOCO7_E3_DEC_E2.md §1` | `results/benchmark/fase6/raw_results.csv` | `src/scripts/fase6/encode_h9c_cq20.py --taus 45` | sim |
| Decomposição de 3 pernas — redundância H9a/H9c | interação média −1,9 pp (soma das partes 16,5% de TS vs empilhado 14,6%) | 4 seqs, H9a-só/H9c-só/H9a+H9c | `RESULTADOS_BLOCO7_E3_DEC_E2.md §2` | `results/benchmark/fase6/raw_results.csv` | `encode_h9adef.py`; `report_e3_dec_e2.py` | sim |

---

## 5. H9d — poda seletiva das partições estendidas (R4)

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Etapa 1 — portão de predizibilidade | ROC-AUC agregado 0,890 (36 atributos) / 0,902 (39 atributos), 792.840 nós; a 50% de busca evitada, 1,1% de vencedores perdidos (39 feat.) | offline, 6 seqs held-out | `RESULTADOS_H9d_predizibilidade.md §2, §2.1` | `results/models/h9d_predictability/students.pt`, `h9d_predictability_h9c/students.pt` | `src/scripts/partition_model/h9d_predictability.py` | sim |
| Cota superior (blanket, AB/4-way desligado por completo) | média +0,89% BD / 1,431× speedup | `AV1_EXT_PART_OFF=1`, 3 seqs teste, 5 quadros, 4 CQ | `RESULTADOS_H9d_cota_superior.md §3` | `results/benchmark/h9d_ub/raw.csv` | `src/scripts/benchmark/h9d_upper_bound.py` | sim |
| Marginal pós-NONE (blanket sobre H9a P_ref) | média +0,798% BD / 1,293× speedup preso em AB/4-way no resíduo | idem, empilhado sobre H9a P_ref | `RESULTADOS_H9d_cota_superior.md §3.7` | `results/benchmark/h9d_marg/raw.csv` | idem `--stack h9a` | sim |
| Etapa 3, Fase 1 (τ=0,30, 5 quadros) | Jockey 1,72%/1,77×; RaceNight 0,90%/1,75×; RiverBank 0,44%/1,45× | 3 seqs teste, 4 CQ | `RESULTADOS_H9d_etapa3_encoder.md §2` | `results/benchmark/h9d_selective/raw.csv` | `src/scripts/benchmark/h9d_selective_sweep.py` | sim |
| Etapa 3, Fase 2 (10 quadros) — confirmação | RaceNight Pareto-domina o knob de τ (0,981%/1,771× vs A2 1,282%/1,757×); Jockey −0,22 pp; RiverBank +0,12 pp | 3 seqs teste, 10 quadros, 4 CQ | `RESULTADOS_H9d_etapa3_encoder.md §4.1` | `results/benchmark/h9d_phase2/raw.csv` | idem | sim |
| Etapa 3, Fase 2b — τ por nível (PL10) | recupera o RiverBank (+0,12→+0,05 pp) sem perder o RaceNight (−0,30 pp); PL10 = τ₁₆0,091/τ₃₂0,103/τ₆₄0,014 | 5 quadros, 4 configs por nível | `RESULTADOS_H9d_etapa3_encoder.md §4.2` | `results/benchmark/h9d_perlevel/raw.csv` | idem | sim |
| CTC final (8 seqs, protocolo CTC) | H9a+H9d(PL10) +0,586%/18,74%/1,238× vs H9a bal +0,568%/17,72%/1,223× | 15 quadros, 4 CQ, cpu0 | `RESULTADOS_H9d_CTC.md §2`; `INVENTARIO_solucoes.md §5.5` | `results/benchmark/fase6/bdrate_average.csv` (config `ml_bal_h9d`) | `src/scripts/fase6/{ctc_h9d.py,report_ctc.py}` | sim |
| Contribuição marginal do H9d | +1,02 pp de TS por +0,018 pp de BD; preço 0,018 pp/pp contra 0,063 pp/pp do knob de τ (~3,5× mais barato) | interpolação do segmento P_rect→A3 no TS do H9d | `RESULTADOS_H9d_CTC.md §3.1`; `SINTESE §5-quater` | saída de `ctc_h9d_marginal.py`, não versionada em CSV próprio; insumo em `results/benchmark/fase6/raw_results.csv` | `src/scripts/fase6/ctc_h9d_marginal.py` | script sim; CSV de saída específico não localizado (ver §4) |
| H9d vence o knob de τ | 6 das 8 sequências; 2 por dominância de Pareto estrita (FoodMarket2, Tango) | idem | `RESULTADOS_H9d_CTC.md §3.2` | idem | idem | idem |
| Fronteira completa 2D (27/07) | P_rect×PL10 (implantado) 0,586%/18,74%, +1,02pp/0,0179 pp-pp, 3,38×; P_rect×PL20 0,651%/19,81%, +2,09pp/0,0399, 1,52×; A3×PL10 1,409%/31,68%, +0,17pp/0,0329, 1,84×; A3×PL20 1,420%/32,16%, +0,65pp/0,0258, 2,35× | 96 encodes novos (P_rect×PL20, A3×PL10, A3×PL20; P_rect×PL10 já medido na campanha CTC anterior), 2 bases H9a × 2 forças H9d, 8 seqs CTC, 128 linhas ao todo | `SINTESE_resultados_metodologia.md §5-quater` | `results/benchmark/fase6/raw_results.csv` (configurações `ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d`, `ml_aggr_h9d_pl20`, 32 linhas cada) | `ctc_h9d.py` (extensão) | sim |
| Base agressiva quase inerte | A3+PL10 rende só +0,17 pp de TS (1/8 seqs acima da resolução de ~0,46 pp; Neon1224 e Tango negativos) | idem | `SINTESE §5-quater`; `INVENTARIO §5.5` | idem | idem | sim |
| Dois estimadores do preço do knob | 0,063 pp/pp (interpolação por sequência, `ctc_h9d_marginal.py`) vs 0,0606 pp/pp (média-das-médias); concordam a ~4% | idem | `INVENTARIO_solucoes.md §5` | idem | idem | idem |

---

## 6. Resultados negativos de valor metodológico (R5)

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Solução 4 — Gate 0 (viabilidade, zero-inflação) | fração de zeros: dim64 59,5%; dim32 83,6%; dim16 98,1% | 10 pkls de treino | `RESULTADOS_solucao4.md §3` | `results/models/regret/gate0.csv` | `src/scripts/partition_model/regret.py` | sim |
| Solução 4 — regressor naïve refutado | 0,00% de redução de custo a 0,5/1/2% de SPLIT-lost; menor SPLIT-lost alcançável 12,4% | Gate 3, validação | `RESULTADOS_solucao4.md §4` | `results/models/regret/gate3_regret.csv` | `train_regret.py` | sim |
| Solução 4 — regressor balanceado (anti-zero-inflação) | 0,00% em todos os r0 ∈ {0,05;0,1;0,2}; menor SPLIT-lost 11,4% | idem, Huber ponderado | `RESULTADOS_solucao4.md §6` | `results/models/regret/gate3b_regret_r0{0.05,0.1,0.2}.csv` | `train_regret.py --balance` | sim |
| Solução 4 — comparação direta com o classificador | H9a-classificador (mesmas 36 features) alcança 51,73% de redução de custo a 0,5% SPLIT-lost | idem | `RESULTADOS_solucao4.md §4, §6` | `results/models/regret/gate3_h9a.csv` | `simulate_pruning.py` | sim |
| B1 — contexto RD hereditário (negativo) | H9a `reg_rel` 24,81 vs H9a_b1 42 atributos `reg_rel` 38,81, a `cost_red` 30% casado | crivo A5, val+teste, 792.840 nós | `RESULTADOS_modelagem_B1_contexto_hereditario.md §2` | `results/models/oracle_regret_b1/ranking.csv` | `train_student_h9.py --feature-set h9a_b1`; `oracle_regret.py` | sim |
| B2 — τ por qindex (positivo, imaterial) | ganho relativo 1,6–2,7× (`reg_frac` 0,006%→0,003% a `cost_red` 30%), abaixo do piso de ruído do encoder | calibração offline, 48.000 superblocos | `RESULTADOS_modelagem_B2_tau_qindex.md §3-4` | `results/models/b2_tau_qindex/b2_frontier.csv` | `src/scripts/partition_model/b2_tau_per_qindex.py` | sim |
| B3 — sinal direcional (confirmado offline, não cravado em política) | acurácia direcional condicional 69,1% agregada (+16,7 pp sobre a maioria 52,4%); recall incondicional HORZ/VERT ~42,9%/41,5% | held-out, 191.909 nós retangulares | `RESULTADOS_modelagem_B3_horz_vert.md §2-3` | `results/models/student_h9a_4cls/b3_report.md` | `src/scripts/partition_model/b3_horz_vert.py` | sim |
| B3 — Etapa 1 pós-NONE (refutado, encerrado) | acurácia direcional plana: 69,2% (controle 36 feat.) → 69,5% (39 feat., +0,3 pp) | controle pareado, 1,86 M nós treino | `RESULTADOS_modelagem_B3_horz_vert.md §7` | `results/models/b3_control36/b3_report.md`, `b3_postnone/b3_report.md` | `b3_horz_vert.py --feature-set {h9a,h9c}` | sim |
| B4 — ponderação de classe por nível (negativo) | SPLIT-recall@16px 0,022→0,361 (16×) mas ECE top 0,011→0,040 (3,6×) e ECE NONE 0,021→0,174 (8×); teto de poda cai 42%→35% de `cost_red` | comparação direta com `student_h9a` | `RESULTADOS_modelagem_B4_ponderacao_classe.md §2-4` | `results/models/student_h9a_cw/students.pt` + `calibration/`, `results/models/oracle_regret/ranking.csv` (linha H9a_cw) | `train_student_h9.py --class-weight`; `calibration.py`; `oracle_regret.py` | sim |
| C5 — poda *soft* não se justifica (negativo) | ver linha equivalente em §3 (fronteira já densa, gaps ≤0,15×) | — | `RESULTADOS_C5_fronteira_tau.md §3` | `results/benchmark/c5_finetau/raw.csv` | `c5_fine_tau.py` | sim |
| Approach B / GNN — refutado no encoder real | ver linha equivalente em §2 (H9a domina o GNN por ~2× em BD) | — | `RESULTADOS_approachB.md §5-6` | `results/benchmark/gnn_frontier/frontier_Jockey.csv` | `gnn_frontier_bench.py` | sim |

---

## 7. Análise integrada (R6)

| afirmação | valor | condição experimental | documento-fonte | artefato numérico | script de reprodução | existe? |
|---|---|---|---|---|---|---|
| Fronteira Pareto global (versão de 3 sequências, superada) | sequência não-dominada `H9c(cpu0) 0,21%/13,9%` → ... → `H9c_cpu3/H9a_cpu3 3,5–4,8%/70–78%` | 3 seqs (Boxing/FoodMarket2/Tango), todos os níveis cpu | `SINTESE_resultados_metodologia.md §6` | não localizado CSV específico desta agregação de 3 seqs (ver §4) | análise ad hoc, não nomeada | não confirmado — ver §4 |
| Conclusão 3 corrigida — H9d soma, H9c não | H9c soma +0,26 pp de TS sobre o H9a; H9d soma +1,02 pp (quatro vezes mais), com informação idêntica (39 atributos) | comparação entre `RESULTADOS_BLOCO7_E1_E4.md` e `RESULTADOS_H9d_CTC.md` | `SINTESE §6` | `results/benchmark/fase6/raw_results.csv` (ambos os levers) | múltiplos, listados em §4-§5 | sim |
| *Frontier-check* combinado (Tango) | TS/BD marginal: nativa cpu1 81,9 > H9c 77,2 > combinado(0,98) 67,9 > combinado(0,95) 65,2 > H9a_bal 29,9 | H9a-conservador + H9c empilhados, 1 seq | `SINTESE §10` | `results/benchmark/fase6_swap_combo/raw_results.csv` | script não nomeado explicitamente (piloto) | sim (dado presente; script de geração não identificado no texto) |
| Custo de inferência agregado — sem alavancagem em direção nenhuma | ver linhas de Metodologia (§1): CNN 0,16–0,21%; H9a 0,26–0,32% do tempo de encode | — | `SINTESE §6`; `RESULTADOS_microbench_pruner.md §6` | `results/benchmark/microbench/pruner_cost.csv` | `microbench_pruner.py` | sim |

---

## 8. Conflitos e divergências numéricas

### 8.1 As duas definições de redução de tempo (TS%) — divergência de até ~3 pp

A tese usa, em documentos diferentes, duas fórmulas para "redução de tempo" (TS%),
e elas **não** produzem o mesmo número para a mesma configuração:

- **Definição canônica** — `TS = média_seq( média_CQ( 1 − t_cfg/t_âncora ) )`. É a
  usada por `report_ctc.py`, `report_swap.py`, `analyze_frontier.py`, e por todos
  os documentos mais recentes: `RESULTADOS_H9d_CTC.md`, `RESULTADOS_BLOCO7_E1_E4.md`,
  `RESULTADOS_fase6_swap_h9c.md`, `INVENTARIO_solucoes.md`.
- **Definição "da síntese"** — `TS = média_seq( 1 − Σ_CQ t_cfg / Σ_CQ t_âncora )`,
  ponderada pelo tempo absoluto e por isso dominada pelo CQ mais caro (cq20). É a
  usada nas tabelas de `SINTESE_resultados_metodologia.md §4` (por exemplo,
  `libaom cpu-used=1` aparece ali como 30,42%/1,440×).

**Divergência medida** (`results/benchmark/fase6_analysis/ts_definitions.csv`,
conferido diretamente nesta auditoria): de **+2,63 pp** (`ml_aggr`: 31,51%
canônica vs 34,14% síntese) a **−2,97 pp** (`h9c_tau95_cpu2`: 40,73% canônica vs
37,76% síntese). O sinal da divergência **não é constante** — inverte conforme a
configuração — e em um caso **reordena** três configurações: sob a canônica,
`ml_aggr` (31,51) < `h9c_tau90_cpu1` (31,65) < `native_cpu1` (32,59); sob a da
síntese, a ordem se inverte inteiramente: `ml_aggr` (34,14) > `native_cpu1`
(30,42) > `h9c_tau90_cpu1` (29,05).

**Qual é a vigente.** A definição **canônica** é a que deve figurar nos capítulos
de Metodologia e Resultados, por três razões documentadas: (i) é a produzida pelo
pipeline de análise (`analyze_frontier.py:281` computa as duas, mas os relatórios
tabulares usam a canônica); (ii) é a adotada pelos documentos de resultado mais
recentes e mais escrutinados (H9d, Bloco 7); (iii) `INVENTARIO_solucoes.md`, a
fonte mestra deste índice, já normalizou todas as suas tabelas para ela e registra
a divergência explicitamente em seu §0. Qualquer número de TS% herdado de
`SINTESE_resultados_metodologia.md §4` ou §8 sem essa checagem deve ser
revalidado contra `ts_definitions.csv` antes de entrar no texto final. Os
**BD-rate não são afetados** — a divergência é só de TS%.

### 8.2 Nomenclatura dos pontos de operação do H9a — Fase 5 vs tabelas mestras

`RESULTADOS_fase5.md §1` relata, por sequência, os pontos **P0, P_ref, A2, A3**
(por exemplo, Jockey: P_ref 0,92%/32,6%). Já `SINTESE §4` e `INVENTARIO §3.1`
relatam, como resultado agregado da Fase 5, os pontos **P0, P_rect, P_ref**
(médias: P0 0,155%/19,05%; P_rect 0,464%/26,49%; P_ref 0,595%/29,53%), com um
ponto `P_rect` que não aparece nomeado em `RESULTADOS_fase5.md`. Os documentos-mestre
não citam o script que produziu especificamente essa agregação de três pontos
(possivelmente `results/benchmark/fase5_final.py`, presente no repositório mas
sem documentação de uso associada). **Isto não foi possível resolver nesta
auditoria** com o material disponível: não há como confirmar, sem executar o
script, se `P_ref` em `RESULTADOS_fase5.md` e `P_ref` nas tabelas mestras
referem-se à mesma configuração de τ. Fica registrado como item a esclarecer antes
de citar os números da Fase 5 no capítulo de Resultados — recomenda-se preferir os
números por sequência de `RESULTADOS_fase5.md`, que têm proveniência direta em
`results/benchmark/h9_test/`.

### 8.3 Preço do knob de τ do H9d — dois estimadores, não um erro

`RESULTADOS_H9d_CTC.md §3.1` reporta o preço do knob de τ do H9a como **0,063
pp/pp**; `SINTESE §5-quater` (nota da fronteira completa de 27/07) reporta **0,0606
pp/pp** para o mesmo knob. Não é um conflito no sentido do §8.1: os dois números
vêm de estimadores diferentes e declarados — interpolação **por sequência**
(`ctc_h9d_marginal.py`) contra a **média-das-médias** sobre as 8 sequências — e os
próprios documentos-fonte alertam que concordam a ~4% e não devem ser misturados
na mesma tabela. Registrado aqui para que o capítulo de Resultados escolha **um**
dos dois e o declare, em vez de citar ambos sem qualificação.

---

## 9. Artefatos citados mas ausentes do repositório

Uma verificação posterior por `Glob`/`ls`, datada de 2026-07-29, corrigiu duas
entradas desta seção que estavam erradas: `results/dataset_h9/manifest.csv.bak-4116`
**existe** (21.043 bytes, não rastreado pelo git), e o script de geração de
`fase6_swap_combo/raw_results.csv`, `src/scripts/fase6/encode_swap_combo.py`,
**existe** e está listado em `docs/RASTREABILIDADE.md`, seção 2.3b. Nenhum dos
dois é, portanto, um artefato ausente, e ambos foram removidos da tabela
abaixo. A entrada sobre a confirmação linha a linha do Gate 3 H9a foi corrigida
de modo análogo: o valor não está em `gate3_h9a.csv`, mas está confirmado em
`results/models/student_h9a/oracle_sim_student.csv` — ver a linha "Gate 3 —
validação (H9a)" na Seção 3.

| artefato citado | citado em | situação verificada |
|---|---|---|
| CSV de saída de `ctc_h9d_marginal.py` (tabela de preço do tempo por sequência, `RESULTADOS_H9d_CTC.md §3`) | `RESULTADOS_H9d_CTC.md §3.1-3.2` | o script existe em `src/scripts/fase6/ctc_h9d_marginal.py`; não foi localizado um CSV de saída versionado especificamente para essa tabela — os números parecem ter sido transcritos de execução direta para o documento |
| Log/saída versionada do custo implantado do pruner (§6.2b de `RESULTADOS_microbench_pruner.md`) | `RESULTADOS_microbench_pruner.md §6` | o script `microbench_pruner.py` e `results/benchmark/microbench/pruner_cost.csv` existem; a tabela específica de §6.2 (Tango/BoxingPractice, acumuladores `g_pt_h9a_feat`) não tem um CSV próprio identificado — parece ser saída de log de execução única, não re-executável a partir de um artefato tabular versionado |
| CSV da agregação Pareto de 3 sequências (Boxing/FoodMarket2/Tango) citada em `SINTESE §6` | `SINTESE_resultados_metodologia.md §6` | não localizado; o próprio documento-fonte já qualifica essa fronteira como "de uma análise anterior" e nota que sua recomposição com o H9d está pendente — tratar como número histórico, não como referência a reproduzir |

Nenhum outro artefato citado nos 25 documentos `RESULTADOS_*.md`, em
`INVENTARIO_solucoes.md`, em `SINTESE_resultados_metodologia.md` ou em
`RASTREABILIDADE.md` deixou de ser localizado nesta varredura: todos os diretórios
de `results/benchmark/` e `results/models/` nomeados nos documentos-fonte foram
confirmados presentes, com os arquivos internos esperados (CSVs de agregação,
`students.pt`, relatórios `.md`), exceto os três itens acima.
