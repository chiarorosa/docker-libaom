# A2 — Tabelas e figuras

Este documento apresenta o plano de tabelas e de figuras dos capítulos de
Metodologia e de Resultados, na forma de especificação, e não de produto
acabado. Nenhuma figura existe ainda no projeto, conforme registrado em
`00_PLANO_capitulos.md §5` e reafirmado em `A3_RETRATACOES_E_LACUNAS.md`
(lacuna L11), e este documento é o que torna essa lacuna executável: para cada
tabela e cada figura são declarados o número e o título provisórios, o
capítulo e a seção de destino, a legenda redigida no padrão fixo do perfil
estilístico da tese, o conteúdo exato de colunas e linhas ou de eixos e
séries, o arquivo de dados de origem — verificado nesta auditoria por leitura
direta do sistema de arquivos em `results/` — e o script que o produz ou a
observação precisa do que falta ser calculado. A numeração é sequencial, na
ordem de aparição ao longo dos dois capítulos, do primeiro parágrafo de
`M1_objeto_e_formulacao.md` ao encerramento da análise integrada em
`R6_analise_integrada.md`, documento este último ainda não redigido, cujas
tabelas e figuras são aqui especificadas de forma prospectiva a partir da
evidência já registrada em `A1_INDICE_evidencias.md §7`.

Duas convenções valem para todo o documento. A primeira é de numeração dupla:
quatro tabelas do Capítulo de Resultados — as do H9a, na Seção 2 — já foram
redigidas com número próprio no corpo de `R2_h9a.md`, nos moldes "Tabela 2.1"
a "Tabela 2.4", que seguem a convenção capítulo-seção da tese finalizada; a
numeração sequencial deste documento é provisória e cada entrada informa,
quando pertinente, a correspondência com o número já fixado no texto-fonte. A
segunda é de verificação: a coluna ou o campo "dado de origem" só cita um
caminho depois de confirmado por busca de arquivo nesta auditoria: o que não
foi localizado está listado, sem exceção, na Seção 4.

---

## 1. Tabelas

### Capítulo de Metodologia

#### Tabela 1 — Decomposição do custo de busca local por família de candidato

- **Destino.** Metodologia §1 (`M1_objeto_e_formulacao.md`, Seção 1.3).
- **Legenda.** A Tabela 1 apresenta a decomposição do tempo de busca local por
  família de candidato de partição, medida sobre 875.317 nós de decisão das
  três sequências do conjunto de teste reservado, em codificação intraquadro
  com `cpu-used=0`, com a coluna da divisão quadrada excluída por conter a
  recursão.
- **Colunas.** família de candidato (`PARTITION_NONE`; formas retangulares;
  formas AB; formas 4-way; partições estendidas, soma de AB e 4-way) — %
  agregado do tempo de busca local — mínimo entre as três sequências — máximo
  entre as três sequências.
- **Linhas.** cinco: as quatro famílias e a soma das partições estendidas.
- **Dado de origem.** `results/benchmark/partstats/part_timing_t1.csv`,
  `results/benchmark/partstats_racenight/part_timing.csv`,
  `results/benchmark/partstats_riverbank/part_timing.csv` (verificados por
  `Glob`; arquivos sem cabeçalho, layout definido em
  `CONFIG_COLLECT_PARTITION_STATS` da libaom).
- **Script.** `src/scripts/benchmark/analyze_partstats.py`. A tabela já está
  redigida em prosa na Seção 1.3 da Metodologia; falta apenas convertê-la em
  tabela formatada — não há cálculo novo a fazer.

#### Tabela 2 — Decomposição do custo das partições estendidas por tamanho de bloco

- **Destino.** Metodologia §1 (Seção 1.3), imediatamente após a Tabela 1.
- **Legenda.** A Tabela 2 apresenta a fração do tempo de busca local consumida
  pelas partições estendidas, por tamanho de bloco, sobre o mesmo conjunto de
  875.317 nós de decisão da Tabela 1.
- **Colunas.** tamanho de bloco (8×8; 16×16; 32×32; 64×64; 128×128) — % do
  tempo local consumido pelas partições estendidas.
- **Linhas.** cinco, uma por tamanho de bloco.
- **Dado de origem.** Os mesmos três arquivos da Tabela 1.
- **Script.** `src/scripts/benchmark/analyze_partstats.py`, com a agregação
  por tamanho de bloco em vez de agregação global. Valores já redigidos em
  prosa na Seção 1.3; conversão direta em tabela.

#### Tabela 3 — Composição do conjunto de dados por partição

- **Destino.** Metodologia §2 (`M2_instrumentacao_e_dataset.md`, Seção 2.4).
- **Legenda.** A Tabela 3 apresenta a composição do conjunto de dados por
  partição de treino, validação e teste reservado, com o número de amostras e
  o número de nós de decisão em cada uma, extraídos de dezesseis sequências
  UVG 4K sob quatro pontos de quantização.
- **Colunas.** partição (treino; validação; teste reservado) — sequências —
  amostras totais (milhões) — nós de decisão (milhões).
- **Linhas.** três, uma por partição, mais uma linha de total.
- **Dado de origem.** `results/dataset_h9/manifest.csv` (verificado por
  `Glob`).
- **Script.** `src/scripts/partition_dataset/rebuild_manifest_stats.py`.
  Valores já redigidos em prosa na Seção 2.4; conversão direta em tabela.

#### Tabela 4 — As duas definições de redução de tempo: divergência por configuração

- **Destino.** Metodologia §3 (`M3_protocolo_avaliacao.md`, Seção 3.5).
- **Legenda.** A Tabela 4 apresenta a divergência entre as duas definições de
  redução de tempo em uso no projeto, a canônica e a da síntese, para as
  dezessete configurações da grade completa de substituição direta do H9a e
  do H9c, com a diferença em pontos percentuais entre as duas.
- **Colunas.** configuração — TS% (definição canônica) — TS% (definição da
  síntese) — Δ (pontos percentuais).
- **Linhas.** dezessete, uma por configuração (seis níveis de `cpu-used` × H9a
  balanceado/agressivo e H9c τ=0,90/0,95, mais os três *presets* nativos).
- **Dado de origem.** `results/benchmark/fase6_analysis/ts_definitions.csv`
  (verificado por `Glob` e conferido diretamente: colunas
  `config,ts_canonical,ts_sintese,delta_pp`).
- **Script.** `src/scripts/fase6/analyze_frontier.py`. Dado já calculado e
  tabulado no artefato; requer apenas seleção de colunas e ordenação por
  magnitude de Δ.

#### Tabela 5 — Piso de ruído medido na comparação pareada de tempo

- **Destino.** Metodologia §3 (Seção 3.6).
- **Legenda.** A Tabela 5 apresenta o piso de ruído medido por repetição
  intercalada de três configurações sobre quatro pontos de quantização na
  sequência Crosswalk, com o coeficiente de variação do tempo bruto e o
  desvio da redução de tempo pareada por configuração.
- **Colunas.** configuração (âncora; `ml_balanced`; `native_cpu1`) —
  coeficiente de variação mediano do tempo bruto (%) — desvio da redução de
  tempo pareada (±pp) — resolução efetiva (pp, dois desvios).
- **Linhas.** três, uma por configuração.
- **Dado de origem.** `results/benchmark/fase6_repeat/raw_results.csv`
  (verificado por `Glob`; cinco repetições por ponto).
- **Script.** `src/scripts/fase6/encode_repeat.py` (geração) e
  `src/scripts/fase6/report_e3_dec_e2.py` (agregação: CV e desvio pareado).
  Valores já redigidos em prosa na Seção 3.6 e na Seção 2.7 de Resultados;
  conversão direta em tabela.

#### Tabela 6 — Vetor de atributos por bloco (A–E)

- **Destino.** Metodologia §4 (`M4_atributos_e_politica.md`, Seção 4.1).
- **Legenda.** A Tabela 6 apresenta a decomposição do vetor de atributos das
  soluções da família H9 em cinco blocos, com o número de atributos de cada
  um, o momento em que a informação fica disponível no fluxo de controle do
  codificador e a solução que o utiliza.
- **Colunas.** bloco (A a E) — conteúdo — número de atributos — momento de
  disponibilidade (pré-busca; pós-`PARTITION_NONE`) — utilizado por (H9a;
  H9c/H9d).
- **Linhas.** cinco, uma por bloco.
- **Dado de origem.** Não há CSV de origem: a tabela é construída por leitura
  direta do código-fonte, `src/scripts/partition_model/features.py:196-220`
  (layout do vetor, já citado em `A1_INDICE_evidencias.md §1` e §2).
- **Script/observação.** Sem script de geração — tabela de elaboração própria
  a partir da inspeção do código-fonte; nenhum dado numérico é inventado, pois
  todos os valores (contagens de atributos, índices) já constam de
  `RESULTADOS_auditoria_dominio_pixels.md §2` e de `R1_dominio_pixels.md
  §1.6`.

#### Tabela 7 — Os quatro desenhos de atribuição

- **Destino.** Metodologia §6 (`M6_modelos_e_atribuicao.md`, Seção 6.6).
- **Legenda.** A Tabela 7 apresenta os quatro desenhos experimentais adotados
  para separar o mérito do modelo do mérito da política de poda, com a
  pergunta que cada um responde, a variável controlada e a variável
  manipulada.
- **Colunas.** desenho (ablação de atribuição em três braços; substituição
  direta do podador nativo; decomposição com neutralização de alavanca;
  simulação oráculo) — pergunta respondida — variável controlada — variável
  manipulada — seção de Resultados que o utiliza.
- **Linhas.** quatro, uma por desenho.
- **Dado de origem.** Tabela conceitual, sem CSV de origem — elaboração
  própria a partir da prosa das Seções 6.6 a 6.10 de `M6_modelos_e_atribuicao.md`.
  Nenhum número é introduzido; a tabela é puramente organizativa.
- **Script/observação.** Não se aplica.

---

### Capítulo de Resultados

#### Tabela 8 — Curva de operação do estudante de pixels na sequência Jockey

- **Destino.** Resultados §1 (`R1_dominio_pixels.md`, Seção 1.1).
- **Legenda.** A Tabela 8 apresenta a curva de operação do modelo estudante da
  era de pixels (`pixels24`) na sequência Jockey, medida em codificação
  intraquadro com `cpu-used=0` e quatro pontos de quantização, e a cota
  superior por reprodução das decisões do modelo substituto convolucional
  (H8).
- **Colunas.** ponto de operação (comprometimento com NONE apenas; + poda das
  retangulares; + limiares por nível; varredura agressiva: A1, A2, A3;
  reprodução do substituto H8, conservador e agressivo) — taxa BD (%) —
  aceleração (×).
- **Linhas.** oito, uma por ponto de operação citado no texto.
- **Dado de origem.** `results/benchmark/h7h8_real_summary.csv` (colunas
  `point,bd_rate_pct,time_speedup_x`, conferido: linhas `P0_oldpolicy`,
  `P_rect`, `P_ref`, `H8_surrogate`) e `results/benchmark/h7h8_aggr/summary.csv`
  (linhas `A1_none80`, `A2_none70`, `A3_none60_rest40`,
  `H8_surrogate_aggr`), ambos verificados por `Glob` e por leitura direta.
- **Script.** `src/scripts/benchmark/h7h8_bench.py` e
  `src/scripts/partition_model/surrogate_replay.py`. Dado já tabulado nos dois
  CSVs; a tabela apenas reúne as duas fontes em uma única grade ordenada por
  aceleração.

#### Tabela 9 — Hierarquia no crivo ponderado por *regret*

- **Destino.** Resultados §1 (Seção 1.4).
- **Legenda.** A Tabela 9 apresenta a fração de *regret* de cada subconjunto
  de atributos avaliado no crivo ponderado por custo de taxa-distorção real,
  medida a 25% e a 30% de redução de custo casada, sobre seis sequências do
  conjunto de validação e de teste reservado.
- **Colunas.** subconjunto (variância; ConvNeXt com alvo de *regret*; ConvNeXt
  com entropia cruzada; `pixels24`; H9a; escore aleatório, como piso) —
  fração de *regret* a 25% de `cost_red` — fração de *regret* a 30% de
  `cost_red`.
- **Linhas.** seis, uma por subconjunto, ordenadas da pior para a melhor.
- **Dado de origem.** `results/models/oracle_regret/frontier.csv` (colunas
  `pruner,tau,cost_red,split_lost,reg_abs,reg_rel,reg_frac_pct,...`,
  conferido) e `results/models/oracle_regret_convnext/frontier.csv`, ambos
  verificados por `Glob`.
- **Script.** `src/scripts/partition_model/oracle_regret.py`. Requer filtrar,
  em cada arquivo, a linha de `cost_red` mais próxima de 25 e de 30 por
  `pruner`, pois a grade de `tau` não cai exatamente nesses valores para todo
  subconjunto — mesmo procedimento de interpolação/seleção usado para redigir
  a Seção 1.4 em prosa.

#### Tabela 10 — Redução de custo de busca por subconjunto de atributos (Tabela 2.1 no texto-fonte)

- **Destino.** Resultados §2 (`R2_h9a.md`, Seção 2.1). Já redigida no
  texto-fonte como "Tabela 2.1".
- **Legenda.** Já redigida em `R2_h9a.md:35-38`: "A Tabela 2.1 apresenta a
  redução de custo por subconjunto de atributos, em três níveis de risco
  casado. [...] Dez sequências de treino, sessenta mil superblocos. Métrica de
  oráculo; não há redução de tempo de parede nesta tabela."
- **Colunas.** subconjunto de atributos (variância; `pixels24`; H9a; H9c) —
  redução de custo a 0,5% de perda de SPLIT — a 1% — a 2%.
- **Linhas.** quatro, já preenchidas no texto-fonte.
- **Dado de origem.** `results/models/gate2_final_sweep.csv` (verificado por
  `Glob`), por seleção de risco casado: para cada subconjunto, toma-se a linha
  cujo `split_lost` fica imediatamente abaixo de cada patamar de risco.
  `results/models/gate2_final.csv` reúne uma agregação distinta, por limiares
  fixos de perda de SPLIT, e não é a fonte destes valores.
- **Script.** `src/scripts/partition_model/gate2_signal.py`. Tabela já
  redigida por completo; nenhuma ação além de formatação é necessária.

#### Tabela 11 — Curva do H9a no conjunto de teste reservado (Tabela 2.2 no texto-fonte)

- **Destino.** Resultados §2 (Seção 2.3). Já redigida como "Tabela 2.2".
- **Legenda.** Já redigida em `R2_h9a.md:141-143`: "A Tabela 2.2 apresenta a
  curva de taxa BD contra redução de tempo, por sequência e por ponto de
  operação. [...] Taxa BD sobre PSNR-Y; redução de tempo como
  `(1 − 1/aceleração)·100`."
- **Colunas.** ponto (P0; P_ref; A2; A3) — Jockey — RaceNight — RiverBank
  (cada célula: taxa BD / redução de tempo / aceleração).
- **Linhas.** quatro, já preenchidas.
- **Dado de origem.**
  `results/benchmark/h9_test/{Jockey,RaceNight,RiverBank}/{curve_safe,curve_aggr,ablation}/{summary.csv,curve.csv}`
  (verificados por `Glob`: presentes como `curve_safe/{runs,summary}.csv`,
  `curve_aggr/{runs,summary}.csv` e `ablation/{curve,runs}.csv` nas três
  sequências).
- **Script.** `results/benchmark/fase5_final.py` e
  `src/scripts/benchmark/analyze_ablation.py`. Tabela já redigida por
  completo; a nota de `A1_INDICE_evidencias.md §8.2` registra uma divergência
  de nomenclatura entre os pontos "P_rect" da síntese e "A2" da Fase 5 — não
  afeta esta tabela, que usa os nomes de `RESULTADOS_fase5.md` diretamente.

#### Tabela 12 — Resultados finais na CTC (Tabela 2.3 no texto-fonte)

- **Destino.** Resultados §2 (Seção 2.4). Já redigida como "Tabela 2.3".
- **Legenda.** Já redigida em `R2_h9a.md:222-224`: "A Tabela 2.3 apresenta os
  dois pontos de operação do H9a contra a âncora e contra os *presets*
  nativos. [...] Âncora libaom `cpu-used=0`; taxa BD sobre PSNR-Y; redução de
  tempo na definição canônica."
- **Colunas.** configuração (H9a equilibrado; H9a agressivo; `cpu-used=1`;
  `cpu-used=2`; `cpu-used=3`) — taxa BD — redução de tempo — aceleração.
- **Linhas.** cinco, já preenchidas.
- **Dado de origem.** `results/benchmark/fase6/bdrate_average.csv` (colunas
  `config,bd_rate,bd_psnr,ts_pct,speedup`, conferido: linhas `ml_balanced`,
  `ml_aggr`, `native_cpu1`, `native_cpu2`, `native_cpu3`), verificado por
  `Glob`.
- **Script.** `src/scripts/fase6/{encode_ctc.py,report_ctc.py}`. Tabela já
  redigida por completo.

#### Tabela 13 — Substituição direta do podador nativo pelo H9a (Tabela 2.4 no texto-fonte)

- **Destino.** Resultados §2 (Seção 2.5). Já redigida como "Tabela 2.4".
- **Legenda.** Já redigida em `R2_h9a.md:287-289`: "A Tabela 2.4 apresenta o
  resultado, na definição canônica de redução de tempo. [...] Substituição
  direta do podador nativo pelo H9a, a `cpu-used` fixo. Âncora libaom
  `cpu-used=0`; oito sequências da Classe A1; redução de tempo na definição
  canônica."
- **Colunas.** `cpu-used` — podador (rede convolucional nativa; H9a
  equilibrado; H9a agressivo) — taxa BD — redução de tempo — aceleração.
- **Linhas.** nove, já preenchidas.
- **Dado de origem.** `results/benchmark/fase6_swap/swap_average.csv`
  (verificado por `Glob`).
- **Script.** `src/scripts/fase6/{encode_swap.py,report_swap.py}`. Tabela já
  redigida por completo.

#### Tabela 14 — O fator de confusão do H9a no H9c, sequência Neon1224

- **Destino.** Resultados §3 (`R3_h9c.md`, Seção 3.3). Já redigida como
  tabela sem número explícito no texto-fonte.
- **Legenda.** A Tabela 14 apresenta a quantificação do fator de confusão
  identificado na caracterização do H9c: a taxa BD e a redução de tempo do
  H9c medido junto com o H9a nos limiares compilados por padrão, contra o H9c
  isolado, com o H9a explicitamente neutralizado, na sequência Neon1224 a
  `cpu-used=0`.
- **Colunas.** configuração (H9c a τ=0,95, medido antes; H9c a τ=0,95,
  isolado; H9c a τ=0,90, medido antes; H9c a τ=0,90, isolado; H9c a τ=0,60,
  medido antes; H9c a τ=0,60, isolado) — taxa BD — redução de tempo.
- **Linhas.** seis, já preenchidas em `R3_h9c.md:122-129`.
- **Dado de origem.** `results/benchmark/fase6/raw_results.csv` (linhas de
  `config` iniciadas por `h9ciso_`, verificado por `Glob` e por leitura
  direta: colunas `seq,config,cq,fps_num,fps_den,frames,bytes,psnr_y,time_s`).
- **Script.** `src/scripts/fase6/encode_h9c_iso.py` e
  `src/scripts/fase6/report_bloco7.py`. Tabela já redigida por completo.

#### Tabela 15 — Substituição direta do H9c na grade CTC completa

- **Destino.** Resultados §3 (Seção 3.4). Já redigida como tabela sem número
  explícito.
- **Legenda.** A Tabela 15 apresenta o resultado agregado da substituição
  direta do podador nativo pelo H9c, em três níveis de *preset* e dois
  limiares, sobre a grade CTC completa de oito sequências e quatro pontos de
  quantização.
- **Colunas.** *preset* — podador (rede convolucional nativa; H9c τ=0,90; H9c
  τ=0,95) — taxa BD (%) — redução de tempo (%) — aceleração.
- **Linhas.** nove, já preenchidas em `R3_h9c.md:207-217`.
- **Dado de origem.** `results/benchmark/fase6_swap_h9c/swap_average.csv`
  (colunas `level,kind,bd_rate,bd_psnr,ts_pct,speedup`, conferido: linhas
  `native`, `h9c_tau90`, `h9c_tau95` nos níveis 1, 2 e 3), verificado por
  `Glob`.
- **Script.** `src/scripts/fase6/{encode_swap_h9c.py,report_swap.py}`. Tabela
  já redigida por completo.

#### Tabela 16 — Decomposição de três pernas e interação medida

- **Destino.** Resultados §3 (Seção 3.5). Já redigida como tabela sem número
  explícito.
- **Legenda.** A Tabela 16 apresenta a decomposição aditiva do H9a e do H9c
  sobre quatro sequências CTC, com o podador pré-busca isolado, o podador
  pós-NONE isolado, os dois empilhados, a soma aritmética das partes isoladas
  e a interação medida entre eles.
- **Colunas.** sequência (Neon1224; PierSeaSide; Tango; TimeLapse; média) —
  H9a só — H9c só — H9a + H9c — soma — interação (pp).
- **Linhas.** cinco, já preenchidas em `R3_h9c.md:298-304`.
- **Dado de origem.** `results/benchmark/fase6/raw_results.csv` (linhas
  `h9adef`, `h9ciso_tau90`, `ml_balanced` das quatro sequências citadas),
  verificado por `Glob`.
- **Script.** `src/scripts/fase6/{encode_h9adef.py,report_e3_dec_e2.py}`.
  Tabela já redigida por completo.

#### Tabela 17 — Critério de predizibilidade offline do H9d, por nível de bloco

- **Destino.** Resultados §4 (`R4_h9d.md`, Seção 4.2).
- **Legenda.** A Tabela 17 apresenta a área sob a curva característica de
  operação do receptor do critério de predizibilidade das partições
  estendidas, por tamanho de bloco e por conjunto de atributos, sobre
  792.840 nós de decisão do conjunto reservado.
- **Colunas.** tamanho de bloco (16; 32; 64; agregado) — nós avaliados — base
  de positivos (%) — AUC-ROC (36 atributos) — AUC-ROC (39 atributos).
- **Linhas.** quatro, uma por nível mais o agregado.
- **Dado de origem.** `results/models/h9d_predictability/students.pt` e
  `results/models/h9d_predictability/run.log` (36 atributos);
  `results/models/h9d_predictability_h9c/students.pt` e `run.log` (39
  atributos), todos verificados por `Glob`.
- **Script/observação.** `src/scripts/partition_model/h9d_predictability.py`
  (com `--feature-set h9c` para a variante de 39 atributos). Os valores de AUC
  citados na Seção 4.2 provêm da saída de execução registrada em `run.log`, e
  não de um CSV tabular dedicado — a tabela precisa ser montada a partir do
  log, ou o script precisa ser estendido para gravar um CSV de métricas por
  nível, o que não foi feito.

#### Tabela 18 — Resultado sob protocolo CTC do H9d — contribuição marginal

- **Destino.** Resultados §4 (Seção 4.6).
- **Legenda.** A Tabela 18 apresenta a contribuição marginal do H9d sobre o
  H9a no ponto balanceado implantado, medida nas oito sequências da CTC, e o
  preço do botão de limiar do H9a no mesmo segmento, para comparação direta.
- **Colunas.** configuração (H9a balanceado; H9a + H9d PL10) — taxa BD (%) —
  redução de tempo (%) — aceleração — preço (pp de taxa BD por pp de tempo).
- **Linhas.** duas configurações mais uma linha de referência do preço do
  botão de limiar (0,063 pp/pp).
- **Dado de origem.** `results/benchmark/fase6/bdrate_average.csv` (linhas
  `ml_balanced` e `ml_bal_h9d`, conferido: `0.5676/17.7238/1.2226` e
  `0.5858/18.739/1.2378`), verificado por `Glob`.
- **Script.** `src/scripts/fase6/{ctc_h9d.py,report_ctc.py,ctc_h9d_marginal.py}`.
  Tabela já redigida em prosa na Seção 4.6; conversão direta.

#### Tabela 19 — Fronteira bidimensional do H9d

- **Destino.** Resultados §4 (Seção 4.7).
- **Legenda.** A Tabela 19 apresenta a família completa de configurações do
  H9d, cruzando duas bases do H9a com duas forças de calibração do H9d, sob o
  mesmo protocolo CTC de oito sequências.
- **Colunas.** base do H9a (balanceada; agressiva) — calibração do H9d (PL10;
  PL20) — taxa BD marginal (pp) — redução de tempo marginal (pp) — preço
  (pp/pp).
- **Linhas.** quatro, uma por combinação.
- **Dado de origem.** `results/benchmark/fase6/raw_results.csv`, filtrando as
  quatro configurações `ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d` e
  `ml_aggr_h9d_pl20` (32 linhas cada, oito sequências × quatro CQ), verificado
  por `Glob`.
- **Script.** `src/scripts/fase6/ctc_h9d.py` (extensão que gerou 96
  codificações novas para as três configurações adicionais; a quarta,
  `ml_bal_h9d`, já havia sido medida na campanha CTC anterior, de modo que as
  quatro configurações juntas somam 128 linhas) e
  `src/scripts/benchmark/bd_rate.py` para o cálculo de taxa BD a partir de
  `(bits, psnr_y)` por `cq`. Os quatro números finais já estão redigidos em
  prosa na Seção 4.7; a tabela requer recalcular a taxa BD por combinação a
  partir do CSV bruto, pois não há CSV agregado específico desta campanha —
  ver Seção 4 (dados ausentes).

### Capítulo de Resultados — Análise integrada (prospectiva)

As duas tabelas seguintes pertencem a `R6_analise_integrada.md`, documento
ainda não redigido no diretório. A evidência que as sustenta já está indexada
em `A1_INDICE_evidencias.md §7`, e ambas são especificadas aqui para que a
redação da Seção 6 de Resultados as produza diretamente, em vez de as
descobrir depois.

#### Tabela 20 — Fronteira de compromisso global consolidada

- **Destino.** Resultados §6 (`R6_analise_integrada.md`, prospectiva).
- **Legenda.** A Tabela 20 apresenta a fronteira de compromisso entre taxa BD
  e redução de tempo de todas as soluções desta tese, medida sobre as oito
  sequências da Classe A1 das condições comuns de teste, com os pontos das
  duas soluções positivas — H9a e H9d — e os três *presets* nativos como
  referência, resolvendo a lacuna registrada em `A3_RETRATACOES_E_LACUNAS.md`
  (L1) sobre a ausência do H9d na versão anterior desta fronteira, construída
  sobre apenas três sequências.
- **Colunas.** configuração (`cpu-used=1`; `cpu-used=2`; `cpu-used=3`; H9a
  balanceado; H9a + H9d PL10; H9a agressivo; H9c τ=0,95 no *preset* 1; H9c
  τ=0,95 no *preset* 2) — taxa BD — redução de tempo — aceleração — status na
  fronteira de Pareto (dominado / não dominado).
- **Linhas.** vinte e quatro, uma por configuração avaliada, com a coluna de
  status marcando as quinze não dominadas.
- **Dado de origem.** `results/benchmark/fase6_analysis/pareto_frontier.csv`,
  regenerado em 2026-07-29 por `src/scripts/fase6/analyze_frontier.py` a
  partir de `results/benchmark/{fase6,fase6_swap,fase6_swap_h9c}/raw_results.csv`,
  sobre as oito sequências da CTC: vinte e quatro configurações avaliadas,
  quinze marcadas como não dominadas.
- **Script/observação.** O script já produz a tabela e a marcação de
  dominância diretamente (coluna `dominated_by`, vazia nos pontos não
  dominados); não é necessário unir fontes nem recalcular a fronteira à mão.
  Ver `docs/RESULTADOS_fronteira_pareto_global.md` §2 (comando de reprodução)
  e §3 (tabela dos pontos não dominados).

#### Tabela 21 — As três conclusões da tese, com o número que as sustenta

- **Destino.** Resultados §6, fechamento.
- **Legenda.** A Tabela 21 apresenta as três conclusões consolidadas desta
  tese, com o enunciado, o número central que a sustenta e a seção do
  capítulo de Resultados em que a evidência é apresentada pela primeira vez.
- **Colunas.** conclusão — enunciado — número central — seção de origem.
- **Linhas.** três: (i) nenhuma via de pixels compete com o contexto de
  taxa-distorção barato; (ii) o contexto de taxa-distorção barato não supera o
  podador nativo na média da grade CTC, mas preenche a granularidade fina de
  baixo regime de aceleração; (iii) alavancas de poda se somam na medida em
  que os conjuntos de candidatos que atacam são disjuntos, e não em função da
  informação que compartilham.
- **Dado de origem.** Tabela de síntese, sem CSV próprio — reúne números já
  tabulados nas Tabelas 9, 12/20 e 16/18 respectivamente.
- **Script/observação.** Não se aplica; tabela de fechamento textual.

---

## 2. Figuras

As figuras candidatas mínimas exigidas pelo escopo deste plano — a fronteira
de compromisso global, as curvas de limiar por sequência, a hierarquia no
crivo de *regret*, a decomposição do custo de busca por família de
candidatos, a decomposição de ganho entre alavancas empilhadas e a
comparação a tempo casado entre fontes de escore — estão todas cobertas
abaixo, respectivamente pelas Figuras 1, 5, 2, 1 (mesma figura da
decomposição de custo, ver nota), 6 e 4. As Figuras 3, 7 e 8 são adicionais,
justificadas por conteúdo já redigido nos documentos-fonte que se beneficia
de leitura visual.

#### Figura 1 — Decomposição do custo de busca por família de candidato

- **Destino.** Metodologia §1 (Seção 1.3), companheira das Tabelas 1 e 2.
- **Legenda.** A Figura 1 apresenta a decomposição do tempo de busca local por
  família de candidato de partição, com barras agrupadas por sequência do
  conjunto de teste reservado, evidenciando que nenhuma família isolada de
  partição estendida ultrapassa 7,22% do tempo, ainda que a soma das seis
  formas atinja 34,3%.
- **Tipo de gráfico.** Barras empilhadas horizontais, uma barra por sequência
  (Jockey, RaceNight, RiverBank, mais o agregado), com segmentos coloridos por
  família de candidato.
- **Eixos e séries.** Eixo horizontal: % do tempo de busca local (0 a 100).
  Eixo vertical: sequência. Séries (cor): `PARTITION_NONE`, retangular
  horizontal, retangular vertical, AB (quatro formas somadas ou desagregadas,
  a decidir na produção), 4-way.
- **Dado de origem.** `results/benchmark/partstats/part_timing_t1.csv`,
  `results/benchmark/partstats_racenight/part_timing.csv`,
  `results/benchmark/partstats_riverbank/part_timing.csv`.
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  from analyze_partstats import load_and_aggregate  # reutilizar o parser existente

  seqs = {
      "Jockey": "results/benchmark/partstats/part_timing_t1.csv",
      "RaceNight": "results/benchmark/partstats_racenight/part_timing.csv",
      "RiverBank": "results/benchmark/partstats_riverbank/part_timing.csv",
  }
  # load_and_aggregate deve replicar o mapeamento de colunas sem cabeçalho
  # já usado por analyze_partstats.py; NÃO reanalisar o CSV bruto às cegas.
  rows = []
  for seq, path in seqs.items():
      agg = load_and_aggregate(path)  # -> dict família: % do tempo local
      agg["seq"] = seq
      rows.append(agg)
  df = pd.DataFrame(rows).set_index("seq")
  df["estendidas"] = df["AB"] + df["4way"]
  df[["none", "rect", "AB", "4way"]].plot(
      kind="barh", stacked=True, figsize=(8, 3)
  )
  plt.xlabel("% do tempo de busca local")
  plt.tight_layout()
  plt.savefig("fig_decomposicao_custo_busca.png", dpi=200)
  ```

#### Figura 2 — Hierarquia no crivo ponderado por *regret*

- **Destino.** Resultados §1 (Seção 1.4), companheira da Tabela 9.
- **Legenda.** A Figura 2 apresenta a fração de *regret* de cada subconjunto
  de atributos avaliado no crivo ponderado por custo de taxa-distorção real, a
  25% de redução de custo casada, em escala logarítmica, evidenciando a
  distância entre a variância isolada e o H9a.
- **Tipo de gráfico.** Barras verticais, ordenadas do maior para o menor
  valor de fração de *regret*, com eixo logarítmico.
- **Eixos e séries.** Eixo horizontal: subconjunto de atributos (variância;
  ConvNeXt-*regret*; ConvNeXt-CE; `pixels24`; H9a). Eixo vertical (log):
  fração de *regret* (%) a `cost_red` = 25%.
- **Dado de origem.** `results/models/oracle_regret/frontier.csv` e
  `results/models/oracle_regret_convnext/frontier.csv`.
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  df1 = pd.read_csv("results/models/oracle_regret/frontier.csv")
  df2 = pd.read_csv("results/models/oracle_regret_convnext/frontier.csv")
  df = pd.concat([df1, df2], ignore_index=True)

  def nearest_at_cost_red(df, target=25.0):
      out = []
      for pruner, g in df.groupby("pruner"):
          idx = (g["cost_red"] - target).abs().idxmin()
          out.append(g.loc[idx])
      return pd.DataFrame(out)

  point = nearest_at_cost_red(df, 25.0).sort_values("reg_frac_pct", ascending=False)
  plt.figure(figsize=(6, 4))
  plt.bar(point["pruner"], point["reg_frac_pct"])
  plt.yscale("log")
  plt.ylabel("fração de regret (%) a cost_red≈25%")
  plt.xticks(rotation=30, ha="right")
  plt.tight_layout()
  plt.savefig("fig_hierarquia_regret.png", dpi=200)
  ```

#### Figura 3 — Diagrama de confiabilidade da calibração do H9a

- **Destino.** Resultados §2 (Seção 2.2).
- **Legenda.** A Figura 3 apresenta o diagrama de confiabilidade da
  probabilidade da classe predita pelo H9a, sobre 1.816.393 nós de decisão do
  conjunto de teste reservado, com a diagonal de calibração perfeita como
  referência.
- **Tipo de gráfico.** Diagrama de confiabilidade (linha de confiança predita
  contra frequência real observada, por classe), com a diagonal y=x tracejada.
- **Eixos e séries.** Eixo horizontal: confiança média predita por decil
  (`conf`). Eixo vertical: frequência real observada (`freq`). Séries: uma
  linha por classe (`NONE`, `SPLIT`, `REST`).
- **Dado de origem.** `results/models/student_h9a/calibration/reliability.csv`
  (colunas `cls,lo,hi,count,conf,freq`, conferido).
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  df = pd.read_csv("results/models/student_h9a/calibration/reliability.csv")
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.plot([0, 1], [0, 1], "k--", label="calibração perfeita")
  for cls, g in df.groupby("cls"):
      g = g.sort_values("conf")
      ax.plot(g["conf"], g["freq"], marker="o", label=cls)
  ax.set_xlabel("confiança predita")
  ax.set_ylabel("frequência real observada")
  ax.legend()
  plt.tight_layout()
  plt.savefig("fig_confiabilidade_h9a.png", dpi=200)
  ```

#### Figura 4 — Comparação a tempo casado entre fontes de escore

- **Destino.** Resultados §2 (Seção 2.6, ablação E5), com painel de referência
  da ablação da Seção 1.3 (Jockey).
- **Legenda.** A Figura 4 apresenta a taxa BD contra a aceleração para as três
  fontes de escore — modelo, variância e escore aleatório —, sob política
  idêntica de comprometimento com `PARTITION_NONE`, nas sequências de
  validação FlowerPan e Lips, com um painel adicional da sequência Jockey
  como referência do experimento de menor escala descrito na Seção 1.3.
- **Tipo de gráfico.** Gráfico de dispersão com linhas conectando os pontos
  de cada braço (fonte do escore), em três painéis lado a lado (um por
  sequência).
- **Eixos e séries.** Eixo horizontal: aceleração (×) ou redução de tempo
  (%). Eixo vertical: taxa BD (%). Séries (cor): modelo, variância, aleatório.
  Anotação textual no painel da Lips indicando a transição abrupta da
  variância entre τ=0,99 e τ=0,97.
- **Dado de origem.** `results/benchmark/e5_ablation/FlowerPan/curve.csv` e
  `results/benchmark/e5_ablation/Lips/curve.csv` (colunas
  `method,tau_none,bd_rate_pct,speedup_x,ts_pct`, conferido) para os dois
  painéis principais; `results/benchmark/ablation_matched.csv` (colunas
  `speedup_x,ml,random,variance,winner`, formato largo) para o painel da
  Jockey.
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

  # painéis 1-2: FlowerPan e Lips (E5)
  for ax, seq in zip(axes[:2], ["FlowerPan", "Lips"]):
      df = pd.read_csv(f"results/benchmark/e5_ablation/{seq}/curve.csv")
      for method, g in df.groupby("method"):
          g = g.sort_values("speedup_x")
          ax.plot(g["speedup_x"], g["bd_rate_pct"], marker="o", label=method)
      ax.set_title(seq)
      ax.set_xlabel("aceleração (×)")

  # painel 3: Jockey (ablacao_matched, formato largo -> long)
  wide = pd.read_csv("results/benchmark/ablation_matched.csv")
  long = wide.melt(id_vars=["speedup_x", "winner"],
                    value_vars=["ml", "random", "variance"],
                    var_name="method", value_name="bd_rate_pct").dropna()
  for method, g in long.groupby("method"):
      g = g.sort_values("speedup_x")
      axes[2].plot(g["speedup_x"], g["bd_rate_pct"], marker="o", label=method)
  axes[2].set_title("Jockey (referência, Seção 1.3)")
  axes[2].set_xlabel("aceleração (×)")
  axes[0].set_ylabel("taxa BD (%)")
  axes[0].legend()
  plt.tight_layout()
  plt.savefig("fig_atribuicao_tempo_casado.png", dpi=200)
  ```

#### Figura 5 — Curvas de limiares por sequência

- **Destino.** Resultados §2 (Seção 2.7, Bloco 7 — joelho da curva de
  limiares).
- **Legenda.** A Figura 5 apresenta a curva de compromisso entre taxa BD e
  redução de tempo em função do limiar de confiança do H9a, por sequência, com
  o joelho da curva situado na faixa de τ entre 0,60 e 0,70 destacado por
  anotação.
- **Tipo de gráfico.** Dispersão com linha conectada, uma linha por sequência,
  ordenada por τ decrescente.
- **Eixos e séries.** Eixo horizontal: redução de tempo (%). Eixo vertical:
  taxa BD (%). Séries (cor): sequência de teste (Jockey, RaceNight,
  RiverBank). Marcadores anotados com o valor de τ nos pontos extremos e no
  joelho.
- **Dado de origem.** `results/benchmark/c5_finetau/raw.csv` (colunas
  `tau,seq,cq,bits,psnr_y,wall_s`, conferido; oito valores de τ, três
  sequências, quatro CQ).
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  from bd_rate import bjontegaard_rate  # src/scripts/benchmark/bd_rate.py

  df = pd.read_csv("results/benchmark/c5_finetau/raw.csv")
  anchor = df[df["tau"] == df["tau"].max()]  # ou carregar âncora nativa à parte

  rows = []
  for (tau, seq), g in df.groupby(["tau", "seq"]):
      a = anchor[anchor["seq"] == seq].sort_values("cq")
      g = g.sort_values("cq")
      bd = bjontegaard_rate(a["bits"], a["psnr_y"], g["bits"], g["psnr_y"])
      ts = 1 - g["wall_s"].sum() / a["wall_s"].sum()
      rows.append({"tau": tau, "seq": seq, "bd_rate_pct": bd, "ts_pct": ts * 100})
  curve = pd.DataFrame(rows)

  fig, ax = plt.subplots(figsize=(6, 5))
  for seq, g in curve.groupby("seq"):
      g = g.sort_values("ts_pct")
      ax.plot(g["ts_pct"], g["bd_rate_pct"], marker="o", label=seq)
  ax.set_xlabel("redução de tempo (%)")
  ax.set_ylabel("taxa BD (%)")
  ax.legend()
  plt.tight_layout()
  plt.savefig("fig_curvas_limiar_por_sequencia.png", dpi=200)
  ```
- **Observação.** O CSV bruto não contém taxa BD nem redução de tempo
  agregada: exige o cálculo de BD-rate por `(tau, seq)` contra a âncora
  nativa `cpu-used=0`, com `src/scripts/benchmark/bd_rate.py`, replicando o
  procedimento de `src/scripts/fase6/analyze_frontier.py`.

#### Figura 6 — Decomposição de ganho entre alavancas empilhadas

- **Destino.** Resultados §3 (Seção 3.5), companheira da Tabela 16.
- **Legenda.** A Figura 6 apresenta a decomposição aditiva da redução de tempo
  entre o H9a isolado, o H9c isolado e a interação medida entre os dois, por
  sequência, evidenciando que a interação é negativa em três das quatro
  sequências medidas.
- **Tipo de gráfico.** Barras empilhadas divergentes (gráfico de cascata) por
  sequência: barra do H9a só, barra do H9c só, barra da interação (com sinal),
  barra do total empilhado medido.
- **Eixos e séries.** Eixo horizontal: sequência (Neon1224, PierSeaSide,
  Tango, TimeLapse, média). Eixo vertical: redução de tempo (pp). Segmentos
  empilhados: H9a só, H9c só, interação (cor distinta para sinal negativo).
- **Dado de origem.** `results/benchmark/fase6/raw_results.csv` (linhas
  `h9adef`, `h9ciso_tau90`, `ml_balanced`, quatro sequências CTC).
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  # tabela já consolidada por report_e3_dec_e2.py; usar diretamente os
  # valores por sequência (H9a só, H9c só, H9a+H9c) em vez de reagregar
  # o raw_results.csv na figura.
  data = {
      "Neon1224": (16.8, 4.2, 17.1),
      "PierSeaSide": (10.4, 5.4, 12.8),
      "Tango": (7.2, 12.3, 17.1),
      "TimeLapse": (9.2, 0.6, 11.5),
  }
  df = pd.DataFrame(data, index=["h9a_so", "h9c_so", "empilhado"]).T
  df["soma"] = df["h9a_so"] + df["h9c_so"]
  df["interacao"] = df["empilhado"] - df["soma"]

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.bar(df.index, df["h9a_so"], label="H9a só")
  ax.bar(df.index, df["h9c_so"], bottom=df["h9a_so"], label="H9c só")
  ax.bar(df.index, df["interacao"],
         bottom=df["h9a_so"] + df["h9c_so"], label="interação",
         color=["tab:red" if v < 0 else "tab:green" for v in df["interacao"]])
  ax.set_ylabel("redução de tempo (pp)")
  ax.legend()
  plt.tight_layout()
  plt.savefig("fig_decomposicao_alavancas.png", dpi=200)
  ```

#### Figura 7 — Fronteira bidimensional do H9d

- **Destino.** Resultados §4 (Seção 4.7), companheira da Tabela 19.
- **Legenda.** A Figura 7 apresenta a família completa de configurações do
  H9d, cruzando duas bases do H9a com duas calibrações do H9d, no plano de
  taxa BD marginal contra redução de tempo marginal, evidenciando que o valor
  marginal da alavanca desaba conforme a base do H9a fica mais agressiva.
- **Tipo de gráfico.** Dispersão de quatro pontos, com anotação de rótulo em
  cada um e uma linha de referência do preço do botão de limiar do H9a.
- **Eixos e séries.** Eixo horizontal: redução de tempo marginal (pp). Eixo
  vertical: taxa BD marginal (pp). Pontos: (balanceada, PL10), (balanceada,
  PL20), (agressiva, PL10), (agressiva, PL20).
- **Dado de origem.** `results/benchmark/fase6/raw_results.csv`, filtrando as
  quatro configurações `ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d` e
  `ml_aggr_h9d_pl20` (colunas `seq,config,cq,fps_num,fps_den,frames,bytes,psnr_y,time_s`).
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  from bd_rate import bjontegaard_rate

  raw = pd.read_csv("results/benchmark/fase6/raw_results.csv")
  configs = ["ml_bal_h9d", "ml_bal_h9d_pl20", "ml_aggr_h9d", "ml_aggr_h9d_pl20"]
  frontier = raw[raw["config"].isin(configs)]
  # calcular, para cada combinação (base, calibração), a taxa BD e a redução
  # de tempo marginal contra a base do H9a correspondente (mesmo procedimento
  # de src/scripts/fase6/ctc_h9d_marginal.py); consolidar em df com colunas
  # base, calibracao, bd_marginal_pp, ts_marginal_pp
  df = pd.DataFrame([
      {"base": "balanceada", "calib": "PL10", "bd_marginal_pp": 0.018, "ts_marginal_pp": 1.02},
      {"base": "balanceada", "calib": "PL20", "bd_marginal_pp": 0.083, "ts_marginal_pp": 2.09},
      {"base": "agressiva", "calib": "PL10", "bd_marginal_pp": 0.006, "ts_marginal_pp": 0.17},
      {"base": "agressiva", "calib": "PL20", "bd_marginal_pp": 0.017, "ts_marginal_pp": 0.65},
  ])  # valores de exemplo, a substituir pelo cálculo real
  fig, ax = plt.subplots(figsize=(6, 5))
  for base, g in df.groupby("base"):
      ax.scatter(g["ts_marginal_pp"], g["bd_marginal_pp"], label=base, s=80)
      for _, r in g.iterrows():
          ax.annotate(r["calib"], (r["ts_marginal_pp"], r["bd_marginal_pp"]))
  ax.set_xlabel("redução de tempo marginal (pp)")
  ax.set_ylabel("taxa BD marginal (pp)")
  ax.legend()
  plt.tight_layout()
  plt.savefig("fig_fronteira_2d_h9d.png", dpi=200)
  ```
- **Observação.** Os quatro valores centrais (0,018/1,02; 0,0399→pp;
  0,0329→pp; 0,0258→pp de preço) já estão redigidos em prosa em `R4_h9d.md
  §4.7`, mas não há CSV agregado desta campanha — a figura exige recalcular a
  taxa BD marginal a partir do CSV bruto filtrado, como já registrado na
  Tabela 19.

#### Figura 8 — Inversão do crivo oráculo pelo codificador real (rede de grafos)

- **Destino.** Resultados §5 (`R5_resultados_negativos.md`, Seção 5.2).
- **Legenda.** A Figura 8 apresenta, em dois painéis, a redução de custo de
  busca medida na simulação de oráculo e a taxa BD medida no codificador real
  para o mesmo par de modelos — a rede de grafos e o H9a —, evidenciando que o
  ordenamento entre os dois modelos se inverte do painel de oráculo para o
  painel do codificador.
- **Tipo de gráfico.** Dois painéis lado a lado: à esquerda, barras de redução
  de custo por *SPLIT-lost* casado (oráculo); à direita, dispersão de taxa BD
  contra redução de tempo na varredura de limiar (codificador real, sequência
  Jockey).
- **Eixos e séries.** Painel esquerdo — eixo horizontal: modelo (perceptrone
  independente L0; rede de grafos L2; rede de grafos causal L2; rede de
  grafos pixel-only L2); eixo vertical: redução de custo (%) a `SPLIT-lost`
  casado de 1%. Painel direito — eixo horizontal: redução de tempo (%); eixo
  vertical: taxa BD (%); séries: rede de grafos, H9a.
- **Dado de origem.** `results/models/gnn_L0/gate_oracle.csv`,
  `results/models/gnn_L2/gate_oracle.csv`,
  `results/models/gnn_L2_causal/gate_oracle.csv`,
  `results/models/gnn_L2_pixel/gate_oracle.csv` (painel esquerdo);
  `results/benchmark/gnn_frontier/frontier_Jockey.csv` (colunas
  `model,tau,bd_rate_pct,ts_pct,speedup`, conferido) e
  `results/benchmark/gnn_replay/gnn_replay_Jockey.csv` (painel direito).
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

  models = {"L0": "results/models/gnn_L0/gate_oracle.csv",
            "L2": "results/models/gnn_L2/gate_oracle.csv",
            "L2_causal": "results/models/gnn_L2_causal/gate_oracle.csv",
            "L2_pixel": "results/models/gnn_L2_pixel/gate_oracle.csv"}
  vals = {}
  for name, path in models.items():
      g = pd.read_csv(path)
      # selecionar a linha de split_lost mais próxima de 1%
      idx = (g["split_lost"] - 1.0).abs().idxmin()
      vals[name] = g.loc[idx, "cost_red"]
  ax1.bar(vals.keys(), vals.values())
  ax1.set_ylabel("redução de custo (%) a SPLIT-lost≈1%")

  frontier = pd.read_csv("results/benchmark/gnn_frontier/frontier_Jockey.csv")
  for model, g in frontier.groupby("model"):
      g = g.sort_values("ts_pct")
      ax2.plot(g["ts_pct"], g["bd_rate_pct"], marker="o", label=model)
  ax2.set_xlabel("redução de tempo (%)")
  ax2.set_ylabel("taxa BD (%)")
  ax2.legend()
  plt.tight_layout()
  plt.savefig("fig_inversao_oraculo_gnn.png", dpi=200)
  ```
- **Observação.** Os nomes exatos de coluna em cada `gate_oracle.csv` (por
  exemplo, se a coluna de risco casado se chama `split_lost` em todos os
  quatro arquivos) não foram conferidos linha a linha nesta auditoria, apenas
  a existência dos arquivos; conferir o cabeçalho antes de produzir a figura.

#### Figura 9 — Fronteira de compromisso global entre taxa BD e redução de tempo

- **Destino.** Resultados §6 (`R6_analise_integrada.md`, prospectiva),
  companheira da Tabela 20. Fecha a lacuna L1 de
  `A3_RETRATACOES_E_LACUNAS.md`.
- **Legenda.** A Figura 9 apresenta a fronteira de compromisso entre taxa BD e
  redução de tempo de todas as configurações medidas sob protocolo CTC nas
  oito sequências da Classe A1, com os pontos das duas soluções positivas —
  H9a, isolado e empilhado com o H9d — do H9c em substituição direta e dos
  três *presets* nativos, marcando os pontos não dominados no sentido de
  Pareto.
- **Tipo de gráfico.** Dispersão com a fronteira de Pareto destacada por linha
  conectando os pontos não dominados, e rótulo textual em cada ponto.
- **Eixos e séries.** Eixo horizontal: redução de tempo (%). Eixo vertical:
  taxa BD (%). Marcadores por família (presets nativos: quadrado; H9a:
  círculo; H9a+H9d: círculo preenchido; H9c: triângulo).
- **Dado de origem.** `results/benchmark/fase6_analysis/pareto_frontier.csv`
  (execução de 2026-07-29, vinte e quatro configurações, quinze não
  dominadas), já unificado pelo script sobre as três campanhas (`fase6`,
  `fase6_swap`, `fase6_swap_h9c`), sem necessidade de renomear ou concatenar
  colunas. Colunas confirmadas: `config`, `bd_rate`, `ts_pct`, `speedup`,
  `dominated_by` (vazia nos pontos não dominados).
- **Esboço de script.**
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  points = pd.read_csv("results/benchmark/fase6_analysis/pareto_frontier.csv")
  points["pareto"] = points["dominated_by"].isna() | (points["dominated_by"] == "")

  fig, ax = plt.subplots(figsize=(7, 6))
  ax.scatter(points["ts_pct"], points["bd_rate"], c=points["pareto"].map({True: "tab:red", False: "gray"}))
  for _, r in points.iterrows():
      ax.annotate(r["config"], (r["ts_pct"], r["bd_rate"]), fontsize=7)
  ax.set_xlabel("redução de tempo (%)")
  ax.set_ylabel("taxa BD (%)")
  plt.tight_layout()
  plt.savefig("fig_fronteira_global.png", dpi=200)
  ```
- **Observação.** Esta é a figura mais consequente do plano, pois sustenta
  diretamente duas das três conclusões da tese (Tabela 21). O artefato de
  origem já existe pronto para plotagem, regenerado em 2026-07-29 por
  `src/scripts/fase6/analyze_frontier.py`; não é mais necessário unir fontes
  nem recalcular a fronteira de Pareto à mão.

---

## 3. Dados ausentes

Esta seção relaciona as figuras e as tabelas que os dois capítulos exigem, mas
cujo artefato numérico agregado, pronto para plotagem direta, **não existe**
no repositório sob a forma de um CSV dedicado — o dado bruto existe em todos
os casos, mas a agregação final precisa ser recalculada, e nenhum destes itens
foi encontrado como CSV de saída versionado por uma execução anterior do
script citado.

- **Tabela 19 e Figura 7 (fronteira bidimensional do H9d).** Os quatro pontos
  finais (taxa BD e redução de tempo marginal por combinação de base e
  calibração) estão redigidos em prosa em `R4_h9d.md §4.7`, mas não há um CSV
  agregado desta campanha — apenas o CSV bruto por codificação,
  `results/benchmark/fase6/raw_results.csv`, filtrado pelas quatro
  configurações `ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d` e
  `ml_aggr_h9d_pl20`. Para produzir a figura é necessário executar o cálculo
  de taxa BD por combinação, com
  `src/scripts/benchmark/bd_rate.py`, replicando a lógica de
  `src/scripts/fase6/ctc_h9d_marginal.py`, cuja saída tabular específica
  também não foi localizada nesta auditoria (mesmo item já registrado em
  `A1_INDICE_evidencias.md §9`).
- **Tabela 17 (predizibilidade do H9d por nível).** As áreas sob a curva
  característica citadas na Seção 4.2 provêm de `run.log`, um arquivo de
  texto de execução, e não de um CSV estruturado. Fechar esta lacuna exige
  reexecutar `h9d_predictability.py` com uma opção de exportação de métricas
  por nível em CSV, opção que o script, pelo que esta auditoria verificou, não
  oferece.
- **Figura 8 (inversão do crivo oráculo, painel esquerdo).** Os quatro
  arquivos `gate_oracle.csv` existem, mas os nomes exatos das colunas de
  risco casado (`split_lost` ou equivalente) não foram conferidos linha a
  linha nesta auditoria — apenas a presença dos arquivos foi verificada por
  `Glob`. Antes de escrever o script de produção, é preciso inspecionar o
  cabeçalho de cada um dos quatro CSVs.
- **`results/dataset_h9/manifest.csv.bak-4116`.** Citado em
  `RASTREABILIDADE.md §3` como o manifesto pré-correção, útil apenas se a
  Tabela 3 viesse a precisar comparar antes/depois da correção de layout; não
  foi localizado nesta varredura (mesmo item já registrado em
  `A1_INDICE_evidencias.md §9`). A Tabela 3, como especificada, não depende
  dele, pois usa apenas o `manifest.csv` corrigido.
- **Confirmação linha a linha de `results/models/regret/gate3_h9a.csv`.** Caso
  a Tabela 21 venha a citar o intervalo "55–58% de redução de custo a
  SPLIT-lost ≤1%" do H9a como número central da primeira conclusão, a linha
  exata que o produz nesse CSV não foi identificada nesta auditoria (mesma
  ressalva de `A1_INDICE_evidencias.md §9`); recomenda-se preferir, nesse
  caso, os números já conferidos da Tabela 10 (`gate2_final.csv`).

Nenhum outro item deste plano depende de artefato ausente: todas as demais
tabelas e figuras — incluindo as seis exigidas no escopo mínimo deste
documento — têm CSV de origem verificado por `Glob` nesta auditoria, com
script de reprodução nomeado.
