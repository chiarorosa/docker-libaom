# Auditoria numérica dos artigos LASCAS 2027 e ISCAS 2027

Data: 2026-08-13. Objeto: verificar se as tabelas de `results/thesis/PAPER_LASCAS_2027.md`
e `results/thesis/PAPER_ISCAS_2027.md` reproduzem fielmente a fonte canônica, e se todos
os resultados de taxa BD e de redução de tempo apresentados nos dois artigos provêm
exclusivamente da grade das condições comuns de teste (CTC) Classe A1.

Motivação: restrição editorial fixada nesta data — os dois artigos só podem apresentar
taxa BD e redução de tempo medidas sobre a CTC A1; o conjunto UVG pode ser citado quanto
à construção do conjunto de dados e a estatísticas de distribuição, jamais quanto a taxa
BD ou redução de tempo.

## 1. Método de verificação

A redução de tempo foi **recalculada do zero** a partir dos tempos brutos de
`results/benchmark/fase6/raw_results.csv`, na definição canônica declarada em
`results/thesis/M3_protocolo_avaliacao.md` §3.4: para cada ponto de quantização,
`1 − t_configuração / t_âncora`; média sobre os quatro pontos; média sobre as sequências.
Nenhum valor foi lido de tabela intermediária.

Comando de reprodução (PowerShell, host ou contêiner `av1_bench`):

```powershell
$rows = Import-Csv results/benchmark/fase6/raw_results.csv
$anchor = @{}
foreach ($r in $rows | Where-Object {$_.config -eq 'anchor'}) { $anchor["$($r.seq)|$($r.cq)"] = [double]$r.time_s }
foreach ($g in $rows | Where-Object {$_.config -ne 'anchor'} | Group-Object config) {
  $perSeq = foreach ($s in $g.Group | Group-Object seq) {
    $ts = $s.Group | ForEach-Object { 1 - ([double]$_.time_s / $anchor["$($_.seq)|$($_.cq)"]) }
    (($ts | Measure-Object -Average).Average)
  }
  "{0,-18} {1} seqs  {2,6:N2}%" -f $g.Name, $perSeq.Count, ((($perSeq | Measure-Object -Average).Average)*100)
}
```

## 2. Conferência das tabelas — nenhuma divergência

| Tabela | Fonte canônica | Artefato | Resultado |
|---|---|---|---|
| LASCAS II (6 linhas) | `R4_h9d.md` §4.6 | `fase6/bdrate_average.csv` | idêntica |
| LASCAS III (4 preços) | `R4_h9d.md` §4.7 | `fase6/raw_results.csv` | idêntica |
| ISCAS I (fator de confusão) | `R3_h9c.md` §3.3 | `fase6/raw_results.csv` | idêntica |
| ISCAS II (9 linhas) | `R3_h9c.md` §3.4 | `fase6_swap_h9c/swap_average.csv` | idêntica |
| ISCAS III (testes pareados) | `R3_h9c.md` §3.4 | `fase6_analysis/paired_tests.csv` | idêntica |
| ISCAS IV (decomposição) | `R3_h9c.md` §3.5 | `fase6/raw_results.csv` | idêntica |

Redução de tempo recalculada, contra o valor publicado: `ml_balanced` 17,72 (17,72);
`ml_bal_h9d` 18,74 (18,74); `ml_aggr` 31,51 (31,51); `ml_aggr_h9d` 31,68; `ml_bal_h9d_pl20`
19,81; `ml_aggr_h9d_pl20` 32,16; `native_cpu1/2/3` 32,59 / 42,72 / 67,94 (idem);
`h9c_tau45` 21,35 (21,4); `h9ciso_tau90` 5,61 (5,6); `h9adef` 10,88 (10,9);
`h9c_tau90` 13,59 (13,6); `h9c_tau95` 12,61 (12,6).

Os dois estimadores do preço do botão de limiar — 0,063 pp/pp por interpolação por
sequência e 0,0606 pp/pp pela média das médias — foram conferidos contra a nota de
`R4_h9d.md` §4.6 e não se misturam em tabela alguma do artigo.

## 3. Conformidade com a restrição CTC A1

### 3.1 ISCAS — conforme

Todas as campanhas citadas foram verificadas sequência a sequência em
`fase6/raw_results.csv` e `fase6_swap_h9c/`. A cobertura é exclusivamente CTC A1
(BoxingPractice, Crosswalk, FoodMarket2, Neon1224, NocturneDance, PierSeaSide, Tango,
TimeLapse). O subconjunto de três sequências da varredura de τ da Seção IX é
Neon1224, PierSeaSide e TimeLapse, também CTC. O único número medido fora da CTC é o
critério de decisão offline (61,2% de redução de custo de busca simulado a 0,20% de perda
de `SPLIT`), que é simulação sobre o conjunto de dados anotado, e não taxa BD nem redução
de tempo.

### 3.2 LASCAS — quatro não conformidades

As campanhas `results/benchmark/c5_finetau/` e `results/benchmark/h9d_ub/` foram medidas
sobre **Jockey, RaceNight e RiverBank**, que são o conjunto de teste reservado do UVG, e
não sequências da CTC. Verificação: nomes dos arquivos `.obu` nos dois diretórios e coluna
`seq` de `c5_finetau/raw.csv` e `h9d_ub/raw.csv`.

| Passagem do artigo | Número apresentado | Campanha | Conjunto |
|---|---|---|---|
| §IV-C, cota superior do desligamento | +0,89% de taxa BD a 1,431×; marginal 1,293× a +0,798% | `h9d_ub` | UVG |
| §IV-B, densidade da curva de limiares | maior vão de aceleração de 0,15× em 21 vizinhanças | `c5_finetau` | UVG |
| §V, cascata de atribuição | faixas de aceleração disjuntas; taxa BD 11, 44 e 94 vezes menor que a da variância | E5/curva de τ | UVG |
| Figura 2 | curva de limiares do H9a | `c5_finetau` | UVG, **sobreposta** a pontos e *presets* de CTC |

Permanecem admissíveis, por serem distribuição e não taxa BD nem redução de tempo: a
Tabela I e a Figura 1 (decomposição do custo de busca sobre 875.317 nós), a calibração da
Seção IV-B (erro esperado de 0,0112; precisão de 95,6% e 96,5%) e a área sob a curva ROC
de 0,902 da Seção IV-C. Todas exigem, por outro lado, rótulo explícito de que provêm do
conjunto de dados construído sobre o UVG, o que o artigo hoje não declara.

### 3.3 Consequência sobre a alegação de granularidade contínua

Retirada a curva de τ, restam em CTC apenas **dois** pontos de operação do H9a: 17,72% e
31,51% de redução de tempo. A faixa de "12% a 22%" citada no resumo e na Seção VI é a
**dispersão entre sequências** do ponto equilibrado, e não uma faixa varrida pelo limiar.
Valores recalculados por sequência do ponto equilibrado: FoodMarket2 12,35; NocturneDance
12,59; PierSeaSide 16,78; TimeLapse 17,31; BoxingPractice 19,39; Crosswalk 20,18; Tango
21,26; Neon1224 21,93. A afirmação de continuidade precisa ser reescrita, ou sustentada
por uma varredura de τ sobre a grade CTC, cujo custo é de aproximadamente 160 codificações.

### 3.4 Estado — não conformidades corrigidas

As quatro não conformidades de §3.2 e a alegação de continuidade de §3.3 foram corrigidas
no texto do LASCAS na mesma data desta auditoria: os números medidos sobre o UVG foram
removidos do corpo, a restrição de escopo foi declarada nas Seções V e VIII, a Figura 2 foi
respecificada sobre dados exclusivamente de CTC, e a Tabela I, a Figura 1 e as estatísticas
offline das Seções III e IV passaram a identificar o conjunto de dados de origem. O ISCAS
recebeu a mesma declaração de escopo nas Seções II e V, sem remoção de resultado algum.

Permanece pendente o alinhamento do resumo em inglês do LASCAS, que ainda cita a faixa de
12%–22% como intervalo contínuo.

## 4. Achado independente da restrição: o marginal de +0,26 pp do H9c

O contraste central do LASCAS — o H9d soma +1,02 ponto percentual onde o H9c somara
+0,26 — aparece no resumo, na Seção I, na Seção VII e na Conclusão do artigo, e provém de
`docs/SINTESE_resultados_metodologia.md` §5-quater e de `docs/ANDAMENTO_tese.md` §8.
O rastreio do valor mostra que ele é o marginal do H9c medido **na sequência Neon1224
apenas**, sobre a base do H9a nos limiares compilados por padrão.

O mesmo marginal, recalculado nas quatro sequências em que a decomposição existe:

| Sequência | H9a só (`h9adef`) | H9a + H9c (`h9c_tau90`) | Marginal |
|---|--:|--:|--:|
| Neon1224 | 16,75% | 17,07% | +0,32 pp |
| PierSeaSide | 10,43% | 12,84% | +2,41 pp |
| Tango | 7,16% | 17,06% | +9,90 pp |
| TimeLapse | 9,17% | 11,49% | +2,33 pp |
| **Média** | **10,88%** | **14,62%** | **+3,74 pp** |

O valor de +0,26 pp é, deste modo, a sequência de **menor** marginal das quatro medidas, ao
passo que o +1,02 pp do H9d é média sobre **oito** sequências e sobre uma base distinta — o
ponto equilibrado implantado, e não os limiares compilados por padrão. Na média de quatro
sequências, o H9c soma +3,74 pp, valor **superior** ao do H9d, e o contraste de "quatro
vezes" se inverte.

Isto não invalida a tese composicional, que se apoia igualmente na interação medida de
−1,9 ponto percentual (`R3_h9c.md` §3.5) e no colapso do ganho do H9d sobre a base
agressiva (`R4_h9d.md` §4.7). Invalida o **par de números** com que a tese é apresentada.
A correção exige uma de três vias: comparar o H9d contra o H9c sobre a mesma base e a
mesma cobertura de sequências, o que não está medido; apresentar o contraste com a média
de quatro sequências, o que o inverte; ou substituir o argumento numérico pela interação
medida e pelo colapso sobre a base agressiva, ambos já disponíveis.

### 4.1 Estado — contraste corrigido

A terceira via foi adotada no texto do LASCAS na mesma data desta auditoria. A razão entre
magnitudes foi retirada do resumo e das Seções I, VII e IX, e substituída pelo **contraste
de sinal** entre as duas composições que consomem entrada idêntica: interação medida de
−1,9 ponto percentual, na composição do podador binário com o pré-busca, contra marginal de
+1,02 ponto percentual do podador seletivo. A assimetria entre as duas medições — quatro
sequências sobre a base de limiares padrão, contra oito sequências sobre o ponto equilibrado
implantado — passou a ser declarada explicitamente na Seção VII do artigo, com o registro de
que uma comparação pareada entre os dois podadores não foi executada. O valor de 0,26 ponto
percentual não aparece mais em passagem alguma do artigo, à exceção do apêndice interno de
conformidade, que documenta a retratação.

Permanece não medido, e é a via que fecharia a questão de forma definitiva: o marginal do
podador binário e o do podador seletivo sobre a **mesma** base do H9a e as **mesmas** oito
sequências da grade CTC.

## 5. Limitações desta auditoria

1. A taxa BD **não** foi recalculada; foi conferida contra `bdrate_average.csv` e
   `swap_average.csv`, que são saída do mesmo `src/scripts/benchmark/bd_rate.py` que
   produziu os números publicados. Um erro dentro desse script não seria detectado aqui.
2. A verificação de conjunto de sequências cobre as campanhas efetivamente citadas nos
   dois artigos, e não a totalidade dos diretórios de `results/benchmark/`.
3. O rastreio do valor de +0,26 pp chegou a `docs/ANDAMENTO_tese.md`; o valor exato de
   0,26 difere do recálculo direto da Neon1224, que dá 0,32 pp, provavelmente por os dois
   provirem de agregações distintas do mesmo par de execuções. A conclusão — sequência
   única, base distinta — não depende desta diferença.
4. Os artefatos de `results/benchmark/` não são versionados; esta auditoria vale para o
   estado local desses arquivos na data indicada.

## 6. Adendo de 2026-09-04 — retirada da comparação com a escada de presets do LASCAS

Restrição editorial acrescentada nesta data, por orientação: o artigo LASCAS **não pode
apresentar comparação com os presets nativos** (`cpu-used` 1, 2 e 3). A justificativa é de
escopo, e é a mesma que o próprio resumo do artigo declara: o SNP-AV1 introduz duas redes
neurais em dois estágios da busca de partição e não toca em etapa alguma fora dela,
enquanto qualquer preset nativo reconfigura dezenas de heurísticas espalhadas por todas as
etapas do codificador. Pôr os dois lado a lado ordena arranjos de escopo distinto, e não
técnicas de busca de partição.

### 6.1 O que saiu do artigo

| Passagem removida | Conteúdo |
|---|---|
| Resumo | a admissão de não dominância dos presets e a alegação de preencher o vão da escada |
| Seção I | a formulação "não é mais um preset de velocidade" |
| Seção VI (resultados) | os dois parágrafos de qualificação da comparação e a distinção arquitetural em relação à rede convolucional nativa |
| Seção VI | o parágrafo do valor de `cpu-used=1` (32,59% a 0,449%) e da granularidade não coberta |
| Seção VI | a menção ao custo do caminho da rede nativa no parágrafo de sobrecarga |
| Conclusão | a ressalva de não dominância da escada |
| Figura 3 | as três séries de preset nativo e a faixa sombreada de vão não coberto |

Permanece uma única menção, de duas sentenças, no primeiro parágrafo de comparação da
Seção VI: a declaração de escopo que justifica a ausência da comparação, sem número algum
de preset. Ela existe para antecipar a pergunta do revisor, e não para comparar.

Permanecem legítimas, por não serem comparação de desempenho, as menções a `cpu-used=0`
(âncora e regime de extração do conjunto de dados) e a `cpu-used=6` (verificação de que
um preset rápido já poda heuristicamente e por isso não serve para rotular).

### 6.2 O que entrou no lugar

O eixo de comparação passou a ser interno, e o argumento central do artigo passou a ser a
**taxa de câmbio** entre tempo e taxa BD das duas alavancas que a solução implantada
oferece ao seu usuário, ambas medidas na mesma grade CTC A1 e visíveis na Figura 3:

| Alavanca | Segmento medido | Preço (pp de taxa BD por pp de tempo) |
|---|---|--:|
| botão de limiar do 1º estágio | `ml_balanced` → `ml_aggr` | 0,0606 |
| 2º estágio, calibração implantada | `ml_balanced` → `ml_bal_h9d` | 0,0179 |
| 2º estágio, calibração forte | `ml_balanced` → `ml_bal_h9d_pl20` | 0,0399 |

A razão publicada é, por conseguinte, **3,4×** (0,0606 / 0,0179 = 3,38), e não os 3,5× da
versão anterior. Os dois valores são o mesmo resultado sob estimadores distintos, já
registrados em `results/thesis/R4_h9d.md` §4.6: 3,5× pelo estimador de interpolação por
sequência (0,063) e 3,38× pela média das médias (0,0606). O artigo passou a usar **apenas
o estimador de média das médias**, que é o único derivável dos pontos que a própria
Figura 3 plota, e o script da figura recalcula os três preços a cada execução e os imprime
para auditoria. O estimador de interpolação por sequência continua sendo a fonte do teste
de não dominância sequência a sequência (6 de 8, duas por dominância de Pareto estrita),
que é contagem e não preço, e os dois não se misturam em tabela alguma.

Comando de reprodução (contêiner `av1_bench`):

```bash
build/venv-ml/bin/python src/scripts/benchmark/plot_operating_space_fig3.py \
    --out-dir results/thesis/figuras
```

### 6.3 Consequências para o restante do texto

O enunciado corrigido da Conclusão 3 da tese — dois podadores se somam na medida em que os
conjuntos de candidatos que retiram são disjuntos, no ponto de operação em que efetivamente
rodam (`R6_analise_integrada.md` §6.4) — passou a ser a espinha argumentativa do artigo, na
Seção V (regra do empilhamento) e na Seção VI (a inércia do 2º estágio sobre a base
agressiva, +0,17 pp, abaixo da resolução de ~0,46 pp). A prova por sobreposição entre o
H9a e o podador binário **não** entra, por pertencer ao artigo irmão.

### 6.4 Limitações deste adendo

1. A retirada é editorial e não altera medição alguma: nenhum número foi recalculado, com a
   exceção declarada dos três preços da tabela acima, que saem do mesmo
   `bdrate_average.csv` já auditado na Seção 2.
2. A tese continua apresentando a fronteira global com os presets nativos, como manda o seu
   próprio escopo (`M1_objeto_e_formulacao.md` §1.5). A restrição vale para o artigo LASCAS.

---

## 7. Adendo de 2026-09-05 — alinhamento terminológico ao *proposal* ICASSP e validação do modelo do LASCAS

Motivação: o orientador entregou, em
`results/thesis/IEEE_Conference_Template/reviews/daniel proposal.tex`, a redação de referência
do artigo ICASSP. Duas coisas dele valem para o LASCAS: a **terminologia canônica da área** e o
**batismo científico dos conjuntos de atributos**, que antes eram os blocos A, B e C do código.
Nada de resultado, nenhum número e nenhuma citação cruzada foram transportados: artigos irmãos
da tese não se citam.

### 7.1 Varredura de terminologia aplicada ao LASCAS

| Antes (LASCAS) | Depois (canônico ICASSP) |
|---|---|
| `evaluates ten candidate shapes` | `evaluates up to ten partition types` |
| `candidate shape`, `rectangular shapes`, `4-way shapes` | `partition type`, `rectangular types`, `4-way types` |
| `attribute`, `attribute vector` | `feature`, `feature set` |
| `causal context` (como nome do conjunto) | `pre-search features` / `causal-neighbor partition context` |
| `the undivided candidate` | `the unpartitioned node` / `PARTITION_NONE` |
| `rate-distortion (RD) cost` (sem sigla do processo) | `rate-distortion optimization (RDO)`, definida na Introdução |
| `pixels` (como dimensão de bloco) | `samples` |
| `CQ 20, 32, 43, 55` | `cq-level 20, 32, 43 and 55`, "four quantization points" |
| `10 bits` | `10-bit depth` |
| `multilayer perceptrons` | `multilayer perceptrons (MLPs)` |
| `rectified units` | `ReLU units` |
| `\section{Block Partitioning in AV1}` | `\section{The AV1 Intra Partition Search}` |
| `\subsection{Attribute Design}` | `\subsection{Pre-Search Feature Sets}` |
| `\subsection{Dataset Construction}` | `\subsection{Reference Encoding and Dataset}` |
| `\subsection{Network Architecture and Training}` | `\subsection{Prediction Models and Training}` |

O léxico métrico do grupo (`computational effort`, `coding efficiency`, `time savings (TS)`,
`BD-BR`, `trade-off`) foi **preservado**, por ser normativo em `CLAUDE.md` B.12 e não conflitar
com o *proposal*.

### 7.2 Conjuntos nomeados BC / NC / CSP

O *proposal* nomeia os três blocos do vetor pré-busca, e os nomes foram adotados no LASCAS
(Seção III-A) e na Figura 1 (fluxograma), substituindo a enumeração corrida:

| Nome | Sigla | Nº | Bloco no código (`features.py`) | Índices |
|---|:--:|--:|---|---|
| Block-content features | BC | 24 | A — luminância + `q_norm` + posição no superbloco | 0–23 |
| Causal-neighbor partition context | NC | 8 | B — vizinhança de particionamento causal | 24–31 |
| Coding-state and position features | CSP | 4 | C — `dc_q`, posição no quadro, profundidade | 32–35 |
| (2º estágio) log da taxa, distorção e custo RD do `PARTITION_NONE` | — | 3 | E | 36–38 |

Ganho colateral: a Seção III-A ficou mais **precisa**, e não só mais curta. A redação anterior
dizia "twenty-four luma descriptors", o que superestimava o bloco A — ele tem 21 descritores de
luminância mais `q_norm`, `pos_r` e `pos_c` (`features.py` linhas 193–201). "Twenty-four
block-content descriptors ... the normalized quantization index and the position inside the
64×64 unit" descreve exatamente o que o código calcula.

### 7.3 Validação: qual modelo o LASCAS discute

Verificação feita contra o código implantado, não contra documento derivado:

| Evidência | Valor |
|---|---|
| `src/scripts/partition_model/features.py:204` | `NUM_FEATURES_H9A = 36` (A+B+C) |
| `src/aom/av1/encoder/partition_student_weights.h:26` | `AV1_PARTITION_STUDENT_NUM_FEATURES 36` |
| Cabeçalho gerado por `export_weights.py` | "features from features.py (NUM_FEATURES=36)" |
| `src/aom/av1/encoder/partition_student_h9d_weights.h:27` | `AV1_PARTITION_STUDENT_H9D_NUM_FEATURES 39` |
| `results/thesis/T_ATRIBUTOS_H9.md` §1 e §4 | H9a = 36; H9c/H9d = 39 |
| Contagem de parâmetros | 3×(36·64+64+64·32+32+32·3+3) = 13.641 mais 3×(39·64+64+64·32+32+32·2+2) = 14.118 → **27.759**, exatamente o número já publicado no artigo |

**Veredito: o LASCAS discute o modelo de 36 atributos (BC+NC+CSP) no 1º estágio e o de 39
(BC+NC+CSP + RD do `PARTITION_NONE`) no 2º.** É esse par que foi compilado em C, medido no
codificador e que produziu toda a taxa BD e toda a redução de tempo do artigo. Nenhuma
correção de número foi necessária.

**Ponto de atenção declarado.** O estudo offline do ICASSP aponta **BC+NC (32)** como a melhor
combinação sob redução de busca casada, e **BC+NC+CSP (36)** um degrau abaixo. Não há
contradição factual entre os dois artigos: o LASCAS relata o artefato **implantado e medido no
codificador**, e não a fronteira offline; e ele não faz, em nenhum ponto, a afirmação de que 36
seja o conjunto ótimo. O que existe é uma tensão de leitura para quem ler os dois — e como
artigos irmãos não se citam, ela não aparece em nenhum dos textos. A decisão de projeto, se
houver oportunidade em versão de periódico, é reavaliar o CSP no codificador; até lá o LASCAS
descreve o que foi de fato executado.

### 7.4 Consequência editorial: o teto de 4 páginas

Restrição do evento fixada nesta data: **4 páginas de conteúdo técnico; a 5ª só admite material
referencial.** Para atingi-la sem retirar item obrigatório de `CLAUDE.md` Partes A e B, foram
feitas nove passadas de compressão sobre perífrase (nenhuma sobre evidência), mais:

- retirada da Figura 1 (as dez formas de partição), cujos nomes voltaram à prosa da Seção II;
- Figura 3 (espaço de operação) de 1,97 in para 1,78 in, mantido o piso de 9 pt;
- `\arraystretch` da Tabela I de 1,15 para 1,05;
- legenda da Tabela I encurtada (a âncora `cpu-used=0` já consta do texto e da Figura 2).

Estado verificado: Seções I–V e Conclusões terminam na página 4; a página 5 contém apenas o
Acknowledgment e as Referências. Nenhum `Overfull` e nenhuma referência indefinida no `.log`.

Comando de reprodução das figuras (contêiner `av1_bench`):

```bash
build/venv-ml/bin/python src/scripts/benchmark/plot_pruner_flowchart_fig2.py --out-dir results/thesis/figuras
build/venv-ml/bin/python src/scripts/benchmark/plot_operating_space_fig3.py --out-dir results/thesis/figuras
```

### 7.5 Limitações deste adendo

1. A varredura é **de forma**: nenhum número, nenhuma tabela e nenhuma figura de dado foi
   recalculado. A única figura regerada por motivo de dado foi a Figura 1 do artigo
   (fluxograma), e apenas para trocar os rótulos dos blocos por BC/NC/CSP — as contagens
   continuam vindo de `SPEC`, que espelha `features.py`.
2. A validação de §7.3 confere o **vetor de entrada** dos modelos implantados. Ela não
   revalida os pesos nem repete a campanha de codificação, que permanecem os auditados na
   Seção 2 deste documento.

### 7.6 Ajustes na Figura 1 do artigo (fluxograma), 2026-09-05

Três pontos levantados na revisão da figura, resolvidos assim:

1. **Variante impressa.** O artigo passa a incluir `figura2_fluxograma_cinza.pdf` (paleta em
   escala de cinza) e não mais a variante colorida. A cor não carregava informação no
   fluxograma: os dois estágios já se distinguem pelo número no marcador, pelo nome e pela
   posição.
2. **Forma das caixas de consequência.** As quatro caixas do ramo "yes" passaram a ter a
   **mesma forma** — retângulo de canto reto, como as caixas de processo e as de entrada. Antes,
   a caixa do compromisso com o `PARTITION_SPLIT` tinha cantos arredondados para marcar que ela
   encerra o nó; a distinção continua, mas agora só pelo preenchimento, que não compete com a
   gramática de formas do diagrama (losango = decisão, retângulo = ação ou processo).
3. **A conexão ausente do "only SPLIT" está correta e é deliberada.** Verificado em
   `src/aom/av1/encoder/encodeframe_utils.h:271`: `av1_set_square_split_only` faz
   `partition_none_allowed = 0`. Sob esse ramo o `PARTITION_NONE` **não é avaliado**, de modo que
   o nó nunca alcança a caixa "NONE is evaluated" nem o ponto de enganche do 2º estágio
   (`av1_prune_after_none`). Ligar essa caixa ao trilho de reencontro seria erro de fato. Para
   que a ausência fosse **legível** em vez de parecer um esquecimento, os conectores das outras
   três caixas foram alongados (`X_ACAO_1` de 0,944 para 0,918 e `X_TRILHO` de 0,974 para 0,992),
   o que mais que dobra o traço de retorno e torna evidente qual caixa não o tem.

### 7.7 Figura 2 do artigo (espaço de operação) também em cinza, 2026-09-05

Para que o artigo não misturasse convenções — um diagrama em cinza ao lado de um gráfico
colorido —, `plot_operating_space_fig3.py` ganhou o mesmo arranjo de duas paletas já usado pelo
fluxograma: `PALETAS = {"cor", "cinza"}` e a opção `--variante`, com as duas saídas geradas por
padrão (`figura3_espaco_operacao.pdf` e `figura3_espaco_operacao_cinza.pdf`). O artigo inclui a
variante em cinza; a colorida permanece disponível para apresentação e para a tese.

Nada se perde na conversão, e a razão é de projeto, não de sorte: as duas séries vivem em
**painéis separados**, com escalas próprias, e já se distinguem pela **forma do marcador**
(círculo em (a), triângulo em (b)) — decisão registrada no comentário do script desde a primeira
versão. Na paleta cinza as duas recebem a mesma tinta (`#141413`). O único par que precisa
continuar separável **dentro** de um painel é a reta tracejada do botão de limiar contra a
poligonal medida, e essa separação nunca dependeu de cor: é traço tracejado cinza-médio
(`#8a8880`) contra traço cheio preto.

Estado do artigo após a mudança: monocromático de ponta a ponta, 4 páginas técnicas, página 5 só
com Acknowledgment e Referências, zero `Overfull`, zero referência indefinida, todas as fontes
embutidas e subsetadas.

### 7.8 Referências trazidas do *proposal* ICASSP e símbolo da equação (1), 2026-09-05

**Origem.** A bibliografia do *proposal* (`reviews/ICASSP daniel proposal.tex`, dados completos
lidos de `reviews/PAPER_ICASSP_2027_AV1_reviewDaniel.pdf`) foi confrontada com as 25 referências
do LASCAS. Quatro entradas de lá são mais atuais do que qualquer equivalente daqui e foram
incorporadas; as demais ou já constavam, ou são específicas do estudo offline (ConvNeXt,
aprendizado sensível a custo) e não têm função no LASCAS.

| Nova | Onde entra | Por quê |
|---|---|---|
| Bender *et al.*, *J. Real-Time Image Process.*, 2023 | M5, sobre a afirmação de custo | A frase "one of the largest individual costs of AV1 intra-frame coding is the block partitioning decision" **não tinha citação alguma**: era asserção. Passa a ter âncora externa, e de uma análise de complexidade do próprio libaom. |
| Song *et al.*, *IEEE Signal Process. Lett.*, 2024 | M6, família VVC | Atualiza um grupo que ia só até 2022. |
| Kherchouche *et al.*, ICASSP 2024 | M6, família VVC | Regressão de custo RD para particionamento intra — o vizinho mais próximo do 2º estágio. |
| Kherchouche *et al.*, DCC 2025 | M6, família VVC | O trabalho mais recente da linha; aproximação de custos RD para particionamento intra de VVC. |

O parágrafo M6 passou a **classificar** a família, como manda `CLAUDE.md` B.4/M6, em vez de apenas
listá-la: "over variance, random forests, gradient boosting, texture complexity and the regression
of RD costs [18]–[23]". O veredito de comparabilidade que já existia cobre as três novas sem
frase adicional — todas foram projetadas para a árvore quaternária com árvore multitipo aninhada
e avaliadas em outro codificador.

*Ressalvas de procedência.* (i) A entrada da DCC 2025 é um resumo de **uma página** (p. 377), como
é praxe da conferência; foi mantida por ser publicação IEEE indexada e por constar do *proposal*.
(ii) Os meses de ICASSP 2024 (Abr.) e DCC 2025 (Mar.) foram completados pelas datas fixas dessas
conferências; o mês do *Signal Processing Letters* não foi inventado e a entrada sai sem ele.
(iii) Bender *et al.* é trabalho publicado do próprio grupo, e não artigo irmão desta tese — a
proibição de citação cruzada de `CLAUDE.md` B.15 não se aplica.

**Símbolo da equação (1).** `t_cfg` passou a `t_SNP`, e a definição, a "the evaluated SNP-AV1
configuration". A troca é legítima porque **todas** as configurações às quais o artigo aplica a
definição de redução de tempo são configurações do próprio SNP-AV1 — os seis pontos de operação
diferem apenas nos limiares e na presença do 2º podador, e nenhum preset nativo entra na
comparação (§6 deste documento). O anchor mantém `t_anc`.

Estado após a mudança: 29 referências, numeração em ordem de primeira citação conferida no PDF
([9] Bender entre [8] Layek e [10] Sullivan; família VVC contígua em [18]–[23], comprimida pelo
`cite.sty`), 4 páginas técnicas, zero `Overfull`, zero referência indefinida.
