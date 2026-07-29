# M2 — Instrumentação do codificador e geração do conjunto de dados

> Seção 2 do Capítulo de Metodologia. Redigida conforme as regras editoriais de
> `00_PLANO_capitulos.md` §6. Todo valor numérico provém de documento ou artefato
> deste projeto; as lacunas estão marcadas como `[completar: ...]`.

---

## 2.1 Abertura

Esta seção apresenta a instrumentação do codificador de referência do AV1 e o
procedimento de geração do conjunto de dados que sustenta toda a investigação
descrita nesta tese. São descritos, nesta ordem, a guarda de compilação que isola
o registro do binário de produção, a estrutura gravada em cada nó de
particionamento, a decisão de extrair os rótulos sob busca de taxa-distorção
completa, o corpus e sua cobertura, o pipeline de extração e consolidação com o
respectivo comando de reprodução e, por fim, o defeito de luminância nula,
tratado aqui como resultado metodológico e não como acidente de engenharia. O fio
condutor é a auditabilidade: cada elemento do conjunto de dados é rastreável até
a linha de código que o produziu e até o commit que o registrou.

---

## 2.2 Instrumentação do libaom sob a guarda `LOG_PARTITION_DATA`

O codificador de referência libaom, versão v3.10.0, foi instrumentado para
registrar, em cada nó da busca recursiva de particionamento de quadros intra, o
rótulo de referência (do inglês *ground truth*) da decisão de taxa-distorção
tomada pelo próprio codificador. O ponto de captura é a função
`av1_rd_pick_partition()`, que implementa o caminho de busca por taxa-distorção
(RD) e é chamada uma vez por nó da árvore quaternária de particionamento. O
escopo do registro é restrito ao modo All-Intra (`cpi->oxcf.mode == ALLINTRA`) e
aos blocos quadrados de interesse, ou seja, `BLOCK_64X64`, `BLOCK_32X32`,
`BLOCK_16X16` e `BLOCK_8X8`.

A captura é dividida em três momentos dentro da mesma chamada, pois os dados que
compõem uma amostra completa não estão disponíveis simultaneamente. Os pixels de
luminância do bloco-fonte são copiados logo após `av1_set_offsets()`, antes de a
recursão mover os ponteiros de plano; o custo RD do candidato `PARTITION_NONE` é
capturado imediatamente após `none_partition_search()`, antes que a estrutura
`this_rdc` seja reaproveitada pelos estágios seguintes; e a decisão final
(`pc_tree->partitioning`) é lida quando `found_best_partition` é verdadeiro,
antes de o `pc_tree` ser liberado. O registro é gravado no fim da função, já com
a decisão consolidada.

Cada amostra é uma estrutura `PartitionSample` de 4144 bytes, *little-endian*,
com os campos escalares ordenados do maior para o menor e o buffer de pixels ao
final, com preenchimento explícito, de modo que o layout não contenha
preenchimento implícito e possa ser lido em Python por um único formato de
`struct`, `"<qqII5H8B6x4096s"`. A estrutura registra: a luminância do bloco em
8 bits, num buffer fixo de 64×64 pixels cuja região válida é `block_dim²` no
canto superior-esquerdo; o rótulo `PARTITION_TYPE` completo, nas suas dez
classes; o contexto de particionamento dos vizinhos causais acima e à esquerda
(`above_bsize`, `left_bsize`, `neigh_avail`); o passo de dequantização DC da
luminância (`dc_q`) e o `base_qindex` do quadro; a posição do bloco em unidades
de mode-info; as dimensões do quadro; e a taxa, a distorção e o custo RD do
candidato `PARTITION_NONE`. O rótulo é gravado em sua forma completa, e não já
reduzido a uma decisão binária, pois a redução preserva menos informação e pode
ser feita a posteriori. Uma verificação de layout em tempo de compilação
(`partition_sample_size_check`) quebra a construção caso a estrutura deixe de
medir exatamente 4144 bytes, e, deste modo, qualquer divergência entre o produtor
em C e o consumidor em Python é detectada antes de gerar dado inválido.

A guarda de compilação é necessária por três razões, todas de método. A primeira
é a integridade da medição de tempo: a contribuição desta tese é avaliada por
redução de tempo de parede, e um binário que escreve em disco a cada nó de
particionamento não pode ser o mesmo binário que mede aceleração. A segunda é a
garantia de inércia: com `LOG_PARTITION_DATA` no valor padrão zero, todo o código
de registro desaparece do pré-processador e a construção de produção é idêntica à
original, propriedade que o projeto verifica byte a byte contra a árvore de
controle `src/aom_baseline`, mantida intocada. A terceira é o determinismo: o
registro utiliza estado global — um descritor de arquivo aberto uma única vez em
modo *append* e um contador `sample_id` por processo — e, portanto, não é seguro
para múltiplas linhas de execução, o que impõe `--threads=1` na extração. Esta
escolha foi deliberada, pois priorizou-se a reprodutibilidade bit a bit sobre o
tempo de extração. O contador `sample_id`, ao reiniciar a cada processo, cumpre
ainda uma segunda função: ele marca a fronteira entre quadros no arquivo
acumulado, e é por esse reset que o validador reconstrói a proveniência de quadro
de cada amostra.

> **Procedência.** Código: `src/aom/av1/encoder/partition_search.c`, linhas 44–158
> (estrutura, verificação de layout e funções `av1_partition_target_dim`,
> `av1_partition_capture_luma`, `av1_partition_log_write`), 5676–5707 (captura de
> luminância e contexto), 5833–5845 (contexto RD do `PARTITION_NONE`) e
> 6012–6015 e 6065–6071 (decisão e escrita). Documentos:
> `docs/RELATORIO_pipeline_dataset_particionamento.md` §2 e §3;
> `docs/GUIA_partition_dataset.md`, seção "Formato do registro";
> `docs/RASTREABILIDADE.md` §2.4. Commits: `a17c525` (instrumentação do contexto
> RD, blocos B/C/E) e `a6748c5` (extração dos atributos H9).

---

## 2.3 A escolha de `cpu-used=0` para a extração

A extração foi executada integralmente com `--cpu-used=0`, ou seja, no regime de
busca RD completa do libaom. Esta decisão é a mais consequente de toda a seção,
uma vez que o rótulo de referência só é fiel se provier de uma busca que
efetivamente avaliou todas as formas de partição candidatas. Qualquer preset mais
rápido já contém poda heurística e terminação antecipada e, deste modo,
contaminaria os rótulos com as decisões de outra heurística, e não com a decisão
RD-ótima que se pretende aprender.

A contaminação foi verificada empiricamente, e não assumida. Na sequência
Beauty, quadro 0, `cq-level=32`, uma extração a `cpu-used=6` produziu contagem
**zero** para as classes `HORZ_A`, `HORZ_B`, `VERT_A`, `VERT_B`, `HORZ_4` e
`VERT_4`, ao passo que a `cpu-used=0` todas as dez classes aparecem. A
`cpu-used=8` nada foi registrado, pois o modo All-Intra deixa de percorrer o
caminho RD e cai no caminho não-RD baseado em variância, onde a instrumentação
simplesmente não existe. A densidade de amostragem acompanha o mesmo efeito:
foram medidas aproximadamente 125.775 amostras por quadro a `cpu-used=0` contra
cerca de 34.593 a `cpu-used=3`, uma razão próxima de 3,6 vezes. Então, conjuntos
de dados extraídos sob presets diferentes **não são comparáveis** entre si como
rótulo de referência, e o valor de `cpu_used` é registrado por linha no
manifesto justamente para tornar essa incomparabilidade explícita e auditável.

O preço desta decisão é elevado e foi pago deliberadamente. A extração completa
consumiu **32,3 horas** de processamento contínuo dentro do contêiner, para
gerar as sessenta e quatro tarefas de codificação que compõem o conjunto de
dados — dezesseis sequências por quatro pontos de quantização —, o que
corresponde a uma mediana de **27,2 minutos por tarefa** de cinco quadros, ou
seja, cerca de **363 segundos por quadro** em resolução 4K. Este custo é,
também, a razão pela qual o arquivo binário intermediário foi tratado como cache
de reconversão, pois recodificar é ordens de grandeza mais caro do que reler.

Cabe registrar como este número foi apurado, uma vez que as duas cifras que o
projeto registrava para o custo da extração não se conciliavam entre si. O valor
acima foi reconstituído a partir dos carimbos de conclusão de cada tarefa,
gravados na coluna `timestamp` do manifesto: a janela vai de 2026-07-09T15:21:11
a 2026-07-10T23:39:56, e a soma dos sessenta e três intervalos consecutivos
iguala a janela total, sem intervalo anômalo, o que confirma que a execução foi
contínua e que nenhum período de ociosidade infla a contagem. Deste modo, a
cifra de aproximadamente 510 segundos por quadro registrada no guia de extração
corresponde ao extremo lento da distribuição, e não à média, ao passo que a
cifra de oito horas registrada no roteiro de reprodução está subestimada em
cerca de quatro vezes e foi corrigida.

> **Procedência.** `docs/RELATORIO_pipeline_dataset_particionamento.md` §5
> (decisões metodológicas) e §6, itens 2 e 4 (achados);
> `docs/GUIA_partition_dataset.md` §3 e §5;
> `docs/SINTESE_resultados_metodologia.md` §2.1; `docs/RASTREABILIDADE.md` §3 e
> §6. Artefato: colunas `cpu_used` e `timestamp` em
> `results/dataset_h9/manifest.csv` (64 linhas). O custo total de 32,3 h foi
> apurado em 2026-07-29 a partir dos carimbos de conclusão desta última coluna,
> conforme descrito acima; reprodução da apuração pela diferença entre o menor e
> o maior valor da coluna, com verificação de que a soma dos intervalos
> consecutivos iguala a janela.

---

## 2.4 O corpus

O conjunto de dados canônico da tese é o `results/dataset_h9/`, gerado a partir
de **dezesseis sequências** do corpus UVG em resolução 4K (3840×2160), 8 bits e
formato 4:2:0, com 600 quadros cada, à exceção de ShakeNDry e SunBath, que têm
300. Cada sequência foi codificada em **quatro pontos de quantização**
(`cq-level` 20, 32, 43 e 55, correspondendo a `base_qindex` 80, 128, 172 e 220,
pela relação `base_qindex = 4·cq`), sobre **cinco quadros** amostrados de forma
temporalmente uniforme ao longo do clipe inteiro — nas sequências de 600 quadros,
os índices 0, 150, 300, 449 e 599. A amostragem temporal espaçada é uma decisão
de método, e não de conveniência, pois em codificação All-Intra quadros
consecutivos são quase idênticos e um conjunto de dados denso no tempo seria
pouco representativo; a diversidade provém, então, de quatro eixos combinados:
espacial, de conteúdo, de quantização e temporal.

O conjunto resultante contém **26,98 milhões de amostras**, uma por nó visitado
da árvore de particionamento, das quais **10,07 milhões são nós de decisão**
(`block_dim ∈ {16,32,64}`). Os 16,91 milhões restantes, ou 62,7% do total, são
blocos 8×8, folhas terminais cujo rótulo é constante — medido: zero amostras
não-NONE em 16,91 milhões —, e por isso são excluídos da modelagem, embora
permaneçam na contagem da árvore, pois podar um bloco 16×16 economiza exatamente
os seus quatro filhos. O desbalanceamento de classe é acentuado e depende
fortemente do tamanho do bloco: nos nós 64×64 o rótulo `SPLIT` responde por
64,12% das amostras e `NONE` por 28,84%; nos 32×32, `NONE` sobe para 47,72% e
`SPLIT` cai para 28,86%; e nos 16×16, `NONE` domina com 73,46% contra apenas
4,40% de `SPLIT`. Este perfil é característica do domínio, e não do pipeline, e
sua consequência para o treino é tratada na seção de arquiteturas de rede
neural.

O armazenamento é feito em **64 arquivos `.pkl`** — um por par (sequência, ponto
de quantização) —, totalizando **31 GB** em disco, com a luminância gravada como
`float32` em [0,1], codificação sem perdas dos pixels originais de 8 bits. O
conjunto não é versionado no repositório git, em função do volume. Acompanha-o o
**`manifest.csv`**, com uma linha por arquivo e vinte e oito colunas de
proveniência e estatística: sequência, dimensões, total de quadros, número e
lista dos quadros usados, `cq_level`, `base_qindex`, `cpu_used`, número de linhas
de execução, total de amostras, histograma por tamanho de bloco, histograma pelas
dez classes de partição, caminhos dos arquivos e carimbo de tempo. O manifesto
não é um acessório: sem ele o `.pkl` perde rastreabilidade, e conjuntos gerados
com `cpu-used` distinto deixariam de ser distinguíveis. Para arquivamento com
identificador digital de objeto (do inglês *digital object identifier* – DOI),
está previsto o formato `.npz` em `uint8`, de aproximadamente 8 a 12 GB, portátil
e independente de *pickle*.

A partição em treino, validação e conjunto de teste reservado é feita **por
sequência**, sem vazamento de conteúdo, e foi congelada antes de qualquer
medição: dez sequências de treino (15,76 M amostras, das quais 6,09 M são nós de
decisão), três de validação — HoneyBee, FlowerPan e Lips — com 5,89 M amostras
(2,07 M de decisão), e três de teste — Jockey, RaceNight e RiverBank — com
5,32 M amostras (1,91 M de decisão).

> **Procedência.** `docs/RASTREABILIDADE.md` §3 (dataset canônico);
> `docs/ZENODO_datasheet.md`, Partes 1 e 2; `docs/SINTESE_resultados_metodologia.md`
> §2.1 e §2.3. Artefatos: `results/dataset_h9/manifest.csv` (64 linhas),
> `results/dataset_h9/label_histogram.csv` e
> `results/dataset_h9/label_distribution.md` §1. Volume de 31 GB medido no disco
> do hospedeiro em `results/dataset_h9/`.

---

## 2.5 O pipeline de extração e consolidação

A extração é organizada em três camadas: a instrumentação em C, compilada num
diretório de construção separado (`build/libaom_logpart`) com
`-DCMAKE_C_FLAGS="-DLOG_PARTITION_DATA=1"` e `AOM_TARGET_CPU=generic`, de modo a
não contaminar o diretório do ciclo diário; os scripts Python em
`src/scripts/partition_dataset/`, que residem sob `src/` porque o
`docker-compose` monta apenas `./src`, `./logs` e `./results` no contêiner; e os
artefatos consolidados em `results/dataset_h9/`.

Os scripts têm papéis disjuntos. O `build_dataset.py` orquestra o produto
cartesiano de sequências por pontos de quantização: lê as dimensões do nome do
arquivo e o número de quadros do seu tamanho, calcula os deslocamentos de
amostragem temporal, pré-extrai cada quadro-alvo uma única vez com `dd` e o
reaproveita entre os pontos de quantização, codifica, converte e escreve o
`manifest.csv`, sendo retomável em caso de interrupção. O
`convert_partition_data.py` transforma o arquivo binário em `.pkl`, validando o
alinhamento em 4144 bytes e expondo luminância, rótulo e contexto RD. O
`validate_partition_data.py` confere a integridade estrutural, exporta imagens
PNG por tamanho e classe e verifica a **acurácia de pixel** contra o arquivo YUV
fonte, separando os segmentos de quadro pelo reset do `sample_id`. O
`rebuild_manifest_stats.py` recalcula as colunas estatísticas do manifesto a
partir dos `.pkl`, rota necessária quando o binário é descartado na geração; o
`analyze_label_histogram.py` produz o relatório de distribuição de rótulos; e o
`pkl_to_npz.py` converte para o formato de arquivamento em `uint8`.

O comando canônico de reprodução, executado dentro do contêiner, é o seguinte:

```bash
venv-ml/bin/python src/scripts/partition_dataset/build_dataset.py \
  --out-dir results/dataset_h9 --qps 20 32 43 55 --frames 5 --cpu-used 0 \
  --aomenc build/libaom_logpart/aomenc

venv-ml/bin/python src/scripts/partition_dataset/rebuild_manifest_stats.py \
  --dataset-dir results/dataset_h9
venv-ml/bin/python src/scripts/partition_dataset/analyze_label_histogram.py \
  --hist results/dataset_h9/label_histogram.csv \
  --out results/dataset_h9/label_distribution.md
```

Duas armadilhas operacionais foram identificadas e documentadas. A primeira é
que `aomenc --limit` conta quadros a partir do início da entrada, **antes** de
`--skip`, de sorte que `--skip=599 --limit=1` codifica zero quadros; na primeira
geração isto fez a amostragem temporal registrar apenas o quadro 0. A segunda é
que `--skip` relê o arquivo desde o início a cada chamada, o que a
aproximadamente 31 MB/s de vazão custava perto de quatro minutos para alcançar o
quadro 599. A pré-extração com `dd`, reaproveitada entre os pontos de
quantização, corrige a primeira e elimina de duas a três horas de entrada e saída
redundante na execução completa.

A validação de acurácia de pixel foi conduzida sobre uma extração reduzida de
quatro sequências, `cq-level=32` e dois quadros: 817.540 amostras, dois segmentos
de quadro detectados em todas as sequências e **zero divergências** em 816.390
blocos internos ao quadro, com todas as dez classes de partição presentes. No
conjunto canônico o binário intermediário foi descartado na geração
(`--no-keep-bin`), e a integridade da luminância é atestada por via independente,
descrita na subseção seguinte. `[completar: verificação de acurácia de pixel
executada diretamente sobre os `.pkl` do `dataset_h9`, se houver registro]`.

Cabe registrar uma correção de auditoria. Até 2026-07-19, as colunas estatísticas
do manifesto (`num_samples`, `dim*`, `part_*`) haviam sido calculadas com o
layout de registro antigo, de 4116 bytes, sobre registros de 4144 bytes, o que
inflava a contagem de amostras em exatamente 4144/4116, ou 0,68%, e invalidava os
histogramas. As colunas foram reconstruídas a partir dos `.pkl` por
`rebuild_manifest_stats.py`, com o original preservado em `manifest.csv.bak-4116`.
Nenhum modelo, taxa BD ou veredito de critério de decisão dependia dessas
colunas, uma vez que o carregador de treino lê apenas `pkl_path`, `sequence` e
`cq_level`.

> **Procedência.** `docs/GUIA_partition_dataset.md` §1, §3, §4, §6 e §7;
> `docs/RELATORIO_pipeline_dataset_particionamento.md` §4, §6, §7 e §10;
> `docs/RASTREABILIDADE.md` §2.2 (inventário de scripts) e §6 (pipeline de
> reprodução, comando a comando). Scripts:
> `src/scripts/partition_dataset/{build_dataset,convert_partition_data,
> validate_partition_data,rebuild_manifest_stats,analyze_label_histogram,
> pkl_to_npz}.py`. Commit do reparo do manifesto: `6e5bcab`.

---

## 2.6 O defeito de luminância nula como resultado metodológico

Esta subseção apresenta o episódio de maior consequência metodológica de toda a
investigação, e ele é apresentado como resultado, e não como nota de rodapé de
engenharia, pois a lição que dele se extrai integra a defesa de validade desta
tese.

O conversor grava a luminância como `float32` normalizado no intervalo [0,1],
codificação sem perdas dos pixels originais de 8 bits. Os consumidores em Python,
por outro lado, assumiam `uint8` no intervalo [0,255]. A incompatibilidade
produzia dois efeitos distintos e igualmente silenciosos: o extrator de atributos
(`features.py`) convertia para inteiro, e todo valor abaixo de um era truncado
para zero, zerando integralmente cada atributo manual; e o carregador do modelo
substituto normalizava por 255 uma segunda vez, entregando à rede uma entrada
quase nula. Ou seja, os modelos foram treinados sobre imagem em branco. Como o
codificador em C lê `uint8` real, na inferência o modelo recebia atributos numa
escala inteiramente distinta da vista em treino, o que produzia previsões
degeneradas e poda catastrófica, medida em **+6,4% de taxa BD** com limiares
inertes. **Toda a cadeia H1–H6 fora treinada e avaliada sobre luminância em
branco**, e suas conclusões — em particular a de que a luminância teria um limite
superior de desempenho estreito — foram invalidadas, pois mediam uma entrada
vazia.

O passo decisivo do diagnóstico foi separar o defeito do dado. Verificou-se que
`round(luma × 255)` reproduz o quadro-fonte **exatamente**, com diferença máxima
igual a zero, o que estabelece que o dado bruto estava íntegro e que o defeito
residia por inteiro nos consumidores. Esta constatação teve consequência
prática direta e considerável: não foi necessário reextrair o conjunto de dados,
operação que custaria novas horas de codificação a `cpu-used=0`; bastou corrigir
o carregador e retreinar, o que é trabalho de unidade de processamento gráfico.

A correção foi implementada como **fonte única de verdade**: a função
`data._denorm_uint8` passou a ser o único ponto do sistema que converte um bloco
armazenado para a escala de 8 bits, aplicando `round(x·255)` com saturação em
[0,255] a blocos de ponto flutuante e repassando sem alteração os blocos já em
`uint8`. Deste modo, todo consumidor — extração de atributos, treino do modelo
substituto, destilação, calibração e simulação — passa a ver exatamente os mesmos
pixels que o codificador vê. Os caches derivados foram invalidados e os conjuntos
de dados da era anterior (`dataset/`, `dataset_new/`, `dataset_reduced_cq32/`,
`dataset_smoke/`) foram removidos do disco e do rastreamento do git, por não
serem comparáveis nem reutilizáveis.

A correção pontual, porém, não é a lição. A lição é a **asserção de guarda**. Foi
introduzida a função `assert_real_luma`, que inspeciona alguns superblocos do
primeiro arquivo de um conjunto e interrompe a execução caso a luminância montada
apresente máximo menor ou igual a um ou variância inferior a 1,0 — ou seja, caso
os dados de treino não tenham textura real. A asserção é barata, roda antes de
qualquer época de treino e passou a guardar dezoito scripts consumidores, entre
eles o treino do modelo substituto, o treino dos modelos estudantes H9a e H9c, a
destilação, a calibração, a construção de alvos de *regret*, os critérios de
decisão *offline* e a simulação de poda. Então, o que era um defeito passou a ser
uma propriedade verificada a cada execução: **asserir que os dados de treino têm
variância não nula**. O efeito da correção sobre a qualidade do modelo foi
mensurável — a macro-F1 do modelo substituto subiu de 0,12 para 0,20 —, e todos
os resultados anteriores à correção foram remedidos, e não meramente reanotados.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §2.2;
> `docs/ANDAMENTO_tese.md` §1, item 4, e §2 (linha "Bug luma-branco");
> `docs/PREH7_analise_alavancas.md` §4 (aviso) e §5;
> `docs/PLANO_H9_contribuicao_tese.md`, linha 39; `docs/RASTREABILIDADE.md` §2.1
> e §8. Código: `src/scripts/partition_model/data.py`, linhas 104–140
> (`_denorm_uint8` e `assert_real_luma`). Commits: `cb9d407` (correção da
> de-normalização) e `63f299c` (retreino sobre luminância real, guarda e pesos
> reais).

---

## 2.7 Fechamento

O conjunto de dados descrito nesta seção fixa o que é observável e o que é
rotulável no problema, mas não decide nada por si: um rótulo de referência fiel
não autoriza, sozinho, nenhuma afirmação sobre aceleração de codificação. Para
que as soluções propostas nos capítulos seguintes possam ser comparadas entre si
e contra o próprio codificador, é necessário fixar, antes de qualquer medição, as
sequências de teste, as métricas, as âncoras e os limiares de aceitação. A seção
seguinte apresenta o protocolo de avaliação congelado por commit e os critérios
de decisão em cascata que governam quando o custo caro — integração em C e
codificação real — pode ser pago.
