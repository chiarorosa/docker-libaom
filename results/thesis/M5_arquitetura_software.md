# 5. Arquitetura de software, paridade e garantia de inércia

Esta seção apresenta a arquitetura de software que sustenta todas as medições
relatadas nesta tese, ou seja, o conjunto de binários do codificador, os pontos
de inserção dos podadores aprendidos na busca recursiva de particionamento, o
mecanismo de execução do modelo dentro do codificador e as duas verificações que
tornam as medições atribuíveis. A exposição segue a ordem em que a evidência foi
construída: primeiro os artefatos compiláveis, depois onde o código novo age,
depois como se prova que ele calcula o que foi treinado e que, desligado, ele não
existe. Por fim, são registrados o custo computacional do podador implantado e
uma lição de arquitetura de software cuja consequência experimental foi a
contaminação de uma campanha inteira de medição.

## 5.1 Os binários do codificador e a guarda de compilação

O laboratório experimental separa fisicamente duas árvores de código-fonte do
codificador de referência AV1 (libaom v3.10.0): `src/aom_baseline`, mantida
intocada como controle cego, e `src/aom`, onde vive todo o código desenvolvido
neste trabalho. Esta separação é a primeira linha de defesa contra contaminação,
pois a âncora de comparação jamais é recompilada a partir da árvore modificada. Os
binários são configurados com o gerador Ninja e com `-DCONFIG_INTERNAL_STATS=1`,
e os de desempenho usam `-DCMAKE_BUILD_TYPE=Release` com `-DENABLE_NASM=ON`, de
modo que a lógica desenvolvida compete contra a âncora sob as mesmas otimizações
vetorizadas do codificador.

Todo o código novo vive sob duas guardas de compilação, ambas com valor padrão
zero: `LOG_PARTITION_DATA`, que ativa a instrumentação de registro das amostras
de particionamento em `partition_search.c`, e `PARTITION_ML_STUDENT`, que ativa a
inferência do modelo, a política de poda e as linhas de base de atribuição em
`partition_strategy.c`. Com as guardas desligadas, o pré-processador remove o
código novo antes da geração de objeto, e o binário resultante é, por construção,
o codificador de referência. Deste modo, a distinção entre binário instrumentado e
binário de referência não depende de disciplina de execução, e sim de uma
propriedade verificável do artefato compilado.

Quatro binários cumprem funções distintas e não intercambiáveis no protocolo. O
`libaom_logpart` (`src/aom` com `-DLOG_PARTITION_DATA=1`) é o binário de
**extração**, utilizado uma única vez para gerar o conjunto de dados de
particionamento. O `libaom_perf` (`src/aom` com `-DPARTITION_ML_STUDENT=1`,
Release) é o binário de **desempenho**, com os podadores H9a, H9c e H9d
embarcados, e é o binário sob teste em todas as campanhas de codificação real. O
`libaom_perf_anchor` (`src/aom_baseline`, Release) é a **âncora**, o codificador
de referência puro contra o qual se medem taxa BD e tempo, e é também a origem dos
presets nativos usados como termo de comparação. O `libaom_ml_check` (`src/aom`,
alvo `generic`, guarda ligada) é o binário de **verificação**, compilado em C puro
para que nenhuma rotina em *assembly* substitua o código sob exame, e é o que
alimenta a verificação de paridade da §5.4. A estes somam-se dois binários
auxiliares: o `libaom_noop` (`src/aom` com a guarda desligada, Release), usado na
verificação de inércia da §5.5, e o `libaom_dev_generic` (`RelWithDebInfo`, C
puro), usado na validação lógica diária e nos testes unitários da AOMedia.

> **Procedência.** `docs/PROTOCOLO_avaliacao.md` §3 (tabela de binários);
> `docs/SINTESE_resultados_metodologia.md` §2.6; `docs/GUIA_builds.md` e
> `docs/guia_cmake.txt` (linhas de configuração `cmake` reproduzíveis);
> `docs/RASTREABILIDADE.md` §2.4 (guardas por arquivo). Código:
> `src/aom/av1/encoder/partition_strategy.c:1605-1608`,
> `src/aom/av1/encoder/partition_search.c:54-58`.

## 5.2 Os pontos de inserção na busca recursiva e o mecanismo de habilitação

A busca de particionamento do libaom é recursiva sobre a *quadtree* do
superbloco, descendo de 64 para 32, 16 e 8 pixels, e a função
`av1_rd_pick_partition` percorre, em cada nó, uma sequência fixa de estágios de
busca. Dois pontos desta sequência recebem os ganchos desenvolvidos neste
trabalho, e a ordem entre eles define quanta informação cada podador tem
disponível e quanto trabalho cada um pode ainda evitar. O primeiro ponto é
`av1_prune_partitions_before_search`, invocada antes de qualquer avaliação de
taxa-distorção; o segundo é `av1_prune_after_none`, invocada imediatamente após
`none_partition_search` ter preenchido o custo real do `PARTITION_NONE` e antes
dos estágios de busca quadrada, retangular, AB e de quatro vias.

No primeiro ponto age o **H9a**, através da função `student_prune_partition`,
que aplica a política de três ações em cascata sobre as probabilidades emitidas
pelo modelo. No segundo ponto agem o **H9c**, através de `student_h9c_decide`, e
o **H9d**, através de `student_h9d_decide`, nesta ordem, ambos partilhando o mesmo
gancho e o mesmo vetor de atributos e diferindo apenas na ação. A ação do H9d não
é consumida no próprio gancho: ela marca o campo `h9d_skip_ext` do estado de
busca, lido adiante nos portões das partições AB e de quatro vias em
`partition_search.c`. Esta indireção existe porque o alvo do H9d é um conjunto de
candidatos avaliado em estágios posteriores, e não a decisão imediata do nó.

O mecanismo de habilitação opera em dois níveis, e a assimetria entre eles é o
tema da §5.7. O primeiro nível é a guarda de compilação `PARTITION_ML_STUDENT`,
comum aos três podadores, sem a qual `av1_prune_after_none` é uma função vazia e
nenhum gancho é sequer compilado. O segundo nível é a variável de ambiente, lida
uma única vez e memorizada em variável estática para não introduzir variação de
tempo: `AV1_STUDENT_H9C_ENABLE` habilita o H9c e `AV1_STUDENT_H9D_ENABLE`
habilita o H9d, ambos desligados por padrão. O H9a **não possui variável de
habilitação**: uma vez compilado o binário, ele é invocado sempre que a condição
`try_student_prune` for satisfeita, condição esta puramente geométrica — quadro
intra, superbloco de ao menos 64 pixels, tamanho de bloco entre 16 e 64 pixels e
unidade de 64 por 64 pixels inteiramente contida no quadro. Os limiares de
decisão são igualmente configuráveis por variável de ambiente, com sufixos por
nível de bloco (`_16`, `_32`, `_64`), para que uma varredura de limiares não exija
recompilação; os valores compilados por padrão são 0,9 para os limiares de NONE e
de SPLIT do H9a e −1,0 para o limiar de descarte das partições retangulares
(valor que desativa esta ação), 0,9 para o H9c e o vetor por nível 0,091, 0,103 e
0,014 para o H9d, correspondente ao ponto de operação PL10.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §2.8 (diagrama do
> fluxo de controle e as duas dimensões do espaço de projeto);
> `docs/ARQUITETURA_pruner_implantado.md` §2 e §4. Código:
> `partition_search.c:5759` e `:5847` (os dois sítios de invocação);
> `partition_strategy.c:2269-2290` (`av1_prune_after_none`), `:2446-2452`
> (`try_student_prune`), `:1702-1760` (habilitação e limiares por variável de
> ambiente).

## 5.3 Execução do modelo em C e exportação dos pesos treinados

A execução do modelo dentro do codificador não introduziu nenhum código de
inferência novo. Os atributos são calculados em C por `student_node_features`, que
preenche o vetor de 36 posições dos blocos A, B e C a partir de dados já
residentes no codificador naquele ponto da busca, e os ganchos pós-NONE acrescentam
as três posições do bloco E derivadas do custo real do `PARTITION_NONE`. A
passagem direta é então executada por `av1_nn_predict`, a rotina nativa do libaom
para perceptrones de múltiplas camadas, seguida de `av1_nn_softmax`. Esta decisão
elimina uma classe inteira de risco de implementação, pois a aritmética de
inferência utilizada em tempo de codificação é exatamente a que o codificador já
emprega nas suas próprias redes.

As topologias são declaradas como estruturas `NN_CONFIG`, uma por tamanho de
bloco, em cabeçalhos gerados automaticamente a partir dos modelos treinados em
Python: 36 → 64 → 32 → 3 para o H9a, 39 → 64 → 32 → 3 para o H9c e
39 → 64 → 32 → 2 para o H9d, todas com duas camadas ocultas e retificação linear.
Os cabeçalhos `partition_student_weights.h`, `partition_student_h9c_weights.h` e
`partition_student_h9d_weights.h` são produzidos por `export_weights.py` e
`h9d_export_weights.py`, que emitem os pesos em notação científica de oito casas
decimais e no arranjo de memória que `av1_nn_predict` espera. O bloco de 8 por 8
pixels não é modelado, uma vez que é terminal na busca e não possui subárvore a
podar.

> **Procedência.** `docs/ARQUITETURA_pruner_implantado.md` §1;
> `docs/SINTESE_resultados_metodologia.md` §2.6. Código:
> `partition_strategy.c:2132-2185` (`student_prune_partition`), `:2192-2243`
> (`student_h9c_decide`), `:2244-2266` (`student_h9d_decide`);
> `partition_student_h9d_weights.h:50-58` (estrutura `NN_CONFIG`). Scripts:
> `src/scripts/partition_model/{export_weights.py, h9d_export_weights.py}`.

## 5.4 Paridade C-Python como critério de validação

A integração em C introduz o risco de que o codificador calcule, em tempo de
execução, atributos numericamente distintos daqueles sobre os quais o modelo foi
treinado, situação em que o modelo implantado operaria sobre uma entrada
deslocada sem que nenhum sintoma óbvio aparecesse. Para eliminar este risco foi
adotada a **paridade C-Python** como critério de validação obrigatório da
integração, o quarto critério de decisão da cascata. O procedimento executa o
binário de verificação `libaom_ml_check` sobre um quadro real da sequência Jockey
a cq20, despeja em arquivo binário, por nó visitado, o vetor de atributos e as
probabilidades calculados em C, e recalcula ambos em Python a partir da fonte de
verdade `features.node_features_h9a`, comparando posição a posição.

As tolerâncias foram fixadas antes da execução: 2 × 10⁻⁶ para cada canal de
atributo e 1 × 10⁻³ para as probabilidades, esta última acomodando a redução de
precisão da rotina nativa de inferência e a margem do ponto flutuante de precisão
simples. O resultado obtido foi de desvio máximo **0,0 × 10⁰ nos 36 atributos**,
ou seja, igualdade exata em todos os canais, com as probabilidades dentro da
tolerância, sobre **588 nós únicos**. Este é o número que autoriza afirmar que o
modelo implantado consome exatamente a entrada sobre a qual foi treinado.

Cabe delimitar, com precisão, o que esta verificação prova e o que ela não prova.
A paridade prova a **aritmética dos atributos**, isto é, que as mesmas entradas
produzem os mesmos valores nas duas implementações. Ela **não** prova a origem do
contexto lido em tempo de execução, pois os campos de vizinhança e de quantização
percorrem o caminho C → despejo → Python, e o desvio nulo destes campos é, neste
caso, tautológico. A origem do contexto é validada por dois outros meios: por
revisão de código contra o idioma nativo do codificador, verificando que os
tamanhos dos blocos acima e à esquerda e a força de quantização são lidos das
mesmas estruturas que a busca de taxa-distorção nativa consulta, e, fim a fim,
pelo benchmark da Fase 5 — uma origem de contexto errada produziria um podador
sem poder de discriminação, e o ganho medido no conjunto de teste reservado seria
indistinguível do ruído. Os sinais efetivamente independentes na verificação de
paridade são os do bloco A e os atributos derivados de granularidade e de
profundidade.

> **Procedência.** `docs/ANDAMENTO_tese.md` §4 (Fase 4, critério 4);
> `results/models/student_h9a/gate4_evidence.txt` (registro literal do
> resultado); `docs/SINTESE_resultados_metodologia.md` §2.6. Script:
> `src/scripts/partition_model/check_feature_parity.py` (tolerâncias em `:80` e
> `:179`). Reprodução em `docs/RASTREABILIDADE.md` §6, passo 4.

## 5.5 A garantia de inércia e a atribuição das diferenças de tempo

A segunda verificação da fase de integração é a **garantia de inércia**: com a
guarda de compilação desligada, o binário gerado a partir de `src/aom` deve
produzir um fluxo de bits byte a byte idêntico ao da âncora. A verificação foi
executada sobre um quadro texturizado de 448 por 256 pixels em codificação
integralmente intra, a cpu-used 0 e cq 32, conteúdo escolhido por exercitar os
candidatos retangulares, AB e de quatro vias além da busca completa de modos. Os
resumos criptográficos dos dois fluxos coincidiram no valor
`b904f11c9fa02d5ea25460cb976ef29d`, e o fluxo produzido com a guarda ligada foi
decodificado com sucesso pelo decodificador da âncora, resultando em 172 076
bytes válidos, com os testes `EncodeAPI.AllIntra` passando 3 de 3.

Esta garantia é a condição que torna legítima toda atribuição de diferença de
tempo à poda. O tempo de codificação medido é uma diferença entre dois programas
distintos, e só é lícito atribuí-la ao podador se estiver estabelecido que os dois
programas são, fora do podador, o mesmo programa. Sem a inércia demonstrada,
qualquer diferença de tempo poderia decorrer de efeitos colaterais da compilação,
de alterações acidentais em estruturas partilhadas ou de mudanças de disposição de
memória, e nenhuma quantidade de repetições de medição separaria estas causas do
efeito da poda. Além disso, a política de poda apenas **remove** candidatos e
jamais força um candidato ilegal, e, deste modo, o fluxo de bits permanece válido
e nem o decodificador nem o formato são alterados: a contribuição é, integralmente,
redução do espaço de busca no codificador.

A mesma disciplina foi aplicada a cada solução acrescentada. Na campanha do H9d,
antes de qualquer medição de contribuição marginal, verificou-se que o
codificador com o H9d desligado reproduz o ponto de operação equilibrado do H9a
de forma byte a byte idêntica, com 1 574 775 bytes e PSNR-Y de 40,9720 dB, o que
estabelece que a única variável alterada entre as duas linhas comparadas é a ação
do próprio H9d.

> **Procedência.** `results/models/student_h9a/gate4_evidence.txt` (resumo
> criptográfico, decodificação e testes unitários);
> `docs/SINTESE_resultados_metodologia.md` §2.6 e o registro de integridade do
> H9d; `docs/ARQUITETURA_pruner_implantado.md` §4;
> `docs/ANDAMENTO_tese.md` §4.

## 5.6 O custo computacional do podador implantado

O custo computacional do podador foi medido em duas grandezas distintas, e a
distinção entre elas importa. A primeira é a **inferência isolada**, ou seja, a
passagem direta do modelo, cronometrada dentro de uma codificação real com
`clock_gettime(CLOCK_MONOTONIC)`, sob a guarda `AV1_PRUNER_TIMING`, com
cpu-used 1, uma única linha de execução e três quadros, em duas execuções
independentes. A rede convolucional nativa do codificador custou 24 763 e
24 606 nanossegundos por chamada, contra 484 e 488 nanossegundos do perceptrone de
múltiplas camadas do H9a e 770 e 745 nanossegundos do H9c, ou seja, o perceptrone
de múltiplas camadas executa a sua predição a cerca de um cinquentavo do custo
computacional da rede convolucional nativa por chamada.

A segunda grandeza é o **custo implantado**, isto é, quanto tempo de parede o
podador inteiro acrescenta à codificação, cobrando também a extração dos
atributos. Nas mesmas duas execuções, a extração custou 3 596 e 3 883
nanossegundos por chamada, dominando a inferência por um fator de 7,2 a 7,8, e os
totais por codificação foram de 210,5 e 203,5 milissegundos de extração contra
29,1 e 26,0 milissegundos de inferência. Medidos contra o tempo de parede das
mesmas codificações — 92,3 segundos para Tango a cq32 e 70,8 segundos para
BoxingPractice a cq43 —, o caminho completo do H9a pesa **0,26% e 0,32%** do tempo
de codificação, e o caminho da rede convolucional nativa pesa 0,16% e 0,21%.

A consequência é registrada aqui com honestidade, pois ela retira um argumento da
proposta. Como o custo próprio do podador é de cerca de um terço de um por cento
da codificação, **o custo de inferência não é alavanca em direção alguma**: não se
alega economia de inferência, nem se sofre custo de inferência, e todo o ganho de
tempo medido provém das decisões de poda, não da leveza do modelo. A alegação de
leveza sai, então, da lista de vantagens da solução, sem por isso converter-se em
desvantagem, uma vez que a soma cai num valor desprezível dos dois lados, e nenhum
resultado de taxa BD contra tempo precisa ser revisto. A razão de cerca de
cinquenta vezes permanece válida para o que mede — o algoritmo de decisão isolado
— e não deve ser citada como vantagem do podador implantado. Vale a ressalva de
escopo: os três podadores coexistem na mesma codificação instrumentada, e a
medição cobre duas sequências, três quadros e uma única linha de execução, o que
basta para razões de ordem de grandeza e não para estimativa de precisão fina.

> **Procedência.** `docs/RESULTADOS_microbench_pruner.md` §2 (inferência
> isolada — 484/488 e 770/745 ns/chamada do H9a e do H9c, e 24 763/24 606 ns/chamada
> da CNN nativa, medição de 2026-07-17 registrada apenas em prosa, sem CSV
> versionado correspondente), §6.2 e §6.2b (custo implantado e peso absoluto,
> adendo de 2026-07-19, cujo ns/chamada do H9a — 497/496 — diverge levemente do
> de §2 por ser uma agregação sobre o encode inteiro, e não a mesma amostra de
> duas execuções), §6.5 (ressalvas). Artefato: `results/benchmark/microbench/pruner_cost.csv`
> (fonte de §6, não de §2). Script: `src/scripts/benchmark/microbench_pruner.py`.
> Instrumentação: `partition_strategy.c:148` e `:156`.

## 5.7 Uma lição de arquitetura de software com consequência experimental

O tema desta subseção é um achado de arquitetura de software cuja consequência foi
experimental, e o seu registro integra a metodologia porque explica tanto uma
retratação quanto uma prática adotada em todas as campanhas subsequentes. O
mecanismo é o seguinte: o gancho do H9a é habilitado exclusivamente por uma
condição geométrica, `try_student_prune`, sem variável de habilitação, ao
contrário do H9c e do H9d, que possuem `AV1_STUDENT_H9C_ENABLE` e
`AV1_STUDENT_H9D_ENABLE`. Compilado o binário de desempenho, o H9a passa a rodar
em todo quadro intra sob os seus limiares compilados por padrão, e a sua atuação é
invisível na leitura do *script* de experimento, que define apenas as variáveis do
podador sob teste.

A contaminação materializou-se nas primeiras campanhas do H9c, cujos *scripts*
definiram somente as variáveis de habilitação e de limiar do H9c, sem neutralizar
os limiares do H9a, que rodou nos seus valores compilados de 0,9 para NONE e para
SPLIT. Deste modo, todas as linhas rotuladas como H9c mediram, na verdade, **H9a e
H9c empilhados**, e não o H9c isolado. A detecção veio da incoerência entre a
redução de tempo atribuída ao H9c e o alcance estrutural de um podador que age
depois da avaliação do `PARTITION_NONE` e que, por construção, jamais economiza
aquele custo; a inspeção do fluxo de controle localizou o gancho sem variável de
habilitação.

A correção adotada foi a **neutralização explícita dos limiares** do podador que
não está sob teste, com re-execução mantendo esta como única variável alterada. Os
limiares do H9a foram fixados em 2 para NONE e para SPLIT e em −1 para o descarte
das partições retangulares, valores que nunca disparam, pois as probabilidades do
softmax nunca excedem a unidade nem ficam abaixo de zero. A quantificação obtida
na sequência Neon1224, a cpu-used 0, é a seguinte: a linha medida antes da
correção com limiar 0,95 registrava 0,267% de taxa BD e 17,15% de redução de
tempo, contra 0,037% e 2,96% do H9c isolado; com limiar 0,90, 0,270% e 17,36%
contra 0,037% e 4,23%; e com limiar 0,60, 0,386% e 20,53% contra 0,100% e 9,31%.
Ou seja, de **82% a 96% da redução de tempo** atribuída ao H9c provinha, na
verdade, do H9a, e o H9c isolado poda muito pouco, entre cerca de 3% e 9%. As
afirmações derivadas das tabelas contaminadas foram formalmente retiradas, e todas
as campanhas posteriores — inclusive o experimento de substituição da rede
convolucional nativa — passaram a neutralizar explicitamente os limiares do H9a
desde o próprio *script*.

A lição generalizável é de projeto de código experimental: em software de
pesquisa, um caminho de execução habilitado por geometria e sem interruptor
explícito é, para efeitos práticos, indistinguível de código inerte na leitura do
experimento, e a sua atuação silenciosa contamina toda comparação que não o
neutralize. Todo gancho experimental deve, então, ter habilitação explícita e
estado observável no registro da execução, requisito que passou a valer para as
soluções acrescentadas depois desta descoberta.

> **Procedência.** `docs/ANDAMENTO_tese.md` §8.1 (mecanismo, prova de código,
> quantificação e retratação) e §8.2 (a prática de neutralização no experimento
> de substituição); `docs/SINTESE_resultados_metodologia.md` §5. Artefatos:
> linhas `h9ciso_*` em `results/benchmark/fase6/raw_results.csv`;
> `results/benchmark/fase6_swap_h9c/`. Scripts:
> `src/scripts/fase6/{encode_h9c_cq20.py, encode_h9c_iso.py, encode_swap_h9c.py}`.
> Código: `partition_strategy.c:2446-2452` (condição geométrica sem variável de
> habilitação) e `:1706` (a variável de habilitação do H9c, ausente no H9a).

## 5.8 Síntese e encaminhamento

Esta seção estabeleceu que as medições dos capítulos seguintes repousam sobre
quatro binários de papéis distintos, sobre guardas de compilação que tornam o
código novo removível pelo pré-processador, sobre uma verificação de paridade que
prova a aritmética dos atributos com desvio máximo nulo em 588 nós e sobre uma
verificação de inércia que prova, por resumo criptográfico, que o codificador
desligado é o codificador de referência. Foram, também, registrados o custo
computacional do podador implantado e a lição de arquitetura de software que a
contaminação da primeira campanha do H9c ensinou. Descrito o substrato de
execução, a próxima seção apresenta as arquiteturas de rede neural propostas e a
metodologia de atribuição que separa o mérito do modelo do mérito da política que
o envolve.

> **Procedência.** Consolidação das subseções 5.1 a 5.7; estrutura conforme
> `results/thesis/00_PLANO_capitulos.md` §3. Branch `ml-partition-dev`; trilha de
> commits em `docs/RASTREABILIDADE.md` §7.
