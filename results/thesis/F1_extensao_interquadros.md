# F1. A extensão à predição interquadros como trabalho futuro

Este documento apresenta a análise da extensão das três soluções de poda
implantadas em C — o H9a, o H9c e o H9d — ao regime de codificação que emprega
predição interquadros, em vez do regime *All-Intra* sob o qual toda a
investigação foi conduzida. A análise é estritamente estática, ou seja,
realizada sobre o código-fonte do codificador e sobre os resultados já medidos,
sem que nenhuma codificação nova tenha sido executada. Ela é destinada ao
capítulo de trabalhos futuros e cumpre três funções: registrar o mecanismo
exato pelo qual a delimitação intraquadro desta tese é uma condição de validade,
e não uma conveniência de escopo; antecipar a objeção previsível de banca sobre
a generalidade da contribuição; e ordenar as quatro extensões possíveis por
custo e por prognóstico. Nenhum número desta análise é resultado de medição
nova, e as passagens que exigiriam medição estão marcadas como lacuna.

## F1.1 A guarda de tipo de quadro no código implantado

Esta seção apresenta o estado atual do código integrado ao libaom no que diz
respeito ao tipo de quadro, uma vez que dele decorre o comportamento observável
das três soluções sob uma configuração de codificação com predição interquadros.

Os dois pontos de inserção das soluções desta tese são condicionados à função
`frame_is_intra_only`, do próprio codificador. O ponto de inserção pré-busca,
que executa o H9a, está condicionado em `partition_strategy.c:2447`; o ponto de
inserção posterior ao candidato `PARTITION_NONE`, que executa o H9c e o H9d,
está condicionado em `partition_strategy.c:2280`. A instrumentação que gerou o
conjunto de dados anotado é condicionada de forma análoga, através da
comparação `cpi->oxcf.mode == ALLINTRA`, em `partition_search.c:5683`.

A consequência operacional é direta e merece registro explícito, pois é a
primeira coisa que se observaria numa tentativa de reprodução. O binário
atualmente compilado executa, sem erro e sem produzir fluxo de bits inválido,
sob qualquer configuração de grupo de quadros: as três soluções simplesmente
não são acionadas nos quadros que empregam predição interquadros, e permanecem
ativas apenas nos quadros intraquadro. Deste modo, num perfil de acesso
aleatório, a poda aprendida agiria somente sobre os quadros-chave, e a redução
de tempo medida nesta tese estaria diluída pela fração do tempo total que estes
quadros representam.

A guarda do H9a não é arbitrária, pois espelha deliberadamente a guarda da rede
neural convolucional intraquadro nativa, em `partition_strategy.c:2423`, que é
a heurística contra a qual o H9a compete de frente. A guarda do H9c e do H9d
carrega, além disso, uma exigência geométrica própria: os atributos de contexto
hierárquico do bloco A leem a região do bloco ascendente, o que obriga a unidade
de 64 por 64 amostras que contém o nó a estar inteiramente dentro do quadro.

> **Procedência.** Código: `src/aom/av1/encoder/partition_strategy.c:2280,
> 2423, 2447`; `src/aom/av1/encoder/partition_search.c:5683`. Documento-fonte:
> `docs/ARQUITETURA_pruner_implantado.md` §2 e §4.

## F1.2 O que a extensão preserva: validade do fluxo de bits e segurança de execução

Esta seção apresenta as propriedades da política de poda que sobrevivem, sem
alteração alguma, à remoção da guarda de tipo de quadro, pois a separação entre
o que é seguro e o que é inadequado na extensão é a distinção central de todo
este documento.

A política de três ações implantada em `partition_strategy.c:2178-2185` é
agnóstica ao tipo de quadro, uma vez que as três primitivas que ela aciona —
`av1_disable_all_splits`, `av1_set_square_split_only` e
`av1_disable_rect_partitions` — apenas manipulam sinalizadores da estrutura
`PartitionSearchState` e apenas removem candidatos do espaço de busca. Nenhuma
delas força um candidato ilegal. Deste modo, a garantia de validade do fluxo de
bits, verificada nesta tese pela identidade byte a byte com a compilação de
referência quando a solução está desativada, é preservada em quadros que
empregam predição interquadros: o decodificador e o formato não mudam, e o que
se altera é exclusivamente a extensão do espaço de busca percorrido pelo
codificador.

A extração de atributos também permanece válida, pois lê o quadro-fonte através
de `x->plane[AOM_PLANE_Y].src.buf` e a vizinhança de particionamento através de
`xd->above_mbmi` e `xd->left_mbmi`, estruturas que estão igualmente definidas em
quadros com predição interquadros.

Cabe destacar, por fim, um risco de correção que foi examinado e descartado. O
H9a age antes da heurística nativa de divisão por busca de movimento
simplificada, condicionada em `partition_strategy.c:2461`, e uma decisão de
`NONE-commit` impediria a execução desta heurística, deixando a estrutura
`sms_tree` sem preenchimento. Os consumidores posteriores destes dados, por
outro lado, os calculam sob demanda e sob a guarda do sinalizador
`sms_none_valid`, em `partition_strategy.c:599` e `:615`, e por isso não
realizam leitura de dado não inicializado. Não há, portanto, defeito de correção
na extensão — o que existe é a substituição de uma heurística informada por
movimento por outra que o ignora, examinada na seção seguinte.

> **Procedência.** Código: `src/aom/av1/encoder/partition_strategy.c:599, 615,
> 2178-2185, 2461`; `src/aom/av1/encoder/context_tree.h:98-100`.
> Documento-fonte: `docs/ARQUITETURA_pruner_implantado.md` §4, quanto à garantia
> de validade do fluxo de bits.

## F1.3 O que a extensão não preserva: a ausência de canal temporal nos atributos

Esta seção apresenta a razão principal pela qual a extensão direta é
inadequada, que é de natureza informacional e não de engenharia.

A auditoria do vetor de trinta e seis atributos, implementado em
`student_node_features` (`partition_strategy.c:1885-1990`), mostra que ele é
integralmente espacial. O bloco A, com as colunas de zero a vinte e três, reúne
estatísticas de luminância do quadro-fonte corrente — variância, variâncias por
quadrante, gradientes horizontais e verticais, perfis de linha e de coluna,
densidade de bordas, componente contínua, contexto hierárquico do bloco
ascendente e dos blocos irmãos, e posição dentro do superbloco. O bloco B, com
as colunas de vinte e quatro a trinta e uma, codifica os tamanhos dos blocos
vizinhos acima e à esquerda. O bloco C, com as colunas de trinta e dois a trinta
e cinco, codifica a quantização efetiva, a posição normalizada no quadro e a
profundidade do nó. Nenhuma coluna codifica quadro de referência, campo de
vetores de movimento, energia do resíduo compensado por movimento, gradiente
temporal, tipo de quadro ou nível da pirâmide temporal.

Esta ausência é decisiva, pois a variável causal da decisão de particionamento
muda de natureza entre os dois regimes. Na predição intraquadro, a textura do
quadro-fonte é a própria variável causal, e por isso o bloco A carrega o maior
salto da escada informacional medida nesta tese, de **0,0374** de perda de
otimalidade para a variância isolada a **0,0053** para o bloco A completo, ou
seja, um fator de **7,0 vezes**. Na predição interquadros, a variável causal
passa a ser a energia do resíduo após compensação de movimento, e a textura do
quadro-fonte torna-se um preditor fraco e enviesado desta energia.

O efeito prático se manifesta em dois modos de erro, cujos custos são
assimétricos. Uma região texturada e estática, como um plano de fundo detalhado
sob câmera parada, apresenta variância elevada e leva o modelo a atribuir
probabilidade elevada à divisão, quando o resíduo compensado é praticamente
nulo e a decisão ótima seria o candidato `PARTITION_NONE`; o custo deste erro é
perda de redução de tempo, e não perda de eficiência de codificação. Uma região
lisa sob movimento complexo, como uma oclusão ou conteúdo novo sobre um
gradiente suave, apresenta variância reduzida e leva o modelo a atribuir
probabilidade elevada ao candidato `PARTITION_NONE`, acionando a ação de
`NONE-commit`, que poda a subárvore inteira — justamente a subárvore de que o
codificador necessitava. O custo deste segundo erro é perda direta de eficiência
de codificação.

Deste modo, a extensão direta desloca os erros de confiança elevada do modelo do
lado barato para o lado caro, e a ação mais agressiva da política, que é a
alavanca primária de redução de tempo em predição intraquadro, torna-se a menos
justificada em predição interquadros.

> **Procedência.** Código: `src/aom/av1/encoder/partition_strategy.c:1885-1990`.
> Documentos-fonte: `docs/RESULTADOS_rpp_escada_informacional.md` §4.6, para os
> valores 0,0374 e 0,0053 e para o fator de 7,0 vezes;
> `results/thesis/M4_atributos_e_politica.md`, para a composição dos blocos A, B
> e C. Artefato numérico: `results/models/rpp_ladder/`.

## F1.4 A inversão do nicho competitivo

Esta seção apresenta o resultado mais consequente da análise, que é a inversão
completa do conjunto de heurísticas nativas ativas quando se muda o tipo de
quadro, pois desta inversão depende o argumento que sustenta a segunda solução
positiva desta tese.

Em regime *All-Intra*, três podadores nativos que agem depois do candidato
`PARTITION_NONE` estão desligados por guarda de tipo de quadro, resultado já
registrado nesta investigação: `av1_ml_predict_breakout`, em
`partition_search.c:4296`; `av1_ml_early_term_after_split`, em
`partition_search.c:4359`; e `av1_ml_prune_rect_partition`, em
`partition_search.c:4372`. É este vazio nativo que o H9c e o H9d ocupam, e é ele
que explica por que o H9c alcança, no regime de qualidade elevada, taxa BD de
**0,065%** contra **0,153%** do degrau nativo `cpu-used=1`.

Em quadros que empregam predição interquadros, a situação se inverte por
completo. Os três podadores acima voltam a atuar, e a eles se somam a divisão
por busca de movimento simplificada, condicionada em
`partition_strategy.c:2461`, e a poda de partições retangulares pela mesma
busca, condicionada em `partition_strategy.c:2483`; ao mesmo tempo, a rede
convolucional intraquadro nativa se desliga. Deste modo, o H9a deixa de competir
com a rede convolucional nativa, mas passa a competir com cinco heurísticas
nativas informadas por movimento e a agir antes delas, preemptando decisões que
elas tomariam com mais informação. E o H9c e o H9d perdem exatamente a
propriedade que os tornou uma contribuição: o nicho de decidir depois de
conhecer o custo real de taxa-distorção do candidato `PARTITION_NONE` deixa de
estar nativamente vazio.

Existe, por outro lado, um contrapeso real e favorável à família posterior ao
candidato `PARTITION_NONE`. Os atributos trinta e seis, trinta e sete e trinta e
oito do H9c e do H9d são a taxa, a distorção e o custo de taxa-distorção
efetivamente obtidos pelo candidato `PARTITION_NONE` já buscado. Em um quadro
que emprega predição interquadros, este custo já incorpora o resultado da
compensação de movimento, uma vez que corresponde à melhor busca interquadros
naquele nó. Deste modo, o H9c e o H9d são as únicas das três soluções que
carregam informação temporal de forma implícita, sem qualquer alteração no vetor
de atributos, e são por isso as candidatas naturais à extensão. Este mesmo
contexto de taxa-distorção posterior ao candidato `PARTITION_NONE` foi medido em
predição intraquadro elevando a área sob a curva ROC do H9d de **0,890** para
**0,902**, o que corrobora o valor informacional destas três colunas.

> **Procedência.** Código: `src/aom/av1/encoder/partition_search.c:4296, 4359,
> 4372`; `src/aom/av1/encoder/partition_strategy.c:2423, 2461, 2483`.
> Documentos-fonte: `docs/RESULTADOS_fase6_swap_h9c.md` §4.2 e §9, para o vazio
> nativo e para a limitação de regime já registrada, e §4, para 0,065% contra
> 0,153%; `docs/RESULTADOS_H9d_predizibilidade.md` §2.1, para 0,890 e 0,902.
> Artefato numérico: `results/benchmark/fase6/raw_results.csv`.

## F1.5 O deslocamento do suporte do índice de quantização

Esta seção apresenta um terceiro obstáculo, de natureza pontual e verificável no
código, que afeta a calibração dos limiares e não apenas a qualidade das
predições.

A instrumentação que gerou o conjunto de dados anotado registrou o índice de
quantização do quadro, através de `cm->quant_params.base_qindex`, em
`partition_search.c:5690`. A inferência em tempo de codificação, por outro lado,
utiliza o índice de quantização do bloco, através de `x->qindex`, em
`partition_strategy.c:1986`, valor que alimenta os atributos dezessete e trinta
e dois. Sob a receita de codificação desta tese, que fixa `--deltaq-mode=0` e
`--enable-tpl-model=0`, os dois valores são idênticos e não existe discrepância
alguma.

Numa configuração de acesso aleatório, por outro lado, o modelo de propagação
temporal e a modulação do passo de quantização estão ativos por padrão, e o
índice de quantização do bloco passa a variar dentro do quadro; além disso, a
pirâmide temporal atribui índices de quantização acentuadamente distintos por
nível. Deste modo, dois dos trinta e seis atributos passariam a operar fora do
suporte observado em treinamento, num perceptron de múltiplas camadas que não
oferece garantia alguma de comportamento em extrapolação. A consequência mais
séria não é o erro pontual de predição, e sim o fato de que os limiares, que
constituem o próprio ponto de operação da solução e foram calibrados sobre a
distribuição intraquadro do índice de quantização, deixariam de designar o mesmo
ponto de operação.

> **Procedência.** Código: `src/aom/av1/encoder/partition_search.c:5690`;
> `src/aom/av1/encoder/partition_strategy.c:1986`. Documento-fonte:
> `docs/DECISOES_escopo.md` §3, para a receita de codificação com
> `--deltaq-mode=0` e `--enable-tpl-model=0`.

## F1.6 Quatro cenários de extensão

Esta seção apresenta os quatro cenários de extensão que a análise identifica,
ordenados por custo crescente de implementação, com o prognóstico de cada um e
com a marcação explícita do que exigiria medição.

**Primeiro cenário — implantação sem alteração alguma, em perfil de acesso
aleatório.** As soluções agiriam apenas sobre os quadros-chave. A redução de
tempo total ficaria diluída pela fração do tempo de codificação que estes
quadros representam, e a perda de eficiência de codificação não diluiria na
mesma proporção, uma vez que o quadro-chave serve de referência a todo o grupo
de quadros e a distorção nele introduzida se propaga. Deste modo, a razão entre
redução de tempo e perda de eficiência tende a piorar em relação ao regime
*All-Intra*, e este é o menos vantajoso dos quatro cenários.
`[completar: a fração do tempo de codificação atribuível aos quadros
intraquadro num perfil de acesso aleatório em 4K não foi medida nesta
investigação, e sem ela nem a redução de tempo nem a perda de eficiência deste
cenário podem ser declaradas]`.

**Segundo cenário — remoção da guarda de tipo de quadro, sem retreinamento.**
Este é o cenário que a análise das Seções F1.3 e F1.5 desaconselha, pois os
modelos foram treinados sobre uma distribuição em que a textura do quadro-fonte
é a variável causal e passariam a operar sobre outra, na qual ela não é. Os
limiares calibrados em predição intraquadro deixariam de designar o mesmo ponto
de operação, e mantê-lo exigiria elevar o limiar da ação de `NONE-commit`, o que
consumiria a redução de tempo que justifica a solução.
`[completar: a fronteira de compromisso deste cenário em quadros com predição
interquadros não foi medida; a afirmação acima é mecanismo derivado do código e
do vetor de atributos, e não resultado]`.

**Terceiro cenário — retreinamento sobre conjunto de dados interquadros, com os
mesmos trinta e seis atributos.** Este cenário recupera a calibração, ou seja, a
distribuição das classes e os limiares, mas não recupera informação alguma, pois
o vetor de atributos permanece integralmente espacial. O prognóstico é o de um
modelo que aprende uma distribuição a priori forte, favorável ao candidato
`PARTITION_NONE`, com poder de discriminação reduzido. O custo de entrada não é
desprezível: exige alterar a guarda `ALLINTRA` da instrumentação, em
`partition_search.c:5683`, e regerar integralmente o conjunto de dados anotado.

**Quarto cenário — extensão do vetor de atributos com canais temporais.** Este é
o único cenário tecnicamente defensável dos quatro, e a Seção F1.7 apresenta os
atributos candidatos, todos já residentes no codificador no ponto de inserção.

> **Procedência.** Análise estática sobre
> `src/aom/av1/encoder/partition_strategy.c` e
> `src/aom/av1/encoder/partition_search.c`, no estado do ramo
> `ml-partition-dev`. Nenhuma codificação nova foi executada para este documento,
> e nenhum dos quatro cenários possui medição associada.

## F1.7 Os atributos temporais já residentes no codificador

Esta seção apresenta o desenho do quarto cenário, uma vez que ele é a linha de
investigação que a análise recomenda e que a sua viabilidade depende de os
atributos necessários serem baratos de obter.

O atributo central é a energia do resíduo compensado por movimento no próprio
nó, disponível em `sms_none_feat[0]` e `sms_none_feat[1]`, campos da estrutura
`SIMPLE_MOTION_DATA_TREE` declarada em `context_tree.h:98`. Estes dois campos
são o análogo interquadros exato do atributo zero, que é a variância do bloco, e
o codificador já os calcula sob demanda para as suas próprias heurísticas
nativas. O mesmo vale para as energias dos quatro nós descendentes, lidas em
`partition_strategy.c:1136-1139`, que compõem precisamente o vetor utilizado
pela heurística nativa `av1_ml_early_term_after_split`.

Aos atributos de movimento se somariam três grupos de leitura direta e custo
desprezível: o tipo de quadro e o nível da pirâmide temporal, obtidos de
`cm->current_frame.frame_type` e da estrutura de grupo de quadros; os vetores de
movimento e os quadros de referência dos blocos vizinhos, obtidos de
`xd->above_mbmi` e `xd->left_mbmi`, estruturas que o bloco B já lê para outra
finalidade; e o sinalizador de bloco não codificado destes mesmos vizinhos.

Cabe destacar que este desenho preserva a propriedade que torna a solução desta
tese implantável, que é o custo de extração praticamente nulo: nenhum dos
atributos propostos exige computação nova, pois todos derivam de dados que o
codificador já mantém no ponto de inserção ou que ele calcularia de todo modo
para as suas heurísticas nativas.

> **Procedência.** Código: `src/aom/av1/encoder/context_tree.h:98-100`;
> `src/aom/av1/encoder/partition_strategy.c:599, 615, 1136-1139`.
> Documento-fonte: `docs/ARQUITETURA_pruner_implantado.md` §3, quanto ao
> princípio de custo de extração praticamente nulo dos atributos do bloco B.

## F1.8 Síntese e ordenação da linha de investigação

Esta seção apresenta a síntese da análise, destinada ao texto do capítulo de
trabalhos futuros, e a ordenação das três soluções por prognóstico de extensão.

A delimitação ao regime *All-Intra* não é uma conveniência de escopo desta tese,
e sim uma condição de validade do mecanismo que sustenta a sua segunda
contribuição positiva. Em predição intraquadro, os três podadores nativos
posteriores ao candidato `PARTITION_NONE` estão desligados por guarda de tipo de
quadro, e o nicho que o H9c e o H9d ocupam é nativamente vazio; em predição
interquadros, este nicho está ocupado. Esta formulação é preferível, no texto
final, a qualquer declaração de escopo apresentada como limitação genérica, pois
é estrutural, verificável no código-fonte do codificador e independente de
medição nova.

Quanto à ordenação, o H9c e o H9d são as soluções com melhor prognóstico de
extensão, uma vez que o contexto de taxa-distorção posterior ao candidato
`PARTITION_NONE` já incorpora, em quadros que empregam predição interquadros, o
resultado da compensação de movimento, e é medidamente informativo, tendo
elevado a área sob a curva ROC de 0,890 para 0,902 em predição intraquadro. O
H9a é a solução com pior prognóstico, pois age antes de qualquer informação de
movimento estar disponível e passaria a preemptar cinco heurísticas nativas mais
bem informadas do que ele naquele ponto da cadeia.

Cabe registrar, por fim, uma advertência quantitativa contra a transposição
direta de qualquer número desta tese ao novo regime. A decomposição do custo de
busca por candidato que justifica o H9d — segundo a qual as partições
assimétricas e de quatro vias somam **34,3%** do trabalho local do nó, sobre
**875.317** nós de decisão — foi medida exclusivamente em regime *All-Intra*
com `cpu-used=0`. Em codificação com predição interquadros, o custo dominante
migra para a estimação de movimento e para a busca de modos sobre múltiplos
quadros de referência, e a árvore de particionamento é mais rasa, o que reduz
por construção a oportunidade de poda. Deste modo, toda a fronteira de
compromisso desta tese exigiria remedição, e nenhum dos seus pontos de operação
pode ser declarado válido no novo regime sem ela.

> **Procedência.** Documentos-fonte: `docs/RESULTADOS_C1_custo_por_candidato.md`
> §3, para 34,3% e 875.317 nós; `docs/RESULTADOS_fase6_swap_h9c.md` §4.2 e §9;
> `docs/RESULTADOS_H9d_predizibilidade.md` §2.1. Artefatos numéricos:
> `results/benchmark/partstats*/part_timing*.csv` e
> `results/benchmark/fase6/raw_results.csv`. Script de análise:
> `src/scripts/benchmark/analyze_partstats.py`.

---

> **Natureza deste documento.** Análise estática de código-fonte e de resultados
> já medidos, produzida em 2026-09-02 no ramo `ml-partition-dev`, sem execução de
> codificação nova. Todas as afirmações de mecanismo remetem a linhas
> identificadas do código-fonte do codificador; todas as afirmações numéricas
> remetem a documentos de resultado desta investigação; e todas as projeções de
> desempenho no novo regime estão marcadas como lacuna, uma vez que não foram
> medidas.
