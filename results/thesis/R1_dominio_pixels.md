# 1. O domínio de pixels: diagnóstico, cota superior e os cinco negativos

Esta seção apresenta os resultados obtidos no **domínio de pixels**, ou seja, no
conjunto de soluções que tomam a luminância do superbloco como fonte primária da
informação de poda. São apresentados, nesta ordem, a curva de operação do modelo
estudante da era de pixels e a medição da cota superior por reprodução das
decisões do modelo substituto convolucional, o defeito de luminância nula que
obrigou à remedição de toda a cadeia exploratória inicial, a ablação de
atribuição que produziu o primeiro resultado negativo forte, a hierarquia medida
no crivo ponderado por perda de otimalidade (do inglês *regret*), a
demonstração de que o modelo substituto
convolucional não estabelece cota superior alguma, a auditoria de composição do
vetor de atributos da solução implantada e, por fim, o quinto e último resultado
negativo desta família.

Estes resultados são, no seu conjunto, negativos, e é deliberadamente com eles
que o Capítulo de Resultados se abre. Eles não são descarte de investigação: são
a evidência medida que motiva e justifica a reformulação do problema apresentada
nas seções seguintes, e é apenas contra eles que o ganho das soluções
sobreviventes adquire significado. A família de pixels encerra com **cinco
tentativas independentes negativas**, e nenhuma via de pixels aprendida de ponta
a ponta foi implantada no codificador.

---

## 1.1 A curva de operação do estudante e a cota superior por reprodução

A primeira solução investigada consistiu em um modelo substituto convolucional
do tipo ConvNeXt, com cabeças por nível espelhando a hierarquia de
particionamento do AV1, do qual foi destilado um modelo estudante de perceptrone
de múltiplas camadas sobre as vinte e quatro colunas manuais do bloco A — vinte e
uma de luminância, mais quantização e posição, conforme a auditoria de composição
da Seção 1.6 —, designado `pixels24` e executado no codificador pela função nativa
`av1_nn_predict`. Esta era de pixels produziu a primeira curva de operação real
do trabalho, medida na sequência Jockey do conjunto de teste reservado, em
codificação intraquadro com `cpu-used=0` e quatro pontos de quantização
(`cq` 20, 32, 43 e 55).

No preset conservador, o estudante entrega **0,249% de taxa BD a 1,033× de
aceleração** apenas com a ação de comprometer o nó com o `PARTITION_NONE`,
**0,492% a 1,048×** quando a poda das formas retangulares é acrescentada e
**0,422% a 1,074×** com os limiares refinados por nível. Na varredura agressiva,
a curva estende-se até **1,617% de taxa BD a 1,285× de aceleração**, passando por
**1,23% a 1,176×**. Deste modo, a era de pixels sustentou uma faixa de operação
de aproximadamente 7% a 29% de aceleração ao custo de 0,4% a 1,6% de taxa BD, e
foi este o resultado que o trabalho carregou antes da reformulação.

A cota superior do domínio foi medida pelo experimento designado H8, que não
treina modelo algum: as probabilidades do modelo substituto são gravadas por nó
fora do codificador e reinjetadas nele por arquivo, de modo que a mesma chamada
de poda aplique exatamente a mesma política com as pontuações do substituto. O
codificador é, então, convertido em instrumento de medição da cota superior do
substituto, sem que uma única convolução seja executada em C. No ponto
conservador, com os limiares 0,90, 0,90 e 0,20, a reprodução das decisões do
substituto entrega **−0,114% de taxa BD a 1,021× de aceleração**, e na varredura
agressiva **0,197% a 1,032×**.

Este resultado é o primeiro sinal desfavorável ao domínio de pixels, e é
frequentemente mal lido. O modelo substituto de alta capacidade, com acesso à
luminância bruta do superbloco inteiro e sem pagar custo algum de inferência,
não converte a sua vantagem de informação em tempo: a aceleração obtida é de
apenas 1,02× a 1,03×, ou seja, praticamente nula. Além disso, e por construção,
**todo resultado H8 é uma cota superior que ignora o custo de inferência**, pois
os 28,1 milhões de parâmetros do modelo substituto seriam avaliados a cada
superbloco e jamais poderiam ser embarcados na libaom em regime algum.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §3 (arquitetura do
> substituto, destilação e medição por reprodução);
> `docs/RASTREABILIDADE.md` §5.1 (pontos operacionais e varredura agressiva);
> `docs/INVENTARIO_solucoes.md` §2 (limiares 0,90/0,90/0,20 e faixa de redução de
> tempo do H8); `docs/RESULTADOS_auditoria_dominio_pixels.md` §4 (o custo de
> inferência nunca foi pago). Artefatos: `results/benchmark/h7h8_real_summary.csv`,
> `results/benchmark/h7h8_real/{runs,summary}.csv` e
> `results/benchmark/h7h8_aggr/summary.csv`. Scripts:
> `src/scripts/benchmark/h7h8_bench.py` (duas imagens por sequência, conforme o
> valor padrão declarado em `:71`) e
> `src/scripts/partition_model/surrogate_replay.py`. Commits `172e669` e `9b2f1d9`.

---

## 1.2 O defeito de luminância nula e a remedição da cadeia H1 a H6

A cadeia exploratória inicial, designada H1 a H6, foi integralmente invalidada
por um defeito de representação de dados descoberto em 2026-07-08. Este episódio
é relatado aqui como resultado, e não como nota de rodapé, uma vez que determina
o que pode e o que não pode ser lido daqueles números.

O conjunto de dados armazenava a luminância como `float32` normalizado no
intervalo de zero a um, ao passo que os consumidores em Python assumiam `uint8`
no intervalo de zero a duzentos e cinquenta e cinco. A extração de atributos
truncava os valores para inteiro, produzindo **blocos inteiramente nulos**. O
carregador do modelo substituto normalizava novamente por duzentos e cinquenta e
cinco, produzindo uma entrada quase nula.

A consequência é categórica: **toda a cadeia H1 a H6 foi treinada sobre imagem em
branco**. As conclusões que dela se extraíram — em particular a de que o sinal
disponível na luminância seria limitado a uma cota superior da ordem de 13% a
18% — eram enunciados sobre uma entrada vazia, e não sobre os pixels.

Cabe destacar que o dado bruto estava correto o tempo todo. A verificação
`round(pkl·255)` reproduz o quadro-fonte com diferença máxima nula. Apenas os
consumidores estavam errados, o que explica por que o defeito atravessou seis
hipóteses sem produzir sintoma evidente.

A correção instituiu `data._denorm_uint8` como fonte única de verdade para a
desnormalização e invalidou todos os caches derivados. Sobretudo, introduziu uma
asserção de luminância real, `assert_real_luma`, que passou a guardar o treino, a
destilação e a simulação.

Toda a cadeia foi, então, retreinada sobre pixels reais, e os resultados
apresentados na Seção 1.1 já são os da medição posterior à correção. Os artefatos
da era anterior foram removidos do controle de versão e do disco, por serem
cientificamente inválidos e não comparáveis ao conjunto de dados canônico.

A lição metodológica que este defeito deixa é parte da defesa de validade desta
tese: **asserir que os dados de treino possuem variância não nula** é uma
verificação de custo desprezível que teria interrompido seis hipóteses antes da
primeira medição de codificação. Por isso a asserção foi convertida em guarda
permanente do arcabouço, e não em correção pontual.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §2.2 (descrição do
> defeito, correção e consequência metodológica); `docs/ANDAMENTO_tese.md` §1,
> itens 2 e 4 (cota superior aparente de 13% a 18% e invalidação de H1 a H6);
> `docs/RASTREABILIDADE.md` §8 (artefatos removidos por invalidade científica).
> Scripts: `src/scripts/partition_model/data.py` (`_denorm_uint8`) e a asserção
> `assert_real_luma`. Commits `cb9d407` (correção) e `63f299c` (retreino sobre
> luminância real).

---

## 1.3 A ablação de atribuição — o primeiro resultado negativo

A pergunta que a curva de operação da Seção 1.1 não responde é a da atribuição. O
ganho medido decorre do aprendizado, ou decorre da política de poda, que
produziria ganho semelhante com qualquer pontuação razoável?

Para respondê-la foi executada uma **ablação de atribuição** que mantém a
política rigorosamente idêntica — comprometimento com o `PARTITION_NONE` no mesmo
ponto de inserção, mesma grade de limiares, mesmo codificador — e varia
**exclusivamente a fonte da pontuação**. Compararam-se o modelo estudante de
pixels, uma pontuação aleatória e um limiar de variância trivial da forma
`exp(−var/1000)`.

O resultado é negativo e consistente. A tempo casado, o limiar de variância
apresenta taxa BD **menor que a do modelo estudante em todos os cinco níveis de
aceleração medidos**: **0,171% contra 0,238%** a 1,05×; **0,408% contra 1,058%** a
1,15×; **0,764% contra 1,393%** a 1,30×; **1,119% contra 1,660%** a 1,45×; e
**1,357% contra 1,895%** a 1,55×.

A pontuação aleatória, por sua vez, é largamente inferior aos dois, com **2,935%**
a 1,30× e **4,745%** a 1,55×. Isso estabelece que existe aprendizado no modelo
estudante, mas não o suficiente para separá-lo de uma estatística de uma única
grandeza.

A extensão exata do que foi medido precisa ser declarada com precisão, uma vez
que é sobre ela que o resultado se sustenta e é ela que limita a sua
generalidade: **uma única sequência (Jockey), duas imagens, `cpu-used=0`, quatro
pontos de quantização**. Este é o experimento de codificação isolado da família
de pixels, e a sua limitação de escala foi registrada desde o primeiro dia.

A leitura de saturação que dele chegou a ser extraída **foi retirada do corpo do
texto** por medição posterior, e é substituída pela hierarquia apresentada na
Seção 1.4. O que permanece é o fato medido: naquele experimento, o limiar de
variância supera o modelo de pixels sob política casada.

O peso deste resultado negativo não está na sua escala, e sim na sua natureza. A
variância do bloco é **um dos vinte e quatro atributos do próprio modelo
estudante**, e um modelo que dispõe dela entre as suas entradas não deveria
perder para ela isolada. Foi esta constatação, e não uma preferência de
arquitetura, que reorientou a investigação para o contexto de taxa-distorção
barato.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §3 (ablação de
> atribuição sob política casada); `docs/RASTREABILIDADE.md` §5.2;
> `docs/INVENTARIO_solucoes.md` §2.1 (declaração explícita de uma sequência e
> duas imagens, e da contradição com o crivo); `docs/ANDAMENTO_tese.md` §1 item 6
> e §1.1 (o negativo como origem do `dataset_h9`). Artefatos:
> `results/benchmark/ablation_matched.csv` e
> `results/benchmark/ablation_attrib/curve.csv`. Scripts:
> `src/scripts/benchmark/ablation_attrib.py` e
> `src/scripts/benchmark/analyze_ablation.py`. Commits `2380e91` e `4c5fb9f`.

---

## 1.4 A hierarquia medida no crivo de perda de otimalidade

A contradição entre a ablação de atribuição e as medições posteriores foi
arbitrada — na medida em que um crivo offline pode arbitrar — pelo crivo
ponderado por perda de otimalidade, que substitui a contagem de nós
comprometidos pelo custo de taxa-distorção real de cada poda.

A grandeza reportada é a **fração de perda de otimalidade**. Ela é a razão
percentual entre a soma da perda de otimalidade absoluta dos nós efetivamente
podados e o custo de taxa-distorção total do conjunto de nós de decisão. No
numerador, a perda de otimalidade absoluta é o sobrecusto de taxa-distorção pago
por podar em vez de tomar a subárvore ótima — grandeza não negativa, e nula
sempre que a decisão correta já era `PARTITION_NONE`.

O denominador é acumulado antes de qualquer limiar ser fixado e permanece
constante ao longo de toda a varredura. Esta é a condição para que a grandeza
seja legível como fronteira: ela tende a zero quando a poda tende a zero, ao
passo que um dano médio por poda permaneceria elevado sob poda escassa.

A grandeza é lida a **redução de custo casada**, de modo que todos os candidatos
sejam comparados no mesmo ponto de operação. A avaliação cobre **seis
sequências** do conjunto de validação e de teste reservado e **3.808.703 nós de
decisão**, contra os modelos treinados nas dez sequências restantes.

> **Correção de registro.** Versões anteriores desta seção citavam 792.840 nós de
> decisão ao lado desta hierarquia. Aquela contagem é a da execução original do
> crivo, que não possuía as pernas convolucionais; a hierarquia com ConvNeXt vem
> de execução posterior, sobre a vara completa de 3.808.703 nós. O mesmo
> parágrafo confirmava a ordenação a 30% com valores da execução de 792.840 nós,
> misturando duas medições sob uma única contagem. Todos os valores abaixo saem
> agora de uma **execução única**, `results/models/oracle_regret_rpp/`.

A hierarquia medida a 25% de redução de custo, na qual o menor valor é o melhor,
é a seguinte: **variância 0,0374; ConvNeXt com entropia cruzada 0,0272; ConvNeXt
com alvo de perda de otimalidade 0,0219; vinte e quatro colunas compactas 0,0053;
e as mesmas vinte e quatro mais a vizinhança causal de particionamento 0,0033**.

Os ganhos marginais entre passos sucessivos são igualmente informativos.
Acrescentar as vinte e três colunas manuais restantes à variância isolada compra
**7,0×**. Acrescentar 28,1 milhões de parâmetros convolucionais sobre pixels
crus rende **0,2×**, ou seja, **piora**, ainda que o modelo profundo disponha de
2.481 vezes mais parâmetros. Acrescentar as oito colunas de vizinhança causal de
particionamento compra **1,60×**, e esta é a única adição de contexto que
compra alguma coisa: as quatro colunas de quantização efetiva, posição no quadro
e profundidade **não acrescentam informação decisória**, e pioram o resultado
sempre que somadas à vizinhança.

A mesma ordenação se reproduz nos pontos de 20% e de 30% de redução de custo, o
que confirma que a hierarquia não é artefato de um ponto de operação isolado.

Esta hierarquia é o enunciado que a tese sustenta a respeito do domínio de
pixels, e ela é incompatível com qualquer leitura de que a informação disponível
nos pixels se esgotaria na variância: as vinte e quatro colunas compactas superam
a variância por 7,0× nesta métrica.

Três ressalvas são declaradas e devem acompanhar a hierarquia em toda leitura.

A primeira é de amostragem, e foi **sanada por remedição**. A grade de limiares
original da variância saltava de 6,14% para 39,46% de redução de custo, de modo
que todos os valores intermediários da sua curva eram interpolados através de um
único vão. Uma grade própria de vinte e seis limiares fechou o vão, e o ponto de
τ igual a 0,980 entrega 25,20% de redução medida. A leitura anterior, de 0,0573,
**superestimava a variância em 1,53 vez** por efeito exclusivo da interpolação.

A segunda é de procedimento, e também foi sanada. Os degraus desta hierarquia são
agora treinados por **receita única** — mesma topologia, mesmo número de épocas,
mesma taxa de aprendizado, mesma amostragem —, variando apenas o subconjunto de
colunas, com três sementes por degrau. A hierarquia anterior comparava um braço
destilado de vinte e quatro colunas contra um braço de entropia cruzada direta de
trinta e seis, e o procedimento de treino sozinho valia de 2,0 a 2,4 vezes.

A terceira é de escopo, e **permanece**. O crivo **não adjudica**: no único par
com verdade de codificação limpa ele diverge do codificador. A sua função
declarada é eliminar candidatos inferiores antes do custo caro de codificação, e
não coroar vencedores.

> **Procedência.** `docs/RESULTADOS_oraculo_regret.md` §2 a §4 (definição da
> fração de perda de otimalidade e do modelo de custo de busca);
> `docs/RESULTADOS_rpp_escada_informacional.md` §3 e §4 (fronteira única, escada
> sob receita fixa, grade densa da variância e valores por semente);
> `docs/RESULTADOS_auditoria_convnext_corpus.md` (corpus dos braços profundos);
> `docs/INVENTARIO_solucoes.md` §2.2. Artefato:
> `results/models/oracle_regret_rpp/{report,ranking.csv,frontier.csv}`, execução
> única de onde saem todos os valores desta seção; os artefatos superados
> `results/models/oracle_regret/` e `oracle_regret_convnext/` são preservados.
> Scripts: `src/scripts/partition_model/oracle_regret.py` e `rpp_ladder.py`.

---

## 1.5 O modelo substituto convolucional não estabelece cota superior

A intenção declarada ao construir o modelo substituto convolucional era a de
estabelecer a **cota superior do domínio de pixels**: capacidade elevada, acesso
à luminância bruta do superbloco inteiro e reprodução exata das suas decisões no
codificador, sem custo de inferência. Esta intenção não sobreviveu à medição, e o
seu fracasso é, ele próprio, um resultado da tese. Três medições independentes o
sustentam.

A primeira é a da hierarquia da Seção 1.4. O modelo substituto de 28,1 milhões de
parâmetros sobre pixels crus **perde para o braço de vinte e quatro colunas**, um
perceptrone de múltiplas camadas sobre colunas manuais das quais vinte e uma são
extraídas da **mesma luminância** e as três restantes são o índice de quantização
e a posição no superbloco, que a rede também recebe pelo seu plano constante de
quantização — por 0,0219 contra 0,0053 a 25% de redução de custo, ou seja,
**4,1×**, com 2.481 vezes mais parâmetros. Um modelo batido por outro cujo acesso
à informação é estritamente menor, e que ele poderia em princípio representar
internamente, **não estabelece cota superior alguma**: o resultado enuncia algo a
respeito do treino realizado, e não a respeito dos pixels.

A segunda é a do objetivo de treino, e ela **inverteu-se por remedição**. O
enunciado anterior desta tese era o de que o retreino contra a perda de
otimalidade, com peso `1 + α·regret_rel` e `α = 3`, teria piorado o modelo em
toda a faixa, por fatores de 1,06× a 3,80×, e que a refutação seria forte por
estarem alinhados objetivo de treino e métrica de avaliação.

Aquela comparação era confundida. Auditoria dos argumentos gravados nos pontos de
controle estabeleceu que o braço de entropia cruzada fora treinado sobre um
conjunto de dados distinto — quatro sequências, um único ponto de quantização e
dois quadros —, com duas das seis sequências da vara de avaliação contaminadas,
ao passo que o braço de perda de otimalidade fora treinado sobre o conjunto
canônico com a partição correta. Mudavam corpus, taxa de aprendizado, decaimento,
número de épocas, tamanho de lote e parada antecipada, e não apenas o objetivo.

A ablação foi refeita com corpus, escalonamento e critério de seleção idênticos,
fazendo `α = 0` degenerar o peso em entropia cruzada pura. Sob esse controle, o
alvo de perda de otimalidade é **melhor** que a entropia cruzada em todos os
pontos de 10% de redução de custo em diante, com 0,0219 contra 0,0272 a 25%. A
hipótese de que o alvo de perda de otimalidade levanta o teto do domínio de
pixels foi, portanto, **confirmada fracamente, e não refutada**: ela levanta o
teto, mas não o suficiente para que a via de pixels compita com a representação
compacta, que continua 4,1 vezes à frente.

A terceira é a da capacidade, e ela **sobrevive à remedição, reforçada**. Sob
entropia cruzada e corpus correto, dobrar a largura de fusão de 128 para 256
piora a perda de validação de 0,895092 para 0,903984, isto é, **1,0%**. O
registro anterior media 0,16% de diferença, mas o fazia entre dois braços de
`α = 3`. A largura maior não apenas deixa de ajudar: ela atrapalha.

Cabe ainda registrar a correção de um erro de registro anterior. Alegou-se que o
ponto de controle do modelo substituto original teria sido colhido em
sobreajuste, e a alegação é falsa: nas trinta épocas completas, o mínimo da perda
de validação (1,7999) e o máximo do macro-F1 (0,2034) ocorrem ambos na época 13,
que é a do ponto de controle salvo. O modelo original está corretamente
selecionado. Ele é apenas fraco em termos absolutos.

O registro honesto do que a tese possui é, deste modo, o seguinte. Há uma **cota
inferior** do domínio de pixels, dada pelo melhor desempenho observado, que é o
do braço de vinte e quatro colunas. E há uma **cota superior genuína apenas no
oráculo**: a decisão de taxa-distorção ótima de perda de otimalidade nula, que
limita qualquer podador, e não apenas os de pixels.

A cota superior do domínio de pixels permanece **não medida**. O modelo
substituto convolucional permanece no arcabouço como instrumento de diagnóstico e
como a tentativa documentada de estabelecê-la.

Cabe, por fim, registrar uma consequência da remedição sobre uma decisão de
escopo. A opção por não levar o braço de perda de otimalidade ao codificador fora
justificada pelo argumento de que ele passaria o critério de decisão sem superar
a sua própria linha de base, o que tornaria o custo de replay injustificado. Esse
argumento **cai**, uma vez que ele supera a linha de base. A decisão é
reenquadrada, como no caso análogo da regressão de perda de otimalidade, em
**assimetria de custo experimental, e não implicação lógica** — e permanece bem
fundamentada por outra via, a de que a reprodução das decisões do substituto no
codificador já mediu aceleração de apenas 1,02× a 1,03×.

> **Procedência.** `docs/RESULTADOS_auditoria_convnext_corpus.md` §1 a §4
> (corpus, contaminação da vara, degeneração do peso em `α = 0` e controle de
> capacidade); `docs/RESULTADOS_rpp_escada_informacional.md` §3 e §4.7 a §4.9
> (fronteira única, razão de 4,1× e inversão do resultado de objetivo);
> `docs/RESULTADOS_convnext_regret.md` (documento superado, com nota de correção
> no cabeçalho) §4 (correção do registro sobre a seleção do ponto de controle);
> `docs/ANDAMENTO_tese.md` §1.3 (cota superior genuína apenas no oráculo).
> Artefatos: `results/models/surrogate_ce_h9/`,
> `results/models/surrogate_ce_h9_f256/`, `results/models/surrogate_regret/`,
> `results/models/surrogate_real/{metrics.csv,surrogate_best.pt}` e
> `results/models/oracle_regret_rpp/`. Scripts:
> `src/scripts/partition_model/train_surrogate_regret.py` e
> `src/scripts/partition_model/oracle_regret.py::score_surrogate`. A correção de
> registro sobre a seleção do ponto de controle incide sobre o commit `0122b53`.

---

## 1.6 A composição auditada — o que a hierarquia de fato enuncia

A leitura da hierarquia da Seção 1.4 exige uma auditoria de composição, uma vez
que os dois conjuntos de atributos comparados **não são disjuntos**.

O layout do vetor está rotulado no próprio código de extração. Os índices 0 a 23
constituem o bloco A, rotulado como bloco de pixels. Os índices 24 a 31
constituem o bloco B, de contexto de particionamento da vizinhança. Os índices 32
a 35 constituem o bloco C, de quantização e posição.

O rótulo do bloco A é, contudo, mais largo do que o seu conteúdo, e a diferença
importa. Das suas vinte e quatro colunas, **vinte e uma derivam de luminância**;
as outras três são o índice de quantização normalizado (`q_norm`, índice 17) e as
duas coordenadas de posição do nó dentro da unidade de 64 px (índices 22 e 23),
conforme `features.py:53-61`.

Como o H9a é definido como a união dos blocos A, B e C, segue que **vinte e uma
das suas trinta e seis colunas são descritores de luminância** —
variância global e por quadrante, gradientes horizontal e vertical, orientação,
densidade de bordas e contraste com o bloco-pai e com os irmãos. Segue também que
**o H9a não contém grandeza alguma de custo de taxa-distorção**, pois estas
constituem o bloco E, exclusivo do H9c. Mais ainda: `pixels24` é definido como os
índices de 0 a 23, ou seja, é **literalmente o bloco A do H9a**.

A consequência é que o enunciado corrente "o contexto de taxa-distorção vence os
pixels" **é falso**, e seria derrubado por qualquer avaliador que abrisse o
arquivo de extração de atributos.

O enunciado correto, e verificável, é triplo. No domínio de pixels, **descritores
manuais compactos vencem uma rede convolucional profunda sobre pixels crus**, por
**4,1×**, com 2.481 vezes menos parâmetros. **As oito colunas de vizinhança
causal de particionamento, de custo praticamente nulo por já estarem residentes
na memória do codificador, acrescentam 1,60× sobre eles.** E **as quatro colunas
de quantização efetiva, posição no quadro e profundidade não acrescentam
informação decisória**: somadas ao bloco A elas não se separam dele, com o dobro
da dispersão entre sementes, e somadas ao bloco B pioram o resultado de forma
consistente, com as três sementes de A+B superando as três de A+B+C. A solução
implantada é, portanto, majoritariamente um modelo de pixels ao qual foi
acrescentado contexto causal barato — e o seu vetor de trinta e seis colunas não
é, offline, o melhor dos quatro subconjuntos medidos.

Cabe registrar que estes números substituem os anteriores, de 1,7× e 3,4×, e por
que substituem. Aqueles comparavam um braço destilado de vinte e quatro colunas
contra um braço de entropia cruzada direta de trinta e seis, de modo que o
procedimento de treino entrava na conta junto com os atributos — e ele sozinho
vale de 2,0 a 2,4 vezes. A escada atual fixa a receita e varia apenas o
subconjunto de colunas, com três sementes por degrau.

Cabe acrescentar uma precisão que fortalece o segundo enunciado. Como o índice de
quantização normalizado **já reside no bloco A**, o retorno de 1,60× das oito
colunas não pode ser atribuído ao modelo passar a dispor da quantização: ela
estava disponível dos dois lados da comparação. O mesmo vale para a rede
convolucional, cuja entrada inclui um plano constante de quantização. E o achado
sobre o bloco C confirma-o por outra via: quando a quantização é acrescentada de
novo, agora na forma do passo de dequantização efetivo, ela nada compra.

Este enunciado é mais forte que o anterior, e não mais fraco. Ele converte um
resultado negativo sobre redes convolucionais em um resultado positivo sobre
**onde a informação está**: nas decisões de particionamento já tomadas pela
vizinhança causal, e não na capacidade de representação sobre o bloco-fonte nem
no estado de quantização e posição do codificador.

A mesma leitura é corroborada pelo critério de decisão offline da fase seguinte.
A ação isolada de comprometimento com o `PARTITION_NONE` rende de 10% a 19% de
redução de custo com o `pixels24` e de 16% a 25% com o H9a. É, portanto, um ganho
**marginal sobre** as vinte e quatro colunas do bloco A, e não um ganho
**competitivo contra** elas. Estes intervalos são lidos de uma varredura de
limiares pela regra de risco casado: para cada subconjunto, toma-se a linha cujo
`split_lost` fica imediatamente abaixo de cada patamar de risco declarado.

> **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §2, §2.1 e §2.2
> (layout do vetor, `pixels24` como bloco A e releitura da hierarquia) e §7
> (correções de registro aplicadas em quatro documentos);
> `docs/RESULTADOS_rpp_escada_informacional.md` §4.2 a §4.6 (confundidor de
> treino, separação por semente dos blocos B e C, e a escada corrigida);
> `docs/RESULTADOS_convnext_regret.md` §5 (nota de correção de 2026-07-26);
> `docs/RASTREABILIDADE.md` §5.3 (redução de custo por subconjunto e nota de
> composição). Fonte auditada: `src/scripts/partition_model/features.py:53–61`,
> `:196–201`, `:204` e `:216`. Artefatos: `results/models/gate2_final_sweep.csv`
> (fonte dos intervalos por risco casado citados acima; `results/models/gate2_final.csv`
> reúne uma agregação distinta, por limiares fixos de perda de SPLIT, e não é a
> fonte destes números). Commits `1f7535d` e `8866757`.

---

## 1.7 O quinto negativo — predizibilidade intra a partir dos vizinhos

A auditoria de composição expôs, além dos achados da Seção 1.6, uma hipótese
**especificada e nunca testada**.

O bloco D fora especificado como o SATD do resíduo de uma predição intra barata
construída a partir dos vizinhos reconstruídos, com a advertência explícita de
que um resíduo de predição constante teria a mesma variância do bloco e, portanto,
nada acrescentaria. A implementação, contudo, calcula o SATD do **bloco-fonte**,
sem predição alguma e sem tocar em vizinho algum.

Deste modo, a configuração reprovada no critério de decisão da fase 2 — com
46,9%, 50,5%, 53,4% e 57,8% de redução de custo contra 47,0%, 49,7%, 52,6% e
57,3% do H9a, ou seja, nula — testou uma estatística derivada apenas da fonte,
fortemente correlacionada com a variância e os gradientes que o bloco A já
contém. Ela não informa sobre a hipótese formulada.

A hipótese especificada foi, então, testada como bloco D', com três colunas:
disponibilidade do vizinho, logaritmo do SATD do resíduo e ganho de SATD. Escolhe-
se a melhor entre as predições DC, vertical, horizontal e PAETH por soma de
diferenças absolutas, descartando o termo DC nas duas pontas para que a
comparação se faça entre energias de corrente alternada.

A bancada é idêntica à do critério de decisão da fase 2, variando **apenas o
conjunto de colunas**, de modo que o veredito incida sobre informação e não sobre
capacidade. A execução cobriu 13.000 superblocos de treino e 3.000 de validação,
com três sementes por braço.

**O critério de decisão não é atingido.** Ele exigia, fixado antes da medição,
entropia cruzada menor **e** área sob a curva maior nos níveis de 16 e de 32
amostras simultaneamente.

Em 32 amostras há um positivo consistente, com entropia cruzada de 0,7422 contra
0,7657 e área sob a curva de 0,868 contra 0,851 — mas apenas 1.869 nós dispõem do
atributo. Em **16 amostras, o nível com 15.855 nós, ou seja, com oito vezes mais
dados, o efeito é nulo**: a entropia cruzada passa de 0,5149 para 0,5144 e a área
sob a curva recua de 0,847 para 0,846. A leitura de redução de custo a risco
casado é igual ou pior em todos os limites, com 6,83%, 6,83%, 14,74% e 21,30%
contra 7,36%, 7,36%, 15,37% e 21,48% do H9a.

A objeção de diluição, segundo a qual o nulo em 16 amostras decorreria de treinar
com nós sem vizinho disponível, **é contrariada pelo padrão observado**. A
disponibilidade é menor em 32 amostras (17,6%) do que em 16 amostras (45,9%), e é
em 32 amostras que o efeito aparece.

Como o atributo custaria uma predição intra e uma transformada de Hadamard por nó
no codificador, um ganho nulo no nível mais numeroso já o desqualifica. Fica
registrado, sem ser perseguido, um resíduo delimitado: o positivo em 32 amostras,
e o fato de que o nível de 64 amostras é **invisível a este crivo**, uma vez que
ali a disponibilidade offline do atributo é de 0,0%.

Com este veredito, a família fecha com **cinco vias de pixels que não foram
implantadas**: o modelo substituto convolucional com entropia cruzada, o mesmo
modelo com alvo de perda de otimalidade, a decisão estruturada por rede de grafos
do Approach B, o bloco D sobre o bloco-fonte e o bloco D' de predizibilidade
intra a partir dos vizinhos.

A contagem de **hipóteses refutadas**, contudo, é de **quatro**, e a distinção é
obrigatória. A hipótese de que o alvo de perda de otimalidade levantaria o teto
do domínio de pixels **não foi refutada**: sob ablação controlada, ela se
confirma, ainda que fracamente, conforme a Seção 1.5. Aquela via não foi
implantada porque o teto que ela levanta permanece 4,1 vezes atrás da
representação compacta, e não porque a hipótese tenha caído.

> **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §3 (divergência
> entre especificação e implementação do bloco D), §6 (atributos, convenção do
> SATD e aproximação do vizinho-fonte), §6.1 (disponibilidade, sinal por nível e
> leitura de redução de custo) e §6.2 (veredito e resíduo delimitado);
> `docs/RESULTADOS_convnext_regret.md` §5 (quadro de encerramento da família);
> `docs/INVENTARIO_solucoes.md` §2; `docs/RASTREABILIDADE.md` §5.3 (valores da
> configuração do bloco D). Artefatos: `results/models/gate_intra_pred.csv` e
> `results/models/gate_intra_pred_sweep.csv`. Scripts:
> `src/scripts/partition_model/gate_intra_pred.py`,
> `src/scripts/partition_model/features_intrapred.py` e
> `src/scripts/partition_model/gate2_signal.py`.

---

## 1.8 Síntese e encaminhamento

O domínio de pixels está, deste modo, caracterizado por medição, e não por
conjectura. Quatro resultados o delimitam.

A era de pixels entregou uma curva de operação real de 1,033× a 1,285× de
aceleração, ao custo de 0,249% a 1,617% de taxa BD. A reprodução das decisões do
modelo substituto convolucional no codificador mostrou que a cota superior
pretendida não converte a sua vantagem de informação em tempo, com aceleração de
apenas 1,02× a 1,03×.

A cadeia exploratória anterior a esta curva foi integralmente remedida em razão
do defeito de luminância nula, e as suas conclusões não podem ser invocadas. Sob
política casada, um limiar de variância trivial supera o modelo estudante de
pixels nos cinco níveis de aceleração medidos de um experimento de uma sequência
e duas imagens. E o modelo substituto convolucional, longe de estabelecer a cota
superior do domínio, perde para um perceptrone sobre vinte e quatro colunas do
bloco A, das quais vinte e uma são extraídas da mesma luminância que ele recebe
crua.

O conjunto destes negativos não conduz a uma conclusão de impossibilidade. Conduz
a uma reorientação precisa da pergunta.

A decisão de particionamento é, por definição, uma decisão de taxa-distorção. A
hierarquia medida indica que o que separa o melhor podador dos demais não é
capacidade de representação sobre o bloco-fonte, e sim **oito colunas de
vizinhança causal de particionamento, de custo praticamente nulo** — as decisões
que os blocos acima e à esquerda já tomaram, que a própria busca nativa já
consulta e que a memória do codificador já contém no momento da decisão. A
decomposição da Seção 1.6 é precisa quanto a isso: das doze colunas de contexto
causal, são as oito de vizinhança que compram o ganho, e as quatro de
quantização, posição e profundidade nada acrescentam. É esta constatação, e não
uma preferência de arquitetura, que motiva e justifica a virada investigada nas
seções seguintes.

A solução que materializa esta virada — o H9a, um podador pré-busca por contexto
de taxa-distorção barato, cujo vetor de trinta e seis atributos contém, ele
próprio, as vinte e quatro colunas do bloco A aqui caracterizadas — é
apresentada na próxima seção, com a sua curva de operação no conjunto de teste
reservado, a sua ablação de atribuição e os seus resultados sob protocolo das
condições comuns de teste.

> **Procedência.** Consolidação das notas das Seções 1.1 a 1.7; nenhum valor novo
> é introduzido nesta síntese.
