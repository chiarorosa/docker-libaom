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
de múltiplas camadas sobre vinte e quatro descritores manuais de luminância,
designado `pixels24` e executado no codificador pela função nativa
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
por um defeito de representação de dados descoberto em 2026-07-08, e este
episódio é relatado aqui como resultado, e não como nota de rodapé, pois ele
determina o que pode e o que não pode ser lido daqueles números. O conjunto de
dados armazenava a luminância como `float32` normalizado no intervalo de zero a
um, enquanto os consumidores em Python assumiam `uint8` no intervalo de zero a
duzentos e cinquenta e cinco: a extração de atributos truncava os valores para
inteiro, produzindo **blocos inteiramente nulos**, e o carregador do modelo
substituto normalizava novamente por duzentos e cinquenta e cinco, produzindo
uma entrada quase nula.

A consequência é categórica: **toda a cadeia H1 a H6 foi treinada sobre imagem
em branco**, e as conclusões que dela se extraíram — em particular a de que o
sinal disponível na luminância seria limitado a uma cota superior da ordem de 13%
a 18% — eram enunciados sobre uma entrada vazia, e não sobre os pixels. Cabe
destacar que o dado bruto estava correto o tempo todo, pois a verificação
`round(pkl·255)` reproduz o quadro-fonte com diferença máxima nula; apenas os
consumidores estavam errados, o que explica por que o defeito atravessou seis
hipóteses sem produzir sintoma evidente.

A correção instituiu `data._denorm_uint8` como fonte única de verdade para a
desnormalização, invalidou todos os caches derivados e, sobretudo, introduziu uma
asserção de luminância real, `assert_real_luma`, que passou a guardar o treino, a
destilação e a simulação. Toda a cadeia foi, então, retreinada sobre pixels
reais, e os resultados apresentados na Seção 1.1 já são os da medição posterior à
correção. Os artefatos da era anterior foram removidos do controle de versão e do
disco, por serem cientificamente inválidos e não comparáveis ao conjunto de dados
canônico.

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

A pergunta que a curva de operação da Seção 1.1 não responde é a da atribuição:
o ganho medido decorre do aprendizado, ou decorre da política de poda, que
produziria ganho semelhante com qualquer pontuação razoável? Para respondê-la
foi executada uma **ablação de atribuição** que mantém a política rigorosamente
idêntica — comprometimento com o `PARTITION_NONE` no mesmo ponto de inserção,
mesma grade de limiares, mesmo codificador — e varia **exclusivamente a fonte
da pontuação**, comparando o modelo estudante de pixels, uma pontuação
aleatória e um limiar de variância trivial da forma `exp(−var/1000)`.

O resultado é negativo e consistente. A tempo casado, o limiar de variância
apresenta taxa BD **menor que a do modelo estudante em todos os cinco níveis de
aceleração medidos**: **0,171% contra 0,238%** a 1,05×, **0,408% contra 1,058%**
a 1,15×, **0,764% contra 1,393%** a 1,30×, **1,119% contra 1,660%** a 1,45× e
**1,357% contra 1,895%** a 1,55×. A pontuação aleatória, por sua vez, é largamente
inferior aos dois, com **2,935%** a 1,30× e **4,745%** a 1,55×, o que estabelece
que existe aprendizado no modelo estudante, mas não o suficiente para separá-lo
de uma estatística de uma única grandeza.

A extensão exata do que foi medido precisa ser declarada com precisão, pois é
sobre ela que o resultado se sustenta e é ela que limita a sua generalidade:
**uma única sequência (Jockey), duas imagens, `cpu-used=0`, quatro pontos de
quantização**. Este é o experimento de codificação isolado da família de pixels,
e a sua limitação de escala foi registrada desde o primeiro dia. A leitura de
saturação que dele chegou a ser extraída **foi retirada do corpo do texto** por
medição posterior, e é substituída pela hierarquia apresentada na Seção 1.4; o
que permanece é o fato medido, ou seja, que naquele experimento o limiar de
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
arbitrada, na medida em que um crivo offline pode arbitrar, pelo crivo ponderado
por perda de otimalidade, que substitui a contagem de nós comprometidos pelo
custo de taxa-distorção real de cada poda. A grandeza reportada é a **fração de
perda de otimalidade**, definida como a razão percentual entre a soma da perda
de otimalidade absoluta dos nós efetivamente podados — ou seja, o sobrecusto de
taxa-distorção pago por podar em vez de tomar a subárvore ótima, grandeza não
negativa e nula sempre que a decisão correta já era `PARTITION_NONE` — e o
custo de taxa-distorção total do conjunto de nós de decisão. Este denominador é
acumulado antes de qualquer limiar ser fixado e permanece constante ao longo de
toda a varredura, o que é a condição para que a grandeza seja legível como
fronteira: ela tende a zero quando a poda tende a zero, ao passo que um dano
médio por poda permaneceria elevado sob poda escassa. A grandeza é lida a
**redução de custo casada**, de modo que todos os candidatos sejam comparados
no mesmo ponto de operação. A avaliação
cobre **seis sequências** do conjunto de validação e de teste reservado e
**792.840 nós de decisão**, contra os modelos treinados nas dez sequências
restantes.

A hierarquia medida a 25% de redução de custo, na qual o menor valor é o melhor,
é a seguinte: **variância 0,0573; ConvNeXt com alvo de perda de otimalidade
0,0219; ConvNeXt
com entropia cruzada 0,0207; `pixels24` 0,0121; e H9a 0,0036**. Os ganhos
marginais entre passos sucessivos são igualmente informativos: acrescentar os
vinte e três descritores manuais restantes à variância isolada compra **4,7×**;
acrescentar 28,1 milhões de parâmetros convolucionais sobre pixels crus rende
**0,6×**, ou seja, **piora**; e acrescentar os doze atributos de vizinhança,
quantização e posição ao `pixels24` compra **3,4×**.

A mesma ordenação se reproduz no ponto de 30% de redução de custo, com H9a em
0,006, `pixels24` em 0,015, variância em 0,060 e pontuação aleatória em 0,612, o que
confirma que a hierarquia não é artefato de um ponto de operação isolado. Esta
hierarquia é o enunciado que a tese sustenta a respeito do domínio de pixels, e
ela é incompatível com qualquer leitura de que a informação disponível nos pixels
se esgotaria na variância: o `pixels24` supera a variância por 4,7× nesta
métrica.

Duas ressalvas são declaradas e devem acompanhar a hierarquia em toda leitura. A
primeira é de amostragem: a grade de limiares da variância salta de 6,14% para
39,46% de redução de custo, de modo que **todos os valores intermediários da sua
curva são interpolados através de um único vão**, e a razão medida contra os
modelos de pixels repousa sobre esta interpolação. A segunda é de escopo: o crivo
**não adjudica**, pois no único par com verdade de codificação limpa ele diverge
do codificador, e a sua função declarada é eliminar candidatos inferiores antes
do custo caro de codificação, e não coroar vencedores.

> **Procedência.** `docs/RESULTADOS_oraculo_regret.md` §2 a §4 (definição da
> fração de perda de otimalidade, extensão do conjunto e fronteira por redução
> de custo);
> `docs/RESULTADOS_convnext_regret.md` §2 e §5 (hierarquia a 25% e ganhos
> marginais) e §3 (ressalva da amostragem da curva da variância);
> `docs/INVENTARIO_solucoes.md` §2.2. Artefatos:
> `results/models/oracle_regret/{report,ranking.csv,frontier.csv}` e
> `results/models/oracle_regret_convnext/`. Script:
> `src/scripts/partition_model/oracle_regret.py`.

---

## 1.5 O modelo substituto convolucional não estabelece cota superior

A intenção declarada ao construir o modelo substituto convolucional era a de
estabelecer a **cota superior do domínio de pixels**: capacidade elevada, acesso
à luminância bruta do superbloco inteiro e reprodução exata das suas decisões no
codificador, sem custo de inferência. Esta intenção não sobreviveu à medição, e o
seu fracasso é, ele próprio, um resultado da tese. Três medições independentes o
sustentam.

A primeira é a da hierarquia da Seção 1.4. O modelo substituto de 28,1 milhões de
parâmetros sobre pixels crus **perde para o `pixels24`**, um perceptrone de
múltiplas camadas sobre vinte e quatro atributos manuais extraídos da **mesma
luminância**, por 0,0207 contra 0,0121 a 25% de redução de custo, ou seja, cerca
de 1,7×. Um modelo batido por outro cujo acesso à informação é estritamente
menor, e que ele poderia em princípio representar internamente, **não estabelece
cota superior alguma**: o resultado enuncia algo a respeito do treino realizado,
e não a respeito dos pixels.

A segunda é a do retreino com o alvo correto. O modelo substituto original fora
treinado com entropia cruzada sobre rótulos duros, ou seja, otimizando acurácia
por nó, quando a própria tese já havia estabelecido que acurácia por nó é mau
substituto do compromisso entre taxa BD e tempo. O retreino contra a perda de
otimalidade, com peso `1 + α·regret_rel` e `α = 3`, mantendo arquitetura,
largura de fusão e
todo o caminho a jusante inalterados, **piorou o modelo em toda a faixa**, por
fatores de 1,06× a 3,80×, com a piora **maior justamente na região conservadora**
que um podador implantável ocuparia. A refutação é forte porque objetivo de
treino e métrica de avaliação estavam alinhados, ou seja, era o caso mais
favorável possível à hipótese.

A terceira é a da capacidade. Dobrar a largura de fusão de 128 para 256 altera a
perda ponderada de validação de 0,9250 para 0,9235, isto é, **0,16%**, o que
indica que a restrição não é de capacidade de modelo. Cabe ainda registrar a
correção de um erro de registro anterior: alegou-se que o ponto de controle do
modelo substituto original teria sido colhido em sobreajuste, e a alegação é
falsa, pois nas trinta épocas completas o mínimo da perda de validação (1,7999) e
o máximo do macro-F1 (0,2034) ocorrem ambos na época 13, que é a do ponto de
controle salvo. O modelo original está corretamente selecionado; ele é apenas
fraco em termos absolutos.

Deste modo, o registro honesto do que a tese possui é o seguinte: uma **cota
inferior** do domínio de pixels, dada pelo melhor desempenho observado, que é o
do `pixels24`, e uma **cota superior genuína apenas no oráculo**, ou seja, na
decisão de taxa-distorção ótima de perda de otimalidade nula, que limita qualquer podador e
não apenas os de pixels. A cota superior do domínio de pixels permanece **não
medida**, e o modelo substituto convolucional permanece no arcabouço como
instrumento de diagnóstico e como a tentativa documentada de estabelecê-la.

> **Procedência.** `docs/RESULTADOS_convnext_regret.md` §1 (método do retreino e
> remoção do confundidor de capacidade), §2 (piora de 1,06× a 3,80×), §2.2
> (derrota para o `pixels24`) e §4 (correção do registro sobre a seleção do ponto
> de controle); `docs/ANDAMENTO_tese.md` §1.3 (cota inferior no `pixels24`, cota
> superior genuína apenas no oráculo);
> `docs/SINTESE_resultados_metodologia.md` §3 (refino final de 2026-07-26).
> Artefatos: `results/models/surrogate_regret/`,
> `results/models/surrogate_real/metrics.csv` e
> `results/models/oracle_regret_convnext/`. Scripts:
> `src/scripts/partition_model/train_surrogate_regret.py` e
> `src/scripts/partition_model/oracle_regret.py::score_surrogate`. A correção de
> registro incide sobre o commit `0122b53`.

---

## 1.6 A composição auditada — o que a hierarquia de fato enuncia

A leitura da hierarquia da Seção 1.4 exige uma auditoria de composição, pois os
dois conjuntos de atributos comparados **não são disjuntos**. O layout do vetor
está rotulado no próprio código de extração: os índices 0 a 23 constituem o bloco
A, de descritores de luminância; os índices 24 a 31 constituem o bloco B, de
contexto de particionamento da vizinhança; e os índices 32 a 35 constituem o
bloco C, de quantização e posição. Como o H9a é definido como a união dos blocos
A, B e C, segue que **vinte e quatro dos seus trinta e seis atributos são
descritores de luminância** — variância global e por quadrante, gradientes
horizontal e vertical, orientação, densidade de bordas e contraste com o bloco
pai e com os irmãos — e que **o H9a não contém nenhuma grandeza de custo de
taxa-distorção**, pois estas constituem o bloco E, exclusivo do H9c. Mais ainda,
`pixels24` é definido como os índices de 0 a 23, ou seja, é **literalmente o
bloco A do H9a**.

A consequência é que o enunciado corrente "o contexto de taxa-distorção vence os
pixels" **é falso**, e seria derrubado por qualquer avaliador que abrisse o
arquivo de extração de atributos. O enunciado correto, e verificável, é duplo:
no domínio de pixels, **descritores manuais compactos de luminância vencem uma
rede convolucional profunda sobre pixels crus**, por cerca de 1,7×; e **doze
atributos causais de vizinhança, quantização e posição, de custo praticamente
nulo por já estarem residentes na memória do codificador, acrescentam 3,4×
sobre eles**. A solução implantada é, portanto, majoritariamente um modelo de
pixels ao qual foi acrescentado contexto causal barato.

Este enunciado é mais forte que o anterior, e não mais fraco. Ele converte um
resultado negativo sobre redes convolucionais em um resultado positivo sobre
**onde a informação está**: na vizinhança causal já codificada, e não na
capacidade de representação sobre o bloco-fonte. A mesma leitura é corroborada
pelo critério de decisão offline da fase seguinte, no qual a ação isolada de
comprometimento com o `PARTITION_NONE` rende de 10% a 19% de redução de custo com
o `pixels24` e de 16% a 25% com o H9a, ou seja, um ganho **marginal sobre** os
vinte e quatro descritores de luminância, e não um ganho **competitivo contra**
eles. Estes intervalos são lidos de uma varredura de limiares pela regra de
risco casado, ou seja, para cada subconjunto toma-se a linha cujo `split_lost`
fica imediatamente abaixo de cada patamar de risco declarado.

> **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §2, §2.1 e §2.2
> (layout do vetor, `pixels24` como bloco A e releitura da hierarquia) e §7
> (correções de registro aplicadas em quatro documentos);
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
**especificada e nunca testada**. O bloco D fora especificado como o SATD do
resíduo de uma predição intra barata construída a partir dos vizinhos
reconstruídos, com a advertência explícita de que um resíduo de predição
constante teria a mesma variância do bloco e, portanto, nada acrescentaria. A
implementação, porém, calcula o SATD do **bloco-fonte**, sem predição alguma e
sem tocar em vizinho algum. Deste modo, a configuração reprovada no critério de
decisão da fase 2 — com 46,9%, 50,5%, 53,4% e 57,8% de redução de custo contra
47,0%, 49,7%, 52,6% e 57,3% do H9a, ou seja, nula — testou uma estatística
derivada apenas da fonte, fortemente correlacionada com a variância e os
gradientes que o bloco A já contém, e não informa sobre a hipótese formulada.

A hipótese especificada foi, então, testada como bloco D', com três colunas —
disponibilidade do vizinho, logaritmo do SATD do resíduo e ganho de SATD —
escolhendo a melhor entre as predições DC, vertical, horizontal e PAETH por soma
de diferenças absolutas, e descartando o termo DC nas duas pontas para que a
comparação se faça entre energias de corrente alternada. A bancada é idêntica à
do critério de decisão da fase 2, variando **apenas o conjunto de colunas**, de
modo que o veredito incida sobre informação e não sobre capacidade. A execução
cobriu 13.000 superblocos de treino e 3.000 de validação, com três sementes por
braço.

**O critério de decisão não é atingido.** O critério, fixado antes da medição,
exigia entropia cruzada menor **e** área sob a curva maior nos níveis de 16 e de
32 amostras simultaneamente. Em 32 amostras há um positivo consistente, com
entropia cruzada de 0,7422 contra 0,7657 e área sob a curva de 0,868 contra
0,851, mas apenas 1.869 nós dispõem do atributo. Em **16 amostras, o nível com
15.855 nós, ou seja, com oito vezes mais dados, o efeito é nulo**: a entropia
cruzada passa de 0,5149 para 0,5144 e a área sob a curva recua de 0,847 para
0,846. A leitura de redução de custo a risco casado é igual ou pior em todos os
limites, com 6,83%, 6,83%, 14,74% e 21,30% contra 7,36%, 7,36%, 15,37% e 21,48%
do H9a.

A objeção de diluição, segundo a qual o nulo em 16 amostras decorreria de treinar
com nós sem vizinho disponível, **é contrariada pelo padrão observado**, pois a
disponibilidade é menor em 32 amostras (17,6%) do que em 16 amostras (45,9%), e é
em 32 amostras que o efeito aparece. Como o atributo custaria uma predição intra e
uma transformada de Hadamard por nó no codificador, um ganho nulo no nível mais
numeroso já o desqualifica. Fica registrado, sem ser perseguido, um resíduo
delimitado: o positivo em 32 amostras e o fato de que o nível de 64 amostras é
**invisível a este crivo**, pois ali a disponibilidade offline do atributo é de
0,0%.

Com este veredito, a família fecha com **cinco tentativas independentes
negativas**: o modelo substituto convolucional com entropia cruzada, o modelo
substituto convolucional com alvo de perda de otimalidade, a decisão estruturada por rede de
grafos do Approach B, o bloco D sobre o bloco-fonte e o bloco D' de
predizibilidade intra a partir dos vizinhos.

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

O domínio de pixels está, então, caracterizado por medição, e não por conjectura.
A era de pixels entregou uma curva de operação real de 1,033× a 1,285× de
aceleração ao custo de 0,249% a 1,617% de taxa BD, e a reprodução das decisões do
modelo substituto convolucional no codificador mostrou que a cota superior
pretendida não converte a sua vantagem de informação em tempo, com aceleração de
apenas 1,02× a 1,03×. A cadeia exploratória anterior a esta curva foi
integralmente remedida em razão do defeito de luminância nula, e as suas
conclusões não podem ser invocadas. Sob política casada, um limiar de variância
trivial supera o modelo estudante de pixels nos cinco níveis de aceleração
medidos de um experimento de uma sequência e duas imagens. E o modelo substituto
convolucional, longe de estabelecer a cota superior do domínio, perde para um
perceptrone sobre vinte e quatro atributos extraídos da mesma luminância.

O conjunto destes negativos não conduz a uma conclusão de impossibilidade, e sim
a uma reorientação precisa da pergunta. A decisão de particionamento é, por
definição, uma decisão de taxa-distorção, e a hierarquia medida indica que o que
separa o melhor podador dos demais não é capacidade de representação sobre o
bloco-fonte, e sim **doze atributos causais de vizinhança, quantização e posição,
de custo praticamente nulo**, que a própria busca nativa já consulta e que a
memória do codificador já contém no momento da decisão. É esta constatação, e não
uma preferência de arquitetura, que motiva e justifica a virada investigada nas
seções seguintes.

A solução que materializa esta virada — o H9a, um podador pré-busca por contexto
de taxa-distorção barato, cujo vetor de trinta e seis atributos contém, ele
próprio, os vinte e quatro descritores de luminância aqui caracterizados — é
apresentada na próxima seção, com a sua curva de operação no conjunto de teste
reservado, a sua ablação de atribuição e os seus resultados sob protocolo das
condições comuns de teste.

> **Procedência.** Consolidação das notas das Seções 1.1 a 1.7; nenhum valor novo
> é introduzido nesta síntese.
