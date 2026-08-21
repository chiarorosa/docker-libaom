# 4. H9d — poda seletiva das partições estendidas

Esta seção apresenta a segunda solução positiva implantada nesta tese, denominada
H9d, um podador aprendido que decide, nó a nó, se vale avaliar as partições
estendidas do AV1 — as quatro formas assimétricas do tipo AB e as duas formas
4-way de proporção 4:1. São apresentados, nesta ordem, o alvo que a solução
persegue e a razão de nenhum podador anterior o ter visado, o critério de
predizibilidade offline que autorizou o custo de integração em C, a cota superior
obtida pelo desligamento total das partições estendidas, a evolução da moldura de
comparação que tornou o resultado interpretável, a varredura de limiares no
codificador, o resultado sob protocolo CTC (do inglês *Common Test Conditions*),
a fronteira bidimensional de configurações e a verificação de integridade que
sustenta a validade de todo o marginal medido.

A moldura experimental precisa ser fixada antes de qualquer número, uma vez que é
ela que define o significado de tudo o que se segue. Ela é, em si mesma, uma
contribuição metodológica desta tese.

O H9d é um **complemento** do H9a, e não um substituto nem um concorrente do
podador nativo. O H9a decide primeiro, na chamada de poda pré-busca, antes que
qualquer avaliação de taxa-distorção seja paga, e um podador de partições
estendidas só pode agir sobre o resíduo de nós que o H9a não comprometeu com
`PARTITION_NONE`.

A única moldura válida combina, por conseguinte, duas medidas: a **contribuição
marginal medida sobre o H9a**, com o mesmo binário e os dois ambientes
empilhados; e o **teste de não-dominância de Pareto contra a curva de limiares do
próprio H9a**, no mesmo arranjo e contra a mesma âncora.

Comparar o H9d contra o codificador nativo dupla-conta o tempo que o H9a já colhe
e superestima o ganho. Compará-lo como substituição direta do H9a o faz parecer
dominado. Ambas as leituras foram percorridas nesta investigação, e a Seção 4.4
relata o percurso.

---

## 4.1 O alvo — um terço do custo de busca que nenhum podador visava

O alvo do H9d foi escolhido por medição, e não por analogia com as soluções
anteriores. A decomposição do custo de busca por família de candidatos,
apresentada na Seção 1.3 do Capítulo de Metodologia, mediu que as partições
estendidas — as quatro formas AB somadas às duas 4-way — consomem **34,3% do
tempo de busca local** do nó, com mínimo de 28,9% e máximo de 41,3% entre as três
sequências do conjunto de teste reservado. Este é um bloco de custo comparável ao
do `PARTITION_NONE` (30,1%) e ao das duas formas retangulares (35,6%), e é,
portanto, grande o bastante para justificar uma solução própria.

Este custo passou despercebido em todas as análises anteriores por uma razão
estrutural. Ele é grande em agregado, mas diluído entre seis candidatos
individualmente modestos: **nenhuma forma estendida isolada ultrapassa 7,22%** do
tempo de busca local.

O custo concentra-se, ademais, nos blocos grandes, o que restringe o alcance útil
do podador e, ao mesmo tempo, explica onde ele deve agir. Nos blocos de 8×8
amostras estas formas não se aplicam e o custo é nulo. Em 16×16 amostras
representam 8,7% do tempo local. Em 32×32 e 64×64 amostras representam,
respectivamente, **50,0%** e **51,0%**. Em 128×128 amostras, 35,3%.

Podador anterior algum visava este conjunto de candidatos, e é esta propriedade
que torna o H9d uma solução distinta, e não um quarto degrau de uma escada de
sofisticação.

O H9a age na chamada de poda pré-busca, e a sua ação é comprometer o nó com o
`PARTITION_NONE` ou restringir a busca à divisão quadrada. O H9c age na chamada de
poda pós-NONE, e a sua ação é encerrar a busca de forma binária.

O H9d partilha com o H9c exatamente o mesmo ponto de inserção
(`av1_prune_after_none`) e exatamente o mesmo vetor de trinta e nove atributos. A
pergunta que formula, contudo, é outra: não *"posso encerrar a busca aqui?"*, e
sim *"vale avaliar as partições estendidas?"*. Deste modo, `PARTITION_NONE`, as
formas retangulares e a divisão quadrada permanecem intocados.

> **Procedência.** `docs/RESULTADOS_C1_custo_por_candidato.md` §3 (decomposição
> agregada, por sequência e por tamanho de bloco);
> `docs/SINTESE_resultados_metodologia.md` §5-quater (alvo e distinção em relação
> ao H9a e ao H9c) e §2.8 (espaço de projeto em duas dimensões);
> `docs/INVENTARIO_solucoes.md` §5. Artefato:
> `results/benchmark/partstats*/part_timing*.csv` (não versionado). Script:
> `src/scripts/benchmark/analyze_partstats.py`.

---

## 4.2 O critério de predizibilidade offline

A viabilidade do H9d foi decidida fora do codificador, antes que qualquer custo
de integração em C fosse pago, conforme a cascata de critérios de decisão
descrita na Seção 3 do Capítulo de Metodologia. O rótulo binário foi definido por
nó como positivo quando a partição ótima pertence ao conjunto estendido — as
formas `HORZ_A`, `HORZ_B`, `VERT_A`, `VERT_B`, `HORZ_4` e `VERT_4` — e negativo
em qualquer outro caso, o que corresponde exatamente à pergunta que o podador
precisa responder. Foram avaliados **792.840 nós de decisão** do conjunto
reservado, formado pelas seis sequências fora do treino, ou seja, as três de
validação (HoneyBee, FlowerPan e Lips) e as três de teste (Jockey, RaceNight e
RiverBank), com um perceptrone de múltiplas camadas por nível de bloco, na mesma
receita já utilizada para o H9a.

A separabilidade obtida é forte e homogênea entre os níveis. Utilizando os trinta
e seis atributos pré-busca do H9a, a área sob a curva característica de operação
do receptor (do inglês *receiver operating characteristic* – ROC) foi de **0,890**
no agregado, com área sob a curva de precisão e revocação de 0,425 sobre uma base
de positivos de apenas 10,2%.

Por nível, o de 16 amostras é o mais separável, com 0,906 sobre 572.213 nós e base
de 9,9%. O de 32 amostras é o menos separável, com 0,817 sobre 172.627 nós e base
de 13,1%. O de 64 amostras atinge 0,864 sobre 48.000 nós, com base de apenas 3,0%.

O refinamento que seguiu para o codificador incorpora o contexto de
taxa-distorção do `PARTITION_NONE`, disponível sem custo adicional no ponto de
inserção pós-NONE, o que eleva o vetor para trinta e nove atributos.

Este refinamento eleva a área sob a curva característica agregada de 0,890 para
**0,902**, com ganho consistente nos níveis de 16 e de 32 amostras (0,919 e 0,829)
e estabilidade no de 64 amostras (0,865).

Em termos da troca que interessa ao podador, o modelo de trinta e nove atributos
evita **69,7%** das buscas estendidas perdendo 10% dos vencedores verdadeiros; ou
evita 50% das buscas perdendo apenas **1,1%** dos vencedores. Isso caracteriza uma
margem confortável e autorizou o avanço para a integração em C.

> **Procedência.** `docs/RESULTADOS_H9d_predizibilidade.md` §2 e §2.1;
> `docs/SINTESE_resultados_metodologia.md` §5-quater;
> `docs/INVENTARIO_solucoes.md` §5.1. Modelo:
> `results/models/h9d_predictability/students.pt`. Script:
> `src/scripts/partition_model/h9d_predictability.py` (com `--feature-set h9c`
> para a variante de trinta e nove atributos). *Limitação registrada na fonte:* a
> fração de busca evitada é computada sobre todos os nós, e não apenas sobre os
> que alcançam o critério de decisão das partições estendidas, o que superestima
> ligeiramente a economia implantável; a separabilidade é o sinal robusto.

---

## 4.3 A cota superior do desligamento total

A cota superior do H9d foi obtida desligando por completo as partições estendidas
em codificações reais, o que delimita simultaneamente as duas fronteiras do
problema.

O desligamento total representa o extremo do eixo. Ele entrega a maior aceleração
possível, uma vez que forma estendida alguma é avaliada, e o pior custo possível
de eficiência de compressão, pois as formas estendidas são descartadas mesmo
quando eram ótimas. Qualquer podador seletivo fica, por construção, dentro deste
envelope.

O experimento utilizou um único binário com o desligamento condicionado a
variável de ambiente e inerte por padrão, o que garante que o braço de referência
é o codificador real. Mediu as três sequências de teste com cinco quadros e quatro
pontos de quantização.

O envelope medido é de **+0,89% de taxa BD por uma aceleração de 1,431×**, com
+1,13% e 1,485× na Jockey, +0,76% e 1,418× na RaceNight e +0,79% e 1,402× na
RiverBank.

A leitura decisiva não está nos valores isolados, e sim na razão entre eles. As
partições estendidas custam cerca de **0,03 ponto percentual de taxa BD por cada
1% de tempo economizado**. São, portanto, candidatos caros e raramente decisivos —
exatamente o perfil que favorece um podador seletivo.

Esta medição de tempo de parede cruza-se, ademais, com a decomposição por
temporizador da Seção 4.1, que previa aceleração próxima de 1,43× a partir dos
34,3% de tempo local. As duas medições independentes concordam, o que valida o
instrumento.

A cota superior foi medida também na moldura marginal, empilhando o desligamento
total sobre o H9a no ponto de referência implantado. Este é o número que delimita
o espaço realmente disponível para a solução.

Mesmo depois de o H9a já ter comprometido parte dos nós com o `PARTITION_NONE`,
restam **1,293× de aceleração** presos nas partições estendidas do resíduo, ao
custo de +0,798% de taxa BD, com 1,367× na Jockey, 1,239× na RaceNight e 1,290×
na RiverBank. O conjunto de candidatos que o H9d ataca é, deste modo, grande
depois do H9a, e não um resto de ruído.

> **Procedência.** `docs/RESULTADOS_H9d_cota_superior.md` §3 (envelope contra o
> codificador nativo) e §3.7 (envelope marginal sobre o H9a);
> `docs/INVENTARIO_solucoes.md` §5.2. Artefatos:
> `results/benchmark/{h9d_ub,h9d_marg}/raw.csv` (não versionados). Scripts:
> `src/scripts/benchmark/h9d_upper_bound.py` (com `--stack h9a` para a medição
> marginal), `src/scripts/benchmark/bd_rate.py`. *Limitação registrada na fonte:*
> cinco quadros por ponto, comparações internamente consistentes contra a mesma
> âncora, cujos valores absolutos não devem ser cruzados com os das campanhas de
> quinze quadros.

---

## 4.4 A evolução da moldura de comparação

Esta subseção apresenta o percurso metodológico que levou da moldura atrativa,
passando pela moldura precipitada, até a moldura válida, pois este percurso é,
em si, um resultado desta tese e organiza a leitura de todas as composições entre
alavancas apresentadas no Capítulo de Resultados. As três molduras utilizam
exatamente os mesmos dados; o que muda é a pergunta que cada uma responde, e a
diferença entre as respostas é de sinal, não de grau.

A primeira moldura compara o desligamento total das partições estendidas
diretamente contra o codificador nativo, e produz 1,431× de aceleração por +0,89%
de taxa BD. Este resultado é atrativo à primeira vista, e é exatamente por isso
que é enganoso: ele contabiliza, como ganho do H9d, todo o tempo que o H9a já
colhe sozinho no mesmo codificador implantado, uma vez que os nós comprometidos
com o `PARTITION_NONE` pelo H9a jamais alcançariam a avaliação das partições
estendidas. A comparação contra o nativo pristino dupla-conta, e, deste modo,
superestima a contribuição.

A segunda moldura corrige a primeira pelo lado errado, tratando o H9d como
substituição direta do H9a, e conduz a uma conclusão precipitada de dominância.

Colocando os pontos de operação do H9a lado a lado com a cota superior do H9d
isolado, todos medidos contra a mesma âncora nativa, existe em cada sequência um
ponto do H9a que domina a cota superior do H9d nos dois eixos. Na Jockey, 1,503× a
0,93% contra 1,485× a 1,13%. Na RaceNight, 1,484× a 0,729% contra 1,418× a 0,76%.
Na RiverBank, 1,434× a 0,226% contra 1,402× a 0,79%.

A leitura imediata foi de alavanca dominada e, portanto, rejeitada. Esta leitura
supõe, contudo, que as duas soluções disputam o mesmo lugar no fluxo de controle,
o que é falso.

A terceira moldura é a válida, e decorre diretamente da ordem de execução dentro
de `av1_rd_pick_partition`. A chamada de poda pré-busca do H9a executa antes da
avaliação do `PARTITION_NONE`, que por sua vez precede as formas retangulares e,
por fim, as formas estendidas. O H9d só age no resíduo.

A medição correta é, por conseguinte, marginal — H9a mais H9d contra H9a, mesmo
binário —, e o teste de mérito é a não-dominância de Pareto contra a curva de
limiares do próprio H9a, que é o botão de velocidade gratuito de que a solução já
dispõe.

Nesta moldura, o ponto empilhado **não é dominado por ponto algum da curva de
limiares em nenhuma das três sequências**. Na Jockey, 2,19% a 2,00×, contra 1,79×
do ponto A2 e 2,47% do ponto A3. Na RaceNight, 1,21% a 1,86×, contra 1,70× do A2
e 1,96% do A3. Na RiverBank, 0,91% a 1,66× — aceleração que ponto algum da curva
de limiares alcança.

O resultado é uma inversão de veredito produzida exclusivamente pela troca da
moldura. Cabe destacar que este ponto ainda é o do desligamento total, ou seja, o
pior caso do H9d, o que faz da não-dominância uma cota inferior do valor da
solução seletiva.

> **Procedência.** `docs/RESULTADOS_H9d_cota_superior.md`, trilha de raciocínio
> do cabeçalho, §3.6 (moldura de substituição, preservada como passo
> intermediário) e §3.7 (moldura marginal e teste de Pareto); referências de
> fluxo de controle em `src/aom/av1/encoder/partition_search.c` (poda pré-busca,
> `PARTITION_NONE`, retangulares e estendidas, nesta ordem);
> `results/thesis/M1_objeto_e_formulacao.md` §1.4. Artefatos:
> `results/benchmark/{h9d_marg,h9d_tau}/raw.csv` (não versionados). Script:
> `src/scripts/benchmark/h9d_tau_curve.py`.

---

## 4.5 A varredura de limiares no codificador e o limiar por nível

A política seletiva foi integrada em C no mesmo ponto de inserção pós-NONE do
H9c. Ela computa os trinta e nove atributos, executa a inferência e marca o nó
para pular as partições estendidas quando a probabilidade estimada fica abaixo do
limiar.

A integração reutiliza o mecanismo de desligamento já validado na cota superior. A
exportação dos pesos foi verificada por comparação de ida e volta: sobre 192
vetores aleatórios, a diferença máxima entre a saída da implementação de treino e
a do arranjo gravado no codificador foi de **1,35×10⁻⁷**, ou seja, exata até o
erro de ponto flutuante. Toda a superfície de controle é acionada por variável de
ambiente e permanece desligada por padrão.

A varredura de limiares foi executada sobre o ponto de referência do H9a nas três
sequências de teste, com o limiar global assumindo os valores 0,05, 0,10, 0,20,
0,30 e 0,45. Dois resultados foram obtidos.

O primeiro é que o **podador seletivo domina o próprio desligamento total nas três
sequências**, o que confirma a projeção do critério de predizibilidade. Na
RiverBank, o desligamento total custa 0,91% a 1,66×, ao passo que o seletivo custa
0,64% a 1,55×. Na RaceNight, 1,21% a 1,86× contra 1,08% a 1,80×. Na Jockey, 2,19%
a 2,00× contra 2,02% a 1,87×.

O segundo é que, contra a curva de limiares do H9a de mesma aceleração, o seletivo
vence em duas das três sequências — de forma clara na RaceNight e modesta na
Jockey —, perdendo levemente na RiverBank, que é justamente o conteúdo em que as
partições estendidas quase não são escolhidas.

A perda na RiverBank foi diagnosticada e corrigida por um refinamento de projeto.
Os limiares que descartam a mesma fração de vencedores diferem muito entre níveis:
o critério de predizibilidade da Seção 4.2 mede, à perda de cerca de 10% dos
vencedores, limiares de 0,091 no nível de 16 amostras, 0,103 no de 32 e apenas
0,014 no de 64. Um limiar global de 0,30 é, deste modo, agressivo demais no nível
de 32 amostras — que é onde as formas estendidas mais vencem e, portanto, onde
mora o custo de eficiência.

A calibração por nível denominada **PL10** recupera a RiverBank, passando de +0,12
para +0,05 ponto percentual em relação à curva de limiares, ou seja, de perda
clara a empate aproximado. Ela não abre mão do ganho na RaceNight, que se mantém
em −0,30 ponto percentual, e deixa a Jockey em −0,02.

As variantes mais agressivas PL20 e PLmix trocam esta robustez por mais tempo
economizado, e não foram escolhidas. A configuração PL10 foi, por isso, gravada
como padrão no codificador: é o único ponto de operação que **nunca perde** para a
curva de limiares em nenhuma das três sequências de teste.

> **Procedência.** `docs/RESULTADOS_H9d_etapa2_C.md` §1 a §4 (integração,
> comparação de ida e volta, superfície de controle e validação de inércia);
> `docs/RESULTADOS_H9d_etapa3_encoder.md` §2 a §4.2 (varredura de limiares,
> confirmação a dez quadros e calibração por nível);
> `docs/INVENTARIO_solucoes.md` §5.3 e §5.4. Artefatos:
> `results/benchmark/h9d_selective/raw.csv` (não versionado). Scripts:
> `src/scripts/partition_model/h9d_export_weights.py`,
> `src/scripts/benchmark/h9d_selective_sweep.py`. *Divergência de registro
> anotada:* `RESULTADOS_H9d_etapa3_encoder.md` §4.2 e §5 registram a calibração
> por nível como medida a cinco quadros, enquanto `INVENTARIO_solucoes.md` §5.4
> a rotula como de dez quadros; a confirmação a dez quadros documentada na fonte
> primária refere-se ao limiar global de 0,30, e adota-se aqui o registro da
> fonte primária. `[completar: confirmação a dez quadros da configuração PL10 nas
> três sequências de teste, apontada na fonte como trabalho futuro barato]`.

---

## 4.6 O resultado sob protocolo CTC

O resultado que entra neste capítulo foi medido nas oito sequências da Classe A1
das condições comuns de teste, em 4K e 10 bits, com quinze quadros, quatro pontos
de quantização e `cpu-used=0`, contra a mesma âncora nativa utilizada para o H9a.

A configuração medida mantém fixa a política do H9a no ponto balanceado implantado
e empilha o H9d na calibração PL10. A diferença contra as codificações já medidas
do H9a é, deste modo, a contribuição marginal pura do H9d. Em valores absolutos, o
H9a balanceado entrega +0,568% de taxa BD a 17,72% de redução de tempo (1,223×), e
o par empilhado entrega **+0,586% a 18,74%** (1,238×).

A contribuição marginal do H9d sobre o H9a é, portanto, de **+1,02 ponto
percentual de redução de tempo ao custo de +0,018 ponto percentual de taxa BD**.
Isso corresponde a um preço de **0,018 ponto percentual de taxa BD por ponto
percentual de tempo economizado**.

Este preço é o número central da seção, uma vez que é diretamente comparável ao do
único mecanismo alternativo disponível ao usuário da solução implantada: afrouxar
o limiar do próprio H9a. O segmento entre o ponto balanceado e o ponto agressivo
desse botão custa **0,063 ponto percentual por ponto percentual**. O H9d compra
tempo, por conseguinte, por cerca de **um terço** do que custaria comprá-lo com o
botão de limiar — aproximadamente **3,5 vezes mais barato**.

O teste de não-dominância por sequência confirma o agregado e revela onde a
alavanca funciona. Contra o ponto de mesma redução de tempo na curva de limiares,
o H9d é melhor em **seis das oito sequências**, com vantagem média de −0,043 ponto
percentual. Perde levemente apenas na Crosswalk (+0,018) e na Neon1224 (+0,041) —
as duas sequências em que o ganho de tempo é quase nulo, de 0,4 e 0,1 ponto
percentual, ou seja, aquelas em que o eixo estendido praticamente não é exercido e
o modelo quase não tem o que podar.

**Duas sequências exibem, ademais, dominância de Pareto estrita.** Na FoodMarket2,
a taxa BD cai de 0,63% para 0,61% com +1,8 ponto percentual de tempo economizado.
Na Tango, cai de 1,15% para 1,14% com +1,6 ponto percentual.

Cabe destacar que os valores de taxa BD são exatos, e não estimativas ruidosas: o
número de bytes e o PSNR-Y são determinísticos para um dado codificador e uma dada
entrada. O único componente com variância entre execuções é o tempo de parede.

> **Procedência.** `docs/RESULTADOS_H9d_CTC.md` §2, §3, §3.1 e §3.2;
> `docs/SINTESE_resultados_metodologia.md` §5-quater;
> `docs/INVENTARIO_solucoes.md` §5.5. Artefatos:
> `results/benchmark/fase6/raw_results.csv` (configuração `ml_bal_h9d`, 32
> linhas), `bdrate_average.csv`, `bdrate_per_seq.csv` e `tables.tex` (não
> versionados). Scripts: `src/scripts/fase6/ctc_h9d.py`,
> `src/scripts/fase6/report_ctc.py`, `src/scripts/fase6/ctc_h9d_marginal.py`.
> *Nota sobre estimadores:* o preço do botão de limiar vale 0,063 ponto
> percentual por ponto percentual pelo estimador de interpolação por sequência e
> 0,0606 pela média das médias sobre as oito sequências, de onde a razão contra o
> H9d sai como 3,5× ou 3,38×; os dois estimadores concordam a cerca de 4% e não
> devem ser misturados na mesma tabela.

---

## 4.7 A fronteira bidimensional e a dependência do ponto de operação

A limitação de um único ponto de operação foi fechada por uma campanha própria,
que mediu a família completa de configurações do H9d em duas dimensões: duas
bases do H9a, o ponto balanceado e o ponto agressivo, cruzadas com duas forças do
H9d, as calibrações PL10 e PL20. Três das quatro combinações — a base balanceada
com PL20 e a base agressiva com PL10 e com PL20 — foram medidas nesta campanha,
totalizando **96 codificações** novas sob o mesmo protocolo CTC; a quarta
combinação, a calibração PL10 sobre a base balanceada, é o par implantado e já
havia sido medida na campanha CTC anterior. As quatro configurações juntas
somam 128 linhas em `results/benchmark/fase6/raw_results.csv`. Os quatro
pontos batem o botão de limiar, com razões de custo
entre 1,52× e 3,38×, e o par implantado — a calibração PL10 sobre a base
balanceada — é o **melhor dos quatro**, a 0,0179 ponto percentual por ponto
percentual, contra 0,0399 da PL20 sobre a base balanceada, 0,0329 da PL10 sobre a
base agressiva e 0,0258 da PL20 sobre a base agressiva. A campanha confirma,
portanto, a escolha de projeto feita antes de ela existir.

A fronteira expõe, contudo, um achado que o ponto único não permitia ver, e que
altera o enunciado teórico da tese: **o valor marginal do H9d desaba conforme a
base do H9a fica agressiva**.

Sobre a base balanceada, o ganho é de +1,02 ponto percentual de redução de tempo.
Sobre a base agressiva, ele cai para **+0,17 ponto percentual** — valor que fica
abaixo da resolução temporal medida do arranjo, de aproximadamente 0,46 ponto
percentual, e que, por isso, não deve ser citado como positivo.

A verificação por sequência é ainda mais nítida. O ganho supera a resolução
temporal em **seis das oito** sequências sobre a base balanceada, e em apenas
**uma das oito** sobre a base agressiva, com Neon1224 e Tango chegando a valores
negativos. Sobre a base agressiva, o H9d é praticamente inerte.

O mecanismo que explica esta inércia é o mesmo que explicava a aditividade, lido
no outro sentido. Com o limiar do `PARTITION_NONE` afrouxado para 0,60, o H9a
compromete os nós tão cedo que eles nunca alcançam o critério de decisão das
partições estendidas. Os dois podadores passam, deste modo, a disputar o mesmo
resíduo, e o H9d volta a se comportar como o H9c.

A disjunção de ação que sustenta a aditividade **não é, por conseguinte,
propriedade absoluta da alavanca, e sim função do ponto de operação sobre o qual
ela age**. O ponto implantado é justamente aquele em que a disjunção é máxima, o
que valida a escolha — mas por uma razão mais estreita do que a simples afirmação
de que as ações são disjuntas.

> **Procedência.** `docs/ANDAMENTO_tese.md` §0.1 (fronteira bidimensional, 96
> codificações e achado da inércia sobre a base agressiva);
> `docs/SINTESE_resultados_metodologia.md` §5-quater (tabela dos quatro pontos,
> consequência para §2.8 e ressalvas); `docs/INVENTARIO_solucoes.md` §5.5;
> `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` (resolução temporal do arranjo, de
> aproximadamente 0,46 ponto percentual). Artefato:
> `results/benchmark/fase6/raw_results.csv` (não versionado).

---

## 4.8 A verificação de integridade

Toda a validade do marginal medido depende de uma condição verificada antes de
qualquer codificação de resultado: o binário que contém o H9d precisa reproduzir
exatamente a base do H9a quando o H9d está desligado.

Esta verificação foi executada por recodificação de um ponto de referência com o
binário novo e o H9d desligado. O fluxo de bits obtido é **byte a byte idêntico**
ao da base: 1.574.775 bytes e PSNR-Y de 40,9720 dB, contra os mesmos 1.574.775
bytes e 40,9720 dB da referência, com diferença de tempo de 0,2%, dentro do ruído
de medição.

A verificação foi repetida no segundo pré-ajuste de limiar, antes da campanha da
fronteira bidimensional, com o mesmo desfecho: 1.579.208 bytes e 40,9600 dB,
também idênticos à referência agressiva.

A consequência desta verificação é o que torna a seção defensável. O código do H9d
é comprovadamente inerte quando desligado. A base do H9a dentro do binário novo é,
deste modo, a mesma base já medida e publicada, e a diferença entre os dois braços
não pode ser atribuída a variação de compilação, a efeito colateral de
instrumentação ou a deriva de configuração.

Todo o marginal medido é, portanto, limpo. A inércia por padrão é a mesma garantia
de projeto adotada para o H9a e para o H9c, descrita na Seção 5 do Capítulo de
Metodologia.

> **Procedência.** `docs/RESULTADOS_H9d_CTC.md` §1.1 (verificação no pré-ajuste
> balanceado); `docs/INVENTARIO_solucoes.md` §5.5 (verificação nos dois
> pré-ajustes); `docs/ANDAMENTO_tese.md` §0.1. Script:
> `src/scripts/fase6/ctc_h9d.py --integrity`.

---

## 4.9 Síntese — o que o H9d estabelece

O H9d está, então, fechado como a segunda solução positiva implantada desta tese:
um podador aprendido, de trinta e nove atributos, inserido na chamada de poda
pós-NONE, que decide seletivamente se vale avaliar as partições estendidas e que
acrescenta **+1,02 ponto percentual de redução de tempo por +0,018 ponto
percentual de taxa BD** sobre o ponto implantado do H9a, nas oito sequências da
CTC, comprando tempo por cerca de um terço do preço do botão de limiar e vencendo
a curva de limiares em seis das oito sequências, duas delas por dominância de
Pareto estrita.

O valor deste resultado para a tese, contudo, não está no tamanho da aceleração,
que é modesto em valor absoluto. Está no que ele refuta.

O enunciado geral de que alavancas de poda não se somam foi construído sobre a
composição entre o H9a e o H9c, e atribuía a não-aditividade a um limite
informacional — à suposição de que dois podadores alimentados pela mesma
informação estariam condenados a colher o mesmo tempo.

O H9d refuta este enunciado por medição direta. Ele soma **+1,02 ponto
percentual** sobre o H9a utilizando **exatamente o mesmo vetor de atributos e
exatamente o mesmo ponto de inserção do H9c**, que somara apenas +0,26 ponto
percentual. Isso seria impossível se a informação partilhada fosse a causa do teto
de composição.

O enunciado correto que substitui o refutado é, por conseguinte, de outra
natureza: **dois podadores se somam na medida em que os seus conjuntos de
candidatos podados são disjuntos, independentemente de partilharem a mesma
informação de entrada** — e, pela evidência da Seção 4.7, na medida em que essa
disjunção se realiza no ponto de operação em que os dois efetivamente rodam.

O H9a e o H9c caçam ambos os blocos fáceis e disputam o mesmo tempo economizável.
O H9d caça um conjunto de candidatos disjunto, e por isso soma. Este enunciado é
prescritivo, uma vez que orienta a procura por ações não disputadas em vez da
procura por mais informação, e é uma das três conclusões consolidadas na Seção 6
deste capítulo.

Duas soluções positivas foram, deste modo, apresentadas. O estudo produziu também
cinco vias encerradas com resultado negativo no conjunto do trabalho — contagem
geral do projeto, e não as cinco vias específicas do domínio de pixels —,
que não sobreviveram aos seus critérios de decisão.

Estas vias não são descarte. São resultado, e carregam valor metodológico próprio,
uma vez que delimitam por medição o que a informação de pixels e as reformulações
do problema não conseguem entregar. Os resultados negativos são apresentados na
próxima seção.

> **Procedência.** Consolidação das notas das Seções 4.1 a 4.8;
> `docs/SINTESE_resultados_metodologia.md` §5-quater (leitura sobre a Conclusão 3
> e a correção do enunciado) e §6 (as três conclusões);
> `docs/ANDAMENTO_tese.md` §0.1 (correção da Conclusão 3: a não-aditividade é
> sobreposição de ação, não limite informacional). Nenhum valor novo é
> introduzido nesta síntese, à exceção do ganho marginal do H9c (+0,26 ponto
> percentual), cuja procedência é a Seção 3 deste capítulo.
