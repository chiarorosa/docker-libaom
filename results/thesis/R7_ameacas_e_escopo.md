# 7. Ameaças à validade, limitações e decisões de escopo

Esta seção encerra o Capítulo de Resultados com o exame das condições sob as
quais os números das seções anteriores devem ser lidos. São apresentadas,
nesta ordem, as ameaças à validade que atravessam a investigação, cada uma com
o mecanismo que a produz, a evidência de sua magnitude e a mitigação adotada;
as decisões de escopo, isto é, o que esta tese deliberadamente não faz e por
que razão; e, por fim, as limitações que permanecem em aberto, com o que as
fecharia e a que custo. Nenhum resultado das seções anteriores é retirado
aqui: o que se declara é o alcance exato de cada afirmação.

## 7.1 Ameaças à validade

Esta parte apresenta seis ameaças à validade dos resultados, cada uma com o
mecanismo que a produz, a evidência de sua magnitude e a mitigação adotada
pelo protocolo desta tese.

### 7.1.1 A simulação oráculo: superestimação e o risco de inversão da ordenação

O protocolo desta tese usa, em cascata, uma simulação sobre o conjunto de dados
anotado — a que este texto chama simulação oráculo — para decidir, a baixo
custo computacional, se um podador merece o investimento de integração em C e
de medição no codificador real. Esta simulação mede a redução de custo de
busca sob um risco de perda de partições ótimas casado entre modelos, e não o
tempo de parede. Historicamente, a margem que a simulação oráculo relata
superestima o ganho real por um fator da ordem de cinco vezes: no critério de
decisão da Fase 2, por exemplo, trinta e cinco por cento de redução de custo no
oráculo corresponderam a cerca de sete por cento de redução de tempo de parede
no codificador. Por isso, o protocolo trata toda margem offline como indício de
ordem de grandeza, e nunca como número final.

A ameaça mais forte, por outro lado, não é a superestimação da magnitude, e sim a
possibilidade de a simulação oráculo inverter a ordenação relativa entre
modelos. O Approach B mediu este fenômeno diretamente: uma rede de grafos
estruturada supera o H9a em vinte e oito pontos percentuais na simulação
oráculo, mas perde para o H9a por um fator de aproximadamente duas vezes em
taxa BD no codificador real, ao longo de toda a varredura de limiares testada.
A fronteira da rede de grafos no codificador é plana, o que indica que o
limite está na qualidade das decisões, e não na calibração dos limiares — a
acurácia por nó e o custo de oráculo são, deste modo, maus indicadores do
compromisso entre taxa BD e tempo real de um podador. A consequência
metodológica é direta: rejeição ou aprovação fora do codificador é indício, não
prova de ordenação, e por isso todo veredito final desta tese passa pelo
codificador. É também por isso que a decisão de não levar a regressão de
*regret* ao codificador foi reenquadrada como assimetria de custo
experimental, e não como implicação lógica de que o codificador confirmaria a
rejeição do oráculo.

### 7.1.2 Ruído de poucos quadros e a resolução do tempo medido

O tempo de parede é a única grandeza desta tese sujeita a ruído de medição; a
taxa BD é exata, pois deriva do número de bytes e do PSNR-Y, ambos
determinísticos para um dado codificador e uma dada entrada. A resolução do
tempo foi medida diretamente, e não suposta: cinco repetições intercaladas de
uma sequência sob três configurações, em execução contínua, deram um
coeficiente de variação mediano de 0,28% e máximo de 0,64%, o que fixa a
resolução da comparação pareada em aproximadamente 0,46 ponto percentual para a
configuração balanceada do H9a e 0,18 ponto percentual para o *preset* nativo
`cpu-used=1` — cerca de quatro vezes menor do que o piso de um a dois por
cento suposto antes da medição. Deste modo, o ganho agregado de +1,02 ponto
percentual de tempo do H9d fica a cerca de 4,4 desvios-padrão acima do ruído, e
é sólido; mas os ganhos por sequência de Neon1224 (+0,1 pp) e Crosswalk (+0,4
pp) ficam abaixo da resolução e não podem ser lidos como positivos
isoladamente.

Esta resolução, por outro lado, vale apenas dentro de uma mesma janela contínua de
execução: as cinco repetições correram na mesma sequência e no mesmo
contêiner sem reinício, e não capturam deriva entre dias, reinícios de
contêiner ou estados térmicos distintos da máquina. Números medidos com
semanas de intervalo, portanto, pedem cautela adicional, e nenhuma comparação
desta tese cruza campanhas distantes no tempo sem esta ressalva.

### 7.1.3 A grade de limiares congelada na validação

Os limiares operacionais de cada podador foram calibrados no conjunto de
validação e congelados antes de qualquer medição no conjunto de teste
reservado, por decisão de protocolo contra a seleção a posteriori. Esta
prática tem um custo: no conjunto de teste reservado, as faixas de aceleração
alcançadas pelo escore aprendido e pela variância trivial saíram disjuntas nas
três sequências, de modo que não houve, ali, um ponto de tempo casado entre os
dois para comparar diretamente a taxa BD. Estender a grade da variância depois
de ver o conjunto de teste resolveria a disjunção, mas violaria o próprio
congelamento que garante a ausência de ajuste sobre dado de teste — e é
exatamente esta violação que o protocolo existe para impedir.

A mitigação adotada foi dupla. Na validação, o experimento E5 estendeu
legitimamente a grade da variância até o extremo conservador, pois ali o papel
declarado do conjunto é escolher limiares, e não medir o resultado final; no
conjunto de teste, manteve-se a atribuição por política casada, que dispensa
sobreposição de velocidade — sob a mesma política, o escore aprendido alcança
taxa BD mínima de 0,008% na RiverBank contra 0,75% da variância, uma razão de
94 vezes, com 11 vezes na Jockey e 44 vezes na RaceNight.

### 7.1.4 A dependência de conteúdo entre sequências

A eficiência relativa de cada podador varia por sequência, e médias sobre o
conjunto podem mascarar esta variação. A FoodMarket2 é um caso identificado de
sequência atípica, favorável ao podador nativo além do padrão das demais
sequências da grade CTC. A Lips, por sua vez, expõe um comportamento quase
bimodal do escore de variância: entre os limiares 0,99 e 0,97, a aceleração
salta de 1,006× para 3,563× e a taxa BD, de 0,019% para 6,58%, sem ponto
intermediário na grade — provavelmente porque a Lips é um *close-up* de rosto,
dominado por regiões de pele lisas e de variância baixa e homogênea, sobre as
quais um limiar de um único momento estatístico se comporta de forma quase
binária.

Além disso, a decomposição que separa a contribuição isolada do H9a, do H9c e
da interação entre ambos cobre quatro das oito sequências da CTC, com
dispersão de 28% a 95% na fração de tempo atribuída ao H9c que de fato era do
H9a; e o único caso de interação positiva entre os dois, na TimeLapse, repousa
sobre um H9c isolado de apenas 0,6% de redução de tempo, número pequeno demais
para sustentar interpretação própria. A mitigação, em todos os casos, é
reportar a dispersão por sequência ao lado da média, e nunca substituir uma
pela outra.

### 7.1.5 A grade de quantização da tese e os índices do guia CTC

Duas divergências entre o comando de codificação desta tese e a especificação
da CTC §4.1 são impostas pela versão do libaom utilizada, e não escolhidas: o
binário `aomenc` da v3.10.0 não expõe o parâmetro `--qp`, de modo que a única
escala de quantização disponível é `--cq-level`, e a opção
`--use-fixed-qp-offsets=1` não existe neste *build*. Sobre estas duas, o
impacto é nulo, pois em modo *All-Intra* com `--kf-max-dist=0` todo quadro é
quadro-chave, e deslocamentos fixos de QP entre tipos de quadro não teriam
efeito algum.

Uma terceira divergência, esta sim uma escolha, é registrada como tal:
manteve-se a grade de quantização já usada na validação (`cq` 20, 32, 43 e 55)
em vez dos índices de quantização exatos do guia CTC, por consistência interna
com os experimentos anteriores desta tese. A grade é aplicada de forma
idêntica a todas as configurações comparadas, inclusive à âncora, o que
preserva a validade de toda comparação interna; a divergência afeta apenas a
comparabilidade direta com submissões externas que sigam os índices exatos do
guia.

### 7.1.6 Riscos que se materializaram e riscos absorvidos

Dois riscos previstos no protocolo se materializaram ao longo da investigação
e foram absorvidos sem alterar o resultado central. O primeiro é que o
critério de decisão do teste reservado, na forma estrita de dominância a
velocidade casada sobre a variância, não fosse atingido — e não foi, pelas
faixas disjuntas descritas na Seção 7.1.3. A investigação absorveu este risco
substituindo o critério estrito pela atribuição a política casada, que
sustenta a mesma conclusão por outro caminho e não depende de sobreposição. O
segundo risco é o custo computacional dos atributos de vizinhança e
quantização, que poderia consumir de volta o ganho de tempo; a medição direta
fechou este risco, pois o podador completo — extração e inferência somadas —
custa no máximo 0,32% do tempo de codificação, sem deslocar nenhum resultado
de taxa BD por tempo.

Um risco permanece vivo, e é retomado na Parte III desta seção: a atribuição do
podador implantado repousava inteiramente em evidência offline até o
experimento E5, que a confirmou no codificador sob política casada em duas
sequências de validação, mas sem atingir a forma estrita do critério
pré-registrado. O par de escores `pixels24` e variância, especificamente,
segue sem árbitro no codificador.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §9, fonte
> principal desta parte, e último item do mesmo parágrafo (grade de
> quantização); `docs/ANDAMENTO_tese.md` §3 (superestimação 35% → 7%), §4.1
> "FULL CONCLUÍDO" (razões 94×, 44× e 11×) e §6 (riscos materializados e
> vivos); `docs/RESULTADOS_approachB.md` §2, §5, §6 e §7 (inversão da
> ordenação); `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3 e §4 (resolução do
> tempo e sua validade intra-execução); `docs/RESULTADOS_BLOCO7_E1_E4.md` §2 e
> §2.1 (dispersão de 28% a 95%); `docs/RESULTADOS_E5_ablacao_validacao.md` §1,
> §3 e §4 (grade estendida, precipício da Lips, atribuição a política casada);
> `docs/DECISOES_escopo.md` §3 (divergências do libaom);
> `docs/RESULTADOS_solucao4.md` §7 (correção D2).

## 7.2 Decisões de escopo

Esta parte apresenta o que esta tese deliberadamente não faz, com a
justificativa registrada no momento em que cada recorte foi fechado. Nenhum
dos itens seguintes é pendência: cada um foi avaliado contra a evidência
disponível e encerrado por decisão, e não por falta de tempo ou de dado.

### 7.2.1 Métrica de qualidade: apenas PSNR-Y

Toda taxa BD desta tese é calculada sobre o PSNR-Y, sem PSNR-U, PSNR-V, SSIM,
MS-SSIM, VMAF, PSNR-HVS-M ou CIEDE2000. A justificativa é dupla: o PSNR-Y é a
convenção dominante na literatura de decisão de particionamento em
codificadores, área em que esta contribuição se situa, e é também o plano
sobre o qual a própria decisão de particionamento opera — o podador consome
atributos derivados da luminância, e a partição do croma é derivada desta
decisão, e não escolhida de forma independente. A CTC especifica um conjunto
métrico mais amplo; esta tese reporta um subconjunto dele, o que não invalida
nenhuma comparação interna, pois todas as configurações são medidas com a
mesma métrica contra a mesma âncora, mas limita a comparabilidade direta com
submissões externas que reportem a métrica combinada de luma e croma.

### 7.2.2 Quinze quadros por sequência: conformidade, não redução

Os quinze quadros usados em toda a validação universal na CTC são exatamente a
especificação da própria CTC para o modo *All-Intra*, e não um recorte próprio
desta tese: a especificação determina o uso dos primeiros quinze quadros para
as classes de vídeo pertinentes, e o registro de mudanças da versão sete da
CTC anota a alteração deliberada de trinta para quinze quadros nesta
configuração. Não há, portanto, déficit de quadros a defender; documentos
anteriores desta investigação descreveram os quinze quadros como redução e
foram corrigidos.

### 7.2.3 Divergências de comando impostas pela versão do libaom

Duas diferenças entre o comando de codificação e a especificação da CTC — o
uso de `--cq-level` no lugar de `--qp`, e a ausência da opção
`--use-fixed-qp-offsets=1` — são impostas pela versão do libaom empregada,
cuja interface é anterior à do AV2/AVM que a CTC v9 tem como alvo, e não
escolhidas por esta tese. A Seção 7.1.5 já registrou o impacto nulo destas
divergências sobre as conclusões.

### 7.2.4 τ adaptativo por qindex: calibrado offline, não levado ao codificador

A calibração de um limiar de comprometimento adaptativo ao índice de
quantização foi executada e classificada como positiva, porém imaterial: o
ganho a custo casado é pequeno demais para deslocar a fronteira de taxa BD por
tempo acima do piso de ruído medido do tempo de parede. A confirmação no
codificador não foi executada, pois gastar horas de codificação para
confirmar um efeito já sabido menor que a resolução do experimento não
produziria um número de tabela defensável.

### 7.2.5 A campanha `h9acomb`: hipótese testada o suficiente

A campanha que combinaria exaustivamente o H9a e o H9c em cerca de cento e
sessenta e oito codificações não foi completada, pois os dados parciais já
tornavam visível o resultado negativo que ela mediria — a interação negativa
entre os dois podadores, documentada na Seção 7.1.4 —, de modo que
codificações adicionais aumentariam apenas a precisão de um número que não
muda conclusão alguma.

### 7.2.6 Reordenação de candidatos e término antecipado: direção não perseguida

Reordenar os candidatos de particionamento e encerrar a busca antecipadamente
por outro critério que não a probabilidade de partição é uma direção que não
foi perseguida, por decisão registrada: exigiria custo de taxa-distorção por
candidato individual, não instrumentado nesta investigação, um novo ponto de
inserção em C, e competiria diretamente com as heurísticas de término
antecipado já nativas do codificador, sem que nenhum critério de decisão
offline pudesse ser aplicado antes de pagar este custo. Fica, então, como
trabalho futuro condicional, fora do escopo desta tese.

> **Procedência.** `docs/DECISOES_escopo.md` §1 a §5 (métrica, quinze
> quadros, divergências do libaom, τ adaptativo e `h9acomb`), fonte principal
> desta parte; `docs/RESULTADOS_approachB.md` §6 ("Direção não perseguida",
> reordenação e término antecipado); `docs/SINTESE_resultados_metodologia.md`
> §9 (registro consolidado das mesmas decisões).

## 7.3 Limitações remanescentes e trabalhos futuros

Esta parte fecha o capítulo com o que permanece genuinamente em aberto — não
decisão, mas pendência —, o que a distingue exatamente por isso das decisões
de escopo da Parte II.

### 7.3.1 A fronteira de compromisso global sem os pontos do H9d

A fronteira de compromisso entre taxa BD e tempo que reúne todos os níveis do
*preset* nativo é de uma análise anterior, sobre três sequências, e não contém o
H9d, medido depois sobre as oito sequências da CTC. A ausência não é
dominância — na medição própria, o H9d é não dominado no sentido de Pareto e
vence a curva de limiares do H9a em seis de oito sequências —, mas a figura,
como está, subrepresenta a segunda solução positiva desta tese na única
comparação que reúne tudo contra tudo. Recompô-la exige os pontos do H9d nos
demais níveis do *preset* nativo, que ainda não foram medidos; o custo desta
recomposição não está registrado em nenhum documento do projeto, embora a
campanha de ordem de grandeza comparável — a fronteira bidimensional do H9d —
tenha custado noventa e seis codificações.

### 7.3.2 O critério estrito do E5 atingido em uma de duas sequências

O critério pré-registrado do experimento E5 pedia dominância a tempo casado
sobre a variância em pelo menos duas das três sequências de validação;
obteve-se em uma das duas sequências efetivamente medidas, a FlowerPan, pois a
Lips não oferece par casado — a variância salta do regime conservador ao
agressivo sem ponto intermediário, como descrito na Seção 7.1.4 — e a
HoneyBee ficou fora por decisão de escopo, por ser a sequência em que a grade
de limiares original foi calibrada. O que decidiria o critério na forma
estrita são três pontos de limiar adicionais na Lips, dentro do intervalo em
que a variância salta de regime, a um custo registrado de aproximadamente
~2,3 h; ambos os desfechos são publicáveis, pois, se a
variância aterrissar num ponto intermediário, há par casado e o critério se
decide na forma estrita, e, se atravessar o intervalo de novo, o comportamento
de transição abrupta fica demonstrado por medição direta, em vez de inferido
de apenas dois pontos.

### 7.3.3 O par `pixels24` e variância sem árbitro no codificador

A discordância entre o crivo offline, que mede o conjunto de vinte e quatro
atributos de luminância superando a variância por um fator de 4,7 vezes sobre
seis sequências e 792 840 nós de decisão, e a ablação original de dois quadros
numa única sequência, em que a variância empata ou supera este mesmo conjunto,
permanece aberta. O experimento E5 não a arbitra: o braço aprendido que ele
mede é o estudante de trinta e seis atributos do H9a, e não o conjunto de vinte
e quatro atributos de luminância isoladamente, de modo que o E5 estabelece que
o H9a vence a variância no codificador, mas não decide especificamente a
ordenação interna ao domínio de pixels em que o crivo offline e a ablação
original discordam. Fechar esta lacuna exigiria uma ablação no codificador
equivalente ao E5, substituindo o braço aprendido pelo conjunto de vinte e
quatro atributos isolado, cujo custo de referência — pela campanha análoga já
executada — é da ordem de cento e quarenta e quatro codificações e vinte e
duas horas.

Esta seção encerra, então, o Capítulo de Resultados. As três partes que ela
reúne — ameaças à validade, decisões de escopo e limitações remanescentes —
não enfraquecem as duas soluções positivas apresentadas nas seções anteriores;
delimitam, com a mesma disciplina de medição que produziu estas soluções,
exatamente até onde cada afirmação desta tese se sustenta.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §6 (fronteira
> global sem o H9d, nota de 26/07 com atualização de 27/07);
> `docs/ANDAMENTO_tese.md` §0.3 e §6; `docs/INVENTARIO_solucoes.md` §8
> (lacunas primeira, segunda e quinta); `docs/RESULTADOS_E5_ablacao_validacao.md`
> §4, §5 e §6 (veredito, o que decidiria o critério e custo de ~2,3 h);
> `docs/DECISOES_escopo.md` (atualização de 28/07, corte de escopo da
> HoneyBee); `docs/RESPOSTAS_contra_argumentos_banca.md` CB-1 e CB-2 (par
> `pixels24` contra variância); `results/thesis/A3_RETRATACOES_E_LACUNAS.md`
> L1, L2 e L3 (consolidação das três lacunas, com procedência cruzada
> completa).
