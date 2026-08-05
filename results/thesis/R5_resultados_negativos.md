# 5. Reformulações do problema: resultados negativos de valor metodológico

Esta seção apresenta as vias de investigação que foram formuladas, construídas,
treinadas e medidas com competência, e que não sobreviveram aos critérios de
decisão fixados antes da medição. Sob o objetivo declarado desta tese — o estudo
de aprendizado de máquina, com foco em redes neurais, para propor soluções e
heurísticas aplicadas ao particionamento de blocos do AV1 em predição
intraquadro —, estes resultados **são o resultado do estudo**, e não material
descartado: cada um deles delimita, por medição, uma região do espaço de
soluções que deixa de ser conjectura e passa a ser território mapeado.

O critério de inclusão desta seção é estrito e foi aplicado sem exceção. Só entram
aqui as vias levadas até um **critério de decisão pré-registrado** — aquelas cujo
limiar de aprovação foi escrito antes de o número existir, e cuja medição foi
executada sobre a mesma partição congelada, os mesmos conjuntos de validação e
teste reservado e as mesmas métricas utilizadas pelas soluções positivas.

Ideias abandonadas por intuição, por falta de tempo ou por inviabilidade de
implementação não são relatadas como resultado negativo, uma vez que não
produziram evidência. Elas aparecem, quando cabe, como decisão de escopo na
Seção 7.

Cada resultado desta seção é, deste modo, comparável ponto a ponto com os
resultados positivos das Seções 2 a 4, pois foi medido contra a mesma régua.

São apresentadas, nesta ordem, duas reformulações do próprio problema — a
regressão da perda de otimalidade (do inglês *regret*) e a decisão estruturada
por rede de grafos —, quatro
ablações de modelagem sobre a solução implantada e três diagnósticos de
engenharia que sustentam decisões tomadas nos capítulos anteriores. A seção
encerra com a síntese do que o conjunto destes negativos estabelece
positivamente sobre a natureza do problema de particionamento.

> **Procedência.** `results/thesis/00_PLANO_capitulos.md` §1 e §4;
> `docs/INVENTARIO_solucoes.md` §6 (família E, reformulações do problema) e §3.5;
> `docs/SINTESE_resultados_metodologia.md` §5-bis e §5-ter.

---

## 5.1 A reformulação por regressão da perda de otimalidade

As soluções apresentadas nas Seções 2 e 3 aprenderam o particionamento como
**classificação** do rótulo de partição, em três classes, e podaram por
**confiança**, ou seja, por um limiar τ aplicado sobre a distribuição de saída do
modelo.

Esta seção apresenta a reformulação que substitui essa formulação por uma
**regressão**: em vez de classificar o rótulo, regredir diretamente o custo de
taxa-distorção de podar — grandeza designada perda de otimalidade — e podar por
custo predito.

A motivação é legítima e precisa de ser registrada como tal. A fronteira entre
taxa BD e tempo é governada por *quanto* de taxa-distorção se perde ao podar, que
é grandeza contínua. Um preditor competente deste custo poderia podar exatamente
onde é barato e, deste modo, acessar um regime de risco inacessível ao
classificador.

A reformulação foi construída sem reextração alguma de dados. O alvo foi
reconstruído a partir da árvore de particionamento comprometida pelo codificador,
definindo-se, para cada nó de decisão, a perda de otimalidade relativa como a
diferença entre o custo de taxa-distorção do `PARTITION_NONE` e o custo da
subárvore ótima, normalizada por este último.

O preditor é um perceptrone de múltiplas camadas por tamanho de bloco, com a mesma
topologia do H9a, cabeça de regressão única e entrada composta pelos **mesmos 36
atributos** do H9a. Isso garante que a comparação isole a formulação, e não a
informação disponível.

O primeiro critério de decisão da cascata, que afere a viabilidade do sinal, foi
atingido — mas com uma ressalva que antecipou o modo de falha.

Sobre as dez sequências de treino, a fração de nós com alvo exato ficou em 51,3%,
66,3% e 83,8% para os blocos de 64, 32 e 16 amostras, e o desvio da perda de
otimalidade ficou em 0,143, 0,066 e 0,019, satisfazendo os limiares
pré-registrados de 40% e de 0,001.

A ressalva registrada foi a **inflação de zeros** do alvo, que cresce com a
profundidade da árvore: 59,5% de zeros em 64 amostras, 83,6% em 32 amostras e
**98,1% em 16 amostras**. O sinal treinável concentra-se, portanto, quase
inteiramente nos blocos grandes.

O preditor ingênuo colapsou exatamente na distribuição a priori, como a inflação
de zeros fazia prever. A perda Huber final ficou em 0,00184, 0,00079 e 0,00027 por
tamanho de bloco — valores trivialmente baixos por serem dominados pela massa de
zeros —, e o modelo passou a prever perda de otimalidade próxima de zero para
quase todos os nós, inclusive para os que exigem divisão.

No terceiro critério de decisão, que mede a redução de custo de busca a risco
casado — sendo o risco a fração de nós verdadeiramente `PARTITION_SPLIT`
comprometidos com `PARTITION_NONE`, designada *SPLIT-lost* —, a regressão entregou
**0,00% de redução de custo** nos três pontos de risco de 0,5%, 1% e 2%, com piso
de *SPLIT-lost* de 12,4%. O classificador H9a, sobre as mesmas entradas, entregou
**51,73%, 51,73% e 62,00%**, com piso de *SPLIT-lost* de 0,01%. Até a linha de
base de variância, muito mais fraca, atingiu 4,46% e 6,35% nos mesmos pontos.

A correção estatística da inflação de zeros foi implementada e medida, em vez de
assumida. Introduziu-se uma perda Huber ponderada que rebalanceia, por tamanho de
bloco, os nós de perda de otimalidade não nula, com pesos de 8,21, 17,76 e 45,56
para as frações de zero de 0,89, 0,95 e 0,98 da população de alvo exato.

O modelo balanceado de fato aprendeu a cauda da distribuição, o que se verifica
pela elevação da perda Huber para 0,00361, 0,00436 e 0,00196. O balanceamento,
contudo, **não resgatou o ordenamento**. Varrendo o parâmetro de escala da
política em 0,05, 0,1 e 0,2, a redução de custo permaneceu em 0,00% em todos os
pontos de risco, e o piso de *SPLIT-lost* permaneceu em cerca de 11%. Existe uma
massa de cerca de 11% de nós verdadeiramente `PARTITION_SPLIT` que o regressor
mapeia para perda de otimalidade próxima de zero, indistinguíveis dos nós seguros.

A conclusão de valor metodológico é firme e independe da magnitude dos números. A
decisão de poda é, no fundo, uma **classificação** entre seguro e inseguro, e a
**magnitude** da perda de otimalidade não é ordenável a partir de atributos
pré-busca baratos.

Ao otimizar essa magnitude, o regressor gasta capacidade em valores que não
sobrevivem à predição, ao passo que o classificador se concentra exatamente na
fronteira de decisão que o ordenamento de poda exige. A probabilidade da decisão
é, por medição, sinal de segurança de poda estritamente melhor que o custo
predito. Isso dá base empírica a *por que* a formulação de classificação adotada
nas Seções 2 a 4 é a correta, em vez de deixar a alternativa intuitiva como
caminho não testado.

Cabe registrar uma correção sobre a regra de parada, uma vez que ela altera a
justificativa, e não o veredito.

O quinto critério de decisão — integração em C e codificação real — não foi pago,
e a justificativa original invocava a superestimação sistemática do oráculo
*offline* para concluir que a codificação real só confirmaria um resultado pior.

Esta justificativa é defeituosa. O próprio resultado da rede de grafos,
apresentado na Seção 5.2, estabelece que o ordenamento entre a simulação e o
codificador **pode inverter-se**; e, se pode inverter-se, rejeição na simulação
não implica rejeição no codificador.

O enquadramento correto é o de decisão sob **assimetria de custo experimental**. O
custo de integrar em C e de executar horas de codificação foi ponderado contra o
valor esperado da informação, dado que o sinal na simulação era fraco. Trata-se de
escolha de alocação de esforço, defensável como tal, e não de implicação lógica.

> **Procedência.** `docs/RESULTADOS_solucao4.md` §§3–7 e a correção de 2026-07-19
> ao final da §7; `docs/SINTESE_resultados_metodologia.md` §5-bis;
> `docs/ANDAMENTO_tese.md`, seção "Solução 4 — regressão de *regret*";
> `docs/INVENTARIO_solucoes.md` §6. Artefatos: `results/models/regret/`
> (`gate0.csv`, `gate3_{regret,h9a,var}.csv`, `gate3b_regret_r0*.csv`) e
> `results/models/regret_balanced/`. Scripts:
> `src/scripts/partition_model/{regret.py,build_regret_targets.py,gate0_regret.py,train_regret.py}`
> e o modo `--regret-bundle` de `simulate_pruning.py`.

---

## 5.2 A decisão estruturada por rede de grafos

Esta seção apresenta a segunda reformulação do problema, que atacou não o alvo,
mas a **estrutura da decisão**.

Todas as soluções anteriores decidem cada nó da árvore de forma independente, por
um perceptrone de múltiplas camadas por tamanho de bloco. A hipótese testada foi a
de que o limite observado seria artefato desta independência, e não limite da
informação disponível.

Para verificá-la foi construído um preditor estruturado que decide o
quadrangulamento recursivo do superbloco de forma **conjunta**, na forma de uma
**rede de grafos** (do inglês *graph neural network* – GNN) com passagem de
mensagens sobre a árvore de particionamento.

O desenho experimental é uma **ablação controlada pelo número de camadas**. Zero
camadas de passagem de mensagens reproduzem exatamente o perceptrone
independente; uma ou mais camadas introduzem a estrutura, com tudo o mais
idêntico — atributos, dados, partição congelada e regime de treino.

Na simulação de oráculo, a estrutura conjunta furou o limite dos nós independentes
de forma inequívoca. Com duas camadas e os mesmos 36 atributos do H9a, a medida
agregada de qualidade de classificação por nó subiu de 0,53, 0,66 e 0,53 para
0,66, 0,80 e 0,67 nos três tamanhos de bloco. A redução de custo de busca a
*SPLIT-lost* casado subiu de 42%, 52% e 52% para **70,5%, 73,5% e 77,8%** nos
pontos de risco de 0,5%, 1% e 2% — ganho de cerca de 28 pontos percentuais.

Uma ablação causal estrita, que autoriza cada nó a receber informação apenas do
pai e dos irmãos já decididos, fez o ganho quase desaparecer, com 46,5%, 51,1% e
57,2%. Esta ablação mostrou-se, contudo, estrita demais: os **pixels** do
superbloco inteiro existem antes da decisão de particionamento, e apenas a
*decisão* do filho é indisponível.

Reformulada com 28 atributos exclusivamente derivados de pixels, quantização e
posição, e invocada como pré-passe por superbloco — o mesmo padrão de invocação da
rede convolucional nativa —, a rede de grafos recuperou de 93% a 100% do ganho,
atingindo 65,8%, 70,6% e 77,8%, contra 31,8%, 37,4% e 37,4% do perceptrone
independente sobre os mesmos 28 atributos. A atribuição do ganho à estrutura é
limpa e vale cerca de 34 pontos percentuais.

O resultado no codificador real inverteu integralmente esta leitura.

A medição foi feita por reinjeção das probabilidades da rede de grafos no
codificador pelo ponto de inserção já existente, o que torna o experimento **fiel
às decisões**. As mesmas probabilidades sob a mesma política produzem exatamente
as mesmas decisões de poda que uma implantação em C produziria, acrescida apenas
do custo de inferência.

Na fronteira real medida sobre a sequência Jockey, com cada modelo no seu melhor
limiar, a rede de grafos entregou 1,50%, 1,53%, 1,59% e 1,56% de taxa BD para
reduções de tempo de 27,6%, 29,0%, 30,4% e 33,3%. O H9a entregou **0,88%, 0,75%,
0,86% e 0,94%** para reduções de tempo de 25,1%, 27,8%, 30,3% e 34,1%.

O H9a domina, portanto, a rede de grafos por cerca de duas vezes em taxa BD ao
longo de toda a varredura do limiar. E nenhum dos dois se aproxima do codificador
nativo, que entrega cerca de 0,45% de taxa BD a 32,6% de redução de tempo.

O ponto mais importante deste resultado não é a magnitude da derrota. É o que ela
estabelece sobre o instrumento de triagem: **a simulação de oráculo inverteu o
ordenamento**. O modelo que vencia fora do codificador por mais de vinte pontos
percentuais perde dentro dele por um fator de dois.

Este é alerta metodológico substancialmente mais forte do que a constatação, já
conhecida, de que a simulação superestima a magnitude do mérito de um podador. A
superestimação de magnitude preserva o ordenamento e permite triagem; a inversão
do ordenamento a destrói.

A raiz diagnosticada é que a **acurácia por nó e o custo de oráculo são maus
indicadores substitutos** do compromisso real entre taxa BD e tempo. A rede de
grafos otimiza acurácia de classificação por nó, vence nesta métrica, e o ganho
não se traduz em valor no codificador. A forma da fronteira confirma o
diagnóstico: a taxa BD da rede de grafos é praticamente plana entre 1,50% e 1,59%
em toda a varredura do limiar, o que situa o limite na **qualidade das decisões**,
e não na calibração.

Cabe registrar, por honestidade de procedência, que o **mecanismo** desta falha
não está provado.

A explicação originalmente proposta — a de que o custo seria dominado por poucas
podas erradas caras em taxa-distorção — foi **retirada** por medição posterior. No
crivo ponderado por perda de otimalidade, as podas da rede de grafos são baratas
tanto por contagem, com *SPLIT-lost* de 0,25%, quanto por custo ponderado, com
sobrecarga de taxa-distorção próxima de zero e a menor de todas as soluções
avaliadas.

A falha real não está, deste modo, na ação de comprometer o nó com o
`PARTITION_NONE`, e a sua causa permanece como pergunta aberta: vazamento de
vizinhança na expressividade do grafo, descasamento entre os rótulos de treino e o
regime de custo da implantação, ou dano por outra ação da política.

A conclusão de que a simulação é mau indicador substituto permanece intacta.
Apenas a explicação do mecanismo foi suavizada. `[completar: causa do desempenho
real da rede de grafos — investigação não realizada, registrada como trabalho
futuro]`

> **Procedência.** `docs/RESULTADOS_approachB.md` §§2–6 e §7 (contribuição
> metodológica), com a correção de 2026-07-20 na §6;
> `docs/RESULTADOS_oraculo_regret.md` §4.2, §4.3 e §5 (retirada da causa
> atribuída); `docs/SINTESE_resultados_metodologia.md` §5-ter;
> `docs/ANDAMENTO_tese.md`, seção "Approach B — decisão estruturada por GNN".
> Artefatos: `results/models/gnn_*/`,
> `results/benchmark/gnn_frontier/frontier_Jockey.csv` e
> `results/benchmark/gnn_replay/gnn_replay_Jockey.csv`. Scripts:
> `src/scripts/partition_model/{graph_data,gnn_model,train_gnn,gate1_gnn,gnn_replay}.py`
> e `src/scripts/benchmark/{gnn_replay_bench,gnn_frontier_bench}.py`.
> *Divergência de registro anotada:* a §7 do documento-fonte não possui
> subdivisão §7.1.

---

## 5.3 As ablações de modelagem sobre o H9a

Esta seção apresenta quatro ablações de modelagem aplicadas diretamente sobre a
solução implantada, todas avaliadas *offline* sobre os conjuntos de validação e
de teste reservado, e nenhuma delas implantada. O valor destas ablações é
justificar por medição as decisões de projeto do H9a que, de outro modo,
permaneceriam como escolhas assumidas.

O **contexto hereditário** testou a hipótese de que o custo de taxa-distorção do
pai e dos irmãos já decididos carrega sinal de profundidade que o nó isolado não
possui, com a vantagem decisiva de ser causalmente disponível no podador
pré-busca.

Foram acrescentados seis atributos ao vetor do H9a, deliberadamente restritos a
**magnitudes** de taxa-distorção medidas e nunca a rótulos ou decisões, elevando a
entrada de 36 para 42 atributos. O critério foi o crivo ponderado por perda de
otimalidade, que mede a qualidade fundamental da decisão a redução de custo
casada.

O resultado é negativo e robusto. A 30% de redução de custo, a sobrecarga relativa
de taxa-distorção sobe de 24,81 para 38,81, e o negativo persiste em toda a
fronteira, com 35,9 contra 59,7 a 34% de redução de custo e 28,0 contra 40,0 a
30%. O contexto hereditário leva o modelo a comprometer o nó com o
`PARTITION_NONE` em nós mais caros em taxa-distorção. Ou seja: um sinal
informativo e causalmente disponível **não é o mesmo que um sinal útil à decisão
de poda**.

O **limiar condicionado à quantização** partiu da constatação de que a mistura de
rótulos varia fortemente com o nível de quantização — de 7,6% de `PARTITION_SPLIT`
em blocos de 16 amostras no regime de alta taxa a 0,14% no regime de baixa taxa —,
o que torna um limiar global mal casado por construção. Como o lado C já lê o
limiar do ambiente, a estratificação seria implantável sem recompilação.

O mecanismo confirma-se: o limiar necessário para atingir a mesma redução de custo
é monotônico no nível de quantização, indo de cerca de 0,60 a 0,90 conforme a
taxa.

O ganho, contudo, é imaterial em termos absolutos. A sobrecarga de taxa-distorção
cai de 0,0061% para 0,0028% a 30% de redução de custo — redução relativa de cerca
de duas vezes, mas redução absoluta inferior a 0,005 ponto percentual. Este valor
está **abaixo do piso de ruído do tempo de parede** do codificador, cuja resolução
medida do tempo pareado é de aproximadamente 0,46 ponto percentual, com
coeficiente de variação mediano de 0,28%.

Esta via é, por conseguinte, registrada como boa prática de calibração, e não como
ganho. Adotá-la custa zero e é mais defensável que um limiar único, mas alegar
melhoria de taxa BD sem medição não se sustenta. `[completar: medição de taxa BD e
tempo do limiar estratificado por quantização no codificador, caso venha a ser
paga]`

O **sinal direcional** entre as formas horizontal e vertical foi investigado
porque o modelo implantado colapsa dez tipos de partição em três classes,
descartando a orientação, e porque o lado C já expõe a ação por direção.

A falsificação inicial falhou de forma clara: um modelo de quatro classes prevê a
direção com 69,1% de acerto sobre 191.909 nós retangulares reservados, contra
52,4% da classe majoritária. Esta acurácia é, porém, **condicional** a saber que o
nó é retangular — e é exatamente nisso que o modelo é fraco, uma vez que cerca de
35% dos nós verdadeiramente retangulares são preditos como `PARTITION_NONE`.

A etapa seguinte testou se os atributos de taxa-distorção reais disponíveis no
ponto de inserção posterior à avaliação do `PARTITION_NONE` supririam esta
distinção, contra um controle pareado de mesma sessão e mesmos dados. O critério
primário — a acurácia condicional subir materialmente acima dos 69% — **não foi
atingido**, ficando em 69,5% contra 69,2% do controle. O critério secundário
melhorou apenas de cerca de 42% para cerca de 47% de sensibilidade.

Com metade dos nós de uma direção não reconhecidos, as duas pontas do compromisso
são ruins ao mesmo tempo: ou se poda pouco, sem redução de tempo, ou se paga
taxa-distorção em excesso. Esta via foi, deste modo, encerrada no critério de
decisão, ao custo de cerca de uma hora de processamento em unidade de
processamento gráfico, em lugar das cerca de oito horas de integração em C e de
codificações que a etapa seguinte exigiria.

A **ponderação de classe** por nível propôs corrigir a inflação de zeros da classe
`PARTITION_SPLIT`. É o caso mais instrutivo do conjunto, uma vez que **funciona
naquilo que se propôs e falha naquilo que importa**.

Como classificador, a ponderação funciona. A sensibilidade à classe
`PARTITION_SPLIT` em blocos de 16 amostras salta dezesseis vezes, de 0,022 para
0,361, e o modelo passa a prever a classe residual em blocos de 64 amostras, onde
antes nunca a previa.

Para um podador por limiar de confiança, contudo, a métrica de implantação não é o
balanço de classificação, e sim a **calibração** da distribuição de saída, que é
sobre a qual o limiar opera. E a calibração é arruinada: o erro de calibração
esperado da classe `PARTITION_NONE` sobe oito vezes, de 0,021 para 0,174, e o
modelo torna-se sistematicamente subconfiante, a ponto de a frequência real de
`PARTITION_NONE` ser de 0,82 onde a probabilidade predita é 0,5.

A consequência é direta. O modelo ponderado poda menos, alcançando no máximo 35%
de redução de custo contra 42% do modelo implantado, sem ganho algum na qualidade
da decisão ponderada por custo, que fica empatada.

Um conserto óbvio revela-se, portanto, um **falso positivo** para podadores por
limiar, e a ablação valida empiricamente a decisão de treinar sem ponderação.

> **Procedência.** `docs/RESULTADOS_modelagem_B1_contexto_hereditario.md` §§1–4;
> `docs/RESULTADOS_modelagem_B2_tau_qindex.md` §§1–6;
> `docs/RESULTADOS_modelagem_B3_horz_vert.md` §§1–5 e §7 (etapa pós-NONE, com o
> controle pareado e a variância entre execuções de ±2 a 3 pontos percentuais);
> `docs/RESULTADOS_modelagem_B4_ponderacao_classe.md` §§1–6;
> `docs/RESULTADOS_oraculo_regret.md` §§2–3 (definição do crivo ponderado por
> perda de otimalidade, 792.840 nós); `docs/INVENTARIO_solucoes.md` §3.5. Artefatos:
> `results/models/{student_h9a_b1,b2_tau_qindex,student_h9a_4cls,b3_postnone,b3_control36,student_h9a_cw,oracle_regret_b1}/`.
> Scripts: `src/scripts/partition_model/{train_student_h9.py,oracle_regret.py,b2_tau_per_qindex.py,b3_horz_vert.py,calibration.py,compare_students.py}`.

---

## 5.4 Os diagnósticos de engenharia da série C

Esta seção apresenta três diagnósticos que não são soluções propostas, e sim
medições dirigidas a decidir, antes de qualquer implementação cara, se uma via
merecia ser perseguida. Eles pertencem a esta seção porque dois deles
**falsificaram** conjecturas de projeto, e porque o terceiro sustenta, por
medição, a solução apresentada na Seção 4.

A **decomposição do custo por candidato** respondeu se as partições estendidas têm
custo a recuperar, com critério de falsificação fixado antes da medição: se as
formas AB somadas às 4-way representassem menos de 10% do tempo de busca local do
nó, a via morreria sem experimento de codificação.

Sobre 875.317 nós de decisão das três sequências de teste reservado, o resultado é
de **34,3%** agregado, com variação de 28,9% a 41,3% entre sequências. O mínimo já
é, portanto, cerca de três vezes o limiar. A conjectura de que estas formas seriam
fatia pequena do custo está errada neste regime, e o resultado é consistente entre
conteúdos.

O diagnóstico localiza ainda **onde** o custo vive. As partições estendidas são
desprezíveis em blocos de 16 amostras, com 8,7%, e representam 50,0% e 51,0% do
trabalho local em blocos de 32 e de 64 amostras — o que restringe o alcance útil
de um podador desta família aos blocos grandes.

A **varredura por nível** decompôs a contribuição da solução implantada,
desligando um nível de cada vez por variável de ambiente, sem recompilação.

O achado é que o ganho **não vem de um único nível**, o que remove a hipótese de
que a comparação com a rede convolucional nativa seria artefato de um nível
dominante.

A varredura revela, ademais, a assimetria de eficiência entre os níveis. O nível
de 64 amostras é o mais agressivo e o mais caro, custando 5,84 pontos percentuais
de taxa BD por cada aceleração unitária na sequência Jockey e 3,27 na RaceNight.
O nível de 32 amostras, ao passo que, é o mais eficiente, chegando a render
aceleração adicional praticamente gratuita.

O mecanismo é inteligível: um comprometimento errado com o `PARTITION_NONE` em um
bloco de 64 amostras descarta uma subárvore inteira, e coloca muita qualidade em
risco por decisão.

A **fronteira de limiares** falsificou a conjectura de que a poda dura por limiar
deixaria a fronteira esburacada, e de que uma modulação contínua do custo de
taxa-distorção preencheria os vazios.

Como esta modulação exigiria cirurgia não trivial na busca recursiva em C, o plano
fixou a falsificação barata: se uma varredura fina do limiar já produzisse
fronteira densa e contínua, a suavização nada acrescentaria.

A varredura de oito limiares mostra que o maior salto de aceleração entre limiares
vizinhos, em vinte e uma vizinhanças, é de 0,15×, e que a maioria fica em torno de
0,03× — sem descontinuidades e com taxa BD suave e monótona. O salto de cerca de
0,32× observado na curva anterior era, deste modo, artefato da amostragem
grosseira, e não da natureza dura da decisão.

Este diagnóstico é, ao mesmo tempo, o encerramento de uma via e um resultado
positivo sobre a solução implantada, cuja fronteira se mostra ajustável de forma
contínua por uma única variável de ambiente, sem retreino.

> **Procedência.** `docs/RESULTADOS_C1_custo_por_candidato.md` §§1–5;
> `docs/RESULTADOS_C2_sweep_niveis.md` §§1–5;
> `docs/RESULTADOS_C5_fronteira_tau.md` §§1–4; `docs/INVENTARIO_solucoes.md`
> §3.6. Artefatos: `results/benchmark/partstats*/part_timing*.csv`,
> `results/benchmark/c2_levels/raw.csv` e
> `results/benchmark/c5_finetau/raw.csv` (não versionados). Scripts:
> `src/scripts/benchmark/{analyze_partstats.py,c2_level_sweep.py,c5_fine_tau.py,bd_rate.py}`.

---

## 5.5 Síntese dos negativos — o que o conjunto estabelece

O conjunto dos resultados desta seção converge para três afirmações positivas
sobre a natureza do problema de particionamento, e é nesta convergência que está
a sua contribuição.

A primeira afirmação é sobre a **formulação correta**. A decisão de poda é uma
classificação entre seguro e inseguro, e não uma estimativa de custo, uma vez que
a magnitude da perda de otimalidade é dominada pelo conteúdo e não é ordenável a
partir de atributos pré-busca baratos.

Esta afirmação não é preferência de projeto. Ela foi obtida construindo a
alternativa, treinando-a com competência, corrigindo o seu modo de falha
estatístico e medindo-a contra o classificador sobre entradas idênticas.

A segunda afirmação é sobre a **estrutura da decisão**. O limite observado não era
artefato de decidir cada nó de forma independente, pois a decisão conjunta extrai
comprovadamente mais sinal fora do codificador. O que se estabelece é que esse
sinal adicional **não é realizável** em taxa BD e tempo reais.

A saturação do sinal de particionamento no nível da solução implantada é, deste
modo, propriedade medida do problema, e não limitação da família de modelos
utilizada.

A terceira afirmação, e a de maior alcance metodológico, é sobre o **estatuto dos
indicadores substitutos**. Um crivo *offline*, mesmo quando ponderado pelo custo
real de taxa-distorção, filtra candidatos claramente inferiores, mas **não adjudica
o vencedor** entre modelos competitivos, uma vez que pode inverter o ordenamento
observado no codificador.

Esta é a justificativa empírica — e não conjecturada — para que o codificador
permaneça o árbitro final de todas as decisões desta tese, e para a cascata de
critérios de decisão descrita no Capítulo de Metodologia, na qual o custo caro só
é pago depois que o sinal se prova fora do codificador. Com a consciência
explícita de que provar-se fora dele é condição necessária, e nunca suficiente.

A estas três afirmações somam-se as justificativas de projeto que as ablações
tornaram medidas em vez de assumidas. O vetor de 36 atributos resiste às duas
extensões mais óbvias. O treino sem ponderação de classe está correto para um
podador por limiar. A estratificação do limiar por quantização é boa prática de
calibração de efeito imaterial. E o terceiro eixo de decisão — a orientação
retangular — carrega sinal aprendível que, medido, não sustenta a política que o
exploraria.

Caracterizar assim o espaço de soluções, com cada fronteira marcada por uma
medição e não por uma impressão, é contribuição em si. Ela transfere para o
registro público o custo experimental de descobrir onde as vias promissoras deixam
de sê-lo.

> **Procedência.** Consolidação das notas das Seções 5.1 a 5.4;
> `docs/RESULTADOS_oraculo_regret.md` §4.3 (limite do crivo *offline*);
> `docs/INVENTARIO_solucoes.md` §7 (resumo do que sobreviveu);
> `docs/SINTESE_resultados_metodologia.md` §5-bis e §5-ter. Nenhum valor novo é
> introduzido nesta síntese.

---

## 5.6 Encaminhamento

As cinco vias encerradas nesta seção são a contagem geral do projeto — e não as
cinco tentativas independentes do domínio de pixels, apresentadas na Seção 1. Elas
delimitam, por medição, o que o problema de particionamento intraquadro **não**
admite, ao passo que as duas soluções positivas apresentadas nas Seções 2 e 4
delimitam o que ele admite.

Resta situar umas e outras sobre a mesma fronteira de compromisso entre taxa BD e
tempo, medida contra o próprio botão de velocidade do codificador, e extrair daí
as conclusões que a tese sustenta. Esta análise integrada é apresentada na próxima
seção.

> **Procedência.** `results/thesis/00_PLANO_capitulos.md` §2 e §4; nenhum valor é
> introduzido neste encaminhamento.
