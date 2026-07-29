# 6. Arquiteturas de rede neural e metodologia de atribuição

Esta seção apresenta as arquiteturas de rede neural investigadas nesta tese e a
metodologia desenvolvida para decidir a quem pertence um ganho de tempo medido.
A seção está organizada em duas partes. A primeira parte descreve os modelos —
o modelo substituto convolucional multinível, a destilação de conhecimento que
dele derivou o modelo estudante da era de pixels, os perceptrones de múltiplas
camadas por tamanho de bloco que constituem os artefatos implantados, a rede de
grafos com passagem de mensagens da abordagem estruturada e a reformulação do
problema por regressão de *regret*. A segunda parte apresenta a **metodologia de
atribuição**, que é a contribuição metodológica deste capítulo: o conjunto de
desenhos experimentais que separa o mérito do modelo do mérito do invólucro que
o executa, e que estabelece, por medição, os limites do que uma avaliação fora
do codificador pode afirmar.

---

## 6.1 O modelo substituto convolucional multinível

O primeiro modelo investigado é um **modelo substituto** convolucional da
família ConvNeXt, desenhado para espelhar a hierarquia da árvore de
particionamento do AV1 e para operar exclusivamente fora do codificador, em
processamento gráfico e em tempo de treino. A entrada é um tensor de dois canais
com 64×64 amostras, formado pela **luminância do superbloco** e por um **plano
constante de quantização**, obtido pela normalização do índice de quantização do
quadro; deste modo, o mesmo conteúdo apresentado sob forças de quantização
distintas produz predições distintas, uma vez que a decisão de particionamento
depende do compromisso de taxa-distorção e não apenas da textura.

O tronco é **compartilhado por todos os níveis da hierarquia**. Três estágios do
extrator convolucional são capturados — um mapa raso de 16×16 células, com campo
receptivo pequeno e informação local, e dois mapas profundos de 4×4 e 2×2
células, que carregam contexto global —, e estes mapas são projetados por
convoluções de núcleo unitário para uma dimensão comum de fusão de **128
canais**. A fusão é descendente: o contexto global é interpolado e somado ao
mapa intermediário, que por sua vez é interpolado e somado ao mapa local, e o
resultado passa por uma convolução de suavização 3×3. Então, cada célula do mapa
de 16×16 dispõe simultaneamente da sua vizinhança imediata e do contexto do
superbloco inteiro, o que é exatamente a informação que a busca recursiva
consome ao descer a árvore.

As **cabeças por nível** operam sobre agrupamentos médios desse mapa fundido, de
modo que cada célula agrega precisamente a região do bloco que lhe corresponde:
um agrupamento 4×4 produz a grade de blocos de 16×16 amostras, um agrupamento
8×8 produz a grade de blocos de 32×32 amostras e a média global produz a
predição única do bloco de 64×64 amostras. O nível de 8×8 amostras é excluído do
modelo, pois é folha terminal da árvore e a decisão ali é sempre
`PARTITION_NONE`. Cada cabeça emite as dez formas de partição do AV1, e as
classes ilegais para o seu tamanho de bloco são suprimidas por uma máscara
aditiva antes da normalização exponencial, o que impede que o modelo distribua
massa de probabilidade em decisões que o codificador jamais tomaria. As
predições de dez classes são, por fim, colapsadas nas três classes que a
política de poda consome, ou seja, `NONE`, `SPLIT` e o agregado das demais
formas.

A perda de treino é uma **entropia cruzada mascarada, sensível a custo e
ponderada por classe**. Ela é mascarada porque apenas as células da grade
correspondentes a nós que a busca de referência efetivamente visitou são
supervisionadas; é sensível a custo porque os alvos são rótulos suaves
construídos a partir de uma matriz de proximidade geométrica entre as formas de
partição, de modo que confundir duas partições semelhantes é penalizado menos do
que confundir partições opostas; e é ponderada por frequência inversa em cada
nível para contrariar o predomínio da classe `NONE`, que responde por cerca de
91% dos rótulos. O critério de seleção declarado é o macro-F1 acompanhado da
revocação da classe `SPLIT`, e nunca a acurácia bruta. Nas trinta épocas
registradas do treino, o ponto de verificação salvo é o da **época 13**, em que
o mínimo da perda de validação, de **1,7999**, e o máximo do macro-F1, de
**0,2034**, coincidem; ou seja, os dois critérios concordam e a seleção não
apresenta sobreajuste na escolha.

O papel declarado deste modelo mudou por medição, e a mudança é registrada aqui
porque afeta a leitura de todo o capítulo. A intenção original era utilizá-lo
como **cota superior do domínio de pixels**, pois ele reúne capacidade elevada,
acesso à luminância bruta e possibilidade de execução exata dentro do
codificador por reinjeção de probabilidades pré-computadas, sem qualquer custo
computacional de inferência convolucional em C. Duas medições retiraram esse
papel. Primeiro, uma ablação controlada do objetivo de treino — a mesma
arquitetura e a mesma dimensão de fusão de 128 canais, alterando somente a perda
para uma entropia cruzada ponderada por nó com peso `1 + α·regret_rel`, com
`α = 3` e normalização por nível pelo percentil 95 — **piorou** o modelo em toda
a faixa avaliada, embora objetivo e métrica de avaliação estivessem alinhados.
Segundo, dobrar a dimensão de fusão altera a perda de validação em apenas
**0,16%**, o que estabelece que a capacidade não é a restrição. Deste modo, o
modelo substituto permanece no arcabouço como **instrumento de diagnóstico** e
como a tentativa documentada de estabelecer a cota superior, e não como a cota
superior: um modelo batido por outro de acesso estritamente menor à informação
não estabelece limite superior de desempenho algum. O único limite superior
genuíno do arcabouço é o **oráculo**, ou seja, a decisão de taxa-distorção ótima
com *regret* nulo, que limita qualquer podador.

> **Procedência.** `src/scripts/partition_model/model.py` (arquitetura
> `PartitionSurrogate`: captura dos estágios, fusão descendente, agrupamentos
> por nível, máscara de legalidade) e `train.py` (perda mascarada, alvos suaves
> geométricos, ponderação por frequência inversa, critério de seleção);
> `docs/SINTESE_resultados_metodologia.md` §3; `docs/ANDAMENTO_tese.md` §1.2 e
> §1.3; `docs/RESULTADOS_convnext_regret.md` §1, §1.1 e §4 (ablação de objetivo,
> dimensão de fusão e correção de registro sobre a seleção do ponto de
> verificação). Artefatos: `results/models/surrogate_real/{surrogate_best.pt,
> metrics.csv}` e `results/models/surrogate_regret/`. Scripts:
> `train.py`, `train_surrogate_regret.py`. *Lacuna:* [completar: confirmação, a
> partir do ponto de verificação `surrogate_best.pt`, da taxa de aprendizado, do
> tamanho do lote e do decaimento de peso efetivamente utilizados, hoje
> conhecidos apenas como valores padrão do script].

---

## 6.2 A destilação de conhecimento e o alcance da ressalva

A versão implantável **da era de pixels** foi obtida por **destilação de
conhecimento** do modelo substituto para um **modelo estudante** raso. Para cada
nó de particionamento, o par de treino reúne os atributos manuais calculados
sobre a luminância daquele bloco e a predição suave que o substituto emite na
célula correspondente do nível equivalente da grade, colapsada em três classes.
O estudante é treinado com uma combinação convexa de dois termos, ou seja,
`α·CE(rótulo duro) + (1−α)·KL(mestre suave)`, com `α = 0,5` e temperatura de
normalização igual a **3,0**. Este arranjo separa deliberadamente a capacidade
de predição do custo computacional de inferência: o substituto observa o
superbloco inteiro por convoluções multirresolução, enquanto o estudante consome
**24 atributos** manuais e executa por uma rotina densa já existente no próprio
codificador.

É obrigatório registrar, com precisão, o alcance desta descrição. A destilação
descreve o modelo estudante da era de pixels, designado **H7**, e **não** o
podador de fato implantado. O podador implantado, designado **H9a**, é treinado
**diretamente** sobre 36 atributos com entropia cruzada de rótulo duro, sem o
modelo substituto no laço de treino e sem qualquer termo de destilação. A razão
é de projeto e foi decidida por evidência: o sinal que separa o H9a dos modelos
anteriores é tabular e não pixélico — contexto de particionamento da vizinhança,
quantização e posição —, e um modelo substituto que observa apenas pixels não
tem como ensiná-lo. Então, atribuir o artefato implantado a uma destilação do
modelo substituto seria erro factual, e esta ressalva é mantida em todas as
descrições do artefato ao longo do texto.

> **Procedência.** `src/scripts/partition_model/distill.py` (construção do
> conjunto de destilação, perda combinada, valores padrão `α = 0,5` e
> temperatura 3,0) e `train_student_h9.py` (treino direto, sem mestre);
> `docs/METODOLOGIA_pipeline_ML.md` §1 e §7 (nota de registro histórico e
> redação corrigida); `docs/SINTESE_resultados_metodologia.md` §3 (ressalva de
> 2026-07-19) e §4; `docs/ANDAMENTO_tese.md` §1.2. Artefatos:
> `results/models/student_real/` (estudante de pixels) e
> `results/models/student_h9a/` (artefato implantado).

---

## 6.3 Os perceptrones de múltiplas camadas por tamanho de bloco

Os artefatos efetivamente implantados são **perceptrones de múltiplas camadas,
um por tamanho de bloco**, treinados de forma independente para os níveis de
16×16, 32×32 e 64×64 amostras. A topologia é comum a todas as variantes e
deliberadamente rasa: duas camadas ocultas de **64 e 32 unidades** com ativação
retificada e uma camada de saída linear, cujo formato de pesos corresponde
exatamente ao esperado pela rotina de inferência nativa do codificador, o que
dispensa qualquer código de inferência novo em C. O que distingue as variantes é
o número de entradas e o número de saídas, conforme o ponto de enganche e a ação
de cada uma. O estudante de pixels consome **24 atributos** e emite três
classes; o H9a consome **36 atributos** e emite três classes; o H9c consome
**39 atributos**, acrescentando a taxa, a distorção e o custo de taxa-distorção
reais do `PARTITION_NONE`, e emite três classes; e o H9d consome os mesmos **39
atributos** do H9c e emite **duas** classes, correspondentes a avaliar ou não
avaliar as partições estendidas.

A função de perda é a **entropia cruzada de rótulo duro sem ponderação de
classe**, e esta escolha é uma decisão de projeto medida, não assumida. A
otimização utiliza o algoritmo AdamW com taxa de aprendizado de `1·10⁻³`,
decaimento de peso de `1·10⁻⁴`, lotes de 4096 amostras e um orçamento fixo de
**30 épocas**, sem parada antecipada. Os atributos são padronizados durante o
treino e a padronização é, ao final, **dobrada na primeira camada linear**, de
modo que o codificador alimente o modelo com atributos crus e a paridade entre
as implementações em Python e em C seja verificável diretamente.

A ponderação de classe foi objeto de uma ablação dedicada, pois a inflação de
zeros da classe `SPLIT` nos blocos de 16×16 amostras torna a correção
aparentemente óbvia. Treinando a mesma topologia com pesos de frequência
inversa por nível, recortados no intervalo `[0,1; 10]`, a **classificação
melhora e a decisão de poda não**: a revocação de `SPLIT` em 16×16 amostras
salta de 0,022 para 0,361 e o macro-F1 sobe de 0,524 para 0,556, mas o erro de
calibração esperado da classe `NONE` degrada de 0,021 para 0,174, ou seja, oito
vezes, e a fração máxima de custo podável cai de 42% para 35%. Como a política é
limiarizada sobre a probabilidade, é a calibração, e não a revocação, que
determina o que o codificador paga. Deste modo, o **critério de seleção de
modelo** adotado não é a acurácia por nó nem o macro-F1, e sim o **crivo
ponderado por *regret* a redução de custo casada**, descrito na Seção 6.10; foi
por este critério que as duas extensões investigadas — a ponderação de classe e
o acréscimo de seis atributos de contexto de taxa-distorção herdado do bloco-pai
e dos irmãos já decididos — foram rejeitadas, e que a estratificação do limiar
por índice de quantização foi aceita como refinamento de calibração de magnitude
imaterial.

> **Procedência.** `src/scripts/partition_model/student.py` (topologia e formato
> de exportação), `distill.py::train_student` (otimizador, lotes, padronização
> dobrada, nota de projeto sobre ponderação de classe), `train_student_h9.py`,
> `train_student_h9c.py` e `h9d_predictability.py` (variantes de entrada e
> saída); `src/scripts/partition_model/features.py:51,204,382` (contagens de 24,
> 36 e 39 atributos); `docs/RESULTADOS_modelagem_B4_ponderacao_classe.md` §2 a
> §5; `docs/RESULTADOS_modelagem_B1_contexto_hereditario.md` §2 e §3;
> `docs/RESULTADOS_modelagem_B2_tau_qindex.md` §3 a §5;
> `docs/SINTESE_resultados_metodologia.md` §4 e §5. Artefatos:
> `results/models/{student_h9a,student_h9a_cw,student_h9a_b1,student_h9c,
> h9d_predictability,b2_tau_qindex}/`.

---

## 6.4 A rede de grafos com passagem de mensagens e a ablação controlada

A última alavanca de modelagem investigada questiona a própria formulação por
nós independentes: se a decisão do quadtree do superbloco fosse tomada de forma
**conjunta**, haveria sinal além do que os perceptrones por nó extraem? Para
responder, foi desenvolvida uma rede de grafos com **passagem de mensagens** (do
inglês *message passing*) sobre a árvore de particionamento do superbloco, em
que cada nó é um vértice portador do mesmo vetor de 36 atributos e as arestas
reproduzem as relações de parentesco e de irmandade. A topologia é um
codificador linear que projeta os atributos em 64 dimensões, seguido de um
número configurável de camadas de convolução em grafo — a variante padrão é a
agregação por amostragem de vizinhança, com as alternativas de atenção e de
isomorfismo disponíveis — e de uma cabeça linear que emite as três classes.

O elemento metodológico decisivo não é a biblioteca nem a escolha de camada, e
sim a **ablação controlada do número de camadas**. Com `n_layers = 0` a rede
degenera, por construção, em um perceptrone independente por nó, uma vez que
nenhuma mensagem é trocada; com `n_layers ≥ 1` a estrutura passa a atuar. Todo o
restante é mantido idêntico, ou seja, os mesmos atributos, os mesmos dados, a
mesma partição por sequência, o mesmo otimizador, a mesma taxa de aprendizado de
`1·10⁻³`, as mesmas 15 épocas e os mesmos lotes de 256 superblocos. Então,
qualquer diferença medida é atribuível à estrutura, e não à capacidade ou ao
regime de treino, o que é precisamente o que uma comparação entre famílias de
modelos publicadas na literatura raramente permite.

Duas variantes adicionais foram construídas para separar o que é ganho legítimo
do que é vazamento de informação. A variante de **arestas completas** agrega
vizinhos que incluem filhos e irmãos futuros, informação que o codificador não
possui no instante da decisão descendente, e constitui, portanto, um limite
superior não causal. A variante de **arestas causais** restringe a agregação ao
bloco-pai e aos irmãos anteriores na ordem de varredura. Por fim, uma variante
**implantável baseada apenas em pixels**, com 28 atributos, descarta o bloco de
vizinhança — a única entrada dependente de decisão — e mantém arestas completas,
o que é legítimo porque os pixels do superbloco inteiro existem antes de
qualquer decisão de particionamento e a agregação ascendente poderia ser
executada como um pré-passe por superbloco, no mesmo padrão de invocação da rede
convolucional nativa.

> **Procedência.** `src/scripts/partition_model/gnn_model.py` (topologia
> `TreeGNN` e a degeneração em perceptrone com `n_layers = 0`), `train_gnn.py`
> (hiperparâmetros da ablação controlada e opções `--causal` e `--feat-mode`) e
> `graph_data.py`; `docs/RESULTADOS_approachB.md` §1 a §4;
> `docs/SINTESE_resultados_metodologia.md` §5-ter. Artefatos:
> `results/models/gnn_{L0,L2,L2_causal,L0_pixel,L2_pixel}/` e
> `results/models/gnn_gate1*.csv`.

---

## 6.5 A reformulação por regressão de *regret*

Todas as formulações anteriores tratam a poda como **classificação** da decisão
do nó. A reformulação investigada substitui o alvo pelo **custo real de podar**:
em vez de predizer qual partição vence, o modelo prediz o *regret*, definido
como a diferença entre o custo de taxa-distorção de comprometer o nó com
`PARTITION_NONE` e o custo da subárvore que a busca completa encontraria, uma
grandeza não negativa que vale zero sempre que a decisão correta já era
`PARTITION_NONE`. A motivação é direta e decorre de um resultado desta própria
investigação: a acurácia por nó demonstrou ser mau indicador do compromisso
entre taxa BD e tempo de um podador, e o *regret* é a grandeza que o podador de
fato arrisca a cada poda.

A arquitetura preserva a topologia de 64 e 32 unidades por tamanho de bloco e
substitui a cabeça de três classes por uma **cabeça de regressão única**. O alvo
é o logaritmo de um mais o *regret* relativo, a perda é a de Huber com parâmetro
unitário, e a otimização utiliza o algoritmo Adam com taxa de aprendizado de
`1·10⁻³` por 200 épocas, sobre os mesmos 36 atributos do H9a, de modo que a
comparação isole a formulação do alvo.

O tratamento da **inflação de zeros** é a decisão metodológica central desta
via. A distribuição do *regret* é dominada por nós de valor exatamente nulo, e
uma regressão ingênua sobre esse alvo degenera, pois minimiza a perda predizendo
zero em quase toda parte. Foram treinadas, então, duas variantes: uma ingênua e
uma **balanceada**, em que as amostras de *regret* não nulo recebem peso igual à
razão entre o número de amostras nulas e não nulas, limitado superiormente, o
que resultou em pesos efetivos entre 8 e 46 vezes. A ressalva que a ablação
expõe é estrutural e não se resolve por reponderação: cerca de **11%** dos nós
cuja decisão verdadeira é `SPLIT` têm *regret* aproximadamente nulo, e são,
portanto, **indistinguíveis dos nós seguros** por qualquer regressão sobre esta
grandeza, o que dá base empírica à afirmação de que a poda é fundamentalmente
uma decisão de classificação.

> **Procedência.** `src/scripts/partition_model/train_regret.py` (cabeça de
> regressão, alvo logarítmico, perda de Huber, reponderação com limite superior)
> e `regret.py` e `build_regret_targets.py` (definição e reconstrução do alvo);
> `docs/SINTESE_resultados_metodologia.md` §5-bis;
> `docs/RESULTADOS_oraculo_regret.md` §2 (definição de `regret_abs`);
> `docs/RESULTADOS_solucao4.md`. Artefatos: `results/models/regret/` e
> `results/models/regret_balanced/`.

---

## 6.6 O problema da atribuição

A segunda parte desta seção apresenta a metodologia de atribuição. O problema
que ela resolve é o seguinte: um podador aprendido é composto por **duas peças
separáveis**, ou seja, uma **fonte de escore**, que é o modelo, e uma
**política**, que converte o escore em ações de poda sobre o espaço de busca.
Qualquer escore, inclusive um escore sem informação alguma, produz alguma
redução de tempo quando alimentado a uma política agressiva, pois é a política
que remove candidatos da busca. Então, um número de aceleração medido, por maior
que seja, **não é evidência de que o modelo aprendeu algo**, e a ausência dessa
distinção é uma fragilidade recorrente da literatura de poda aprendida.

A posição adotada nesta tese é que um ganho de tempo só é atribuível ao modelo
se **sobrevive à comparação com fontes de escore alternativas executadas sob a
mesma política e no mesmo codificador**. Deste modo, a unidade de comparação
deixa de ser o par formado por modelo e política e passa a ser exclusivamente a
fonte do escore, que é a única variável manipulada. As seções seguintes
descrevem os quatro desenhos que materializam esta posição — a ablação de
atribuição em três braços, a substituição direta do podador nativo, a
decomposição com neutralização de alavanca e a simulação oráculo —, cada um
respondendo a uma pergunta distinta de atribuição.

> **Procedência.** `docs/METODOLOGIA_pipeline_ML.md` §5 (formulação da pergunta
> de atribuição e desenho da ablação); `docs/RESULTADOS_E5_ablacao_validacao.md`
> §1; `results/thesis/00_PLANO_capitulos.md` §2.

---

## 6.7 A ablação de atribuição em três braços

A ablação de atribuição mantém **a política idêntica** e **o mesmo codificador**,
variando somente a fonte da probabilidade `P(NONE)`, selecionada por variável de
ambiente. A política empregada é a de comprometimento com `PARTITION_NONE` em
forma pura, ou seja, o limiar de forçamento de divisão é fixado em valor
inatingível e a poda de partições retangulares é desligada, de modo que a única
ação em operação seja a que os três braços podem exercer igualmente. Os braços
são três. O braço **modelo** utiliza o estudante implantado de 36 atributos. O
braço **limiar de variância** utiliza `P(NONE) = exp(−var/V₀)`, ou seja, a
heurística manual óbvia segundo a qual um bloco liso não deve ser dividido,
construída sobre o atributo isolado mais informativo e que é, ele próprio, uma
das entradas do modelo — o que torna a comparação deliberadamente severa, pois
um modelo bem ajustado não deveria ser dominado por um de seus próprios
atributos. O braço **escore aleatório** utiliza uma função de dispersão uniforme
e determinística da identidade do nó, o que poda a **mesma fração** de nós que
os demais, escolhidos ao acaso, isolando o efeito do invólucro.

Três modos de comparação são distinguidos, e a distinção é essencial. A
comparação **a tempo casado** confronta a taxa BD dos braços no mesmo valor de
aceleração; a comparação **a taxa BD casada** confronta a aceleração no mesmo
valor de taxa BD; e a comparação **a política casada** confronta os braços
inteiros sob a política idêntica, tomando as curvas completas como o objeto de
comparação. Os dois primeiros modos exigem que as faixas de operação dos braços
**se sobreponham**, e esta exigência não é procedimental: na campanha do
conjunto de teste reservado as faixas de aceleração do modelo e da variância
saíram disjuntas nas três sequências, o que impediu qualquer par casado.

A comparação a política casada é o argumento mais forte por duas razões. A
primeira é que ela **controla o invólucro inteiro** — mesma política, mesmos
limiares de ação, mesmo codificador, mesmo âncora, mesma sequência —, deixando a
fonte do escore como única variável manipulada, o que é a condição lógica da
atribuição. A segunda é que ela **dispensa a sobreposição de faixas**, pois a
ausência de ponto de operação de um escore na região implantável deixa de ser
uma limitação da medição e passa a ser um **resultado sobre o escore**. Este é
exatamente o caso observado na sequência Lips, em que a curva da variância
apresenta uma **transição abrupta** entre os limiares 0,99 e 0,97: a aceleração
salta de **1,006× para 3,563×** e a taxa BD de **0,019% para 6,580%**, sem
qualquer ponto intermediário, enquanto o escore do modelo gradua continuamente a
faixa de **1,074× a 1,283×**. Neste caso, não há par casado a comparar, e ainda
assim a comparação a política casada estabelece a afirmação mais forte, ou seja,
que a não sobreposição é propriedade do escore da variância e não da grade de
limiares escolhida.

Duas salvaguardas de honestidade completam o desenho. A grade de limiares do
braço do modelo é mantida **congelada**, e apenas a grade do braço adversário foi
estendida ao extremo conservador, de modo que a alteração beneficie o adversário
e nunca a hipótese; e essa extensão foi executada no **conjunto de validação**,
cujo papel declarado no protocolo é a escolha de limiares operacionais, e não no
conjunto de teste reservado. Além disso, o critério de decisão pré-registrado
exigia dominância a tempo casado em pelo menos duas de três sequências de
validação, e foi atingido em uma de duas, uma vez que uma sequência não produziu
par casado e outra não foi executada por decisão de escopo; este veredito é
declarado como não atingido na forma estrita, com o que de fato se obteve.

> **Procedência.** `docs/METODOLOGIA_pipeline_ML.md` §5 (desenho original de
> três braços e critério de sucesso); `docs/RESULTADOS_E5_ablacao_validacao.md`
> §1 a §4 e §6 (política casada, grades de limiar, transição abrupta na
> sequência Lips, veredito do critério de decisão e limitações);
> `docs/SINTESE_resultados_metodologia.md` §3 (refino de 2026-07-20 e nota do
> E5). Artefatos: `results/benchmark/ablation_matched.csv` e
> `results/benchmark/e5_ablation/{FlowerPan,Lips}/{curve,runs}.csv` (não
> versionados). Scripts: `src/scripts/benchmark/ablation_attrib.py`,
> `analyze_ablation.py` e `run_e5_validation.sh`.

---

## 6.8 A substituição direta do podador nativo

A comparação contra os presets de velocidade do codificador de referência não é
uma comparação de categoria correta, pois cada preset altera simultaneamente
dezenas de heurísticas não aprendidas, além da rede convolucional nativa de poda
de partição. Para isolar o mérito do podador aprendido foi desenvolvida a
**substituição direta** (do inglês *swap*): fixa-se o nível de velocidade,
**desliga-se explicitamente a rede convolucional nativa** por variável de
ambiente e instala-se o podador aprendido como **único** podador de partição
intraquadro. Deste modo, a diferença medida entre as duas configurações é
atribuível ao algoritmo de decisão, e não ao conjunto de heurísticas que o
acompanha.

O desenho exige uma segunda neutralização, e omiti-la produziu um fator de
confusão real nesta investigação. Como a compilação de teste executa o podador
pré-busca **incondicionalmente** em quadros intraquadro, medir um podador
pós-NONE sem desativar o pré-busca mede a pilha dos dois, e não o podador
isolado. A neutralização é feita fixando os limiares do outro podador em valores
inatingíveis, e é executada desde o próprio roteiro de medição, como **única
variável alterada** em relação à configuração de referência.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §4 (substituição
> direta com `AV1_DISABLE_NATIVE_CNN=1`) e §5 (substituição limpa com o podador
> pré-busca neutralizado desde o roteiro); `docs/ANDAMENTO_tese.md` §8.1 e §8.2;
> `docs/RESULTADOS_fase6_swap_h9c.md`. Artefatos:
> `results/benchmark/fase6*/` (não versionados). Scripts:
> `src/scripts/benchmark/analyze_frontier.py`.

---

## 6.9 A decomposição com neutralização de alavanca

Quando dois podadores coexistem no mesmo binário, o ganho medido é o da pilha, e
a repartição entre as alavancas não é observável sem um desenho próprio. A
**decomposição com neutralização de alavanca** mede três configurações sobre a
mesma sequência e a mesma âncora: cada podador isolado, com o outro neutralizado
por limiares inatingíveis, e a pilha completa. A aplicação deste desenho ao
podador pós-NONE quantificou o fator de confusão descrito na seção anterior: o
podador pré-busca sozinho entrega **17,10%** de redução de tempo, a pilha entrega
**17,36%**, e o podador pós-NONE **isolado** entrega apenas **4,23%**, ou seja,
de 82% a 96% da redução de tempo antes atribuída a ele provinha, na verdade, da
alavanca que rodava por baixo.

O mesmo desenho estabelece a forma correta de avaliar uma alavanca nova. Uma
alavanca acrescentada a um podador já implantado deve ser medida **marginalmente
sobre ele** e comparada ao **preço do tempo** que se obteria simplesmente
afrouxando o limiar do podador já existente, preço estimado por interpolação do
segmento correspondente da curva de limiares. Além disso, a integridade do
binário é verificada antes de cada campanha: com a alavanca nova desligada, o
codificador reproduz a configuração de referência de forma **byte a byte
idêntica**, o que foi confirmado com **1 574 775 bytes** de fluxo e PSNR-Y de
**40,9720 dB**. Deste modo, qualquer diferença medida é da alavanca, e não de
uma alteração inadvertida do caminho de código.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §5 (decomposição com
> o podador pré-busca neutralizado por limiares 2/2/−1) e §5-quater (medição
> marginal, preço do tempo por interpolação do segmento de limiares e
> verificação de integridade byte a byte); `docs/ANDAMENTO_tese.md` §8.1 e §8.3;
> `docs/RESULTADOS_H9d_CTC.md`; `docs/RESULTADOS_H9d_etapa3_encoder.md`.
> Scripts: `src/scripts/benchmark/ctc_h9d_marginal.py` e
> `src/scripts/benchmark/h9d_tau_curve.py`.

---

## 6.10 A simulação oráculo e o alerta metodológico

Cada campanha de codificação real custa horas de processamento, e por isso todas
as etapas de validação anteriores à integração em C são resolvidas por
**simulação oráculo**: a política de poda é reproduzida em Python sobre o dado
de referência do conjunto reservado, nó a nó, estimando o que a busca teria
economizado e o risco que teria assumido, sem codificar um único quadro. Três
grandezas são produzidas por ponto de operação. A **redução de nós** é a fração
dos nós registrados cuja busca seria eliminada, e é mantida apenas por
comparabilidade histórica. A **redução de custo** pondera cada nó pelo trabalho
que ele realmente faz, uma vez que a busca completa avalia dez formas
candidatas por nó acima de 8×8 amostras — a contagem verificada no código e
resolvida na Seção 1.1 da Metodologia —, cada uma proporcional ao número de
amostras do bloco; esta é a grandeza que se correlaciona com o tempo. O **risco**
mede quantas vezes a política contradiz a decisão verdadeira da busca de
taxa-distorção.

A medição de risco por contagem foi posteriormente substituída por um **crivo
ponderado por *regret***, pois tratar como equivalentes uma poda errada em um
bloco liso, cujo custo é praticamente nulo, e uma poda errada em um bloco
texturizado, cujo custo é elevado, é indefensável. O crivo reporta a fração de
sobrecarga de taxa-distorção, ou seja, a soma dos *regrets* normalizada pelo
custo total, ao longo de uma fronteira de redução de custo, evitando comparar
soluções em extremos opostos das suas faixas; ele foi aplicado a **792.840 nós
de decisão** das seis sequências de validação e teste, com os modelos treinados
nas dez restantes.

Duas propriedades desta simulação são carregadas como ressalva até o fim do
texto. A primeira é o **fator sistemático de superestimação**: a redução de custo
do oráculo supera o tempo de parede medido em cerca de **cinco vezes**, sendo
observado historicamente que 35% de redução de custo simulada correspondem a
cerca de 7% de tempo real. Neste caso, o que vale é a **margem relativa** entre
soluções, e nunca a magnitude absoluta. A segunda propriedade é mais severa e
constitui o alerta metodológico mais forte desta tese: a **ordenação relativa
entre modelos pode se inverter** entre a simulação e o codificador real. A rede
de grafos da Seção 6.4 supera o perceptrone implantado por ampla margem no
oráculo e, medida no codificador pelo mesmo gancho de reinjeção de
probabilidades, perde para ele por cerca de **duas vezes** em taxa BD ao longo de
toda a varredura de limiares; e o próprio crivo ponderado por *regret* diverge do
codificador justamente neste único par com terreno de comparação limpo. Então,
uma avaliação fora do codificador **filtra perdedores, mas não coroa
vencedores**, e as margens relativas obtidas fora dele são **indício e não prova**
de ordenação. Duas consequências práticas decorrem disto e são respeitadas em
todo o texto: o codificador permanece o árbitro final de qualquer veredito, e a
rejeição de uma via no oráculo é tratada como **assimetria de custo**, ou seja,
como razão econômica para não pagar a campanha real, e não como implicação
lógica de que a via também falharia no codificador.

> **Procedência.** `src/scripts/partition_model/simulate_pruning.py` (política
> replicada, modelo de custo por candidatos e por área, e as três grandezas
> reportadas); `docs/RESULTADOS_oraculo_regret.md` §1 a §4 e §6 (crítica da
> métrica por contagem, definição do crivo, cobertura de 792.840 nós, limite do
> crivo e divergência no par com terreno limpo);
> `docs/RESULTADOS_approachB.md` §5 a §7 (inversão de ordenação medida por
> reinjeção de probabilidades); `docs/SINTESE_resultados_metodologia.md` §9
> (fator de cerca de cinco vezes e ameaça de inversão); `docs/ANDAMENTO_tese.md`
> §4 e §4.1. Artefatos: `results/models/oracle_regret/{report,ranking.csv,
> frontier.csv}` e `results/benchmark/gnn_frontier/frontier_Jockey.csv` (não
> versionado). Scripts: `oracle_regret.py`, `gnn_replay.py` e
> `src/scripts/benchmark/gnn_replay_bench.py`.

---

## 6.11 Síntese e encaminhamento

As arquiteturas investigadas cobrem, então, quatro famílias distintas sobre o
mesmo problema e o mesmo conjunto de dados: um modelo substituto convolucional
multinível que espelha a hierarquia de particionamento, perceptrones rasos por
tamanho de bloco em quatro variantes de entrada e de ação, uma rede de grafos
com passagem de mensagens sobre a árvore do superbloco e uma reformulação por
regressão do custo de podar. Todas foram treinadas sobre a mesma partição
congelada por sequência e avaliadas pelo mesmo crivo, o que torna a comparação
entre elas uma comparação de formulações, e não de orçamentos de treino.

A metodologia de atribuição é o que permite que os números do próximo capítulo
sejam lidos como afirmações sobre modelos, e não sobre políticas de poda. A
ablação em três braços sob política casada isola a fonte do escore; a
substituição direta com neutralização explícita mede um podador isoladamente
contra o podador nativo; a decomposição com neutralização de alavanca separa o
ganho de podadores empilhados; e a simulação oráculo triaria candidatos a um
custo baixo, sob a ressalva medida de que pode inverter a ordenação relativa
entre modelos competitivos. Por fim, os resultados obtidos por cada um destes
desenhos, incluindo as cinco vias encerradas com resultado negativo no conjunto
do trabalho — contagem geral do projeto, e não as cinco tentativas
independentes específicas do domínio de pixels —, são apresentados no próximo
capítulo.

> **Procedência.** Consolidação das notas das Seções 6.1 a 6.10; nenhum valor
> novo é introduzido nesta síntese.
