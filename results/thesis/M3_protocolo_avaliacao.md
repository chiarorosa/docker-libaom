# 3 Protocolo de avaliação congelado e critérios de decisão em cascata

Esta seção apresenta o protocolo sob o qual todas as medições desta tese foram
realizadas: a partição das sequências, a cobertura de codificação, as métricas
reportadas, o piso de ruído do tempo de parede e a cascata de critérios
quantitativos que autorizou, ou interrompeu, cada etapa da investigação. O
protocolo foi fixado **antes** de existir qualquer resultado da família de
soluções que a tese defende, e é este ordenamento temporal — não a qualidade dos
números que vieram depois — que sustenta a validade das conclusões apresentadas
no capítulo de resultados.

---

## 3.1 O congelamento por commit, antes de qualquer medição de teste

O protocolo de avaliação foi congelado em um documento versionado e datado
(`docs/PROTOCOLO_avaliacao.md`), registrado no repositório pelo commit
`aabbbee` em 9 de julho de 2026, antes de qualquer codificação do conjunto de
teste. Este documento fixa, *a priori*, a partição de dados, as sequências de
teste, os pontos de quantização, a contagem de quadros, os binários de
codificação e as métricas, e declara no seu próprio corpo que qualquer desvio
posterior deve ser justificado por escrito em commit subsequente.

O objetivo do congelamento é, justamente, eliminar a objeção de seleção *a
posteriori*: uma investigação que percorre dezenas de configurações produz, por
construção, um espectro largo de resultados, e a escolha do recorte mais
favorável **depois** de ver os números é indistinguível, na leitura final, de uma
descoberta genuína. Fixar o recorte antes torna esta confusão impossível, pois os
números da tese saem exclusivamente do conjunto de teste reservado, que nunca
participou de nenhuma decisão de modelo, de atributo ou de limiar.

O único adendo posterior, de 19 de julho de 2026, **não altera nada do corpo**:
registra apenas que a qualificação "não-implantável", atribuída ao podador
pós-NONE, foi superada pelos fatos. O item original permanece escrito como
estava, precisamente porque a defesa contra seleção *a posteriori* exige que o
texto congelado não seja reescrito à luz do que se aprendeu depois.

> **Procedência.** Documento-fonte: `docs/PROTOCOLO_avaliacao.md` (corpo e adendo
> de 2026-07-19); `docs/SINTESE_resultados_metodologia.md` §2.3. Commit de
> congelamento: `aabbbeee72de2b5f2bac5b8ac6283e36cb487c02`, de 2026-07-09.
> *Registre-se uma correção factual:* documentos internos descrevem o
> congelamento como feito por "commit assinado"; a verificação do repositório
> mostra que o commit **não** possui assinatura criptográfica. O que existe, e é o
> que o argumento requer, é um commit datado e imutável no histórico.

---

## 3.2 A partição por sequência, sem vazamento

A partição dos dados foi feita **por sequência**, e não por quadro ou por bloco,
de modo que nenhuma sequência do conjunto de teste reservado aparece no treino ou
na validação. Este cuidado é essencial no domínio de vídeo, pois quadros vizinhos
de uma mesma sequência compartilham conteúdo, textura e estatística de
quantização, e uma partição temporal permitiria ao modelo memorizar a sequência
em vez de aprender a decisão de particionamento.

As dezesseis sequências UVG em resolução 4K disponíveis em `src/samples/` foram
distribuídas em três conjuntos, nomeados no protocolo congelado:

- **Treino (10 sequências)** – Beauty, Bosphorus, CityAlley, FlowerFocus,
  FlowerKids, ReadySetGo, ShakeNDry, SunBath, Twilight e YachtRide, utilizadas
  para o treino do modelo substituto e do modelo estudante.
- **Validação (3 sequências)** – HoneyBee, FlowerPan e Lips, utilizadas para a
  seleção de modelo, de atributos e dos limiares operacionais.
- **Teste reservado (3 sequências)** – Jockey, RaceNight e RiverBank, utilizadas
  exclusivamente para os números finais da tese.

A escolha das três sequências de teste foi justificada por cobertura de
fenômenos, também *a priori*: Jockey aporta movimento rápido, e foi mantida por
continuidade com a exploração anterior; RaceNight aporta baixa luminosidade,
movimento e grão; e RiverBank aporta textura natural detalhada em tomada
panorâmica. Deste modo, o conjunto reservado cobre movimento, ruído e textura, os
três regimes que mais afetam a decisão de particionamento.

A disciplina da partição foi observada mesmo quando isso custou resultado. A
grade de limiares do braço de variância precisou ser estendida para o extremo
conservador, pois as faixas de aceleração saíam disjuntas e não havia par casado
a comparar; a extensão foi feita **na validação**, cujo papel declarado é
exatamente escolher limiares operacionais, e o braço da proposta ficou intocado
na sua grade congelada.

> **Procedência.** Documento-fonte: `docs/PROTOCOLO_avaliacao.md` §1;
> `docs/SINTESE_resultados_metodologia.md` §2.3; `docs/ANDAMENTO_tese.md` §0.2-bis
> (extensão da grade na validação). Artefato: sequências em `src/samples/*.yuv`.
> Commit: `aabbbee`.

---

## 3.3 Cobertura experimental: quantização, quadros e a grade da tese

A cobertura de codificação foi fixada em quatro pontos de quantização,
`cq-level` 20, 32, 43 e 55, com controle de taxa por qualidade constante
(`--end-usage=q`), correspondendo a `base_qindex` igual a quatro vezes o valor de
`cq-level`. Quatro pontos são o mínimo para uma curva de taxa-distorção estável
sob a integração de Bjøntegaard, e a faixa cobre desde alta qualidade até baixa
taxa.

A contagem de quadros varia com o papel do conjunto. Treino e validação usam ao
menos cinco quadros por sequência, com amostragem temporal via `--skip`, o que
basta para gerar milhões de nós de decisão. O teste reservado usa ao menos dez
quadros por sequência, contagem elevada deliberadamente para estabilizar a taxa
BD e corrigir a ressalva de ruído da ablação anterior, que operava com apenas
dois quadros. A extração de rótulos e as medições de tempo são feitas em
`--cpu-used=0`, único regime em que todas as classes de partição são exploradas
por busca de taxa-distorção completa, e com `--threads=1`, para que a medição de
tempo seja determinística.

É preciso distinguir esta grade da especificação das *Common Test Conditions*
(CTC). O cenário universal desta tese reproduz literalmente os parâmetros da §4.1
do documento CTC — `--cpu-used=0 --passes=1 --end-usage=q --kf-min-dist=0
--kf-max-dist=0 --deltaq-mode=0 --enable-tpl-model=0
--enable-keyframe-filtering=0 --obu`, mais a regra de ladrilhamento 4K da Classe
A1 — e adota os **quinze quadros** que a CTC especifica para o cenário
intraquadro, que são a especificação e não um recorte próprio. Duas divergências
são impostas pelo codificador: o `aomenc` do libaom v3.10.0 não expõe `--qp`,
restando `--cq-level` como única escala de quantização, e a opção
`--use-fixed-qp-offsets=1` inexiste neste *build*. Uma terceira é decisão
metodológica declarada: manteve-se a grade 20/32/43/55 da tese, por consistência
com a validação, em vez dos valores de `qindex` exatos do guia CTC. Como todo
quadro é quadro-chave sob `--kf-max-dist=0`, deslocamentos fixos de quantização
entre tipos de quadro não teriam efeito, e a grade é aplicada de forma idêntica a
todas as configurações, inclusive à âncora.

> **Procedência.** Documento-fonte: `docs/PROTOCOLO_avaliacao.md` §2;
> `docs/DECISOES_escopo.md` §2 e §3. Especificação externa:
> `src/samples/aomctc_test_set/CWG-G082_AV2_CTC_v9.pdf` §4 e §4.1. Script de
> reprodução: `src/scripts/benchmark/run_benchmark.py`. Commit: `aabbbee`.

---

## 3.4 Os três pilares de métrica contra a âncora

Todo resultado é reportado em três pilares complementares, sempre contra a mesma
âncora: o libaom v3.10.0 original, compilado a partir da árvore intocada
`src/aom_baseline` como `libaom_perf_anchor`, executando em `--cpu-used=0`. A
âncora é um controle cego, e a compilação padrão da árvore instrumentada, com as
guardas de compilação desligadas, é verificada byte a byte idêntica a ela antes
de qualquer medição.

O primeiro pilar é a **taxa BD** (Bjøntegaard) sobre PSNR-Y, que mede o custo em
eficiência de compressão, sendo melhor quanto menor. O cálculo segue a formulação
padrão: interpola-se `log10` da taxa como função do PSNR por polinômio cúbico
monótono por partes, integra-se a diferença entre as duas curvas sobre a faixa de
PSNR em que ambas se sobrepõem, com cem amostras, e converte-se a média da
diferença logarítmica em diferença percentual de taxa. O PSNR-Y é calculado
**externamente**, decodificando-se o fluxo e comparando-o com a fonte em `numpy`,
o que dispensa um *build* com estatísticas internas e impede que a medição de
tempo seja poluída pela opção `--psnr`.

O segundo pilar é a **redução de tempo**, denotada TS, que expressa em pontos
percentuais a economia de tempo de parede em relação à âncora. O terceiro é a
**aceleração**, razão entre o tempo da âncora e o tempo da configuração, agregada
como média sobre os pontos de quantização das razões por ponto. A estes três
soma-se uma métrica auxiliar de compromisso, **TS/BD**, que é o número de pontos
de redução de tempo obtidos por ponto percentual de taxa BD cedido, e responde,
em um único número, quanto tempo se compra por unidade de qualidade entregue.
Uma vez que cada limiar produz o seu próprio ponto de operação, a comparação
entre métodos é feita a **aceleração casada**, interpolando linearmente a taxa BD
de cada método numa grade comum de acelerações; o método com menor taxa BD em
todos os níveis comparáveis é o dominante.

> **Procedência.** Documento-fonte: `docs/PROTOCOLO_avaliacao.md` §3 e §4;
> `docs/SINTESE_resultados_metodologia.md` §2.4; `docs/DECISOES_escopo.md` §1
> (decisão por PSNR-Y apenas). Scripts de reprodução:
> `src/scripts/benchmark/bd_rate.py` (taxa BD),
> `src/scripts/benchmark/run_benchmark.py` (codificação, decodificação e PSNR-Y),
> `src/scripts/benchmark/analyze_ablation.py` (comparação a aceleração casada).

### 3.4.1 As métricas do crivo offline e a sua não conversão nos três pilares

Os três pilares acima são medidos **dentro do codificador**. A investigação
recorre, porém, a um segundo conjunto de métricas, computadas **fora** dele, por
simulação sobre árvores de decisão previamente gravadas, e a distinção entre os
dois conjuntos precisa ser mantida com rigor, sob pena de se atribuir a um
resultado offline uma consequência que ele não mede.

A **redução de custo de busca**, denotada `cost_red` e reportada em porcentagem,
é a métrica de economia do crivo offline. Ela não é tempo de parede nem
estimativa de tempo de parede: é um **contador analítico de trabalho**, definido
como o número de candidatos de forma que uma busca completa avaliaria no nó,
ponderado pela área do bloco em pixels. Na formulação empregada, o custo de um
nó de dimensão *n* é o produto do número de candidatos por *n*², com nove
candidatos para blocos de 64, 32 e 16 px e quatro para blocos de 8 px, e a
redução é a diferença percentual entre o custo somado sobre os nós que a
política visita e o custo da busca completa. A grandeza foi construída para
**casar pontos de operação** entre candidatos — comparar o dano de políticas que
podam quantidades diferentes de trabalho não seria informativo —, e não para
prever a economia de tempo que o codificador realizaria.

A **fração de perda de otimalidade**, `reg_frac`, é a métrica de dano do mesmo
crivo, definida na Seção 1.4 do Capítulo de Resultados. Cabe registrar aqui uma
propriedade do seu denominador que restringe a sua leitura: a soma percorre os
nós de decisão dos três níveis da hierarquia — 64, 32 e 16 px —, os quais
**recobrem a mesma área da imagem**, ao passo que a codificação efetiva emprega
apenas um nível por região. Um superbloco contribui, assim, com vinte e um nós
para o denominador, dos quais somente um subconjunto disjunto sobrevive na
partição escolhida. O denominador não é, portanto, o custo de codificar o
quadro, e sim a soma de custos hipotéticos mutuamente excludentes, o que o torna
sistematicamente maior e, por consequência, torna `reg_frac` sistematicamente
menor do que qualquer perda de eficiência observável no codificador.

Uma segunda razão, de natureza distinta, separa as duas famílias. A simulação
offline repassa uma árvore gravada e desconta valores registrados: podar um nó
subtrai o sobrecusto que aquele nó apresentava na gravação. No codificador, ao
contrário, podar um nó altera o contexto de tudo o que se segue — a reconstrução
que serve de referência à predição intraquadro dos blocos posteriores, os
contextos de codificação entrópica e a vizinhança causal dos nós vizinhos. Esta
propagação está inteiramente ausente do crivo.

Deste modo, **nenhum fator de conversão entre as duas famílias é postulado nesta
tese, e nenhum resultado offline é apresentado como previsão de taxa BD ou de
tempo**. As métricas offline são empregadas em regime estritamente **ordinal**:
elas ordenam candidatos a um custo de segundos de processamento gráfico, e
apenas os candidatos sobreviventes são levados à medição em codificador, cujo
custo é de horas. Toda razão extraída delas é uma razão entre candidatos ao
mesmo ponto de operação, jamais um valor absoluto de perda. O caso designado B2
ilustra a necessidade desta disciplina: positivo no crivo, com ganho relativo de
1,6× a 2,7×, e declarado imaterial no codificador por situar-se abaixo do piso
de ruído estabelecido na Seção 3.6.

> **Procedência.** Definição de `cost_red`:
> `src/scripts/partition_model/simulate_pruning.py`, dicionário `CANDS` e função
> `node_cost` (linhas 59 a 63), com o comentário de derivação dos nove candidatos
> nas linhas 55 a 58. Níveis de decisão do denominador de `reg_frac`:
> `src/scripts/partition_model/regret.py`, constante `DECISION_DIMS` (linha 16);
> acumulação em `src/scripts/partition_model/oracle_regret.py` (linhas 119, 139 e
> 297). O caráter ordinal do crivo consta de `oracle_regret.py`, docstring do
> módulo (linhas 2 e 18), e a sua aplicação ao caso B2 de
> `docs/RESULTADOS_modelagem_B2_tau_qindex.md` §4.

---

## 3.5 As duas definições de redução de tempo — e qual foi padronizada

Duas definições de redução de tempo coexistem nos documentos deste projeto, e a
sua distinção é indispensável para a leitura correta de todas as tabelas dos
capítulos. Ambas são computadas pelo mesmo script de análise
(`src/scripts/fase6/analyze_frontier.py`), que as reporta lado a lado exatamente
para tornar a divergência visível:

- **Definição canônica** – calcula-se, para cada ponto de quantização, a fração
  de tempo poupada `1 − t_configuração / t_âncora`, tira-se a média sobre os
  quatro pontos e, então, a média sobre as sequências. Cada ponto de quantização
  pesa igualmente.
- **Definição ponderada pelo tempo** – calcula-se, por sequência,
  `1 − Σ t_configuração / Σ t_âncora`, somando os tempos sobre os quatro pontos
  antes de dividir, e tira-se a média sobre as sequências. Neste caso, o
  resultado é dominado pelo ponto de quantização mais custoso, isto é, `cq` 20.

A divergência entre as duas não é decorativa: ela chega a cerca de três pontos
percentuais, magnitude idêntica à das diferenças que os capítulos discutem. No
artefato medido, o *preset* nativo `cpu-used=1` registra 32,59% na definição
canônica e 30,42% na ponderada, uma diferença de 2,17 pontos percentuais; o maior
afastamento observado é de 2,97 pontos percentuais, na configuração de troca do
podador pós-NONE em `cpu-used=2` (40,73% contra 37,76%). O sinal da diferença não
é sequer constante entre configurações, pois o podador agressivo aparece com
31,51% na canônica e 34,14% na ponderada.

A definição **canônica foi padronizada** e é a que a tese adota. Os documentos
mais recentes do projeto já a empregam, e a tabela consolidada de todas as
soluções está integralmente convertida a ela. Cabe destacar, por fim, que as
taxas BD **não** são afetadas por esta escolha, uma vez que dependem apenas de
bytes e de PSNR, que são determinísticos; a ambiguidade é exclusiva do eixo de
tempo.

> **Procedência.** Documento-fonte: `docs/SINTESE_resultados_metodologia.md` §8
> (aviso sobre as duas definições). Script de reprodução:
> `src/scripts/fase6/analyze_frontier.py`, linhas 279–307. Artefato numérico:
> `results/benchmark/fase6_analysis/ts_definitions.csv` (17 configurações).
> Tabela já padronizada na definição canônica: `docs/INVENTARIO_solucoes.md`.

---

## 3.6 O piso de ruído medido e a resolução da comparação pareada

Uma medição de tempo de parede só é interpretável se o seu piso de ruído for
conhecido, pois a alternativa é discutir diferenças que a própria repetição do
experimento produziria. Este piso foi medido diretamente, e não estimado: cinco
repetições intercaladas de três configurações — âncora, podador equilibrado e
*preset* nativo `cpu-used=1` — sobre quatro pontos de quantização, em execução
contínua.

A dispersão do tempo bruto é pequena. O coeficiente de variação mediano é de
**0,28%**, com máximo de 0,64% entre as células medidas. O plano original supunha
um desvio da ordem de 1 a 2% por codificação, inferido de violações de
monotonicidades que valem por construção; a medição direta mostra que esta
estimativa era pessimista por um fator próximo de quatro, o que torna resolvíveis
diversas comparações antes descartadas como indistinguíveis do ruído.

O número que de fato governa a leitura das tabelas é o desvio padrão da redução
de tempo pareada. Para o podador equilibrado, cuja redução medida é de 20,29%, o
desvio sobre as cinco repetições é de **±0,23 pontos percentuais**, com erro
padrão de 0,10 ponto, o que estabelece uma **resolução efetiva de comparação
pareada de aproximadamente 0,46 ponto percentual**, tomada como dois desvios. Para
o *preset* nativo `cpu-used=1`, cuja redução medida é de 32,84%, o desvio é ainda
menor, de ±0,09 ponto, com resolução de cerca de 0,18 ponto. Diferenças de
redução de tempo inferiores a esta resolução não são citadas como positivas em
nenhum ponto desta tese, e as que a superam com folga são reportadas com a
margem explícita — o ganho marginal de 1,02 ponto percentual da segunda solução
implantada, por exemplo, situa-se a cerca de 4,4 desvios acima do ruído.

Uma ressalva acompanha o número e deve ser carregada em toda leitura. As cinco
repetições ocorreram numa **mesma janela contínua de execução**, na mesma
sequência e no mesmo contêiner, sem reinício, de modo que o desvio medido é
intra-execução e não captura deriva entre dias, reinícios de contêiner ou estados
térmicos distintos. Como as campanhas desta tese rodaram em janelas contínuas
análogas, é o desvio pertinente para as comparações internas; mas comparar
números medidos com semanas de intervalo exige cautela adicional. A homogeneidade
dos coeficientes de variação sugere que o valor não é idiossincrático da
sequência medida, ainda que isso não esteja demonstrado em outros conteúdos.

> **Procedência.** Documento-fonte: `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3 e §4.
> Artefato numérico: `results/benchmark/fase6_repeat/raw_results.csv`. Scripts de
> reprodução: `src/scripts/fase6/encode_repeat.py --seq Crosswalk --reps 5` e
> `src/scripts/fase6/report_e3_dec_e2.py`.

---

## 3.7 Os critérios de decisão em cascata e a regra de parada

O fio condutor metodológico desta tese é a auditabilidade da cadeia de decisões, e
o instrumento que a produz é uma cascata de critérios quantitativos, cada um
fixado antes da medição que o resolveria. A economia da cascata é explícita: o
custo caro — integração em C, codificação real e medição de tempo de parede — só
é pago **depois** que o sinal se provou fora do codificador, em simulação de
oráculo sobre nós já extraídos, que custa horas de processamento gráfico em vez
de dias de codificação. Os degraus, na ordem em que foram percorridos, são os
seguintes.

- **Critério 2 — sinal fora do codificador.** Exigia que o conjunto de atributos
  proposto superasse a variância trivial na simulação de oráculo, a risco casado,
  por margem clara. Foi atingido: em redução de custo a perdas de SPLIT de
  0,5%, 1% e 2%, a variância entrega 0 em todos os pontos, os vinte e quatro
  descritores de luminância entregam 10,1%, 15,3% e 18,9%, e o conjunto completo
  entrega 15,7%, 20,1% e 24,9%. Estes valores são obtidos por uma regra de risco
  casado: para cada subconjunto, toma-se, na varredura de limiares, a linha cujo
  `split_lost` fica imediatamente abaixo de cada patamar de risco declarado, e o
  `cost_red` correspondente é o valor reportado.
- **Critério 3 — validação.** Sobre HoneyBee, FlowerPan e Lips, o conjunto
  completo alcança de 55% a 58% de redução de custo a perda de SPLIT de até 1%,
  com risco mínimo alcançável de 0,19%, enquanto a variância não opera em regime
  de risco baixo, tendo 22,6% como menor perda alcançável.
- **Critério 4 — integração em C.** Exigia paridade bit a bit entre a extração de
  atributos em C e em Python nos trinta e seis atributos, ausência de efeito com
  a guarda desligada, verificada por identidade byte a byte do fluxo, e aprovação
  nos testes intraquadro do codificador. Foi atingido nos três itens.
- **Critério 5 — o número da tese.** Exigia que, no conjunto de teste reservado, a
  curva do modelo dominasse a variância em taxa BD a aceleração casada, por
  margem além do ruído, em ao menos duas das três sequências.

A **regra de parada foi pré-registrada** no próprio protocolo congelado: caso o
critério 2 falhasse, a tese reportaria o diagnóstico e a caracterização de limite
superior, e **não haveria integração em C**. A reformulação por regressão de
custo de poda, rejeitada no critério 3, não teve a sua codificação real
executada, decisão reenquadrada posteriormente como assimetria de custo e não
como implicação lógica, pois ficou estabelecido que a simulação de oráculo pode
**inverter** a ordenação relativa entre podadores no codificador real.

O critério 5 merece registro explícito, pela regra de honestidade dos vereditos:
ele **não foi atingido na sua forma estrita**. No conjunto de teste reservado as
faixas de aceleração dos dois braços saíram disjuntas nas três sequências, de
modo que não existia par casado a comparar. A campanha que produziu a primeira
comparação a tempo casado da tese foi executada na validação, com 144
codificações e cerca de vinte e duas horas de execução, e obteve dominância em
uma de duas sequências: na FlowerPan o modelo vence a variância por fatores de
4,6 e 1,85 em taxa BD a tempo casado, e na Lips a variância não possui ponto de
operação na região implantável, o que impede o par casado. A terceira sequência
foi cortada por decisão de escopo, por ser aquela em que a grade de limiares
original havia sido calibrada.

Cabe acrescentar que o ponto de operação da tese **é** um limiar sobre a saída
probabilística do modelo, e que a leitura deste limiar como grau de confiança foi
verificada, e não suposta. Sobre 1 816 393 nós de decisão do conjunto de teste
reservado, o erro esperado de calibração da classe predita é de **0,0112**, e a
precisão real no limiar implantado de 0,90 é de 95,6% para a classe NONE e de
96,5% para a classe SPLIT. Esta verificação é análise posterior, que não
realimenta nenhuma decisão de projeto.

> **Procedência.** Documento-fonte: `docs/PROTOCOLO_avaliacao.md` §6;
> `docs/ANDAMENTO_tese.md` §2 (status por fase) e §3 (veredito do critério 2);
> `docs/RESULTADOS_calibracao.md` §2; `docs/DECISOES_escopo.md` (corte da terceira
> sequência); `docs/SINTESE_resultados_metodologia.md` §9 (inversão de ordenação
> pela simulação de oráculo). Artefatos numéricos: `results/models/gate2_final_sweep.csv`
> (varredura de τ, de onde os valores do critério 2 são selecionados pela regra
> de risco casado descrita acima; `results/models/gate2_final.csv` reúne uma
> agregação distinta, por limiares fixos de perda de SPLIT, e não é a fonte
> destes números); `results/models/student_h9a/gate4_evidence.txt`;
> `results/benchmark/e5_ablation/<seq>/curve.csv`;
> `results/models/student_h9a/calibration/`. Scripts de reprodução:
> `src/scripts/partition_model/calibration.py`;
> `src/scripts/benchmark/run_e5_validation.sh`;
> `src/scripts/benchmark/ablation_attrib.py`.

---

## 3.8 Os dois cenários experimentais e a sua função distinta

O protocolo opera sobre dois cenários experimentais, cuja separação é estrutural e
não deve ser dissolvida na leitura, pois eles não medem a mesma coisa.

O primeiro é o **universo do próprio aprendizado de máquina**: a partição de dez,
três e três sobre as dezesseis sequências UVG em 4K, com a grade de quatro pontos
de quantização e dez quadros no conjunto reservado. A sua função é **validar e
atribuir**. É nele que se prova que a redução de tempo é atribuível à seleção de
nós feita pelo modelo, e não à política de poda que o envolve, pois a ablação de
atribuição mantém a política, o codificador e a sequência fixos, trocando
**apenas a fonte da pontuação** entre o modelo, a variância e uma pontuação aleatória.
Sem este cenário, um ganho de tempo seria apenas um ganho de tempo; com ele, o
ganho tem uma causa medida.

O segundo é a **CTC**, com as oito sequências da Classe A1 em 4K e dez bits, em
configuração intraquadro e quinze quadros. A sua função é produzir os
**resultados finais**, comparáveis à literatura da área e medidos contra o próprio
botão de velocidade do codificador, isto é, contra os *presets* nativos. Por outro
lado, é um cenário que **não** isola mérito, pois os *presets* nativos misturam a
rede convolucional intraquadro com dezenas de heurísticas que não são de
aprendizado de máquina; uma comparação de categoria correta exige, então,
desligar a rede nativa e colocar o modelo proposto como único podador.

Confundir os dois cenários produz exatamente as duas leituras equivocadas que esta
tese precisa evitar: tomar a validação metodológica por superioridade prática, ou
tomar a derrota na fronteira agregada da CTC por ausência de contribuição. As
perguntas são separadas — *o ganho é atribuível ao modelo?* e *o ganho é útil ao
praticante?* — e cada uma tem o seu próprio conjunto de sequências, a sua própria
contagem de quadros e o seu próprio referencial de comparação.

> **Procedência.** Documento-fonte: `results/thesis/00_PLANO_capitulos.md` §4;
> `docs/SINTESE_resultados_metodologia.md` §4 (validação universal e comparação por
> troca do podador); `docs/DECISOES_escopo.md` §2 e §3. Artefatos numéricos:
> `results/benchmark/fase6/raw_results.csv`,
> `results/benchmark/fase6_swap/raw_results.csv`,
> `results/benchmark/fase6_swap_h9c/raw_results.csv`. Script de reprodução:
> `src/scripts/fase6/analyze_frontier.py`.

---

## 3.9 Síntese e encaminhamento

O protocolo apresentado nesta seção define o que foi medido, contra o quê, com que
precisão e sob que condição de parada: a partição por sequência sem vazamento, a
cobertura de quatro pontos de quantização, a âncora em busca completa de
taxa-distorção, os três pilares de métrica com a definição canônica de redução de
tempo, o piso de ruído de aproximadamente 0,46 ponto percentual na comparação
pareada e a cascata de critérios que autorizou cada etapa. Todos esses elementos
foram fixados antes das medições que decidem, e é esta anterioridade que permite
apresentar os resultados — inclusive os negativos, inclusive o critério não
atingido na forma estrita — como resultados, e não como recorte.

O protocolo é, por outro lado, neutro quanto ao conteúdo da decisão: ele diz como
se mede, não o que o modelo observa nem o que ele faz com o que observa. A seção
seguinte apresenta justamente isso, o projeto de atributos e a política de poda —
quais grandezas são extraídas de cada nó de particionamento, quanto custa
obtê-las dentro do codificador, e como a saída probabilística do modelo é
convertida em ações de poda por limiares que traçam a curva de compromisso entre
redução de tempo e taxa BD.

> **Procedência.** Síntese das subseções 3.1 a 3.8; nenhum número novo é
> introduzido. Encaminhamento para `results/thesis/M4_atributos_e_politica.md`,
> conforme `results/thesis/00_PLANO_capitulos.md` §3.
