# 6. Análise integrada: fronteira de compromisso e as três conclusões

Esta seção apresenta o fechamento do Capítulo de Resultados: a análise
integrada que reúne, sobre a mesma régua de comparação, as três soluções
implantadas e medidas nesta tese — H9a, H9c e H9d — e as contrapõe ao próprio
botão de velocidade do codificador de referência. São apresentados, nesta
ordem, a fronteira de compromisso global, com a ressalva obrigatória sobre a
sua cobertura experimental; as três conclusões que a fronteira sustenta, a
terceira no enunciado corrigido; o argumento transversal de custo
computacional; e a síntese do que sobreviveu. A seção fecha com o
encaminhamento para as ameaças à validade que qualificam esta leitura.

---

## 6.1 A fronteira de compromisso global

A fronteira de compromisso global entre taxa BD e redução de tempo foi
construída sobre três sequências da grade das condições comuns de teste (do
inglês *Common Test Conditions* – CTC) — BoxingPractice, FoodMarket2 e Tango —, no arranjo de substituição direta que isola cada podador da rede
convolucional nativa, cobrindo todos os quatro níveis de `cpu-used` (de 0 a 3)
contra a mesma âncora libaom `cpu-used=0`. A definição de redução de tempo
empregada é a canônica, adotada por todos os documentos mais recentes do
projeto: a média, sobre os quatro pontos de quantização, da razão entre o tempo
economizado e o tempo da âncora, calculada primeiro por sequência e depois pela
média entre sequências. Esta cobertura de três sequências, e não das oito da
CTC completa, é uma limitação declarada da figura, que reflete o custo de
campanha da análise de fronteira, executada antes de o H9d existir como
solução. Os valores desta fronteira são específicos das três sequências que a
compõem e não devem ser confundidos com os valores canônicos sobre as oito
sequências da CTC citados adiante para os mesmos presets nativos: a diferença
é de amostra, não de definição — ambas usam a redução de tempo canônica.

Os pontos não dominados da fronteira, do menor para o maior custo em taxa BD,
são os seguintes: o H9c a τ = 0,95 em `cpu-used=0`, a 0,21% de taxa BD por 13,9%
de redução de tempo; o H9c a τ = 0,90 no mesmo preset, a 0,23% por 15,0%; o
preset nativo `cpu-used=1`, a 0,37% por 28,6%; o H9c a τ = 0,90 empilhado no
preset 1, a 0,39% por 29,6%; o preset nativo `cpu-used=2`, a 0,41% por 38,2% —
o ponto de maior razão entre redução de tempo e taxa BD de toda a fronteira, com
valor de 94, e por isso o pico de eficiência da solução nativa; o H9c empilhado
no preset 2, a 0,44% por 39,1%; o H9a balanceado empilhado no preset 2, a 1,13%
por 47,1%; o H9a agressivo empilhado nos presets 1 e 2, entre 1,9% e 2,0% de
taxa BD por 52% a 60% de redução de tempo; o preset nativo `cpu-used=3`, a
2,80% por 66,4%; e, por fim, o H9c e o H9a no preset 3, entre 3,5% e 4,8% de
taxa BD por 70% a 78% de redução de tempo.

Esta fronteira **não contém os pontos do H9d**, e a ausência precisa de ser lida
com precisão. O H9d foi medido depois da análise de fronteira global, sobre um
arranjo distinto — as oito sequências da CTC, no ponto de inserção pós-NONE,
empilhado sobre o H9a —, e a sua contribuição marginal, apresentada na Seção 4
deste capítulo, tem métrica própria: a comparação contra a curva de limiares do
H9a, e não contra a fronteira de todos os presets e todos os níveis de
`cpu-used`. Recompor a fronteira global com os pontos do H9d exigiria
codificá-lo nos demais níveis de `cpu-used`, o que não foi feito. A ausência do
H9d nesta figura, portanto, **não significa dominância**: na sua medição
própria, o H9d é não dominado no sentido de Pareto e vence a curva de limiares
do H9a em seis das oito sequências CTC, duas delas por dominância estrita. A
figura desta seção subrepresenta a segunda solução positiva da tese, e não a
invalida.

> **Procedência.** `docs/ANDAMENTO_tese.md` §8.3 (fronteira Pareto global, três
> sequências, todos os níveis de `cpu-used`); `docs/SINTESE_resultados_metodologia.md`
> §6 e §8 (definição canônica de redução de tempo); `results/thesis/A3_RETRATACOES_E_LACUNAS.md`
> L1 (lacuna sobre a ausência do H9d). Artefatos: `results/benchmark/fase6_swap/`
> e `results/benchmark/fase6_swap_h9c/` (não versionados).

---

## 6.2 Conclusão 1 — nenhum ponto de aprendizado de máquina domina a rede convolucional nativa

A primeira conclusão que a fronteira sustenta é negativa e precisa: nenhum
ponto de aprendizado de máquina, entre os medidos, é estritamente melhor que a
rede convolucional nativa, ou seja, nenhum entrega mais redução de tempo à
mesma taxa BD ou menos. O H9c chega a empatar tecnicamente com a nativa nos
presets 1 e 2 — os dois pontos vizinhos da fronteira trocam uma fração de taxa
BD por uma fração de redução de tempo, sem que um domine o outro —, mas em
nenhum nível de `cpu-used` um ponto de aprendizado de máquina ocupa a posição
que a nativa ocupa nos dois eixos ao mesmo tempo. A rede convolucional nativa
mantém, ao longo de toda a fronteira, o pico de eficiência de compromisso,
medido pela razão entre redução de tempo e taxa BD, que varia de 78 a 94
conforme o preset, sendo 94 o valor do preset `cpu-used=2`, o ponto de maior
eficiência de toda a fronteira global.

Esta leitura se confirma, com amostra maior, na comparação de substituição
direta sobre as oito sequências da CTC apresentada na Seção 3 deste capítulo: o
H9c empata com a rede nativa nos presets 1 e 2 em toda a grade de quantização,
com diferença estatisticamente significativa a favor do H9c apenas no regime de
alta qualidade — onde os podadores nativos que agem depois do `PARTITION_NONE`
estão nativamente desligados em codificação intraquadro —, e perde no preset 3.
Em nenhuma configuração testada, sob nenhum arranjo, um podador aprendido supera
a rede convolucional nativa nos dois eixos do compromisso ao mesmo tempo. O
espaço que resta ao aprendizado de máquina não é superar este pico, e sim
ocupar o que a nativa deixa descoberto — o objeto da Conclusão 2.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §6, Conclusão 1;
> `docs/ANDAMENTO_tese.md` §8.3 (item 1 das três conclusões);
> `results/thesis/R3_h9c.md` §3.3 e §3.7 (empate estatístico por regime de
> quantização sobre as oito sequências CTC, com p ≤ 0,043).

---

## 6.3 Conclusão 2 — o extremo de baixa taxa BD pertence às soluções aprendidas

A segunda conclusão é positiva e delimita com precisão onde está o valor
prático das soluções aprendidas. Os pontos de menor taxa BD de toda a fronteira
global pertencem ao H9c em `cpu-used=0`, entre 0,21% e 0,23% de taxa BD por
13,9% a 15,0% de redução de tempo — uma região de compromisso que a escada de
presets nativos simplesmente não alcança, pois o preset nativo salta de
`cpu-used=0`, sem nenhuma redução de tempo, para `cpu-used=1`, a aproximadamente
32,6% de redução de tempo sob a definição canônica. Todo o regime entre 0% e
32,6% de redução de tempo fica, então, descoberto pela escada discreta do
codificador, e é exatamente este regime que as soluções aprendidas preenchem de
forma contínua, por variação do limiar de decisão, sem retreino e sem
recompilação.

A mesma leitura se confirma, com maior amplitude, na grade completa das oito
sequências da CTC: as configurações do H9c em `cpu-used=0` nunca excedem 0,27%
de taxa BD, entre 0,160% e 0,171% nos dois limiares medidos — menos de um terço
do custo do ponto do H9a implantado, que é de 0,568% —, e chegam a 0,018% na
sequência Crosswalk e a 0,006% na NocturneDance, valores que são, na prática,
aceleração quase gratuita. A conclusão vale, portanto, sobre as oito sequências
da CTC, e não apenas sobre as três da fronteira global ilustrada na Seção 6.1.

Cabe registrar, com o mesmo rigor da Conclusão 1, o que este resultado **não**
afirma. O valor prático medido é a granularidade fina em baixo regime de
aceleração, e não a superação do pico de eficiência da solução nativa: nenhum
ponto do H9c neste regime domina o preset `cpu-used=1` ou `cpu-used=2` nos dois
eixos simultaneamente, e a comparação correta é contra o vazio que a escada
discreta deixa, não contra o pico já ocupado pela nativa. É esta a forma
corrigida da afirmação de suficiência do Capítulo de Metodologia: o contexto de
taxa-distorção barato é suficiente para superar a cota superior do domínio de
pixels sob política casada, mas não se estende a superar o podador nativo na
média da grade CTC.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §6, Conclusão 2;
> `docs/ANDAMENTO_tese.md` §8.3 (item 2); `results/thesis/R3_h9c.md` §3.7
> (extremo de baixa taxa BD sobre 8/8 sequências CTC); `docs/RESULTADOS_BLOCO7_E1_E4.md`
> §1 e §3; `results/thesis/M1_objeto_e_formulacao.md` §1.5 (escada de presets
> nativos e valor canônico do preset `cpu-used=1`).

---

## 6.4 Conclusão 3, no enunciado corrigido — a não-aditividade é sobreposição de ação

A terceira conclusão foi originalmente redigida na forma geral de que
alavancas de poda não se somam, atribuída a um limite informacional
compartilhado entre os podadores. Esta forma **é refutada** pela própria
evidência que este capítulo reúne, e o enunciado corrigido — registrado em
`results/thesis/A3_RETRATACOES_E_LACUNAS.md` como retratação R8 — é o que esta
seção apresenta: a não-aditividade entre podadores **não é um limite
informacional**, é **sobreposição de ação**. Dois podadores competem pelo mesmo
tempo economizável quando retiram da busca o mesmo conjunto de candidatos,
independentemente de partilharem ou não a mesma informação de entrada; e
somam-se quando os conjuntos que retiram são disjuntos.

A primeira prova é a fração do tempo indevidamente atribuída quando duas
alavancas de mesma ação se empilham. O H9a e o H9c decidem em pontos
diferentes da dimensão de *quando* — um antes de qualquer avaliação de
taxa-distorção, o outro depois de conhecer o custo real do `PARTITION_NONE` —,
mas ocupam a mesma posição na dimensão de *o quê*, pois ambos perguntam se o
nó pode ser encerrado ali. A campanha E4 mediu esta sobreposição sobre quatro
sequências da CTC: em média, **64%** da redução de tempo atribuída ao H9c
isolado era, na verdade, o H9a rodando nos seus limiares padrão, com dispersão
de 28% na Tango a 95% na TimeLapse. A decomposição de três pernas — H9a puro,
H9c puro e o termo de interação — fecha o balanço e mede a interação em **−1,9
ponto percentual** na média: cerca de **12% do ganho potencial evapora na
sobreposição**.

A segunda prova é o H9d somando sobre o H9a com informação idêntica à do H9c,
porque a sua ação é disjunta. O H9d partilha com o H9c exatamente o mesmo
ponto de inserção, pós-NONE, e exatamente o mesmo vetor de trinta e nove
atributos. E, ainda assim, o H9d soma **+1,02 ponto percentual** de redução de
tempo sobre o H9a no ponto implantado, quatro vezes o que o H9c somou (+0,26
ponto percentual), pois a sua pergunta não é *"posso encerrar a busca aqui?"*,
e sim *"vale avaliar as partições estendidas?"* — as formas AB e as formas
4-way, que consomem 34,3% do tempo de busca local e que nenhum dos outros dois
podadores jamais visava. Se a não-aditividade fosse um limite informacional,
este resultado seria impossível: dois podadores com o mesmo ponto de inserção
e o mesmo vetor de entrada produziram contribuições marginais quatro vezes
distintas, e só o conjunto de candidatos que cada um retira da busca difere
entre eles.

O enunciado correto e prescritivo é, portanto: **dois podadores se somam na
medida em que os seus conjuntos de candidatos podados são disjuntos**,
independentemente de partilharem a mesma informação de entrada. Este enunciado
indica onde procurar novos ganhos — em ações ainda não disputadas, e não em
mais informação —, e organiza toda a leitura de composição entre alavancas
apresentada neste capítulo. Por outro lado, uma ressalva precisa de ser
registrada, pois foi obtida por medição posterior e qualifica o enunciado sem o
invalidar: a disjunção de ação **não é uma propriedade absoluta** do H9d, e sim
uma função do ponto de operação sobre o qual o H9a age. Quando o H9a opera no
seu ponto agressivo, o limiar de comprometimento com o `PARTITION_NONE` é tão
frouxo que os nós raramente alcançam o critério de decisão das partições
estendidas, e os dois podadores voltam a disputar o mesmo resíduo: o ganho
marginal do H9d desaba de +1,02 para **+0,17 ponto percentual**, valor abaixo
da resolução temporal medida do experimento, de aproximadamente 0,46 ponto
percentual, que por isso não deve ser lido como positivo. O enunciado ganha,
então, uma cláusula final: dois podadores se somam na medida em que os seus
conjuntos de candidatos podados são disjuntos **no ponto de operação em que
efetivamente rodam**.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §6 (correção do
> enunciado) e §2.8; `docs/ANDAMENTO_tese.md` §8.3 e §0.1;
> `results/thesis/A3_RETRATACOES_E_LACUNAS.md` R8 (forma geral refutada) e R9
> (qualificação pelo ponto de operação); `docs/RESULTADOS_BLOCO7_E1_E4.md` §2
> (E4, 64% de sobreposição); `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §2 e §3.2
> (interação −1,9 pp e resolução temporal ~0,46 pp); `docs/RESULTADOS_H9d_CTC.md`
> §3; `results/thesis/R4_h9d.md` §4.7 e §4.9.

---

## 6.5 O argumento transversal do custo computacional

O argumento transversal desta tese sobre custo computacional compara a
inferência isolada do perceptrone de múltiplas camadas à da rede convolucional
nativa, medida dentro de um encode real e não em laço sintético. O resultado é
que a inferência do perceptrone é cerca de **cinquenta vezes mais barata por
chamada**: aproximadamente 486 nanossegundos contra aproximadamente 24.700
nanossegundos da rede convolucional, diferença estável entre duas execuções
independentes. O escopo desta razão precisa de ser declarado com a mesma
precisão com que ela é citada: mede-se apenas a passagem direta do modelo, e
exclui-se deliberadamente a extração de atributos do perceptrone e a
frequência de invocação de cada modelo, pois a rede convolucional é chamada
uma vez por superbloco enquanto o perceptrone é chamado uma vez por nó. Esta
razão caracteriza o algoritmo de decisão isolado, e não o custo do podador
implantado.

O custo do podador implantado, com a extração de atributos e a inferência
somadas, foi medido diretamente no codificador contra o tempo de parede do
mesmo encode, e é o número que importa para a tese. O caminho da rede
convolucional nativa custa entre 0,16% e 0,21% do tempo de codificação; o
caminho do H9a, extração e inferência incluídas, custa entre 0,26% e 0,32%.
Ambos os podadores, portanto, custam **menos de um terço de um por cento** do
tempo de codificação, e a diferença entre eles é imaterial frente ao que a poda
economiza ou desperdiça na busca de taxa-distorção.

A consequência declarada é dupla, e precisa de ser enunciada nos dois
sentidos. Nenhum resultado de taxa BD contra tempo apresentado neste capítulo
precisa ser revisto: os ganhos de redução de tempo vêm inteiramente das
decisões de poda, e não da leveza de cada inferência, uma vez que o custo de
inferência é ruído frente ao custo de busca que a poda evita ou preserva. Ao
mesmo tempo, a alegação de leveza de inferência **sai da lista de vantagens**
desta proposta, pois não é isso que sustenta o ganho medido; mas não se
transforma em desvantagem, uma vez que o custo absoluto, em qualquer dos dois
modelos, é desprezível frente ao tempo total de codificação.

> **Procedência.** `docs/RESULTADOS_microbench_pruner.md` §2 a §4 (inferência
> isolada, ~486 ns contra ~24.700 ns, escopo declarado) e §6.2b e §6.3 (custo
> implantado, ≤0,32% do tempo de codificação); `docs/SINTESE_resultados_metodologia.md`
> §6, argumento transversal de custo de inferência;
> `results/thesis/A3_RETRATACOES_E_LACUNAS.md` R10 (retratação da leveza de
> inferência como vantagem).

---

## 6.6 Síntese do que sobreviveu

Duas soluções positivas foram implantadas em C e medidas sob protocolo CTC
nesta investigação. O **H9a**, a solução principal, decide na chamada de poda
pré-busca, sobre trinta e seis atributos de pixels e de contexto de
taxa-distorção barato, e entrega, no ponto balanceado implantado, 0,568% de
taxa BD por 17,72% de redução de tempo sob a definição canônica, e, no ponto
agressivo, 1,403% por 31,51%. O **H9d**, a segunda solução positiva, é um
complemento do H9a: decide na chamada de poda pós-NONE se vale avaliar as
partições estendidas, e soma **+1,02 ponto percentual** de redução de tempo ao
H9a balanceado ao custo de **+0,018 ponto percentual** de taxa BD, cerca de 3,5
vezes mais barato que afrouxar o limiar do próprio H9a, vencendo esta
alternativa em seis das oito sequências CTC, duas delas por dominância de
Pareto estrita.

O **H9c** está implantado e mede-se bem contra a rede convolucional nativa nos
presets 1 e 2, com vantagem estatisticamente significativa no regime de alta
qualidade, mas **não sobrevive como contribuição autônoma**: a fração
majoritária da redução de tempo que lhe era atribuída pertencia ao H9a rodando
por baixo, conforme a Conclusão 3 estabelece.

Cinco vias foram encerradas nos seus critérios de decisão, cada uma com
resultado negativo medido, e cada uma delimitando por medição uma região do
espaço de soluções que deixa de ser conjectura. O domínio de pixels fecha com
cinco tentativas independentes negativas — o modelo substituto convolucional
por entropia cruzada e por *regret*, a rede de grafos estrutural e as duas
variantes do bloco de atributos D —, nenhuma competindo com o contexto de
taxa-distorção barato do H9a. A reformulação por regressão de *regret* foi
refutada na validação: entregou 0,00% de redução de custo a risco casado,
contra 51,73% do classificador H9a sobre as mesmas entradas. E a rede de
grafos, embora tenha furado a cota superior medida no oráculo por vinte e oito
pontos percentuais, foi refutada no codificador real, onde o H9a a domina por
cerca de duas vezes em taxa BD ao longo de toda a varredura do limiar — o
alerta metodológico mais forte desta tese sobre o estatuto dos indicadores
substitutos.

> **Procedência.** `docs/INVENTARIO_solucoes.md` §7 (resumo por família) e §1
> (âncora e presets nativos); `results/thesis/00_PLANO_capitulos.md` §2;
> `docs/RESULTADOS_H9d_CTC.md` §2 e §4; `results/thesis/R5_resultados_negativos.md`
> §5.5; `results/thesis/A3_RETRATACOES_E_LACUNAS.md` R15 (as cinco tentativas
> independentes negativas do domínio de pixels).

---

## 6.7 Encaminhamento

Esta análise integrada apoia-se sobre decisões de escopo e limitações medidas
que precisam de ser declaradas com o mesmo rigor dos resultados: a cobertura de
três sequências da fronteira global, a resolução temporal do experimento
pareado, a dependência da Conclusão 3 ao ponto de operação, e a natureza dos
indicadores substitutos que a Conclusão 3 e a síntese dos negativos colocam
sob suspeita. Estas ameaças à validade, e os limites de escopo que decorrem
delas, são apresentadas na próxima e última seção deste capítulo.

> **Procedência.** `results/thesis/00_PLANO_capitulos.md` §4; nenhum valor novo
> é introduzido neste encaminhamento.
