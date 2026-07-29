# A3 — Retratações e lacunas

> **Documento de proteção da escrita.** Reúne as afirmações que foram **retiradas ou
> corrigidas** no decorrer da investigação e que, por decisão editorial fixada em
> `00_PLANO_capitulos.md` §6.5, **não podem reaparecer** em nenhum documento dos
> capítulos; e as **lacunas conhecidas**, com o custo de fechá-las quando o projeto o
> registra. Criado em **2026-07-29**, branch `ml-partition-dev`. Nenhum número deste
> documento é estimado: cada valor provém de um documento ou artefato do projeto, citado
> na própria entrada.

Este documento apresenta o registro corretivo do projeto em três partes. A Parte I trata
das afirmações retiradas ou corrigidas, cada uma com o enunciado antigo em forma literal,
a medição que o derrubou, o enunciado vigente e a procedência da correção. A Parte II
trata das lacunas conhecidas e das pendências, com o capítulo afetado e o custo de
fechamento. A Parte III converte as duas primeiras partes em regras imperativas de
redação, que são o instrumento de uso diário durante a escrita.

**Convenção de leitura.** Os enunciados antigos são reproduzidos entre aspas angulares
(«…») e **na redação original**, inclusive quando esta emprega termos que a norma
terminológica da tese hoje proíbe (por exemplo "teto", "gate", "professor"). A citação
literal é deliberada, pois é ela que permite reconhecer a afirmação caso reapareça. Fora
das citações, este documento segue a terminologia fixada em `00_PLANO_capitulos.md` §6.2.

---

# PARTE I — Afirmações retiradas ou corrigidas

Esta parte apresenta vinte e três afirmações que circularam nos documentos do projeto,
em alguns casos já commitadas, e que medições posteriores derrubaram ou obrigaram a
reformular. Elas estão agrupadas por objeto: o domínio de pixels (R1 a R5), a família de
podadores H9 (R6 a R9), o custo computacional (R10), o protocolo (R11 e R12), o
enquadramento da contribuição (R13 e R14), a instrumentação de atributos (R15), os
resultados negativos (R16 a R19) e as suposições de método que a medição substituiu (R20
a R23).

---

## R1 — A saturação dos pixels na variância

- **Enunciado antigo (literal).** «o particionamento satura, no domínio de pixels, numa
  estatística trivial (variância)»; e, na forma da espinha dorsal, «o domínio de pixels
  **satura na variância** […] Isso transforma "o modelo não conseguiu" em "**o sinal não
  está nos pixels**"».
- **Por que caiu.** A ablação que a sustentava é de **dois quadros numa única sequência**
  (Jockey, `cpu-used=0`), e o crivo A5 — 6 sequências, 792 840 nós de decisão — mede o
  contrário: a `reg_frac` a `cost_red` casado de 25% é **0,0573** para a variância e
  **0,0121** para o `pixels24`, ou seja, os descritores de luma batem a variância por
  **4,7×**. Deste modo, a afirmação repousava sobre a medição mais fina do projeto e era
  contradita pela mais larga.
- **Enunciado vigente.** A tese reporta a **hierarquia medida** no crivo A5, do pior para
  o melhor — variância (0,0573) < `convnext_ce` (0,0207) < `pixels24` (0,0121) < H9a
  (0,0036) —, declarando a contradição entre o crivo e a ablação de dois quadros. O
  negativo que se sustenta é mais estreito e mais seguro: **nenhuma via adicional no
  domínio de pixels compete com o conjunto de atributos do H9a**, com cinco tentativas
  independentes negativas.
- **Data e documento.** Retirada em **2026-07-26**; `docs/ANDAMENTO_tese.md` §0.4 e §1
  (item 6 do arco) e §1.1; refino anterior de **2026-07-20** em
  `docs/SINTESE_resultados_metodologia.md` §3; refino final de 2026-07-26 no mesmo §3;
  `docs/RESULTADOS_convnext_regret.md` §5.1; `docs/RESPOSTAS_contra_argumentos_banca.md`
  CB-1 e CB-2.

## R2 — O ConvNeXt como cota superior do domínio de pixels

- **Enunciado antigo (literal).** «O ConvNeXt […] atua como mestre da destilação e,
  medido diretamente por *replay*, estabelece o limite superior de desempenho (H8)»; e
  «É a **referência de limite superior** do domínio de pixels».
- **Por que caiu.** Medido no mesmo crivo, o ConvNeXt de **28,1 M de parâmetros** sobre
  pixels crus **perde para o `pixels24`**, um perceptrão de múltiplas camadas sobre 24
  atributos manuais derivados da *mesma* luminância: **0,0207 contra 0,0121** de
  `reg_frac` em `cost_red` 25%, cerca de **1,7×**. Um modelo batido por outro de acesso
  estritamente menor à informação não delimita cota superior alguma, pois o resultado
  enuncia algo sobre o treino e não sobre os pixels. Some-se que treiná-lo com o alvo de
  *regret* **piorou-o em toda a faixa (1,06× a 3,80×)** e que dobrar a largura de fusão
  altera a perda de validação em **0,16%**, ou seja, capacidade não é a restrição.
- **Enunciado vigente.** O ConvNeXt permanece como **instrumento de diagnóstico** e como
  a tentativa documentada de estabelecer a cota superior, não como a cota. A tese tem uma
  **cota inferior** do domínio de pixels — o `pixels24` — e uma cota superior genuína
  apenas no **oráculo**, a decisão RD-ótima de *regret* zero, que limita qualquer podador
  e não só os de pixels. A cota superior do domínio de pixels permanece **não medida**.
- **Data e documento.** Corrigido em **2026-07-26**; `docs/ANDAMENTO_tese.md` §1.3
  (reescrito nessa data), `docs/SINTESE_resultados_metodologia.md` §3 (refino final) e
  `docs/RESULTADOS_convnext_regret.md` §5.

## R3 — O sobreajuste na seleção do checkpoint do ConvNeXt

- **Enunciado antigo (literal).** «o teto de pixels foi medido com modelo **sobreajustado**
  e objetivo errado», detalhado como «checkpoint selecionado por macro-F1 na época 27, com
  a perda de validação 15% acima do mínimo».
- **Por que caiu.** A alegação vinha de **leitura parcial** do `metrics.csv`. Nas 30
  épocas completas de `results/models/surrogate_real/metrics.csv`, o mínimo de `val_loss`
  (**1,7999**) e o máximo de macro-F1 (**0,2034**) ocorrem **ambos na época 13**, que é
  exatamente a do checkpoint salvo. Neste caso, os dois critérios concordam e a seleção
  estava correta.
- **Enunciado vigente.** O modelo substituto original está **bem selecionado**; ele é
  apenas **fraco em absoluto** (macro-F1 **0,203**) e foi treinado contra o objetivo
  errado. O retreino com o alvo de *regret* justificava-se por esse motivo único, e o
  motivo caiu junto com o resultado.
- **Data e documento.** Corrigido em **2026-07-26**; `docs/ANDAMENTO_tese.md` §0.4 (nota
  "Correção (2026-07-26)", que retifica o commit `0122b53`) e
  `docs/RESULTADOS_convnext_regret.md` §4. O registro em `docs/DECISOES_escopo.md`, item 2
  da tabela de itens em aberto, marca a premissa como **falsa**.

## R4 — A composição disjunta entre o H9a e o `pixels24`

- **Enunciado antigo (literal).** «o podador implantado (H9a) usa contexto de
  taxa-distorção, **não pixels**»; e o rótulo de tabela «H9a (contexto RD grátis)»
  contraposto a «pixels24».
- **Por que caiu.** O layout do vetor está rotulado no próprio código: `A 0..23 pixels`,
  `B 24..31` vizinhança, `C 32..35` quantização e posição (`features.py:196-201`), com
  `H9a = A+B+C` (`features.py:204`) e `pixels24 = list(range(24))` (`features.py:216`).
  Logo, **24 dos 36 atributos do H9a são descritores de luminância**, o H9a **não contém
  nenhuma grandeza de custo de taxa-distorção** — estas são o bloco E, exclusivo do H9c —
  e o `pixels24` é **literalmente o bloco A do H9a**. Os dois braços comparados não são
  conjuntos disjuntos: um contém o outro.
- **Enunciado vigente.** A linha do H9a mede os **12 atributos adicionais** de
  vizinhança, quantização e posição **sobre** os 24 de luminância. A leitura correta é:
  no domínio de pixels, descritores manuais compactos vencem uma rede convolucional
  profunda sobre pixels crus, e o que separa o campeão do resto não é capacidade de
  modelo, e sim **contexto causal de vizinhança**, doze atributos de custo praticamente
  nulo que acrescentam **3,4×** sobre o `pixels24`.
- **Data e documento.** Corrigido em **2026-07-26**;
  `docs/RESULTADOS_auditoria_dominio_pixels.md` §2, §2.2 e §7 (tabela de correções de
  registro), com a nota de composição replicada em `docs/ANDAMENTO_tese.md` §3 e a
  correção em `docs/RESULTADOS_convnext_regret.md` §5. A auditoria registra que **três
  afirmações já commitadas** tiveram de ser corrigidas por este achado.

## R5 — O ganho do contexto sobre os pixels como margem competitiva

- **Enunciado antigo (literal).** «contexto RD grátis supera pixels ~50 % relativo».
- **Por que caiu.** Pelo mesmo achado de composição de R4: como o `pixels24` está contido
  no H9a, a margem não separa duas propostas concorrentes, e sim mede o **retorno
  marginal** de doze atributos acrescentados a um conjunto que já contém os outros
  vinte e quatro.
- **Enunciado vigente.** O ganho é **marginal, não competitivo**, e deve ser reportado
  como retorno marginal de blocos de atributos, na forma da tabela de `reg_frac` do crivo
  A5.
- **Data e documento.** Corrigido em **2026-07-26**;
  `docs/RESULTADOS_auditoria_dominio_pixels.md` §7, aplicado a `docs/RASTREABILIDADE.md`
  §5.3.

## R6 — O H9c como duas a quatro vezes mais eficiente que o H9a

- **Enunciado antigo (literal).** «o H9c é 2–4× mais eficiente que o H9a» e «supera o
  nativo em eficiência».
- **Por que caiu.** As medições `h9c_tau*` habilitavam apenas as variáveis de ambiente do
  H9c e deixavam o estudante H9a nos seus **defaults compilados**
  (`tau_none = tau_split = 0,9`), pois o H9a roda sob um critério puramente geométrico,
  sem sinalizador de habilitação (`partition_strategy.c:2306-2311`). As linhas mediam,
  portanto, **H9a@0,9 + H9c empilhados**. A re-execução com o H9a neutralizado
  (τ = 2/2/−1, única variável alterada) mostra o H9c isolado podando **2,96%** em τ=0,95,
  **4,23%** em τ=0,90 e **9,31%** em τ=0,60 de redução de tempo, contra 17,15%, 17,36% e
  20,53% das linhas confundidas — ou seja, **82% a 96% da redução de tempo atribuída ao
  H9c vinha do H9a**.
- **Enunciado vigente.** O H9c **não é contribuição autônoma**. A decomposição E4
  generaliza o confundimento para quatro sequências: em média **64%** da redução de tempo
  atribuída ao H9c era do H9a, com dispersão de **28%** (Tango) a **95%** (TimeLapse); o
  H9c isolado entrega **5,6%** de redução de tempo, não 14,6%. O que se pode afirmar é
  que, como **substituto direto da rede convolucional nativa** com o H9a neutralizado, o
  H9c empata com ela em `cpu-used` 1 e 2 (a cpu1, nativa 0,368%/28,60% contra
  `h9c_tau95` 0,370%/27,72%) e vence no regime de alta taxa.
- **Data e documento.** Retratação registrada em **2026-07-16** em
  `docs/ANDAMENTO_tese.md` §8.1 e formalizada na **retratação do topo da §7**
  (**2026-07-19**) do mesmo documento; generalização em
  `docs/RESULTADOS_BLOCO7_E1_E4.md` §2 e §2.1 (**2026-07-25**); resultado limpo em
  `docs/RESULTADOS_fase6_swap_h9c.md` e `docs/ANDAMENTO_tese.md` §8.2.

## R7 — A cota superior mais informativa que não se traduziria em vantagem

- **Enunciado antigo (literal).** «mesmo o teto mais informativo testado (rdcost real do
  NONE) não se traduz em vantagem de tempo de parede sobre o já otimizado pruner nativo»;
  e o veredito de seção «implementado, testado, **não sobrevive ao piloto real**».
- **Por que caiu.** O veredito apoiava-se **inteiramente** no piloto Jockey de dois
  quadros, cuja tabela a §8.1 do mesmo documento prova **contaminada** pelo confundimento
  descrito em R6: a linha «H9c (τ=0,95)» mede H9a@0,9 + H9c e não o H9c isolado, de modo
  que as duas linhas da comparação medem coisas diferentes.
- **Enunciado vigente.** A conclusão é **falsa como enunciada**: vale para o piloto
  contaminado, não para o experimento de substituição limpo. Sobre as **8 sequências
  CTC**, o H9c como substituto da rede convolucional nativa **empata na grade completa** e
  **vence no regime de alta taxa** (CQ 20 e 32: p = 0,043 em cpu1 e p = 0,015 em cpu2). O
  texto original da §7 permanece nos documentos como registro histórico da decisão de
  parada tomada em 2026-07-15, que era correta com a informação daquele momento.
- **Data e documento.** Retratado em **2026-07-19**; `docs/ANDAMENTO_tese.md` §7 (bloco
  de retratação no topo) e §8.2, com o resultado consolidado em
  `docs/RESULTADOS_fase6_swap_h9c.md`.

## R8 — O enunciado geral de que alavancas de poda não se somam

- **Enunciado antigo (literal).** «**Levers que disputam a MESMA ação não se somam**: H9a
  (pixels+contexto), H9c (rdcost pós-NONE) e a CNN nativa exploram o mesmo sinal
  correlacionado», na forma geral «os *levers* não se somam (teto informacional)», e a
  leitura de que se trataria da «mesma história de saturação que a Solução 1 estabeleceu
  no domínio de pixels, agora confirmada no domínio RD».
- **Por que caiu.** O H9d **soma +1,02 pp** de redução de tempo sobre o H9a — quatro
  vezes o que o H9c somou (+0,26 pp) — usando **informação idêntica à do H9c**, pois
  reusa o mesmo vetor de 39 atributos e o mesmo gancho. Se a não-aditividade fosse um
  limite **informacional**, isso seria impossível.
- **Enunciado vigente.** A não-aditividade **não é um limite informacional**, é
  **sobreposição de ação**: o H9a e o H9c perguntam ambos "este bloco é fácil, posso parar
  aqui?", e por isso caçam os mesmos blocos lisos; o H9d pergunta "vale avaliar AB e
  4-way?", que é ação disjunta sobre candidatos que consomem **34,3%** do custo de busca.
  O enunciado correto é: **dois podadores se somam na medida em que seus conjuntos de
  candidatos podados são disjuntos**, independentemente de terem ou não a mesma
  informação de entrada. Esta forma é mais estreita, verdadeira e **prescritiva**, pois
  indica que a via para novos ganhos é procurar ações ainda não disputadas, e não mais
  informação. A medição direta da sobreposição está no E4 (64% da redução de tempo
  atribuída ao H9c era do H9a) e na decomposição de três pernas, cuja **interação média é
  de −1,9 pp**, ou seja, cerca de **12% do ganho potencial evapora na sobreposição**.
- **Data e documento.** Corrigido em **2026-07-26**;
  `docs/SINTESE_resultados_metodologia.md` §6 (nota "Correção do enunciado") e §2.8, e
  `docs/ANDAMENTO_tese.md` §8.3 (nota de correção sob a Conclusão 3); evidência em
  `docs/RESULTADOS_BLOCO7_E1_E4.md` §2 e `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §2.

## R9 — A aditividade do H9d como propriedade absoluta da sua ação

- **Enunciado antigo (literal).** «o H9d escapa porque sua ação é **disjunta**», tomado
  como propriedade do mecanismo, e a leitura de que a disjunção de ação explicaria a
  aditividade em qualquer configuração.
- **Por que caiu.** A fronteira de 96 codificações mostra que **o valor marginal do H9d
  desaba conforme a base do H9a fica agressiva**: de **+1,02 pp** de redução de tempo
  sobre a base equilibrada (P_rect) para **+0,17 pp** sobre a base agressiva (A3). Por
  sequência, o ganho supera a resolução temporal medida (~0,46 pp) em **6 de 8** na base
  equilibrada e em apenas **1 de 8** na agressiva, com Neon1224 e Tango **negativos**.
- **Enunciado vigente.** A disjunção de ação **não é absoluta, é dependente do ponto de
  operação**: com τ_none = 0,60 (A3), o H9a encerra a busca tão cedo que os nós nunca
  chegam ao critério de AB e 4-way, e os dois levantes passam a disputar o mesmo resíduo.
  O enunciado ganha uma cláusula: *dois podadores se somam na medida em que seus conjuntos
  de candidatos podados são disjuntos **no ponto de operação em que rodam***. Sobre a base
  agressiva, o H9d é, na prática, **inerte**, e o seu +0,17 pp **não deve ser citado como
  positivo**, pois está abaixo da resolução temporal medida.
- **Data e documento.** Estabelecido em **2026-07-27**;
  `docs/SINTESE_resultados_metodologia.md` §5-quater (tabela da fronteira completa e nota
  "Consequência para §2.8 e a Conclusão 3"), `docs/ANDAMENTO_tese.md` §0.1 e §0.3, e
  `docs/INVENTARIO_solucoes.md` §7 e §8.

## R10 — A leveza de inferência como vantagem

- **Enunciado antigo (literal).** «O modelo aprendido leve executa sua predição a fração
  do custo da CNN de produção — este é o **ângulo de custo defensável**», apoiado na razão
  de que «a **inferência do MLP é ~50× mais barata por chamada**» (≈486 ns contra
  ≈24 700 ns).
- **Por que caiu.** O custo implantado foi medido no codificador: o podador inteiro,
  extração **e** inferência, é **≤0,32% do tempo de codificação** (rede convolucional
  nativa 0,16% a 0,21%; H9a 0,26% a 0,32%). Neste caso, o custo de inferência **não é
  alavanca em direção nenhuma**, pois a redução de tempo vem das decisões de poda, e não
  do preço de cada predição.
- **Enunciado vigente.** Nenhum resultado de taxa BD contra tempo muda, mas **perde-se o
  direito de alegar leveza de inferência como vantagem**, sem que isso vire desvantagem. A
  razão de ~50× por chamada mede o **algoritmo de decisão isolado**, e não o podador
  implantado, e não deve ser citada como vantagem competitiva; a extração também não é
  custo intrínseco da solução, pois parte é leitura gratuita de estado do codificador e
  parte é uma cópia otimizável a quase zero — se algum dia importar, o alvo é a extração
  (210 ms) e não o modelo (29 ms).
- **Data e documento.** Corrigido em **2026-07-19**;
  `docs/RESULTADOS_microbench_pruner.md` §6, com as notas de correção em
  `docs/RESULTADOS_fase6.md` (nota "Correção (2026-07-19)"),
  `docs/SINTESE_resultados_metodologia.md` §6 (argumento transversal) e
  `docs/ANDAMENTO_tese.md` §5 e §6.

## R11 — Os quinze quadros como recorte próprio da tese

- **Enunciado antigo (literal).** «15 quadros, **não os 60 do CTC pleno**», descrito como
  redução metodológica própria e defendido como limitação.
- **Por que caiu.** A CTC §4.1 especifica exatamente esse número para a configuração *All
  Intra*: «*the test following this configuration uses the first 15 frames*» e «*for video
  test data (Class A, Class B and Class G), `--limit=15` should be used*». O changelog da
  versão 7.0 registra a mudança deliberada «*AI frame count: 30 → 15 frames
  (`--limit=15`)*».
- **Enunciado vigente.** Os quinze quadros são **conformidade com a especificação**, e não
  redução: **não há déficit de quadros a defender**. O item entra no texto como **decisão
  de escopo com justificativa normativa**, na seção de ameaças à validade, e nunca como
  limitação.
- **Data e documento.** Corrigido em **2026-07-26**; `docs/DECISOES_escopo.md` §2
  ("Correção de registro"), aplicado em `docs/RESULTADOS_H9d_CTC.md` e
  `docs/RESULTADOS_BLOCO7_E1_E4.md` §4, e registrado em `docs/ANDAMENTO_tese.md` §0.1 e
  §5.

## R12 — As conclusões H1 a H6 sobre o sinal da luminância

- **Enunciado antigo (literal).** «o sinal da luminância parecia limitado (teto ~13–18%)»,
  apresentado como caracterização do que os pixels ofereceriam.
- **Por que caiu.** O conjunto de dados armazenava a luminância como `float32`
  normalizado em [0,1] enquanto os consumidores em Python assumiam `uint8` em [0,255]:
  `features.py` truncava para inteiro, produzindo **blocos todos zero**, e o carregador do
  modelo substituto normalizava outra vez por 255, produzindo entrada quase nula. **Toda a
  cadeia H1 a H6 fora treinada sobre luminância em branco.** O dado bruto estava correto
  (`round(pkl·255)` reproduz o quadro-fonte, com `maxdiff = 0`); apenas os consumidores
  estavam errados.
- **Enunciado vigente.** As conclusões H1 a H6 estão **invalidadas** — eram sobre entrada
  vazia — e todo resultado anterior à correção foi **re-medido**. A cadeia foi retreinada
  sobre pixels reais, com `data._denorm_uint8` como fonte única de verdade, invalidação de
  caches e uma asserção de luminância real (`assert_real_luma`) guardando treino,
  destilação e simulação. A curva de operação válida é a de H7 e H8. A lição
  metodológica — asserir que os dados de treino têm variância não nula — entra no texto
  como parte da defesa de validade.
- **Data e documento.** Defeito descoberto em **2026-07-08**;
  `docs/SINTESE_resultados_metodologia.md` §2.2 e `docs/ANDAMENTO_tese.md` §1 (item 4 do
  arco) e §2; commits `cb9d407` (correção), `63f299c` (retreino sobre luminância real e
  guarda) e `9b2f1d9` (marcação das tabelas antigas como superadas), listados em
  `docs/RASTREABILIDADE.md`.

## R13 — O contexto necessário **e suficiente** para superar a cota de pixels

- **Enunciado antigo (literal).** «necessário **e suficiente** para superar esse teto».
- **Por que caiu.** "Suficiente" não sobreviveu à Fase 6 na média da grade CTC
  (`RESULTADOS_fase6.md:242-246`), pois ali o H9a compete de frente com a rede
  convolucional intra nativa e perde. "Necessário" nunca foi testado como tal.
- **Enunciado vigente.** A suficiência vale **sob política casada contra a cota do domínio
  de pixels**, e não contra o podador nativo. A afirmação **não** se estende a superar o
  podador nativo na média da grade CTC: ali o valor prático medido é a **granularidade
  fina em baixo regime de aceleração** que a escada discreta dos presets não oferece.
- **Data e documento.** Corrigido em **2026-07-19**; `docs/ANDAMENTO_tese.md` §1 (nota
  "Correção (2026-07-19)") e `docs/RESULTADOS_fase6.md`; consolidado na Conclusão 1 de
  `docs/SINTESE_resultados_metodologia.md` §6.

## R14 — O podador implantado como modelo destilado do ConvNeXt

- **Enunciado antigo (literal).** «A heurística implantada é um perceptron multicamadas
  leve, **destilado de um modelo substituto ConvNeXt** e executado pela rotina
  `av1_nn_predict` do próprio libaom a cada nó de particionamento»; e, na nomenclatura de
  artefatos, «Professor da destilação» para o `surrogate_real` e «Estudante implantado»
  para o `student_real`.
- **Por que caiu.** O podador implantado é o `student_h9a`, treinado **diretamente** sobre
  os 36 atributos com entropia cruzada de rótulo duro, **sem** o ConvNeXt no laço. A
  destilação foi um passo da era de pixels (H7), superado. Escrever "destilado de um
  modelo substituto ConvNeXt" na tese seria **erro factual** sobre o artefato final.
- **Enunciado vigente.** O `student_h9a` é treinado sem modelo substituto no laço; o
  `surrogate_real` é **modelo de referência de cota superior** por *replay*, e não
  "professor" do artefato final; o termo "professor" está proscrito pela norma
  terminológica da tese.
- **Data e documento.** Corrigido em **2026-07-19**;
  `docs/METODOLOGIA_pipeline_ML.md` §7 (bloco "NÃO USAR como está", com a redação antiga
  riscada), `docs/RASTREABILIDADE.md` §4 ("Correção de nomenclatura") e
  `docs/SINTESE_resultados_metodologia.md` §3 (ressalva de 2026-07-19).

## R15 — O bloco D como proxy de resíduo de predição intra

- **Enunciado antigo (literal).** o bloco D descrito como «proxy de resíduo intra», isto
  é, «SATD do **resíduo de uma predição intra barata a partir dos vizinhos
  reconstruídos**», com o alerta do próprio plano de que um resíduo de predição
  **constante** teria a mesma variância do bloco e não agregaria.
- **Por que caiu.** A implementação (`features.py:252-260, 305-308`) calcula
  `block_satd(block)`, o Hadamard do **bloco-fonte**, sem predição alguma e sem tocar em
  vizinho: `f[37] = satd / (var·dim² + 1)` é razão entre duas estatísticas da própria
  fonte, e não existe predição intra em lugar nenhum de
  `src/scripts/partition_model/`. A advertência do plano foi violada pela implementação.
- **Enunciado vigente.** O «H9b» reprovado no Gate 2 testou uma estatística **só da
  fonte**, fortemente correlacionada com a variância e os gradientes que o bloco A já
  contém; o seu resultado nulo era esperado e **não informa sobre a hipótese que o plano
  formulou**. A hipótese especificada foi enfim testada em separado, como bloco D', e o
  seu critério de decisão **não passa**: a área de comprometimento cai em 16 px, o nível
  com 15 855 nós e 8,5× mais dados, onde a entropia cruzada piora 0,1% e a área sob a
  curva cai 0,001. Deste modo, o domínio de pixels fecha por uma via a mais, com **cinco**
  tentativas independentes negativas — ConvNeXt-CE, ConvNeXt-*regret*, GNN do Approach B,
  bloco D e bloco D'.
- **Data e documento.** Registrado em **2026-07-26**;
  `docs/RESULTADOS_auditoria_dominio_pixels.md` §3, §6, §6.1, §6.2 e §7.

## R16 — A regra de parada da Solução 4 como implicação lógica

- **Enunciado antigo (literal).** «Como o oráculo já **rejeita** a regressão de *regret* a
  risco casado, um benchmark real […] só confirmaria um resultado pior», apoiado no
  precedente de que o H9c «venceu o oráculo e foi refutado no benchmark real».
- **Por que caiu.** O precedente invocado **está contaminado**, pois descreve o piloto
  Jockey de dois quadros que a §8.1 prova ser H9a + H9c empilhados (ver R6 e R7); e a
  implicação lógica não vale, pois o próprio Approach B estabeleceu que a ordenação entre
  podadores **pode inverter** entre o oráculo e o codificador real.
- **Enunciado vigente.** Pular o Gate 5 foi uma decisão sob **assimetria de custo
  experimental** — o custo de integrar em C e rodar horas de codificação contra o valor
  esperado da informação, dado que o sinal offline era fraco. É escolha de alocação de
  esforço, defensável como tal, e **não** uma implicação de que o codificador confirmaria
  a rejeição. O motivo substantivo de não priorizar a regressão de *regret* é a
  zero-inflação, independente do argumento do oráculo. A decisão não muda; muda a razão.
- **Data e documento.** Corrigido em **2026-07-19**; `docs/RESULTADOS_solucao4.md` §7
  (bloco "Correção (2026-07-19) — a justificativa acima tem dois defeitos", identificado
  como **correção D2**), com a consequência metodológica registrada em
  `docs/SINTESE_resultados_metodologia.md` §9 e a resposta de defesa em
  `docs/RESPOSTAS_contra_argumentos_banca.md` CB-5.

## R17 — A causa da derrota do modelo estruturado no codificador

- **Enunciado antigo (literal).** a derrota real do GNN atribuída a «poucas podas erradas
  **caras em RD**».
- **Por que caiu.** A medição do A5 **não sustenta** essa causa: as podas NONE do GNN são
  baratas por contagem (`split_lost` **0,25%**) **e** por *regret* ponderado
  (`reg_frac` ≈ 0, o menor de todos). Logo, a falha do GNN **não está na ação de
  encerramento em NONE**.
- **Enunciado vigente.** A causa do fenômeno fica como **pergunta aberta** — vazamento de
  vizinhança, descasamento entre `cpu-used` 0 e 1, ou outra ação da política —, e não como
  raiz provada. A conclusão de que o oráculo é mau proxy do produto taxa BD por tempo
  **permanece**; apenas a explicação do mecanismo foi suavizada.
- **Data e documento.** Corrigido em **2026-07-20**; `docs/RESULTADOS_approachB.md` §6
  (bloco "Correção (2026-07-20)"), com a evidência em
  `docs/RESULTADOS_oraculo_regret.md` §5.

## R18 — A forma forte da Conclusão 3, com a independência dos nós como limite

- **Enunciado antigo (literal).** a Conclusão 3 na leitura de que o limite seria
  **informacional** e de que a decisão por nós **independentes** seria o que o produzia.
- **Por que caiu.** A estrutura conjunta extrai **+28 pp** no oráculo sobre os nós
  independentes, o que derruba a forma forte no oráculo; mas o ganho **não sobrevive ao
  codificador**, pois o H9a domina o GNN por cerca de **2×** em taxa BD ao longo de toda a
  varredura de τ (GNN ~1,50% a 1,59% contra H9a 0,75% a 0,94% a tempo casado), e a
  fronteira do GNN é **plana**, o que indica que o limite é a qualidade das decisões e não
  a calibração.
- **Enunciado vigente.** A estrutura conjunta *extrai* mais sinal offline, por nó e no
  oráculo, mas esse sinal é **não realizável** em taxa BD por tempo real. Deste modo, o
  limite **não** era artefato de independência, e a classificação simples por nó (H9a) é a
  formulação correta. Some-se o alerta metodológico mais forte que a Fase 5 produzira: o
  oráculo **pode inverter o ranqueamento**, e não apenas superestimar a magnitude.
- **Data e documento.** Estabelecido em **2026-07-18**; `docs/RESULTADOS_approachB.md`
  §2, §5 e §7.1, com a consequência metodológica em
  `docs/SINTESE_resultados_metodologia.md` §9 e a resposta de defesa em
  `docs/RESPOSTAS_contra_argumentos_banca.md` CB-3.

## R19 — A fronteira esburacada que a poda suave preencheria

- **Enunciado antigo (literal).** «a decisão **dura** deixa uma fronteira esburacada que a
  modulação **soft** preencheria».
- **Por que caiu.** O varrimento fino de τ mostra que a fronteira **já é densa e
  contínua**: o maior intervalo entre pontos vizinhos, em 21 vizinhanças, é de **0,15×**
  (RaceNight, na ponta agressiva), e a maioria fica em torno de **0,03×**, com taxa BD
  suave e monótona em τ. O salto de ~0,32× da curva grossa anterior era **artefato da
  amostragem**, e não natureza dura da decisão.
- **Enunciado vigente.** A poda dura do H9a, via τ, é efetivamente um **controle contínuo
  de ponto de operação** da fronteira de taxa BD por tempo, obtido sem retreino, por
  variável de ambiente. Este é um resultado **positivo** para a tese, e a implementação de
  poda suave **não se justifica**.
- **Data e documento.** `docs/RESULTADOS_C5_fronteira_tau.md` §2 e §3, que fecha o Bloco 6,
  em convergência com o achado independente do Approach B de que «τ não era o gargalo».

## R20 — O piso de ruído do tempo de parede suposto em 1% a 2%

- **Enunciado antigo (literal).** «σ ≈ 1–2% por encode», inferido de violações de
  monotonicidades que valem por construção, e a prática de descartar como «dentro do
  ruído» as comparações abaixo desse piso.
- **Por que caiu.** A medição direta, com cinco repetições intercaladas na Crosswalk sobre
  três configurações e quatro pontos de quantização, dá **CV mediano de 0,28%** e máximo
  de 0,64%; o desvio do tempo pareado é de **±0,23 pp** para `ml_balanced` e **±0,09 pp**
  para `native_cpu1`, o que fixa a resolução em **~0,46 pp** e **~0,18 pp**
  respectivamente. A estimativa anterior era pessimista por um fator de cerca de **4×**.
- **Enunciado vigente.** A resolução do experimento pareado é de **~0,46 pp**. Deste modo,
  o marginal de +1,02 pp do H9d está a **~4,4 σ** e é sólido; os ganhos por sequência de
  Neon1224 (+0,1 pp) e Crosswalk (+0,4 pp) **não são distinguíveis de zero** em
  codificação única; e, simetricamente, **muitas comparações antes descartadas como
  "dentro do ruído" são de fato resolvíveis**. As diferenças de taxa BD continuam
  **exatas**, pois bytes e PSNR são determinísticos, de modo que as conclusões de Pareto
  não se alteram.
- **Data e documento.** Medido em **2026-07-26**;
  `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3, §3.1, §3.2 e §3.3, com registro em
  `docs/ANDAMENTO_tese.md` §0.1 e §6.

## R21 — O joelho da curva de τ dentro do vão não explorado

- **Enunciado antigo (literal).** a suposição de que o joelho da curva estaria **dentro do
  vão (30, 60)**, deixando um ponto de operação atrativo por colher em τ = 45.
- **Por que caiu.** O preço do degrau, em pontos percentuais de taxa BD por ponto
  percentual de redução de tempo, fica entre **0,013 e 0,042** de τ=95 a τ=70 e salta para
  **0,107** a partir de τ=60, o que situa o joelho na fronteira **τ ≈ 60–70**. E τ=45 não
  está sobre inflexão favorável: a corda entre τ=60 e τ=30 prevê taxa BD de **0,395%** no
  tempo que τ=45 atinge, contra **0,430%** medidos, ou seja, **0,035 pp acima** da
  interpolação.
- **Enunciado vigente.** A faixa inexplorada foi explorada e **não contém nada**. Sobre as
  8 sequências, τ=45 entrega **+0,643% de taxa BD a 21,4%** de redução de tempo, contra
  **+0,449% a 32,6%** do preset nativo `cpu-used=1` — ou seja, **τ=45 é estritamente
  dominado pelo botão de velocidade nativo**. É um negativo limpo, que fecha a questão da
  curva de τ e confirma por medição a previsão registrada no plano.
- **Data e documento.** Medido em **2026-07-26**;
  `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §1 e §1.1, com registro em
  `docs/ANDAMENTO_tese.md` §0.1.

## R22 — A subsunção do E5 pelo retreino do modelo substituto

- **Enunciado antigo (literal).** «O **E5 está pausado por decisão**, não descartado:
  parte do que ele responderia é subsumida pelo item 2 (que testa o **modelo-teto** com o
  objetivo certo, enquanto o E5 testa apenas o estudante)».
- **Por que caiu.** A premissa da subsunção **não se confirmou**. Descobriu-se que o braço
  `ml` da ablação de atribuição usa as **36 features do H9a**
  (`partition_strategy.c:2164`), e não um estudante de pixels, de modo que o E5 mede a
  atribuição do **podador implantado**, e não da cota superior. Some-se que, com o capítulo
  passando a apoiar-se na hierarquia do crivo A5, que é declaradamente
  **não-adjudicante**, a confirmação no codificador ficou **mais** necessária, e não
  menos.
- **Enunciado vigente.** O E5 é experimento **independente e necessário**, e foi executado
  em 27 e 28 de julho de 2026, com 144 codificações em duas sequências de validação. O
  retreino do modelo substituto não responde à pergunta do E5.
- **Data e documento.** Corrigido em **2026-07-27**; `docs/DECISOES_escopo.md`
  (registro histórico de 26/07 marcado como superado, seguido da atualização de 27/07),
  com a execução relatada em `docs/RESULTADOS_E5_ablacao_validacao.md` e
  `docs/ANDAMENTO_tese.md` §0.2-bis.

## R23 — A comparabilidade direta entre números de redução de tempo

- **Enunciado antigo (literal).** o uso das reduções de tempo de fases distintas como se
  fossem a mesma grandeza, em tabelas que misturam, por exemplo, «H9a bal 0,568%/19,3%» e
  «H9a bal 0,568%/17,7%» para a mesma configuração.
- **Por que caiu.** Há **duas definições de redução de tempo** computadas pelo mesmo
  script (`analyze_frontier.py:281`): a **canônica**, média sobre os pontos de
  quantização de `1 − t/t_âncora` e depois sobre as sequências; e a **ponderada pelo
  tempo**, `1 − Σ_CQ t / Σ_CQ t_âncora`, dominada pelo ponto CQ 20. Elas divergem em até
  **~3 pp**: o preset nativo `cpu-used=1` é **30,4%** numa e **32,6%** na outra.
- **Enunciado vigente.** O texto adota a definição **canônica**, que é a usada pelos
  documentos mais recentes (H9d e Bloco 7) e pela tabela já padronizada de
  `INVENTARIO_solucoes.md`, e declara a definição na legenda de toda tabela de tempo.
  Números das duas definições **nunca** aparecem na mesma tabela nem em comparação direta.
  As taxas BD **não** são afetadas.
- **Data e documento.** `docs/SINTESE_resultados_metodologia.md` §8 (bloco "Atenção ao
  copiar estes números"), com o artefato de divergência em
  `results/benchmark/fase6_analysis/ts_definitions.csv`.

---

# PARTE II — Lacunas conhecidas e pendências

Esta parte apresenta o que o projeto sabe que **não** mediu, e que por isso o texto não
pode afirmar. Cada entrada declara o que falta, por que a falta importa para a redação,
qual capítulo é afetado e o custo de fechamento **quando o projeto o registra** — pois o
custo não registrado é declarado como tal, e não estimado. As lacunas L1 a L3 são as que
mais restringem enunciados dos capítulos; as demais são ressalvas que o texto deve
carregar.

---

## L1 — A fronteira de compromisso global não contém os pontos do H9d

- **O que falta.** A fronteira de Pareto global de taxa BD por tempo, que reúne todos os
  níveis de `cpu-used`, é de uma análise anterior sobre **3 sequências**
  (Boxing, FoodMarket2 e Tango) e **não contém o H9d**, medido depois sobre as 8
  sequências da CTC. Recompô-la exige os pontos do H9d nos demais níveis de `cpu-used`,
  que **não foram rodados**.
- **Por que importa.** A ausência do H9d nessa fronteira **não é dominância**; mas a
  figura, como está, **subrepresenta a contribuição**, pois omite a segunda solução
  positiva justamente na única figura que compara tudo contra tudo. Na medição própria, o
  H9d é não dominado no sentido de Pareto e vence o botão de τ em 6 de 8 sequências.
- **Capítulo afetado.** Resultados §6 (`R6_analise_integrada.md`), e por consequência a
  figura da fronteira global planejada em `A2_TABELAS_E_FIGURAS.md`.
- **Custo de fechamento.** **Não registrado pelo projeto.** Como referência de ordem de
  grandeza da mesma família, a fronteira bidimensional do H9d — duas bases do H9a por duas
  forças do H9d — custou **96 codificações**; o número de codificações da recomposição
  global não foi calculado em documento algum.
- **Procedência.** `docs/SINTESE_resultados_metodologia.md` §6 (nota de 26/07 com
  atualização de 27/07), `docs/ANDAMENTO_tese.md` §0.3 (último item da fila) e §6 (riscos
  vivos), `docs/INVENTARIO_solucoes.md` §8.

## L2 — O critério de decisão estrito do E5 foi atingido em uma de duas sequências

- **O que falta.** O critério pré-registrado do E5 pedia dominância a tempo casado sobre a
  variância em **≥2 de 3** sequências de validação; obteve-se em **1 de 2** (FlowerPan),
  pois na Lips **não há par casado** — a variância salta de **1,006× para 3,563×** entre
  τ=0,99 e τ=0,97, e o `ml` vive inteiro dentro desse vão — e a HoneyBee ficou fora por
  decisão de escopo. O que decidiria são **três pontos de τ na Lips**, dentro da transição
  abrupta da curva: **τ = 0,985, 0,98 e 0,975**.
- **Por que importa.** O texto **não pode** declarar o critério atingido. O que se pode
  afirmar é substancial e deve ser afirmado com precisão: a **primeira comparação a tempo
  casado** contra a variância da tese, com o `ml` vencendo em todos os pares (**4,6×** a
  1,15× e **1,85×** a 1,27×); atribuição limpa contra o aleatório em **2 de 2**, com
  margens de **7× a 158×**; taxa BD negativa em **2 de 2** (−0,015% e −0,073%); e o
  fechamento da explicação alternativa da Fase 5, pois a não-sobreposição é **propriedade
  do escore**, e não do grid de τ.
- **Capítulo afetado.** Metodologia §3 (`M3_protocolo_avaliacao.md`, critérios em
  cascata), Resultados §2 (`R2_h9a.md`) e Resultados §7
  (`R7_ameacas_e_escopo.md`).
- **Custo de fechamento.** **~2,3 h**, registrado, para os três pontos de τ na Lips.
  Ambos os desfechos são publicáveis, o que torna o experimento barato em risco: se a
  variância aterrissar em 1,07× a 1,28×, há par casado e o critério se decide na forma
  estrita; se atravessar o vão outra vez, a transição abrupta fica **demonstrada** em vez
  de inferida de dois pontos.
- **Procedência.** `docs/RESULTADOS_E5_ablacao_validacao.md` §4, §5 e §6;
  `docs/ANDAMENTO_tese.md` §0.2-bis e §0.3; `docs/DECISOES_escopo.md` (item 4).

## L3 — O par `pixels24` contra variância segue sem árbitro no codificador

- **O que falta.** A discordância entre o crivo A5 (offline, 6 sequências, 792 840 nós,
  em que o `pixels24` vence a variância) e a ablação CB-1 (codificador, 2 quadros, 1
  sequência, em que a variância empata ou supera o modelo de pixels) **continua aberta**.
  O E5 **não** a arbitra, pois o seu braço `ml` é o estudante **H9a de 36 atributos**, e
  não o `pixels24`: ele decide *H9a contra variância*, e não *`pixels24` contra
  variância*.
- **Por que importa.** É a ordenação **interna ao domínio de pixels**, e é exatamente o
  ponto sobre o qual a banca pressiona em CB-1 e CB-2. Sem árbitro, o texto deve reportar
  a hierarquia do crivo **declarando a contradição**, e nunca resolvê-la em favor de um
  dos lados. A lacuna foi **estreitada, não fechada**.
- **Capítulo afetado.** Resultados §1 (`R1_dominio_pixels.md`) e Resultados §7
  (`R7_ameacas_e_escopo.md`); afeta também a Metodologia §6
  (`M6_modelos_e_atribuicao.md`), onde a metodologia de atribuição é descrita.
- **Custo de fechamento.** **Não registrado como número próprio.** O requisito está
  especificado — uma ablação no codificador com **≥10 quadros e ≥2 sequências**, com o
  braço `pixels24` no lugar do H9a —, e a campanha análoga já executada, o E5, custou
  **144 codificações e cerca de 22 h** em duas sequências de validação, o que serve de
  referência de ordem de grandeza.
- **Procedência.** `docs/RESULTADOS_convnext_regret.md` §5.1;
  `docs/RESPOSTAS_contra_argumentos_banca.md` CB-1 (ressalva 1), CB-2 e "Consequência
  transversal"; `docs/INVENTARIO_solucoes.md` §8 (primeiro item);
  `docs/ANDAMENTO_tese.md` §6 (riscos vivos) e §0.4.

## L4 — A rede convolucional nativa não tem linha isolada de taxa BD e tempo

- **O que falta.** O podador convolucional nativo **só é observável por diferença** nos
  experimentos de substituição; não existe uma linha própria que meça a sua contribuição
  isolada de taxa BD e redução de tempo.
- **Por que importa.** Toda comparação contra a rede nativa é, por construção,
  sistema contra sistema. O texto não pode atribuir a ela um par (taxa BD, tempo) próprio,
  nem afirmar dominância isolada em nenhum dos dois eixos fora do arranjo de substituição.
- **Capítulo afetado.** Resultados §3 (`R3_h9c.md`) e §6 (`R6_analise_integrada.md`).
- **Custo de fechamento.** **Não registrado pelo projeto.**
- **Procedência.** `docs/INVENTARIO_solucoes.md` §8 (terceiro item).

## L5 — O custo de inferência do modelo substituto ConvNeXt nunca foi pago

- **O que falta.** O caminho de medição por *replay* é explícito quanto a isso:
  «*no convolutional inference in C*» (`surrogate_replay.py:4-8`). As decisões do
  substituto são pré-computadas fora do codificador e reinjetadas pelo mesmo gancho de
  poda, de modo que **todo resultado H8 e derivado é cota superior que ignora o custo de
  inferência**.
- **Por que importa.** É metodologicamente correto para medir uma cota, e está declarado
  como tal; mas precisa ser **dito explicitamente nos Resultados**, pois 28,1 M de
  parâmetros por superbloco não são implantáveis na libaom em regime algum. Hoje o ponto é
  inócuo, uma vez que a qualidade do substituto já não passa (ver R2); mas é exatamente o
  tipo de limitação que uma banca aponta.
- **Capítulo afetado.** Metodologia §6 (`M6_modelos_e_atribuicao.md`), Resultados §1
  (`R1_dominio_pixels.md`) e Resultados §7 (`R7_ameacas_e_escopo.md`).
- **Custo de fechamento.** **Não registrado**, e sem valor experimental: medir o custo de
  inferência de um modelo que já perde em qualidade não altera conclusão alguma.
- **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §4;
  `docs/INVENTARIO_solucoes.md` §8 (quarto item).

## L6 — A decomposição de três pernas cobre quatro das oito sequências

- **O que falta.** A decomposição que separa H9a puro, H9c puro e interação existe em
  **4 das 8** sequências (Neon1224, PierSeaSide, Tango e TimeLapse), pois o terceiro termo
  — `h9adef`, o H9a nos seus defaults sozinho — só foi medido nelas. E o E4, que quantifica
  o confundimento, cobre as mesmas quatro.
- **Por que importa.** A média de **64%** de redução de tempo atribuível ao H9a repousa
  sobre quatro amostras com dispersão de **28% a 95%**, de modo que a conclusão robusta é
  a **direção e a ordem de grandeza**, e não o valor central. Do mesmo modo, a interação
  média de **−1,9 pp** tem, no TimeLapse, o único caso positivo (+1,7 pp) apoiado sobre um
  H9c isolado de apenas 0,6%, número pequeno demais para sustentar interpretação.
- **Capítulo afetado.** Resultados §3 (`R3_h9c.md`) e §6 (`R6_analise_integrada.md`).
- **Custo de fechamento.** **12 codificações, cerca de 1 h 30**, registrado, para levar o
  terceiro termo às demais sequências.
- **Procedência.** `docs/RESULTADOS_BLOCO7_E1_E4.md` §4 (limitações);
  `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §2 e §4.

## L7 — O H9d é inerte sobre a base agressiva

- **O que falta.** Nada a medir: a lacuna anterior — a fronteira do H9d com um único ponto
  de operação — foi **fechada em 2026-07-27** com 96 codificações e 4 pontos. Ela foi
  **substituída por uma limitação nova**, que é a inércia do H9d sobre a base agressiva
  (+0,17 pp de redução de tempo, com apenas 1 de 8 sequências acima da resolução).
- **Por que importa.** O texto não pode apresentar a aditividade do H9d como propriedade
  do mecanismo (ver R9), e deve declarar que ela **depende do ponto de operação**. O ponto
  implantado (P_rect × PL10) é justamente onde a disjunção é maior, o que valida a escolha
  por uma razão mais estreita do que "as ações são disjuntas".
- **Capítulo afetado.** Resultados §4 (`R4_h9d.md`) e §6 (`R6_analise_integrada.md`);
  Metodologia §4 (`M4_atributos_e_politica.md`), onde a política por nível é descrita.
- **Custo de fechamento.** Não se aplica: é limitação medida, e não pendência.
- **Procedência.** `docs/INVENTARIO_solucoes.md` §7 e §8 (segundo item);
  `docs/SINTESE_resultados_metodologia.md` §5-quater; `docs/DECISOES_escopo.md` (item 3).

## L8 — A resolução temporal limita a leitura sequência a sequência

- **O que falta.** O desvio do tempo de parede foi medido, mas é **intra-execução**: cinco
  repetições numa janela contínua, na mesma sequência e no mesmo contêiner sem reinício, o
  que **não** captura deriva entre dias, reinícios de contêiner ou estados térmicos
  distintos. Além disso, a medição é de **uma sequência** (Crosswalk) e três
  configurações.
- **Por que importa.** Diferenças de tempo abaixo de **~0,46 pp** não são resolvíveis e
  não podem ser reportadas como positivas — o que atinge, nominalmente, o +0,17 pp do H9d
  sobre a base agressiva e os ganhos por sequência de Neon1224 (+0,1 pp) e Crosswalk
  (+0,4 pp). E números medidos com semanas de intervalo pedem cautela adicional, pois o σ
  medido não os cobre.
- **Capítulo afetado.** Metodologia §3 (`M3_protocolo_avaliacao.md`), Resultados §4
  (`R4_h9d.md`) e §7 (`R7_ameacas_e_escopo.md`).
- **Custo de fechamento.** **Não registrado.** Fechar a parte inter-execução exigiria
  repetições distribuídas por dias, cujo desenho não está especificado em documento algum.
- **Procedência.** `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3.3 e §4;
  `docs/ANDAMENTO_tese.md` §6 (riscos vivos); `docs/INVENTARIO_solucoes.md` §8 (quinto
  item).

## L9 — O resíduo do critério D' em 32 px e a invisibilidade do nível de 64 px

- **O que falta.** O critério de decisão do bloco D' não passa, mas deixa um **resíduo
  delimitado e não perseguido**: um positivo consistente em 32 px (entropia cruzada −3,1%,
  área sob a curva +0,017), com um oitavo dos nós e sem corroboração na leitura de redução
  de custo a risco casado. E o nível de **64 px é invisível a este critério**, pois a
  disponibilidade do atributo ali é de **0,0%** — nenhum nó de 64 px tem ambos os vizinhos
  dentro do superbloco.
- **Por que importa.** O texto deve declarar que a via de pixels fecha **no nível desta
  evidência**, e não em termos absolutos, uma vez que um acerto em 64 px vale mais, pois
  poda a subárvore inteira. A ordenação observada em 32 px, `D' > H9b > H9a`, é a que a
  hipótese prediz.
- **Capítulo afetado.** Resultados §1 (`R1_dominio_pixels.md`) e §7
  (`R7_ameacas_e_escopo.md`); trabalhos futuros da Conclusão.
- **Custo de fechamento.** **Não quantificado**, mas o caminho está especificado e é
  declaradamente barato: **não** re-extrair o conjunto de dados, e sim costurar superblocos
  adjacentes do mesmo quadro (`mi_row` e `mi_col` estão no `ctx`) para obter vizinhos-fonte
  nas bordas, o que elevaria a disponibilidade em 32 px e tornaria 64 px mensurável. Só
  depois disso, e só se o sinal sobreviver, se justificaria instrumentar o codificador para
  ler vizinhos **reconstruídos**.
- **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §6.1, §6.2 e §8.

## L10 — A grade de τ congelada e a não-sobreposição no conjunto de teste

- **O que falta.** A grade de τ foi calibrada na validação e **congelada**; no conjunto de
  teste reservado, as faixas de aceleração do `ml` e da variância saíram **disjuntas em 3
  de 3** sequências, de modo que não houve comparação a tempo casado ali. Estender a grade
  da variância no teste **não foi feito**, pois seria ajustar configuração vendo dado de
  teste, o que viola o congelamento anti-seleção *a posteriori*.
- **Por que importa.** A forma estrita do Gate 5 **não** foi atingida no teste, e o texto
  deve dizê-lo. O que a sustenta é a atribuição **a política casada**, que dispensa
  sobreposição: sob a mesma política, o escore do modelo alcança taxa BD mínima de 0,008%
  na RiverBank contra 0,75% da variância — razão de **94×**, com **11×** na Jockey e
  **44×** na RaceNight.
- **Capítulo afetado.** Metodologia §3 (`M3_protocolo_avaliacao.md`), Resultados §2
  (`R2_h9a.md`) e §7 (`R7_ameacas_e_escopo.md`).
- **Custo de fechamento.** Não se aplica no conjunto de teste, por decisão de protocolo. Na
  validação, a lacuna foi endereçada pelo E5, cujo resíduo está em L2.
- **Procedência.** `docs/ANDAMENTO_tese.md` §4.1 e §6;
  `docs/SINTESE_resultados_metodologia.md` §9; `docs/RESULTADOS_fase5.md` §5.

## L11 — Nenhuma figura existe ainda no projeto

- **O que falta.** O plano das tabelas e figuras dos dois capítulos está redigido, com
  legenda, dados de origem e script sugerido para cada item; mas **nenhuma figura existe
  ainda no projeto**.
- **Por que importa.** As afirmações de fronteira e de granularidade fina — que são as
  Conclusões 1 e 2 — dependem de figura para serem lidas, e a figura da fronteira global é
  a mesma atingida por L1.
- **Capítulo afetado.** Ambos, por meio de `A2_TABELAS_E_FIGURAS.md`.
- **Custo de fechamento.** **Não registrado** como estimativa de tempo; o plano registra o
  script sugerido por figura.
- **Procedência.** `results/thesis/00_PLANO_capitulos.md` §5.

## L12 — A publicação do conjunto de dados não foi executada

- **O que falta.** A folha de dados de publicação está pronta (`docs/ZENODO_datasheet.md`),
  mas a conversão para `.npz` em `uint8` (`pkl_to_npz.py`) **não foi executada**, pois o
  backup bruto em `bin` e `pkl` segue primeiro para armazenamento externo.
- **Por que importa.** A tese descreve o conjunto de dados como contribuição de
  infraestrutura; a redação deve declarar o estado real da publicação, e não prometê-la
  como concluída.
- **Capítulo afetado.** Metodologia §2 (`M2_instrumentacao_e_dataset.md`).
- **Custo de fechamento.** **Não registrado.**
- **Procedência.** `docs/ANDAMENTO_tese.md` §5 (decisões em aberto).

---

# PARTE III — Regras de redação derivadas

Esta parte converte as Partes I e II em pauta imperativa. Cada regra remete à entrada que
a origina, e a violação de qualquer uma delas reintroduz no texto uma afirmação já
derrubada pelo próprio projeto.

**Sobre o domínio de pixels**

1. Não escrever «os pixels saturam na variância» nem «o sinal não está nos pixels».
   Escrever a **hierarquia medida** no crivo A5, com os quatro valores de `reg_frac`, e
   **declarar a contradição** com a ablação de dois quadros. (R1, L3)
2. Não chamar o ConvNeXt de «teto», «cota superior» ou «referência de limite superior» do
   domínio de pixels. Escrever **instrumento de diagnóstico** e **tentativa documentada**
   de estabelecer a cota; registrar que a cota superior do domínio de pixels permanece
   **não medida** e que a única cota genuína do arcabouço é o **oráculo**. (R2)
3. Não escrever que o `pixels24` é uma **cota superior**: ele é a **cota inferior** do
   domínio de pixels, o melhor desempenho ali observado. (R2)
4. Não escrever «o podador implantado usa contexto de taxa-distorção, não pixels», nem
   opor «H9a» a «pixels24» como conjuntos disjuntos. Escrever que **24 dos 36 atributos do
   H9a são descritores de luminância**, que o `pixels24` é o bloco A do H9a e que os 12
   atributos de vizinhança, quantização e posição são um **retorno marginal** sobre eles.
   (R4, R5)
5. Não escrever «contexto RD grátis supera pixels ~50% relativo». Escrever **retorno
   marginal**. (R5)
6. Não escrever que o modelo substituto era **sobreajustado** ou mal selecionado. Escrever
   que ele é **bem selecionado e fraco em absoluto**, com macro-F1 de 0,203, e treinado
   contra o objetivo errado. (R3)
7. Escrever sempre **cinco** tentativas independentes negativas no domínio de pixels —
   ConvNeXt-CE, ConvNeXt-*regret*, GNN do Approach B, bloco D e bloco D' —, nunca quatro.
   (R15)
8. Declarar explicitamente que os resultados H8 e derivados **ignoram o custo de
   inferência**, pois nenhuma inferência convolucional foi executada em C. (L5)
9. Escrever que a via de pixels fecha **no nível desta evidência**, registrando que o
   nível de 64 px é invisível ao critério D'. (L9)

**Sobre a família H9**

10. Não escrever que o H9c é «2 a 4× mais eficiente que o H9a» nem que «supera o nativo em
    eficiência». Escrever que **82% a 96%** da redução de tempo que lhe fora atribuída era
    do H9a, e que a média sobre quatro sequências é de **64%**. (R6)
11. Não citar as tabelas pré-busca `h9c_tau*` como medição do H9c: elas medem **H9a@0,9 +
    H9c empilhados**. Citar as linhas `h9ciso_*`, com o H9a neutralizado. (R6)
12. Não escrever que «mesmo a cota mais informativa não se traduz em vantagem de tempo».
    Escrever que, como **substituto** da rede convolucional nativa, o H9c **empata na
    grade completa** e **vence no regime de alta taxa**. (R7)
13. Não escrever «alavancas de poda não se somam» na forma geral, nem invocar «limite
    informacional». Escrever que a não-aditividade é **sobreposição de ação**, e que dois
    podadores se somam na medida em que seus **conjuntos de candidatos podados** são
    disjuntos **no ponto de operação em que rodam**. (R8, R9)
14. Não apresentar a aditividade do H9d como propriedade absoluta da sua ação, e **não
    citar como positivo** o +0,17 pp sobre a base agressiva, que está abaixo da resolução
    temporal. (R9, L7, L8)
15. Não escrever que o contexto barato é «necessário e suficiente» para superar a cota de
    pixels. Escrever **suficiente sob política casada contra a cota do domínio de
    pixels**, e nunca contra o podador nativo. (R13)
16. Não afirmar que qualquer ponto de aprendizado de máquina **domina** a rede
    convolucional nativa. Escrever que o valor prático é a **granularidade fina em baixo
    regime de aceleração** que a escada discreta dos presets não oferece. (R13)

**Sobre custo, protocolo e números**

17. Não alegar **leveza de inferência como vantagem**, nem citar a razão de ~50× por
    chamada como argumento competitivo. Escrever que o podador inteiro é **≤0,32% do tempo
    de codificação** e que o custo de inferência **não é alavanca em direção nenhuma**.
    (R10)
18. Não descrever os quinze quadros como recorte próprio ou como limitação. Escrever
    **conformidade com a CTC §4.1**, citando `--limit=15`. (R11)
19. Não citar resultado algum de H1 a H6 como caracterização do sinal da luminância, nem a
    cota de «~13 a 18%». Citar apenas H7 e H8, sobre dado corrigido, e usar o defeito de
    luminância nula como **lição metodológica**. (R12)
20. Não descrever o podador implantado como **destilado** de um modelo substituto, e não
    empregar o termo **professor** em nenhuma hipótese. Escrever que o `student_h9a` é
    treinado **diretamente** sobre os 36 atributos, sem modelo substituto no laço. (R14)
21. Não descrever o bloco D implementado como **proxy de resíduo de predição intra**:
    é o SATD do **bloco-fonte**. Registrar a divergência entre especificação e
    implementação, e que o «H9b» reprovado **não** testou a hipótese formulada. (R15)
22. Declarar em toda tabela de tempo **qual** das duas definições de redução de tempo é
    usada, adotar a **canônica** e nunca misturar as duas na mesma tabela ou comparação.
    (R23)
23. Não descartar comparação alguma como «dentro do ruído» com base no piso suposto de 1%
    a 2%. Usar a resolução **medida** de ~0,46 pp, e lembrar que as diferenças de taxa BD
    são **exatas**. (R20)
24. Não apresentar τ = 45 como ponto de operação atrativo nem supor joelho dentro do vão
    (30, 60). Escrever que o joelho está em **τ ≈ 60–70** e que τ = 45 é **estritamente
    dominado** pelo botão de velocidade nativo. (R21)

**Sobre resultados negativos e vereditos**

25. Não escrever que a rejeição no oráculo implica rejeição no codificador. Escrever que
    pular o Gate 5 da Solução 4 foi decisão sob **assimetria de custo experimental**, e que
    o oráculo **pode inverter o ranqueamento**. (R16, R18)
26. Não atribuir a derrota do modelo estruturado a «poucas podas erradas caras em RD».
    Escrever que a causa fica como **pergunta aberta**, preservando a conclusão de que o
    oráculo é mau proxy. (R17)
27. Não escrever que a fronteira do knob de τ é esburacada nem propor poda suave como
    preenchimento. Escrever que a fronteira **já é densa e contínua**, com intervalos
    ≤ 0,15×. (R19)
28. Declarar o critério de decisão do E5 como **não atingido na forma estrita**, com o que
    de fato se obteve, e sem apresentar a HoneyBee como sequência que falhou — é **decisão
    de escopo**. (L2)
29. Declarar que a fronteira de compromisso global **não contém o H9d**, e que essa
    ausência **não é dominância**. (L1)
30. Declarar que o par `pixels24` contra variância **continua sem árbitro no codificador**,
    e que o E5 decide *H9a contra variância*, não aquele par. (L3)

---

*Documento redigido a partir de `docs/ANDAMENTO_tese.md`,
`docs/SINTESE_resultados_metodologia.md`, `docs/RESULTADOS_auditoria_dominio_pixels.md`,
`docs/RESULTADOS_convnext_regret.md`, `docs/RESULTADOS_BLOCO7_E1_E4.md`,
`docs/RESULTADOS_BLOCO7_E3_DEC_E2.md`, `docs/RESULTADOS_solucao4.md`,
`docs/RESULTADOS_approachB.md`, `docs/RESULTADOS_E5_ablacao_validacao.md`,
`docs/RESULTADOS_C5_fronteira_tau.md`, `docs/RESULTADOS_fase5.md`,
`docs/RESULTADOS_fase6.md`, `docs/RESULTADOS_microbench_pruner.md`,
`docs/METODOLOGIA_pipeline_ML.md`, `docs/RASTREABILIDADE.md`, `docs/DECISOES_escopo.md`,
`docs/RESPOSTAS_contra_argumentos_banca.md` e `docs/INVENTARIO_solucoes.md`.*
