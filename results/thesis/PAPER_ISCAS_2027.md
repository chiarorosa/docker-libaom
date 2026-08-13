# Post-NONE Neural Partition Pruning in AV1 All-Intra Coding

**Abstract** — AV1's native All-Intra encoder embeds a convolutional neural network (CNN) that prunes partition candidates before the recursive rate-distortion (RD) search, but comparing a learned pruner against the encoder's speed presets does not isolate its merit: each preset couples the CNN with dozens of unrelated, non-learned heuristics. This paper reports a same-category comparison — machine learning (ML) against ML — for a lightweight post-NONE pruner: a 39-feature multilayer perceptron (MLP) inserted immediately after the `PARTITION_NONE` candidate has been evaluated, deciding whether the remaining nine candidates are searched at all. The comparison uses direct substitution: at a fixed speed level, the native CNN is disabled by an environment variable and the MLP becomes the sole partition pruner, with a second, previously undocumented lever — the encoder's pre-search pruner, which otherwise runs unconditionally on intra frames — neutralized from the measurement script itself. Omitting this second neutralization was found to attribute 82%–96% of the measured time reduction, on one sequence, and 64% on average across four Common Test Conditions (CTC) sequences, to the wrong pruner, an episode reported in full as a lesson for experimental software architecture. Across 192 encodes (six configurations, eight CTC Class A1 sequences, four quantization points, `cpu-used=0` anchor), the isolated post-NONE pruner reaches statistical parity with the native CNN at speed presets 1 and 2 (paired t-test, n=8, p > 0.05 on the full grid) and a statistically significant rate-distortion advantage in the high-quality regime, replicated at both preset levels (p = 0.043 and p = 0.015), while losing significantly at preset 3. Source-code inspection shows that all three native post-NONE pruners are disabled by a frame-type guard in All-Intra coding, so the decision niche the proposed pruner occupies is natively empty — parity is achieved against an encoder that, structurally, is not competing there. The 39-feature MLP's isolated inference cost is about 32 times lower per call than the CNN's (≈758 ns vs. ≈24,700 ns per call, on the same bench); the related 36-feature pre-search pruner is even cheaper (≈486 ns, ≈50×). Both ratios exclude feature extraction and call-frequency differences and are never the source of the measured rate advantage; the pre-search pruner's deployed path — the only one with a directly measured breakdown — costs at most 0.32% of encoding time.

**Index Terms** — AV1, All-Intra coding, partition pruning, neural network inference, convolutional neural network, multilayer perceptron, rate-distortion optimization, confounding factor.

---

## I. Introdução

A decisão de particionamento de blocos do codificador AV1 em predição intraquadro é um dos maiores custos computacionais do codificador de referência. Em cada nó de uma árvore recursiva, o codificador avalia dez formas de partição candidatas por busca exaustiva de custo de taxa-distorção (do inglês *rate-distortion* – RD), mantendo a de menor custo e descendo recursivamente quando a divisão vence. A poda aprendida restringe esta busca: um modelo de aprendizado de máquina (do inglês *machine learning* – ML) é consultado em cada nó e elimina candidatos improváveis antes que o codificador pague o custo de avaliá-los, e o próprio codificador de referência já embarca, nos seus níveis de velocidade padrão (*presets*), uma rede convolucional (do inglês *convolutional neural network* – CNN) para exatamente esta função.

Avaliar um podador aprendido contra os *presets* nativos de velocidade não é, contudo, uma comparação de categoria correta. Cada *preset* altera simultaneamente dezenas de heurísticas não aprendidas — atalhos de busca de movimento, término antecipado de modos, simplificações de transformada —, além da rede convolucional de poda de partição. Uma diferença de tempo ou de taxa BD medida contra um *preset* não é, deste modo, atribuível ao algoritmo de decisão de particionamento isoladamente, e sim ao pacote inteiro de heurísticas que o *preset* ativa.

Este artigo relata uma comparação isolada de categoria correta — ML contra ML — para um podador pós-NONE: um perceptrone de múltiplas camadas (do inglês *multilayer perceptron* – MLP) leve, de trinta e nove atributos, inserido imediatamente depois de o candidato `PARTITION_NONE` ter sido avaliado, que decide se os nove candidatos restantes são sequer buscados. A comparação é feita por **substituição direta**: fixa-se o nível de velocidade, desliga-se explicitamente a rede convolucional nativa por variável de ambiente e instala-se o podador aprendido como único podador de partição intraquadro. O desenho exigiu, contudo, uma segunda neutralização, não documentada nas primeiras campanhas: o podador pré-busca do próprio codificador roda incondicionalmente em quadros intraquadro, de modo que medir o podador pós-NONE sem desativá-lo mede a pilha dos dois, e não o podador isolado. Omitir esta segunda neutralização atribuiu, nas primeiras medições, entre 82% e 96% do tempo economizado ao podador errado — episódio relatado neste artigo por inteiro, pois a lição de arquitetura de software que ele expõe generaliza para além deste podador específico.

Sob as duas neutralizações, em 192 codificações distribuídas em seis configurações, oito sequências das condições comuns de teste (do inglês *Common Test Conditions* – CTC) Classe A1 e quatro pontos de quantização, o podador pós-NONE isolado alcança **paridade estatística** com a rede convolucional nativa nos *presets* 1 e 2, e vantagem de qualidade estatisticamente significativa no regime de alta taxa, replicada em dois níveis independentes de *preset*, ao custo de derrota clara no *preset* 3. A inspeção do código-fonte mostra que os três podadores nativos que agem depois do `PARTITION_NONE` estão desligados por guarda de tipo de quadro em codificação *All-Intra*: o nicho de decisão que o podador proposto ocupa é, deste modo, nativamente vazio, o que explica a paridade sem exigir capacidade de modelo equivalente à da rede convolucional.

O restante deste artigo está organizado como segue. A Seção II descreve o podador pós-NONE — o ponto de inserção, o vetor de atributos e a ação. A Seção III apresenta o desenho de substituição direta e a dupla neutralização. A Seção IV relata o fator de confusão medido e a lição de arquitetura de software. A Seção V descreve o protocolo experimental. As Seções VI e VII apresentam os resultados agregados, os testes pareados e a decomposição por regime de quantização. A Seção VIII trata do custo computacional, e a Seção IX discute limitações e encerra o artigo.

---

## II. O podador pós-NONE

O ponto de inserção do podador é a chamada `av1_prune_after_none`, executada dentro da função de busca recursiva `av1_rd_pick_partition` imediatamente após `none_partition_search()` ter preenchido a taxa, a distorção e o custo RD reais da avaliação de `PARTITION_NONE`, e antes dos estágios de busca da divisão quadrada, das duas formas retangulares, das quatro formas assimétricas do tipo AB e das duas formas de proporção 4:1 (4-way).

O vetor de atributos tem **trinta e nove posições**. As primeiras trinta e seis reproduzem o vetor de um podador pré-busca já existente no mesmo codificador: vinte e quatro descritores de luminância do bloco (bloco A), oito atributos de contexto de particionamento da vizinhança causal já codificada (bloco B) e quatro atributos de quantização e de posição no quadro (bloco C). As três posições finais (bloco E) são exclusivas do podador pós-NONE e trazem exatamente a taxa, a distorção e o custo RD reais da hipótese `PARTITION_NONE` que acabou de ser avaliada, sob transformação logarítmica. É esta informação — indisponível a qualquer podador que decida antes da busca — que distingue o ponto de inserção pós-NONE.

A arquitetura é um perceptrone de múltiplas camadas de dimensões 39→64→32→3, com duas camadas ocultas de ativação retificada e uma camada de saída linear seguida de normalização exponencial, executado pela rotina nativa de inferência do próprio codificador (`av1_nn_predict`). A ação é **binária**: com base num limiar de confiança τ aplicado à saída do modelo, o nó é encerrado ali, reaproveitando o mecanismo `av1_disable_all_splits` já existente no codificador, ou a busca prossegue pelos nove candidatos restantes.

A habilitação opera em dois níveis. O primeiro é a guarda de compilação comum aos podadores desta família, sem a qual a chamada é uma função vazia. O segundo é a variável de ambiente `AV1_STUDENT_H9C_ENABLE`, desligada por padrão, com o limiar τ configurável por variável de ambiente e valor compilado por padrão de 0,9. Com a guarda desligada, o codificador produz fluxo de bits byte a byte idêntico ao da árvore de referência intocada, o que estabelece que toda diferença medida com a guarda ligada é atribuível ao podador, e não a um efeito colateral de compilação.

Antes de qualquer medição em codificação real, o podador foi submetido a um critério de decisão offline por simulação sobre o conjunto de dados anotado, exigido antes de se pagar o custo de integração em C. Este conjunto de dados foi extraído de dezesseis sequências do conjunto UVG em 4K, particionadas por sequência sem vazamento de conteúdo, e é distinto da grade de avaliação descrita na Seção V; ele comparece neste artigo apenas como origem de estatísticas de simulação, e nenhum valor de taxa BD ou de redução de tempo medido sobre ele é reportado aqui. O resultado foi o melhor obtido por qualquer solução da investigação da qual este trabalho deriva: **61,2% de redução de custo de busca simulado a apenas 0,20% de perda de partições `SPLIT` ótimas**, cinco vezes menos risco do que a cota de 1% admitida pelo protocolo. Este descolamento entre a promessa da simulação offline e o que o codificador viria efetivamente a medir é o assunto da seção seguinte.

---

## III. O desenho de substituição direta e a dupla neutralização obrigatória

Para isolar o mérito do podador pós-NONE contra a rede convolucional nativa, este trabalho adota a **substituição direta**. Fixa-se o nível de velocidade (`cpu-used`), desliga-se explicitamente a rede convolucional nativa por variável de ambiente e instala-se o podador aprendido como único podador de partição intraquadro naquele nível. A diferença medida entre as duas configurações — mesmo *preset*, mesmas demais heurísticas de velocidade, único podador de partição trocado — é, deste modo, atribuível ao algoritmo de decisão de particionamento, e não ao conjunto de heurísticas não aprendidas que o *preset* também ativa.

O desenho exige, contudo, uma segunda neutralização, sem a qual a comparação mede outra coisa. A compilação de desempenho do codificador executa o podador pré-busca **incondicionalmente** em quadros intraquadro: uma vez compilado o binário, este podador é invocado sempre que uma condição puramente geométrica é satisfeita, sem variável de habilitação alguma. Medir o podador pós-NONE sem desativar explicitamente o podador pré-busca mede, portanto, a pilha dos dois — e não o podador pós-NONE isolado, que é o objeto deste artigo.

A neutralização é feita fixando os limiares do podador pré-busca em valores inatingíveis (2,0 para os limiares de comprometimento e de forçamento de divisão, e −1,0 para o descarte de partições retangulares), valores que nunca disparam, uma vez que as probabilidades da normalização exponencial do modelo nunca excedem a unidade nem ficam abaixo de zero. A neutralização é executada desde o próprio roteiro de medição, como **única variável alterada** em relação à configuração de referência, e a integridade do binário deste arranjo é verificada por uma garantia de inércia: com a guarda de compilação desligada, o binário gerado a partir da árvore modificada deve produzir fluxo de bits byte a byte idêntico ao da âncora. Esta garantia foi verificada sobre um quadro texturizado de 448 por 256 pixels, em codificação integralmente intraquadro a `cpu-used=0` e `cq=32`: os resumos criptográficos dos dois fluxos coincidiram no valor `b904f11c9fa02d5ea25460cb976ef29d`, o fluxo produzido com a guarda ligada foi decodificado com sucesso pelo decodificador da âncora, resultando em 172.076 bytes válidos, e os testes `EncodeAPI.AllIntra` passaram 3 de 3.

---

## IV. O fator de confusão medido e a lição de arquitetura de código experimental

Omitir a segunda neutralização não é uma hipótese teórica: produziu um fator de confusão real nas primeiras campanhas de medição do podador pós-NONE, e o episódio é relatado aqui por inteiro, pois a auditabilidade da cadeia de decisões é o instrumento metodológico que sustenta este trabalho, e um episódio de autocorreção medido é evidência dessa auditabilidade, não exceção a ela.

**O mecanismo.** O podador pré-busca é habilitado exclusivamente por uma condição geométrica sem sinalizador de ambiente algum, ao contrário do podador pós-NONE, cuja ativação depende de uma variável de habilitação explícita. Compilado o binário de desempenho, o podador pré-busca passa a rodar em todo quadro intraquadro sob os seus limiares compilados por padrão, e a sua atuação é invisível na leitura de um roteiro de experimento que define apenas as variáveis do podador sob teste. Os primeiros roteiros de medição do podador pós-NONE definiram unicamente as suas próprias variáveis de ambiente, deixando o podador pré-busca nos limiares compilados por padrão. Toda linha rotulada como "podador pós-NONE" mediu, deste modo, os dois podadores empilhados.

**A quantificação.** A Tabela I reproduz a quantificação obtida na sequência Neon1224, a `cpu-used=0`, contrapondo a medição contaminada (podador pré-busca em limiares padrão, empilhado) à medição corrigida (podador pré-busca explicitamente neutralizado).

**Tabela I.** Quantificação do fator de confusão, sequência Neon1224, `cpu-used=0`.

| Configuração | Taxa BD (%) | Redução de tempo (%) |
|---|--:|--:|
| τ=0,95, medido antes (empilhado) | 0,267 | 17,15 |
| τ=0,95, isolado | 0,037 | **2,96** |
| τ=0,90, medido antes (empilhado) | 0,270 | 17,36 |
| τ=0,90, isolado | 0,037 | **4,23** |
| τ=0,60, medido antes (empilhado) | 0,386 | 20,53 |
| τ=0,60, isolado | 0,100 | **9,31** |

*Nota de procedência.* `results/thesis/R3_h9c.md` §3.3; artefato `results/benchmark/fase6/raw_results.csv` (linhas `h9ciso_*`); scripts `src/scripts/fase6/{encode_h9c_iso.py, encode_h9adef.py, report_bloco7.py}`.

O podador pós-NONE isolado poda muito pouco: entre 2,96% e 9,31% de redução de tempo, conforme o limiar. **De 82% a 96% da redução de tempo antes atribuída a ele provinha, na verdade, do podador pré-busca** rodando por baixo, nos seus limiares compilados por padrão. A quantificação foi generalizada em quatro sequências CTC adicionais no mesmo ponto de operação: a fração indevidamente atribuída foi de 75% em Neon1224, 58% em PierSeaSide, 28% em Tango e 95% em TimeLapse, com **média de 64%** e dispersão de 28% a 95% — dispersão que estabelece que não se trata de um viés constante descontável por um fator único, e que a contribuição real do podador pós-NONE varia de desprezível a moderada conforme o conteúdo. A configuração empilhada entrega, na média das quatro sequências, 14,6% de redução de tempo; o podador pós-NONE isolado entrega 5,6% — diferença de 9,0 pontos percentuais, muito acima do piso de ruído do experimento pareado, de aproximadamente 0,46 ponto percentual (Seção V).

Como consequência direta desta contaminação, três afirmações preliminares foram formalmente retiradas e não são reproduzidas neste artigo: a de que o podador pós-NONE seria de duas a quatro vezes mais eficiente do que o podador pré-busca; a de que superaria a rede convolucional nativa em eficiência; e o veredito preliminar de que "não sobreviveria ao piloto real", apoiado inteiramente num piloto de dois quadros cuja tabela media exatamente a mesma pilha contaminada.

**A lição generalizável.** Em software de pesquisa, um caminho de execução habilitado exclusivamente por uma condição geométrica, sem interruptor explícito e sem estado observável no registro da execução, é, para efeitos práticos, **indistinguível de código inerte** na leitura de um roteiro de experimento — e a sua atuação silenciosa contamina toda comparação que não o neutralize explicitamente. A prática corretiva adotada, e generalizada a toda medição subsequente deste trabalho, é a neutralização explícita de qualquer alavanca não sob teste, desde o próprio roteiro de medição, como única variável alterada.

---

## V. Protocolo experimental

O escopo experimental restringe-se à predição intraquadro do AV1 em modo *All-Intra*, tendo como âncora o codificador de referência libaom v3.10.0 configurado com `cpu-used=0`, regime em que a busca de partição é exaustiva. A métrica de qualidade é o PSNR-Y; a métrica de eficiência de compressão é a **taxa BD** (Bjøntegaard) sobre esta métrica; a métrica de custo é o tempo de parede, reportado como **redução de tempo (TS)** e como aceleração.

A campanha de substituição direta usa as **oito sequências da Classe A1 das CTC**, em 4K e dez bits, com quinze quadros por sequência — a especificação normativa da própria CTC para o cenário intraquadro —, sob quatro pontos de quantização (`cq-level` 20, 32, 43 e 55), com `--threads=1` para medição determinística de tempo. A campanha soma **192 codificações**, em **seis configurações**, sobre as oito sequências e os quatro pontos de quantização.

Toda taxa BD e toda redução de tempo apresentadas neste artigo provêm desta grade, sem exceção, e isso vale igualmente para as medições auxiliares: a quantificação do fator de confusão da Seção IV, a decomposição de três pernas, a exploração da faixa de limiares e a bancada de custo computacional da Seção VIII foram todas conduzidas sobre sequências desta mesma Classe A1. O conjunto de dados de aprendizado de máquina citado na Seção II não fornece número algum de eficiência de compressão ou de custo de codificação a este texto.

A redução de tempo adota, em todo este artigo, a **definição canônica**: calcula-se, para cada ponto de quantização, a fração de tempo poupada em relação à âncora, tira-se a média sobre os quatro pontos e, então, a média sobre as sequências, de modo que cada ponto de quantização pesa igualmente. Uma definição alternativa, ponderada pelo tempo total somado antes da divisão, circula em documentos internos deste projeto, diverge da canônica em até três pontos percentuais e **reordena o ranqueamento** entre configurações; ela não é usada neste artigo, e as duas definições nunca são combinadas na mesma tabela.

A resolução da comparação pareada de tempo foi medida diretamente, e não suposta: cinco repetições intercaladas de três configurações sobre quatro pontos de quantização, em execução contínua no mesmo contêiner, deram coeficiente de variação mediano de 0,28% do tempo bruto. Para uma configuração de podador equilibrado, o desvio padrão da redução de tempo pareada foi de ±0,23 ponto percentual, o que fixa a **resolução efetiva da comparação pareada em aproximadamente 0,46 ponto percentual** (dois desvios); para o *preset* nativo `cpu-used=1`, a resolução foi de aproximadamente 0,18 ponto percentual. Diferenças de redução de tempo inferiores a esta resolução não são citadas como positivas neste artigo. A medição desta resolução é intra-execução, numa única sequência, e não captura deriva entre dias ou reinícios de contêiner — ressalva retomada na Seção IX.

---

## VI. Resultados agregados e testes pareados

A Tabela II apresenta o agregado da substituição direta sobre a grade CTC completa, contra a âncora `cpu-used=0`, nos três *presets* práticos.

**Tabela II.** Substituição direta do podador nativo pelo podador pós-NONE, grade CTC completa (8 sequências × 4 pontos de quantização).

| *Preset* | Podador | Taxa BD (%) | Redução de tempo (%) | Aceleração |
|:--:|---|--:|--:|--:|
| 1 | CNN nativa | 0,449 | 32,59 | 1,508× |
| 1 | τ=0,90 | 0,448 | 31,65 | 1,498× |
| 1 | τ=0,95 | **0,414** | 30,34 | 1,469× |
| 2 | CNN nativa | 0,536 | 42,72 | 1,788× |
| 2 | τ=0,90 | 0,539 | 42,07 | 1,787× |
| 2 | τ=0,95 | **0,516** | 40,73 | 1,746× |
| 3 | CNN nativa | **2,722** | 67,94 | 3,159× |
| 3 | τ=0,90 | 3,397 | 70,67 | 3,474× |
| 3 | τ=0,95 | 3,384 | 70,25 | 3,419× |

*Nota de procedência.* `results/thesis/R3_h9c.md` §3.4; artefato `results/benchmark/fase6_swap_h9c/swap_average.csv`; scripts `src/scripts/fase6/{encode_swap_h9c.py, report_swap.py}`. Redução de tempo na definição canônica (Seção V).

Na média da grade completa, o podador pós-NONE e a rede convolucional nativa **empatam** nos *presets* 1 e 2, e o podador aprendido é claramente pior no *preset* 3. Os testes t pareados por sequência, com n = 8, sustentam exatamente esta leitura, e não uma leitura mais forte (Tabela III).

**Tabela III.** Testes t pareados (n = 8) contra a rede convolucional nativa.

| Comparação | Δ taxa BD (pp) | p | Sequências favoráveis |
|---|--:|--:|:--:|
| Preset 1, grade completa (τ=0,95) | −0,034 | 0,278 | 4/8 |
| Preset 2, grade completa (τ=0,95) | −0,020 | 0,291 | 5/8 |
| Preset 3, grade completa (τ=0,95) | +0,662 | 0,004 | 0/8 |
| Preset 1, alta qualidade (CQ 20/32) | −0,088 | 0,043 | 6/8 |
| Preset 2, alta qualidade (CQ 20/32) | −0,086 | 0,015 | 7/8 |
| Preset 1, baixa qualidade (CQ 43/55) | +0,009 | não significativo | — |
| Preset 2, baixa qualidade (CQ 43/55) | +0,027 | não significativo | — |
| Substituições análogas do podador pré-busca, todos os regimes | +0,47 a +1,63 | ≤ 0,01 | 0/8 |

*Nota de procedência.* `results/thesis/R3_h9c.md` §3.4; artefatos `results/benchmark/fase6_analysis/{cq_decomposition,ts_per_cq,paired_tests}.csv`.

Nada é significativo na grade completa nos *presets* 1 e 2: a afirmação defensável é de **paridade**, e não de superioridade. No *preset* 3, a substituição é significativamente pior, com nenhuma das oito sequências favorável. Para referência, todas as substituições análogas com o podador pré-busca são significativamente **piores** que a rede nativa em todos os regimes, com diferenças de +0,47 a +1,63 ponto percentual e sequência favorável alguma — o que situa o podador pós-NONE como o único competitivo neste cenário entre os avaliados nesta investigação.

**O podador pós-NONE não é contribuição autônoma quando empilhado sobre o pré-busca.** A Tabela IV decompõe, em quatro sequências CTC, o ganho de tempo de cada podador isolado e a interação entre ambos, medindo o termo que faltava para fechar o balanço `tempo(pré-busca) + tempo(pós-NONE) + interação = tempo(empilhado)`.

**Tabela IV.** Decomposição de três pernas e interação medida, quatro sequências CTC.

| Sequência | Pré-busca só | Pós-NONE só | Empilhados | Soma | Interação |
|---|--:|--:|--:|--:|--:|
| Neon1224 | 16,8% | 4,2% | 17,1% | 21,0% | −3,9 pp |
| PierSeaSide | 10,4% | 5,4% | 12,8% | 15,8% | −3,0 pp |
| Tango | 7,2% | 12,3% | 17,1% | 19,4% | −2,4 pp |
| TimeLapse | 9,2% | 0,6% | 11,5% | 9,8% | +1,7 pp |
| **Média** | **10,9%** | **5,6%** | **14,6%** | **16,5%** | **−1,9 pp** |

*Nota de procedência.* `results/thesis/R3_h9c.md` §3.5; artefato `results/benchmark/fase6/raw_results.csv`; scripts `src/scripts/fase6/{encode_h9adef.py, report_e3_dec_e2.py}`.

A interação é negativa na média, com magnitude de **−1,9 ponto percentual**: os dois podadores empilhados entregam 14,6% de redução de tempo, contra 16,5% da soma das partes, ou seja, cerca de 12% do ganho potencial evapora na sobreposição. Isso ocorre porque os dois podadores, embora decidam em pontos distintos do fluxo de controle, formulam a mesma pergunta sobre o nó — se é possível encerrar a busca ali — e por isso disputam parcialmente o mesmo tempo economizável. Como camada extra sobre o podador pré-busca, o podador pós-NONE entrega, deste modo, um ponto de operação real e implantável, mas **não pode ser creditado como contribuição autônoma** de tempo: a economia que lhe era atribuída pertencia, em média, majoritariamente ao outro podador.

---

## VII. Decomposição por regime de quantização e mecanismo estrutural no código

A média sobre a grade completa anula um efeito real e de sinal oposto entre os dois regimes de quantização, que a decomposição por regime de quantização expõe. No regime de **alta qualidade** (níveis 20 e 32), o podador pós-NONE a τ=0,95 custa menos da metade da taxa BD da rede nativa nos dois *presets* testados: 0,065% contra 0,153% no *preset* 1, e 0,173% contra 0,259% no *preset* 2. A vantagem é estatisticamente significativa e **replica em dois níveis independentes de *preset*** (Tabela III): p = 0,043 com vantagem em 6 de 8 sequências no *preset* 1, e p = 0,015 com vantagem em 7 de 8 sequências no *preset* 2. No regime de **baixa qualidade** (níveis 43 e 55), a vantagem desaparece, com diferenças de +0,009 e +0,027 ponto percentual, nenhuma significativa.

O mecanismo desta dependência de regime foi **verificado no código-fonte do codificador**, e não apenas inferido do padrão empírico. Em codificação *All-Intra*, os três podadores nativos que agem **após** o candidato `PARTITION_NONE` — `av1_ml_predict_breakout`, `av1_ml_early_term_after_split` e `av1_ml_prune_rect_partition` — estão desligados por uma guarda de tipo de quadro, todos sob a condição `!frame_is_intra_only(cm)`. O nicho de decisão que o podador pós-NONE deste artigo ocupa — decidir depois de conhecer o custo RD real do `PARTITION_NONE` — é, deste modo, **nativamente vazio** neste regime de codificação: a rede convolucional nativa atua antes da busca, não depois, e o podador proposto não compete com um mecanismo nativo equivalente em capacidade, e sim preenche um espaço estrutural que a arquitetura de codificação intraquadro do próprio codificador deixa desocupado.

Esta constatação é a que dá a paridade medida na Tabela II o seu sentido correto. O empate não decorre de o perceptrone de trinta e nove atributos igualar a capacidade da rede convolucional; decorre de o codificador, sob este tipo de quadro, não estar disputando o mesmo ponto do fluxo de controle.

A vantagem de qualidade em alta taxa vem, contudo, acompanhada de custo de tempo, e a fronteira de compromisso é o instrumento correto para julgar esta troca, e não a comparação isolada de taxa BD. Nos níveis 20 e 32 do *preset* 1, o podador pós-NONE economiza 20,97% de tempo, contra 25,26% da rede nativa: trocam-se cerca de 4,3 pontos percentuais de tempo por cerca de 0,09 ponto percentual de taxa BD.

---

## VIII. Custo computacional, com escopo declarado

O custo computacional foi medido em duas grandezas distintas, cronometradas dentro de codificações reais com `clock_gettime(CLOCK_MONOTONIC)`, e a distinção entre elas é indispensável para não sobrestimar o argumento (Tabela V). A coluna do podador pré-busca, de trinta e seis atributos, é incluída apenas como referência de contexto, medida na mesma bancada; o objeto deste artigo é o podador pós-NONE, de trinta e nove atributos.

**Tabela V.** Custo computacional por chamada e custo implantado.

| Grandeza | CNN nativa | Perceptrone (podador pré-busca, 36 atributos) | Perceptrone (podador pós-NONE, 39 atributos) |
|---|--:|--:|--:|
| Inferência isolada (ns/chamada) | ≈24.700 (24.763 / 24.606) | ≈486 (484 / 488) | ≈758 (770 / 745) |
| Razão vs. CNN nativa | 1× | ≈1/50 | ≈1/32 |
| Custo implantado, % do tempo de codificação | 0,16%–0,21% | 0,26%–0,32% | [completar: custo implantado específico do podador pós-NONE não reportado nas fontes desta investigação] |

*Nota de procedência.* `docs/RESULTADOS_microbench_pruner.md` §2 (inferência isolada, medição de 2026-07-17 registrada apenas em prosa, sem CSV versionado correspondente) e §6.2/§6.2b (custo implantado, adendo de 2026-07-19), via `results/thesis/M5_arquitetura_software.md` §5.6 e `results/thesis/R6_analise_integrada.md` §6.5; artefato `results/benchmark/microbench/pruner_cost.csv` (fonte apenas do custo implantado de §6, não da inferência isolada de §2); instrumentação `partition_strategy.c:148,156`; script `src/scripts/benchmark/microbench_pruner.py`. A medição cobre duas sequências (Tango a `cq32`, BoxingPractice a `cq43`), três quadros e uma única linha de execução — suficiente para uma razão de ordem de grandeza, e não para uma estimativa de precisão fina.

A inferência do podador pós-NONE, de trinta e nove atributos, é cerca de **trinta e duas vezes mais barata por chamada** do que a da rede convolucional nativa — aproximadamente 758 nanossegundos contra aproximadamente 24.700 nanossegundos. Para referência, o podador pré-busca correlato, de trinta e seis atributos, é ainda mais barato na mesma bancada, a cerca de 486 nanossegundos, ou cerca de um cinquentavo do custo da rede nativa; a diferença entre os dois perceptrones reflete o vetor de entrada maior do podador pós-NONE, e não uma ineficiência de implementação. O **escopo desta razão precisa ser declarado com a mesma precisão com que ela é citada**: mede-se apenas a passagem direta do modelo, excluindo-se deliberadamente a extração de atributos e a frequência de invocação de cada arquitetura — a rede convolucional é chamada uma vez por superbloco, ao passo que o perceptrone é chamado uma vez por nó de decisão. A razão caracteriza, portanto, o algoritmo de decisão isolado, e não o custo do podador implantado.

O **custo implantado**, com a extração de atributos somada à inferência, foi medido diretamente contra o tempo de parede da mesma codificação apenas para o podador pré-busca: entre 0,26% e 0,32% do tempo de codificação, contra 0,16% a 0,21% da rede convolucional nativa. Ambos os caminhos custam, deste modo, **menos de um terço de um por cento** do tempo total de codificação, valor desprezível frente ao que a poda economiza ou desperdiça na busca de taxa-distorção. Um custo implantado equivalente, específico do podador pós-NONE, não foi reportado nas fontes desta investigação (Tabela V). Nenhum resultado de taxa BD contra tempo relatado nas Seções VI e VII precisa ser revisto em função deste custo: os ganhos vêm inteiramente das decisões de poda, e não da leveza de cada inferência. A razão de custo por chamada — trinta e duas vezes para o podador pós-NONE, cinquenta vezes para o pré-busca — **nunca deve ser citada como origem da vantagem de qualidade em alta taxa** relatada na Seção VII; essa vantagem é estrutural, e decorre do nicho nativamente vazio descrito na mesma seção.

---

## IX. Discussão, limitações declaradas e conclusão

**Exploração combinada e do espaço de limiares.** Duas verificações adicionais confirmaram, sem alterar, a leitura anterior. A combinação do podador pré-busca em configuração conservadora com o podador pós-NONE, medida na sequência Tango, não altera a fronteira de compromisso: a rede convolucional nativa permanece no topo da eficiência (razão entre redução de tempo e taxa BD de 81,9 no *preset* 1), o podador pós-NONE isolado vem em seguida (77,2), e as combinações ficam abaixo (67,9 e 65,2). A exploração da faixa de limiares entre τ = 0,30 e τ = 0,60, motivada pela suspeita de um ponto de operação atrativo ali, é um **resultado negativo limpo**: o preço do degrau salta de uma faixa de 0,013–0,042 (τ = 0,95 a 0,70) para 0,107 a partir de τ = 0,60, e o ponto τ = 0,45, medido nas oito sequências, entrega +0,643% de taxa BD a 21,4% de redução de tempo — **estritamente dominado** pelo primeiro degrau nativo (+0,449% a 32,6%). A faixa inexplorada foi explorada e não contém ponto de operação atrativo algum.

**Posição na fronteira de compromisso.** Sobre a fronteira de Pareto recalculada nas oito sequências CTC, o ponto do podador pós-NONE a τ=0,95 no *preset* 1 é não dominado e ocupa taxa BD abaixo do primeiro degrau nativo — 0,414% contra 0,449% — com 93% da redução de tempo correspondente. A rede convolucional nativa mantém, ao longo de toda a fronteira, o pico de eficiência de compromisso, entre 78 e 94 conforme o *preset*; em nenhum nível de velocidade testado um ponto de aprendizado de máquina ocupa a posição que a rede nativa ocupa nos dois eixos simultaneamente. O que a Tabela II e a Tabela III estabelecem é um empate técnico nos dois pontos vizinhos da fronteira nos *presets* 1 e 2 — cada configuração troca uma fração de taxa BD por uma fração de redução de tempo, sem que uma domine estritamente a outra —, e não uma posição superior à da rede nativa.

**Limitações declaradas.** Três limitações restringem o alcance das afirmações deste artigo. Primeiro, a rede convolucional nativa **não possui linha de medição isolada** de taxa BD e tempo: ela só é observável por diferença nos experimentos de substituição, de modo que toda comparação relatada é, por construção, sistema contra sistema, e nenhum par (taxa BD, tempo) próprio pode ser atribuído a ela fora do arranjo de substituição. Segundo, a **decomposição de três pernas** da Tabela IV cobre quatro das oito sequências CTC, com dispersão de 28% a 95% na fração de tempo indevidamente atribuída; a conclusão robusta desta decomposição é a direção e a ordem de grandeza do efeito, e não o valor central exato. Terceiro, a **resolução temporal** de aproximadamente 0,46 ponto percentual foi medida numa única sequência, em janela contínua de execução no mesmo contêiner, e não captura deriva entre dias, reinícios de contêiner ou estados térmicos distintos; comparações entre campanhas distantes no tempo exigem, por isso, cautela adicional.

**Conclusão.** Este artigo isolou, por substituição direta com dupla neutralização explícita, o mérito de um podador de particionamento pós-NONE de trinta e nove atributos contra a rede convolucional nativa de poda de partição do AV1 em codificação intraquadro. Um episódio de fator de confusão — a atuação silenciosa de um segundo podador habilitado apenas por geometria, sem interruptor explícito — foi identificado, quantificado em até 96% do tempo indevidamente atribuído numa sequência e 64% em média sobre quatro, e corrigido, deixando como lição generalizável de arquitetura de software que caminhos de execução experimentais precisam de habilitação explícita e estado observável. Sob a comparação corrigida, o podador pós-NONE **empata** com a rede convolucional nativa nos *presets* práticos 1 e 2, sem que a média da grade CTC sustente superioridade, e vence de forma estatisticamente significativa no regime de alta taxa, replicado em dois níveis independentes de *preset*; perde de forma clara no *preset* 3. A causa não é capacidade de modelo: é que os três podadores nativos que agem depois do `PARTITION_NONE` estão estruturalmente desligados em codificação *All-Intra*, de modo que o nicho que o podador pós-NONE ocupa é nativamente vazio. Um perceptrone de trinta e nove atributos, cuja inferência é cerca de trinta e duas vezes mais barata por chamada — sob o escopo declarado, e nunca como origem da vantagem de taxa BD —, alcançar paridade com uma rede convolucional sintonizada e embarcada no codificador de referência é, por si, resultado; superioridade na média da grade não o é, e não é aqui reivindicada. Como camada extra sobre o podador pré-busca do mesmo codificador, este podador pós-NONE **não constitui contribuição autônoma** de tempo de codificação, uma vez que a maior parte do ganho antes atribuído a ele pertencia ao outro podador.

---

## X. Referências

[1] [completar: referência normativa do codificador AV1 / especificação AOMedia]

[2] [completar: referência do libaom, versão v3.10.0]

[3] [completar: referência do método de taxa BD de Bjøntegaard]

[4] [completar: referência da especificação de condições comuns de teste (CTC) da AOM utilizada, versão 9]

[5] [completar: referência(s) de trabalhos relacionados de poda aprendida de particionamento em codificadores de vídeo, a levantar em revisão de literatura]

[6] [completar: confirmar prazo de submissão, número exato de páginas e demais requisitos formais no CFP do ISCAS 2027]

---

## Especificação de figuras

Nenhuma figura existe ainda no projeto (lacuna L11 de `A3_RETRATACOES_E_LACUNAS.md`). Esta seção especifica as figuras mínimas exigidas pela argumentação deste artigo, para produção posterior.

**F1 — Antes e depois da neutralização, por sequência.**
*Legenda proposta.* A Figura 1 apresenta a redução de tempo atribuída ao podador pós-NONE antes e depois da neutralização explícita do podador pré-busca, por sequência, evidenciando que a maior parte do valor medido antes da correção pertencia ao outro podador.
*O que plota.* Barras pareadas por sequência (Neon1224, PierSeaSide, Tango, TimeLapse): barra "medido antes" (configuração empilhada, podador pré-busca em limiares compilados por padrão) contra barra "isolado" (podador pré-busca neutralizado), em redução de tempo (%); anotação da fração atribuída incorretamente (75%, 58%, 28%, 95%).
*Dado de origem.* `results/benchmark/fase6/raw_results.csv` (linhas `h9adef`, `h9ciso_tau90`, `ml_balanced`, quatro sequências CTC), conforme Tabela IV deste artigo.
*Roteiro sugerido.* Barras agrupadas por sequência com `matplotlib`, reaproveitando o carregamento já usado por `src/scripts/fase6/report_e3_dec_e2.py`; eixo vertical em pontos percentuais de redução de tempo, sem recálculo de agregação nova.

**F2 — Taxa BD por regime de quantização, nativa contra aprendida, nos dois presets, com barras de significância.**
*Legenda proposta.* A Figura 2 apresenta a taxa BD da rede convolucional nativa e do podador pós-NONE (τ=0,95), separada por regime de quantização (alta qualidade: CQ 20/32; baixa qualidade: CQ 43/55), nos *presets* 1 e 2, com indicação do valor de p do teste t pareado (n = 8) em cada par de barras.
*O que plota.* Quatro grupos de barras pareadas (preset 1 alta, preset 1 baixa, preset 2 alta, preset 2 baixa), cada grupo com duas barras (nativa, podador pós-NONE) e um asterisco de significância (p < 0,05) sobre os dois grupos de alta qualidade.
*Dado de origem.* `results/benchmark/fase6_analysis/{cq_decomposition,ts_per_cq,paired_tests}.csv`, conforme Tabela III deste artigo.
*Roteiro sugerido.* Barras agrupadas com `matplotlib`, leitura direta do CSV de decomposição por CQ já produzido por `src/scripts/benchmark/analyze_frontier.py`; anotação de significância a partir da coluna de p-valor de `paired_tests.csv`.

**F3 — Fronteira de compromisso com os pontos de substituição.**
*Legenda proposta.* A Figura 3 apresenta a fronteira de compromisso entre taxa BD e redução de tempo nos três *presets* práticos, com os pontos da rede convolucional nativa e do podador pós-NONE (τ=0,90 e τ=0,95) marcados, evidenciando o empate técnico nos *presets* 1 e 2 e a derrota no *preset* 3.
*O que plota.* Dispersão com taxa BD (%) no eixo vertical e redução de tempo (%) no eixo horizontal, um ponto por combinação de *preset* e podador (nove pontos, conforme a Tabela II), com rótulo de *preset* em cada ponto e a fronteira não dominada destacada por linha.
*Dado de origem.* `results/benchmark/fase6_swap_h9c/swap_average.csv` e, para a posição relativa na fronteira global de referência, `results/benchmark/fase6_analysis/pareto_frontier.csv`.
*Roteiro sugerido.* Dispersão com `matplotlib`, seguindo o padrão já especificado para a fronteira global em `results/thesis/A2_TABELAS_E_FIGURAS.md` (Figura 9), restrito às configurações da Tabela II deste artigo.

---

## Conformidade

> **Apêndice interno de trabalho — NÃO faz parte da submissão.** Esta seção existe para auditoria interna do texto contra as retratações e lacunas registradas na tese de origem, e deve ser removida antes de qualquer envio. Ela é o único ponto deste arquivo em que decisões editoriais e a existência de outros trabalhos derivados da mesma tese são mencionadas.

Esta seção declara, conforme exigido pelas regras editoriais de `00_PLANO_capitulos.md` §6.5, as retratações de `A3_RETRATACOES_E_LACUNAS.md` verificadas contra este texto, as lacunas conhecidas declaradas e os itens `[completar: ...]` pendentes.

**(a) Retratações verificadas e forma de respeito no texto.**

- **R6** (o podador pós-NONE seria "2–4× mais eficiente" que o pré-busca, ou superior à rede nativa em eficiência). Não reproduzida. O artigo cita apenas a paridade estatística nos *presets* 1 e 2 e a derrota no *preset* 3 (Seções VI–VII, Tabelas II e III).
- **R7** (a cota mais informativa "não se traduz em vantagem de tempo de parede", apoiada no piloto contaminado). Não reproduzida. O artigo relata explicitamente que o veredito preliminar foi retirado (Seção IV) e substitui-o pela medição limpa de substituição direta.
- **R8/R9** (não-aditividade de podadores como limite informacional / propriedade absoluta da ação). Não invocadas: o artigo atribui a interação negativa da Tabela IV a sobreposição de ação entre podadores que perguntam a mesma coisa sobre o nó (Seção VI), nunca a limite informacional.
- **R10** (leveza de inferência como vantagem). Não reproduzida. A Seção VIII declara explicitamente o escopo da razão de ~50× e afirma que o custo de inferência "nunca deve ser citado como origem da vantagem de qualidade em alta taxa".
- **R20** (piso de ruído suposto de 1–2%). Não reproduzida. A Seção V usa exclusivamente a resolução medida de ~0,46 ponto percentual.
- **R21** (joelho da curva suposto dentro do vão [30,60] e τ=0,45 como ponto atrativo). Não reproduzida. A Seção IX declara o joelho em τ ≈ 60–70 e τ=0,45 como estritamente dominado pelo *preset* nativo.
- **R23** (mistura das duas definições de redução de tempo). Não reproduzida. A Seção V declara e adota exclusivamente a definição canônica; a definição alternativa é mencionada apenas para registrar a divergência e a reordenação de ranqueamento que produziria, nunca usada em tabela.

**(b) Lacunas declaradas explicitamente no texto.**

- **L4** — a rede convolucional nativa não possui linha de medição isolada de taxa BD e tempo; toda comparação é sistema contra sistema (declarado na Seção IX, "Limitações declaradas").
- **L6** — a decomposição de três pernas cobre quatro das oito sequências CTC, com dispersão de 28% a 95% (declarado na Seção IX e na nota da Tabela IV).
- **L8** — a resolução temporal de ~0,46 ponto percentual é intra-execução, medida numa única sequência, e não cobre deriva entre dias ou reinícios de contêiner (declarado na Seção V e retomado na Seção IX).
- **L11** — nenhuma figura existe ainda no projeto; as três figuras deste artigo são especificações para produção posterior, não gráficos entregues.

**(c) Itens `[completar: ...]` pendentes neste artigo.**

1. Custo implantado específico do podador pós-NONE (extração + inferência, % do tempo de codificação) — não reportado nas fontes permitidas para este podador especificamente; apenas a inferência isolada (≈758 ns/chamada) e o custo implantado do podador pré-busca (0,26%–0,32%) estão medidos (Tabela V).
2. Seis referências bibliográficas da Seção X, incluindo a confirmação de prazo, número de páginas e demais requisitos formais do CFP do ISCAS 2027.

**(d) Correções de procedência aplicadas nesta revisão.**

1. **Razão de custo de inferência atribuída ao modelo errado.** O objeto deste artigo é o podador pós-NONE, de trinta e nove atributos, cuja inferência isolada mede ≈758 ns/chamada (770 e 745 ns nas duas execuções registradas, `M5_arquitetura_software.md` §5.6) — cerca de um trinta e dois avos do custo da rede convolucional nativa, e não um cinquentavo. O abstract, a Seção VIII e a Conclusão citavam, por engano, o valor de ≈486 ns e a razão de ≈1/50, que pertencem ao podador pré-busca correlato, de trinta e seis atributos. Corrigido em todos os quatro pontos: o valor de ≈486 ns/≈1/50 permanece apenas na Tabela V e na Seção VIII como linha de contexto do outro podador, explicitamente rotulada como tal, e nunca mais como propriedade do modelo deste artigo.
2. **Evidência de inércia atribuída a outra campanha.** A Seção III citava, para o arranjo de substituição direta deste artigo, uma verificação byte a byte (1.574.775 bytes, PSNR-Y de 40,9720 dB) que pertence à campanha de outra alavanca (`M5_arquitetura_software.md` §5.5, verificação de integridade do H9d, não do arranjo de substituição do podador pós-NONE). Corrigida para a garantia de inércia geral que de fato cobre este arranjo: quadro de 448×256 pixels, intraquadro a `cpu-used=0` e `cq=32`, resumo criptográfico `b904f11c9fa02d5ea25460cb976ef29d` idêntico entre os dois fluxos, 172.076 bytes decodificados com sucesso e testes `EncodeAPI.AllIntra` 3 de 3.
3. **Nota de procedência da Tabela V incompleta quanto à fonte do dado.** A nota citava `results/benchmark/microbench/pruner_cost.csv` como se fosse a origem dos valores de inferência isolada (ns/chamada). `M5_arquitetura_software.md` §5.6 registra que esse CSV é fonte apenas do custo implantado (§6 do documento de origem), e que a inferência isolada (§2) está registrada apenas em prosa, sem CSV versionado correspondente. A nota foi corrigida para refletir esta distinção e para declarar o escopo da medição — duas sequências, três quadros, uma única linha de execução —, suficiente para uma razão de ordem de grandeza e não para uma estimativa de precisão fina.
