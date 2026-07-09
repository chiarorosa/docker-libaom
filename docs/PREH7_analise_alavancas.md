# Pré-H7 — Análise crítica: alavancas de redução de busca não exploradas

**Pergunta.** Existe algo não explorado — na arquitetura ConvNeXt treinada do zero
(validada em H1–H6) ou no estudante MLP — capaz de viabilizar uma redução de busca
substancialmente maior, mantendo alinhamento com a infraestrutura do libaom?

**Resposta-síntese.** Sim, e a maior alavanca **não é o modelo**: é o **espaço de
ações da política de poda**. A política atual usa apenas duas das três saídas do
estudante e a métrica da simulação oráculo conta **nós**, não **custo de busca**.
Com isso, o termo dominante do custo em cpu-used=0 — a avaliação dos candidatos
retangulares/AB/4-way em cada nó — é invisível e intocado. Corrigir o espaço de
ações e a métrica multiplica a redução alcançável **sem exigir mais informação da
entrada**, isto é, sem colidir com o limite de informação da luminância
estabelecido em H3.

---

## 1. Diagnóstico quantitativo do custo (por que a política atual satura)

Em cpu-used=0 All-Intra, cada nó de particionamento com dimensão ≥16 px avalia
até **9 candidatos de forma** (NONE, HORZ, VERT, HORZ_A, HORZ_B, VERT_A, VERT_B,
HORZ_4, VERT_4); o SPLIT em si não é uma forma avaliada no nó — seu custo está nos
filhos. Cada candidato custa aproximadamente a área do bloco (todas as formas
cobrem os n² pixels do bloco). Modelo de custo por superbloco 64×64:

| Nível | Nós | Cand./nó | Custo (unid. pixel) | Fração |
|------:|----:|---------:|--------------------:|-------:|
| 64    | 1   | 9        | 36 864              | 29 %   |
| 32    | 4   | 9        | 36 864              | 29 %   |
| 16    | 16  | 9        | 36 864              | 29 %   |
| 8     | 64  | 4        | 16 384              | 13 %   |

Duas consequências:

1. **Os candidatos rect/AB/4-way são 8 dos 9 candidatos** em todos os níveis ≥16,
   ou seja, ~77 % do custo total modelado. A política atual (NONE-commit e
   force-SPLIT) só elimina **subárvores**; um nó não podado paga os 9 candidatos
   integralmente.
2. A métrica `search_reduction_pct` da simulação conta nós eliminados — um proxy
   que **subestima** o ganho das ações de nó (NONE-commit também elimina 8/9 dos
   candidatos do próprio nó) e **não enxerga** qualquer ação em nível de
   candidato. O ponto operacional "8,5 % de redução" mede outra grandeza que não
   o tempo.

## 2. Alavancas identificadas (ranqueadas por ganho esperado / esforço)

### A1 — Poda de candidatos retangulares via P(REST) — **a alavanca dominante**
O estudante já emite `[P(NONE), P(SPLIT), P(REST)]`; a política descarta P(REST).
Nova ação: `P(REST) < τ_rest → av1_disable_rect_partitions()` (mantém a busca
NONE vs SPLIT). Verificado no código: essa chamada desabilita HORZ/VERT **e**,
por dependência de `do_rectangular_split`, também AB e 4-way
(`partition_search.c:4090/4149/5077`) — mapeamento nativo, sem código novo de
busca. Fundamentos:

- **A priori favorável:** os rótulos REST são ~5–10 % dos nós; em ~90 % dos nós o
  codificador avalia 8 candidatos apenas para rejeitá-los. Prever "REST é
  improvável" é um subproblema mais fácil do que prever a partição exata — o
  limite de informação de H3 refere-se à decisão completa, não a este binário
  desbalanceado.
- **Risco limitado e gradual:** desabilitar rect nunca invalida o fluxo; o custo
  em taxa BD só ocorre quando a partição ótima era rect/AB/4-way, e nesses casos
  a alternativa NONE/SPLIT é frequentemente um quase-empate em RD (evidência:
  matriz de custo geométrico — confusões HORZ↔HORZ_A são baratas).
- **Escala do ganho:** poda de rect em 70 % dos nós ≈ 54 % do custo modelado,
  contra 8,5 % (em nós) do ponto atual. Mesmo com cobertura conservadora, o
  ganho de tempo esperado é múltiplos do atual.

### A2 — Métrica de custo ponderado na simulação oráculo (pré-requisito de A1)
Substituir a contagem de nós por **unidades de custo**: Σ (candidatos avaliados ×
n²) sobre a árvore, com as três ações refletidas (NONE-commit corta subárvore + 8/9
do nó; force-SPLIT corta 8/9 do nó; rect-off corta 7/9 — mantém NONE e a recursão).
Sem isso, o ponto operacional de H7 é escolhido contra o alvo errado. A contagem
de nós permanece reportada por comparabilidade com H1–H6.

### A3 — τ por nível de bloco + calibração (destrava a faixa intermediária de risco)
A curva do estudante é uma função-degrau (valores idênticos para τ∈[0,84–0,88]):
as probabilidades saturam e um único τ global força um compromisso único para
64/32/16, que têm distribuições e custos muito distintos. Ações: limiares
independentes por nível (`AV1_STUDENT_TAU_{NONE,SPLIT,REST}_{64,32,16}`) e
varredura fina. Custo de implementação trivial; é a resposta direta à "transição
abrupta" apontada em H6.

### A4 — Contexto hierárquico nos atributos do estudante (pai/irmãos/posição)
Parte do hiato estudante (8,5 %) → substituto (13,6 %) é **contexto espacial**: o
ConvNeXt vê o superbloco inteiro; o MLP vê só o próprio bloco. Atributos novos,
computáveis identicamente do pkl e em C (pixels do quadro-fonte, aritmética de
ponteiros a partir de mi_row/mi_col): variância do bloco-pai (2n×2n), contraste
com irmãos (demais quadrantes do pai), posição do bloco no superbloco. Custo de
inferência desprezível; nenhuma re-extração de dataset.

### A5 — Capacidade do MLP ([24,16] → [64,32])
O estudante atual tem ~700 parâmetros por tamanho; os NN_CONFIGs nativos do
libaom usam camadas de 64+. O custo do `av1_nn_predict` é desprezível diante de
uma única avaliação RD de candidato. Elevar a capacidade junto com A4 ataca o
gargalo de "capacidade descritiva" apontado em H5/H6.

### A6 — (diferida → H9) Gancho pós-NONE com atributos de RD
A única alavanca que **eleva o teto de informação**: mover a decisão para depois
da avaliação de PARTITION_NONE e alimentar o estudante com rdcost/taxa/distorção
do NONE — exatamente o padrão dos podadores nativos (`ml_early_term_after_part_split`,
`av1_ml_prune_rect_partition`). Exige re-instrumentação (registrar RD por nó) e
novo dataset; fica como plano de contingência se H7 (com A1–A5) for insuficiente.

### A7 — (diferida) Contexto de vizinhança na entrada do substituto
Entrada 96×96 ou atributos do superbloco vizinho. Mexe no teto (H9/H10), esforço
alto, não bloqueia H7.

## 3. Decisão para o H7

Incorporar **A1 + A2 + A3 + A4 + A5** (nenhuma exige novo dataset; todas mapeiam
em mecanismos nativos do libaom) e então executar o H7 completo: sincronizar os
atributos em C, re-destilar, escolher o ponto operacional pela métrica de custo
ponderado, verificar equivalência bit-a-bit com a flag desligada e medir taxa BD +
tempo reais em Jockey. O teto do substituto (13,6 % em nós @ 0,66 % de SPLIT
perdido) permanece o referencial para H8.

**Risco/limitação declarada:** o modelo de custo (candidatos × n²) é um proxy; o
árbitro final é o tempo de parede do H7. A poda de rect não é registrada pela
contagem de nós — os dois números serão reportados lado a lado.

---

## 4. Resultados da simulação oráculo v2 (Jockey held-out, 1,4 M nós)

> **AVISO (2026-07-09): as tabelas desta seção foram geradas com modelos
> treinados sobre luma corrompida (todo zero — ver bug abaixo) e estão
> SUPERADAS.** Os números reais (modelos re-treinados sobre pixels de verdade)
> estão na §6. Mantidas aqui apenas para registro histórico do raciocínio das
> alavancas.
>
> **Bug da luma em branco.** Descobriu-se, ao validar o H7 fim-a-fim, que o
> dataset armazena a luma como `float32` em [0,1], mas os consumidores de treino
> assumiam `uint8` [0,255]: `features.py` fazia `astype(int64)` (0,11 → 0,
> zerando todo atributo manual) e o carregador do substituto dividia por 255 de
> novo. **Substituto e estudante treinaram sobre imagem em branco.** O encoder em
> C lê `uint8` real, então na inferência o modelo recebia atributos numa escala
> totalmente distinta → previsões degeneradas → poda catastrófica (+6,4 % de taxa
> BD, limiares inertes). Verificou-se que `round(luma×255)` reproduz o quadro-
> fonte **exatamente** (o dado é íntegro; sem re-extração); corrigiu-se no
> carregador (`data._denorm_uint8`) e re-treinou-se a cadeia. **Consequência
> teórica: o "teto de informação da luminância" de H1–H6 foi medido sobre entrada
> vazia e precisa ser relido à luz da §6.**

Confirmação quantitativa das alavancas. Colunas: redução de **nós** (métrica
antiga, comparável a H1–H6) e de **custo** (proxy de tempo, A2); risco de RD.

**Substituto (teto de informação da luminância):**

| τ_none | τ_rest | nós% | **custo%** | splitLost% | rectLost% | rectOff⊘% |
|------:|------:|----:|----:|----:|----:|----:|
| 0,80 | −1 (política antiga) | 13,6 | 9,2 | 0,66 | 8,1 | — |
| 0,80 | 0,20 | 13,6 | **29,1** | 0,66 | 8,1 | 8,4 |
| 0,90 | 0,20 | 0,0 | **27,9** | **0,00** | **0,00** | 5,1 |

**Estudante `student_ctx` (24 atributos, o que roda em C):**

| τ_none | τ_rest | nós% | **custo%** | splitLost% | rectLost% | rectOff⊘% |
|------:|------:|----:|----:|----:|----:|----:|
| 0,84 | −1 (política antiga) | 15,2 | 10,5 | 0,84 | 10,8 | — |
| 0,95 | 0,20 | 0,0 | **41,2** | **0,00** | **0,00** | 6,3 |
| 0,88 | 0,20 | 3,8 | 41,6 | 0,07 | 3,3 | 7,7 |
| refinado por nível | | 16,0 | 24,2 | 0,93 | 11,4 | 7,9 |

**Leitura.**
1. **A métrica antiga enganava.** O "8,5–13,6 %" que reportávamos era redução de
   *nós*; na grandeza que corresponde a tempo (custo), a política antiga entrega
   só ~9–10,5 %.
2. **A poda de retangulares (A1) é a alavanca dominante e barata.** Em
   τ_none=0,95/τ_rest=0,20 o estudante corta **~41 % do custo com zero** partição
   NONE forçada errada e apenas 6,3 % das desativações de rect sobre um nó cuja
   forma ótima era retangular. É a evidência mais forte da tese Pré-H7.
3. **O contexto hierárquico (A4) fechou o hiato.** O estudante v3 iguala/supera o
   substituto no eixo de custo (o gargalo de H5/H6 era descritividade, não só
   informação da entrada).

**Pontos operacionais escolhidos para o H7 (benchmark BD-rate + tempo, Jockey):**
- **P0 (âncora da nossa política antiga):** τ=0,84 / 0,90 / −1.
- **P_rect (agressivo, tese Pré-H7):** τ=0,95 / 0,90 / 0,20 — poda de rect pura.
- **P_ref (refinado por nível):** limiares 0,95/0,95/0,80 · 0,90/0,90/0,90 ·
  0,10/0,20/−1 (níveis 16/32/64).

---

## 5. Correção do bug e re-medição (§4 → §6)

A validação fim-a-fim do H7 revelou o bug da luma em branco (detalhado no aviso
da §4). Após corrigir o carregador e re-treinar toda a cadeia sobre pixels reais:

- **Substituto:** macro-F1 subiu de **0,12 → 0,20**. Há sinal real na luminância
  que o artefato de entrada vazia escondia; ainda assim o problema permanece
  difícil (0,20 em 10 classes), coerente com a decisão de particionamento ser
  guiada por taxa-distorção/contexto, não só pela textura.
- **Simulação oráculo (estudante real, Jockey):** agora os limiares **controlam**
  o compromisso (o estudante cego era inerte). Ponto τ_none=0,95/τ_rest=0,20:
  **34,7 % de redução de custo a 0,01 % de SPLIT perdido e 1,5 % de rect-off
  errado**; ponto refinado 36,9 % a 0,88 % de SPLIT perdido.

## 6. Resultados reais do H7 + H8 (Jockey held-out, cpu-used=0)

Benchmark fim-a-fim com a cadeia corrigida (taxa BD vs. âncora libaom v3.10.0 de
controle; speedup = tempo âncora / tempo teste; ponto de operação seguro):

| ponto | política | taxa BD % | speedup |
|---|---|---:|---:|
| P0 | só NONE-commit (política antiga) | 0,25 | 1,03× |
| P_rect | + poda de retangulares (A1) | 0,49 | 1,05× |
| **P_ref** | refinado por nível | **0,42** | **1,07×** |
| H8 | substituto (teto, via replay) | −0,11 | 1,02× |

**Leitura honesta.**
1. **A poda é segura.** Corrigido o bug, a taxa BD é **desprezível (~0,4 %)**; o
   H8 (substituto) chega a um ganho marginal (−0,11 %), confirmando que o modelo
   real é preciso — quase nunca poda uma decisão que a busca RD não tomaria.
2. **O ganho de tempo é modesto (~3–7 %).** Bem abaixo dos ~35 % de "redução de
   custo" previstos pela simulação. O modelo de custo (candidatos × n²)
   **superestima** o tempo real: no cpu-used=0 a maior parte do tempo por nó está
   na busca de modo/transformada (aproximadamente constante por bloco), não no
   número de formas candidatas; desativar retangulares elimina candidatos
   relativamente baratos, mantendo a recursão de SPLIT (a parte cara).
3. **A alavanca do Pré-H7 ajuda, mas pouco.** P_rect > P0 (1,05× vs 1,03×), e o
   refinado por nível é o melhor compromisso (1,07× a 0,42 %). O teto útil da poda
   guiada só por luminância é baixo — o que corrobora, agora com dado limpo, uma
   versão matizada de H1–H3: a luminância informa, mas o grande ganho exige o
   contexto de taxa-distorção na entrada (H9).
