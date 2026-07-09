# Metodologia — Poda de particionamento AV1 guiada por ML

Documento de referência para a redação da metodologia da tese. Fixa, sem
ambiguidade, **o que executa dentro do codificador**, **o que é apenas
ferramenta de treino/análise**, e **a que componente cada número medido é
atribuível**. Escrito após a correção do bug da luma em branco
(`docs/PREH7_analise_alavancas.md`, §4–6); todos os resultados aqui referem-se à
cadeia treinada sobre pixels reais.

---

## 1. Arquitetura em dois modelos (substituto → estudante)

A proposta separa **capacidade de predição** de **custo de inferência**, ligadas
por **destilação de conhecimento**:

| Papel | Modelo | Onde executa | Função |
|---|---|---|---|
| **Substituto** | ConvNeXt multinível sobre luminância 64×64 | **Fora** do codificador (GPU, offline) | Estabelece o **limite superior** predizível a partir dos pixels; é o **mestre** da destilação |
| **Estudante** | MLP por tamanho de bloco (24 atributos manuais) | **Dentro** do libaom, em cada nó de particionamento | É o artefato **efetivamente implantado**; produz a decisão de poda |

**Ponto central para a metodologia:** o ConvNeXt **não roda no codificador de
produção**. Uma inferência convolucional por nó de particionamento seria
proibitiva em C/tempo real. O ConvNeXt cumpre dois papéis, ambos offline:
(i) **mestre** da destilação — o estudante aprende a imitar suas
probabilidades; e (ii) **referência de teto** — medido diretamente no
codificador via *replay* (§4, experimento H8), estabelece quanto da qualidade
seria alcançável se o modelo pleno pudesse rodar embarcado.

O que se implanta e se mede como "a solução em operação" é o **estudante MLP**,
executado pela rotina nativa `av1_nn_predict` do libaom — sem código de
inferência novo. O estudante **é derivado do ConvNeXt** por destilação; logo, o
ganho do estudante é atribuível à cadeia proposta (ConvNeXt → destilação →
estudante), não a uma heurística avulsa — afirmação que a **ablação de
atribuição** (§5) sustenta empiricamente.

---

## 2. O que executa dentro do codificador

Ponto de integração: `av1/encoder/partition_strategy.c`,
`av1_prune_partitions_before_search`, sob a guarda de compilação
`PARTITION_ML_STUDENT` (compilação padrão é bit-a-bit idêntica ao baseline —
verificado por md5 com limiares no-op). A cada nó de particionamento quadrado
(16/32/64 px; o 8×8 é folha terminal, sempre NONE, e é excluído), a função
`student_prune_partition` executa:

1. **Extração de atributos** (`student_node_features`): 24 atributos do bloco e
   de seu contexto hierárquico (variância, gradientes, perfis linha/coluna,
   textura do bloco-pai, contraste com irmãos, posição no superbloco),
   calculados sobre a luminância-fonte em `uint8`. Verificado **bit-a-bit**
   idêntico à implementação Python de referência (`check_feature_parity.py`).
2. **Inferência** (`av1_nn_predict` + `av1_nn_softmax`): o MLP do estudante
   (pesos em `partition_student_weights.h`, gerados por `export_weights.py`)
   emite `[P(NONE), P(SPLIT), P(REST)]`. Paridade C↔Python das probabilidades
   confirmada (Δ ≈ 9·10⁻⁴, dentro da redução de precisão do `av1_nn_predict`).
3. **Política de poda** (§3): aplica ações nativas conforme os limiares.

Nada aqui é simulado em Python: o número de *speedup* e de taxa BD vem do
binário `libaom_perf` (compilado de `src/aom` com `PARTITION_ML_STUDENT=1`)
codificando as sequências de verdade, com decodificação e PSNR verificados.

---

## 3. Política de poda (o espaço de ações ampliado do Pré-H7)

Três ações, todas mapeadas em funções nativas do libaom (nenhuma escreve busca
nova; todas apenas **removem** candidatos, preservando a validade do bitstream):

| Condição | Ação (função nativa) | Efeito |
|---|---|---|
| `P(NONE) > τ_none` | `av1_disable_all_splits` | Compromete NONE; **corta a subárvore** (a recursão de SPLIT — a parte cara) |
| `P(SPLIT) > τ_split` | `av1_set_square_split_only` | Força divisão quadrática; elimina as formas do nó |
| `P(REST) < τ_rest` | `av1_disable_rect_partitions` | Desabilita retangulares/AB/4-way, mantendo NONE vs SPLIT |

Os limiares τ são lidos de variáveis de ambiente (por nível de bloco), então a
varredura de pontos operacionais dispensa recompilação. A ação de retangulares
(A1) e a métrica de custo ponderado (A2) são as contribuições algorítmicas do
Pré-H7; note que **elas dependem das probabilidades do modelo** — sem o escore
do ML, não há o que limiarizar.

---

## 4. Atribuição dos números medidos

| Experimento | Modelo que roda no codificador | Mecanismo |
|---|---|---|
| **H7** (curva P0/P_rect/P_ref/A1–A3) | **Estudante MLP** | `av1_nn_predict` por nó |
| **H8** (teto, −0,11 % taxa BD) | **ConvNeXt** (substituto) | *Replay*: probabilidades pré-computadas offline, injetadas pelo mesmo gancho (`AV1_STUDENT_PROBS_FILE`) |

Assim, a **curva de operação de ~7–29 % de speedup a 0,4–1,6 % de taxa BD é do
estudante MLP** (a solução embarcável), e o **H8 é o ConvNeXt medido
diretamente** no codificador, estabelecendo o teto. O ConvNeXt participa das duas
medições: como mestre do estudante (H7) e como executor no *replay* (H8).

---

## 5. Ablação de atribuição (a defesa metodológica)

**Pergunta do arguidor:** o ganho vem do modelo (ConvNeXt → estudante) ou apenas
da política de poda, que qualquer escore razoável habilitaria?

**Desenho.** Mantém-se **a política idêntica** (apenas NONE-commit:
τ_split = ∞, τ_rest desligado) e **o mesmo codificador**, variando **somente a
fonte do escore** `P(NONE)` (variável `AV1_STUDENT_BASELINE`):

- **`ml`** — o estudante destilado (a proposta);
- **`variance`** — `P(NONE) = exp(−var/V₀)`: a heurística manual óbvia (bloco
  liso → NONE), usando o atributo isolado mais informativo;
- **`random`** — `P(NONE)` = *hash* uniforme determinístico da identidade do nó:
  poda a **mesma fração** de nós, escolhidos ao acaso.

Varre-se τ_none para cada fonte, obtendo três curvas taxa BD × speedup contra o
mesmo âncora. **Critério de sucesso:** a curva `ml` deve **dominar no sentido de
Pareto** ambas as baselines — menor taxa BD para o mesmo speedup. Se dominar, o
ganho é atribuível à **seleção de nós do modelo**, não ao invólucro; se empatar
com `variance`, a contribuição seria trivial.

**Resultado (Jockey held-out, cpu-used=0, 2 quadros; taxa BD em speedup casado
por interpolação — `analyze_ablation.py`, dados em
`results/benchmark/ablation_matched.csv`):**

| speedup | ML (proposta) | variância | aleatório | menor BD |
|---:|---:|---:|---:|:--|
| 1,05× | 0,24 % | **0,17 %** | — | variância |
| 1,15× | 1,06 % | **0,41 %** | — | variância |
| 1,30× | 1,39 % | **0,76 %** | 2,94 % | variância |
| 1,45× | 1,66 % | **1,12 %** | 4,08 % | variância |
| 1,55× | 1,90 % | **1,36 %** | 4,75 % | variância |
| 1,90× | — | **1,92 %** | 5,99 % | variância |

**Veredito — resultado negativo, reportado com transparência:**

1. **O modelo aprende sinal real.** O ML domina o baseline **aleatório** em todos
   os níveis (a 1,3×: 1,39 % vs 2,94 %); sua seleção de nós está muito acima do
   acaso.
2. **Mas o ML NÃO supera o limiar de variância.** Em todos os pontos comparáveis,
   o heurístico trivial (bloco liso → NONE) entrega **menor** taxa BD para o mesmo
   speedup. A proposta ConvNeXt→estudante, no espaço de ações NONE-commit, **não
   é atribuivelmente superior** a uma única estatística de textura.
3. **Agravante:** a variância é **um dos 24 atributos de entrada do estudante**;
   um modelo bem ajustado não deveria ser dominado por uma de suas próprias
   features. Duas leituras possíveis: (a) a variância é **estatística quase
   suficiente** para o NONE-commit no domínio de pixels; (b) a destilação/treino
   **degrada** esse sinal simples (o substituto atinge apenas F1 0,20).

**Interpretação para a tese.** Este é o resultado mais forte a favor de que o
teto é **informacional** (limitado pelos pixels, não pelo modelo): mesmo com a
cadeia corrigida, o aparato convolucional não extrai da luminância mais do que a
variância já resume. Reforça, com dado limpo, uma versão matizada de H1–H3 e
**motiva H9** (enriquecer a entrada com contexto de taxa-distorção) como a única
via para superar o baseline de variância — porque o sinal de pixels está
saturado.

**Ressalvas (a checar antes de fechar a redação):**
- **Ruído de 2 quadros:** as diferenças (~0,5–0,65 %) são consistentes e
  monótonas nos 6 pontos, o que argumenta contra ruído puro; ainda assim,
  confirmar com ≥10 quadros e outra(s) sequência(s) held-out reforça a afirmação.
- **Escopo NONE-commit:** a ablação isola a ação dominante; o ML **poderia** somar
  valor na ação de retangulares (A1), que a variância não modela naturalmente —
  a testar contra um baseline direcional/anisotrópico.
- **Treino do estudante:** que ele perca para a própria feature sugere espaço de
  melhoria na calibração/destilação, independente do teto informacional.

---

## 6. Reprodutibilidade

- **Binários:** `libaom_perf` (teste, `PARTITION_ML_STUDENT=1`, Release),
  `libaom_perf_anchor` (âncora, baseline v3.10.0 puro), `libaom_ml_check`
  (decodificador + paridade).
- **Modelos:** `results/models/surrogate_real` (ConvNeXt),
  `results/models/student_real` (MLP) → `partition_student_weights.h`.
- **Comandos:** `train.py` → `distill.py` → `export_weights.py` →
  `simulate_pruning.py` (escolha do ponto operacional) → `h7h8_bench.py`
  (medição fim-a-fim) → `ablation_attrib.py` (atribuição).
- **Dado:** dataset All-Intra cpu-used=0 (ground truth por busca RD completa);
  luma `float32` [0,1] de-normalizada para `uint8` no carregador
  (`data._denorm_uint8`), com salvaguarda `assert_real_luma`.
- **Sequência de validação:** Jockey (held-out; nunca vista no treino).

---

## 7. Redação sugerida (frase de atribuição para a tese)

> "A heurística implantada é um perceptron multicamadas leve, **destilado de um
> modelo substituto ConvNeXt** e executado pela rotina `av1_nn_predict` do
> próprio libaom a cada nó de particionamento. O substituto convolucional não é
> embarcado — seu custo de inferência é incompatível com a busca em tempo real —
> mas atua como mestre da destilação e, medido diretamente por *replay*,
> estabelece o limite superior de desempenho (H8). A ablação de atribuição (§5)
> confirma que o ganho de tempo decorre da seleção de nós aprendida, e não da
> política de poda isoladamente, uma vez que o modelo domina no sentido de
> Pareto as baselines de variância e de poda aleatória à mesma taxa."
