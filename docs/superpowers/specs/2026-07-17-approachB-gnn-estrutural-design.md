# Approach B — teste estrutural do teto (GNN diagnóstico → implantável) (design)

> **Documento de design (spec).** Data: 2026-07-17. Branch `ml-partition-dev`.
> Contexto: `docs/SINTESE_resultados_metodologia.md` (Conclusão 3),
> `docs/RESULTADOS_solucao4.md` (classificação > regressão),
> `docs/superpowers/specs/2026-07-17-solucao4-regret-regression-design.md` §11
> (Approach B como trabalho futuro). Define **estrutura, interfaces e gates** —
> hiperparâmetros/limiares são propostas iniciais, calibradas em cada gate.

---

## 1. Descrição breve (a contribuição)

**Um classificador NN estruturado (GNN) sobre o quadtree do superbloco**, que
decide as partições dos nós **conjuntamente** (via *message-passing* pai↔filho↔
irmão) em vez de **independentemente** (como os MLPs por nó do H9a). Serve, no
**Stage 1**, como **instrumento de diagnóstico** para responder a pergunta que a
tese deixou em aberto: o teto de desempenho (Conclusão 3) é **informacional**, ou
apenas um **artefato de decisões independentes por nó**? Para que um resultado
**nulo** seja convincente, dá-se à estrutura o **melhor tiro possível** — GNNs
expressivas de biblioteca (PyTorch Geometric: GraphSAGE, GAT, GIN). Só se houver
folga passa ao **Stage 2** (versão implantável causal top-down).

**Nota de honestidade metodológica:** a escolha *biblioteca (PyG) vs
message-passing à mão* **não é metodológica** — a validade científica está na
**ablação controlada** (MLP independente vs GNN, tudo o mais igual), que independe
da implementação. PyG é adotado por ser o caminho **mais forte** (camadas
expressivas testadas, varredura fácil de arquiteturas) — o que importa
metodologicamente é **não subdimensionar** o modelo estruturado, para que um nulo
signifique "a estrutura não agrega" e não "meu GNN era fraco".

---

## 2. Motivação e enquadramento

A **Conclusão 3** da tese (levers correlacionados = teto informacional) foi
estabelecida **empilhando levers escalares** (H9a + H9c + τ) — **nunca** com um
modelo que decida a árvore **conjuntamente**. E o H9a, embora use contexto **causal**
de vizinhança (bloco B: tamanhos das partições acima/esquerda), decide **cada nó de
forma independente**. Fica aberta a possibilidade de que a estrutura do quadtree
(dependências pai/filho/irmão, não-causais) carregue sinal que os nós independentes
não capturam.

A **Solução 4** acrescentou a lição de que, para a decisão de poda,
**classificação > regressão**. Portanto a Approach B é um **classificador
estruturado** (a versão em árvore do H9a), não um regressor.

**Barra (definida pelo usuário): diagnóstico-primeiro, implantar só se houver
folga.** Segue a disciplina de gates: o custo caro (integração C, benchmark real)
só se paga se o diagnóstico barato mostrar que a estrutura agrega sinal.

---

## 3. Dados e construção do grafo (reuso, sem re-extração)

- **Nós.** Os nós de decisão avaliados (níveis 64/32/16; 8px excluído, como o
  H9a) de cada superbloco, obtidos de `data.iter_superblock_members` sobre os pkls
  `dataset_h9` existentes.
- **Features de nó.** As **36 features H9a** (`features.node_features_h9a`) —
  **idênticas** às do baseline independente, para isolar a contribuição da
  estrutura.
- **Rótulo de nó.** 3 classes N/S/R (`student.collapse_label`), com máscara de
  legalidade por nível (`partition_defs.legality_mask`).
- **Arestas** (derivadas das chaves `(dim, r, c)`, sem estrutura de dados nova):
  - **pai↔filho:** pai de `(dim,r,c)` é `(2·dim, r//2, c//2)`;
  - **irmão↔irmão:** nós que compartilham o mesmo pai.
  Arestas só entre nós **presentes** no superbloco (filhos ausentes são tolerados).
  Grafo minúsculo: ≤ 1+4+16 = 21 nós de decisão por superbloco.

A reconstrução da árvore reusa a lógica de `regret.py` (`_children`, mapeamento por
chave).

---

## 4. Arquitetura GNN e o teste controlado (o ponto crucial)

- **Encoder:** `h = MLP(feat)` (as 36 features → dim oculta).
- **Camadas de grafo (PyTorch Geometric, `L` camadas, bidirecional/não-causal):**
  `GraphSAGE` (padrão), `GAT` (atenção) ou `GIN` — sobre as arestas pai/filho/irmão.
  Adotam-se camadas de biblioteca (testadas, expressivas) para dar à estrutura o
  melhor tiro. PyG **verificado compatível** com o `venv-ml` (torch 2.5.1+cu121,
  CUDA intacto).
- **Head:** `Linear → 3 logits` por nó. (Máscara de legalidade é **no-op** nos
  níveis 64/32/16 — N/S/R todos legais — logo omitida.)
- **Perda:** entropia cruzada de rótulo duro (a **mesma** do H9a, comparação justa).

**O teste controlado (a essência do diagnóstico):** o **mesmo modelo** com
**`L=0` camadas de grafo** é, por construção, um **MLP independente por nó** — o
baseline (não consome arestas). Comparado com **`L≥1` (com estrutura)**, *tudo o
mais idêntico* (features, dados, treino, split congelado). **Se `L≥1` não supera
`L=0`, a estrutura não agrega → o teto é informacional.** Cross-check externo contra
o bundle H9a existente (`results/models/student_h9a/students.pt`) para validade
externa.

**Protocolo forte-primeiro (decisão do usuário):**
1. **Primário (forte):** GNN expressiva (SAGE/GAT) vs MLP baseline (`L=0`).
2. **Se der nulo → endurecer:** varrer variantes (layer ∈ {SAGE,GAT,GIN},
   `L ∈ {1,2,3}`, `hidden`) — fácil em PyG — para confirmar que o nulo **não** é
   subdimensionamento antes de declarar teto informacional.
3. **Message-passing à mão** fica reservado para (a) fallback se o PyG não instalar
   e (b) a **portabilidade em C do Stage 2** (PyG não roda no encoder), a ser
   re-implementado só se o Gate 1-B passar.

---

## 5. Gate 1-B — o marco diagnóstico

Na **validação congelada** (HoneyBee/FlowerPan/Lips), comparar **GNN (`L≥1`) vs
gêmeo independente (`L=0`, MLP)** em duas métricas complementares:

1. **Qualidade de classificação por nó:** macro-F1 e **SPLIT-recall** por tamanho.
2. **Redução de custo no oráculo a SPLIT-lost casado** (a métrica-padrão da tese):
   novo modo `--gnn-bundle` no `simulate_pruning.py` que monta o grafo de cada
   superbloco a partir de `sbs["nodes"]`, roda o GNN e anexa as **probs conjuntas**
   por nó (`prob = softmax(logits)`), alimentando a **mesma** política NONE-commit
   e a **mesma** métrica de SPLIT-lost casado — *apples-to-apples* com H9a,
   variância e regret.

**Critério (calibrado no gate):** Gate 1-B **PASSA** (folga → Stage 2) se `L≥1`
supera `L=0` por margem **relevante e consistente** em SPLIT-recall **e** em redução
de custo a risco casado — sob a variante GNN **mais forte** varrida. Caso contrário
**FALHA** → **teto informacional confirmado** (após a varredura de robustez), e a
Approach B fecha como **caracterização forte** (como a Solução 4), sem pagar Stage 2.

---

## 6. Stage 2 — implantável, condicional e adiado

**Só se o Gate 1-B mostrar folga.** Um preditor estruturado **causal top-down**
(decide pai→filhos com contexto já decidido, respeitando a ordem do codificador),
implantável em C, com **spec e plano próprios** escritos **após** o Stage 1. Pelos
priors (Conclusão 3 + Solução 4), provavelmente **não é alcançado**; fica
registrado como condicional. **Nada de C, export ou benchmark no Stage 1.**

---

## 7. Cadeia de gates (metodologia)

| Gate | Conteúdo | Critério |
|---|---|---|
| **0-B** | Sanidade de grafo/dados (grafos reconstroem; features/rótulos/arestas alinham; `n_layers=0` reproduz um MLP independente) | grafos válidos; `L=0` ≈ MLP por nó |
| **1-B** | Ablação **`L=0` vs `L≥1`** na validação (macro-F1, SPLIT-recall, redução de custo no oráculo a SPLIT-lost casado) + cross-check vs H9a; varredura de robustez em caso de nulo | PASSA se estrutura agrega margem relevante; senão teto informacional |
| **(2-B+)** | *Condicional ao Stage 2* — integração C causal, paridade, no-op, benchmark real | só se Gate 1-B passar |

Ressalva metodológica herdada: o oráculo **superestima** o mérito (~5× histórico;
precedente H9c/Solução 4). No Stage 1 isso é aceitável porque a comparação é
**relativa** (`L≥1` vs `L=0` sob o **mesmo** oráculo) — o viés cancela.

---

## 8. Entregáveis (Stage 1) e estrutura de arquivos

**Criar** (em `src/scripts/partition_model/`):
- `graph_data.py` — monta listas de nós/arestas do quadtree a partir dos
  superblocos (chaves `(dim,r,c)`); reusa `data.iter_superblock_members`,
  `features.node_features_h9a`, `student.collapse_label`.
- `gnn_model.py` — encoder + camadas PyG (SAGE/GAT/GIN, `L` camadas + `--layer`) +
  head; `L=0` = baseline MLP independente.
- `train_gnn.py` — treina `L=0` e `L≥1` com hiperparâmetros idênticos; salva bundles.
- `gate1_gnn.py` — ablação `L=0` vs `L≥1` (macro-F1, SPLIT-recall) + orquestra o oráculo.
- `tests/test_graph.py` — grafo (arestas pai/filho/irmão corretas das chaves;
  `n_layers=0` reproduz MLP independente; a camada de grafo propaga como esperado
  num grafo de brinquedo).

**Modificar:**
- `simulate_pruning.py` — modo `--gnn-bundle` (`score_with_gnn`), montando o grafo
  de `sbs["nodes"]` e anexando probs conjuntas; reusa `simulate`/`metrics`/`report`.

**Docs:** `docs/RESULTADOS_approachB.md`; atualização de
`SINTESE_resultados_metodologia.md` e `ANDAMENTO_tese.md`.

**Reuso:** `regret.py` (lógica de árvore), `features`, `data`, `partition_defs`,
`student.collapse_label`, harness do `simulate_pruning`. **Dependência nova:**
`torch_geometric` no `venv-ml` (verificado compatível com torch 2.5.1+cu121);
fallback para message-passing à mão se a instalação quebrar o ambiente.

---

## 9. Riscos e limitações (honestidade)

- **Prior forte de saturação** (Conclusão 3 + Solução 4): a estrutura pode não
  bater o independente. **Mitigação de valor:** a ablação **`L=0` vs `L≥1`** torna
  até o resultado nulo **nítido e defensável** ("a estrutura não agrega → o teto é
  informacional, não artefato de independência") — é um resultado de tese de
  qualquer forma, e barato.
- **Subdimensionar o modelo estruturado (o risco metodológico real):** um GNN fraco
  produziria um nulo falso. **Mitigação:** camadas PyG expressivas (SAGE/GAT/GIN) e,
  em caso de nulo, uma **varredura** de layer/`L`/`hidden` antes de declarar teto —
  para dar à estrutura o melhor tiro. A escolha PyG-vs-manual **não** é
  metodológica; a ablação controlada é.
- **Dependência (`torch_geometric`):** verificada compatível com o `venv-ml`; se
  quebrasse o ambiente, o fallback é o message-passing à mão (mesma ablação).
- **Diagnóstico não-causal ≠ implantável:** o Stage 1 usa contexto do quadtree
  inteiro (não-causal), então é um **limite superior**; um pruner real (Stage 2)
  só tem contexto causal top-down. Por isso Stage 2 é condicional e separado.
- **Filhos ausentes / árvores parciais:** o grafo tolera arestas só entre nós
  presentes; o Gate 0-B verifica a reconstrução.
- **Comparação justa:** `L=0` e `L≥1` devem partilhar features, dados, split e treino;
  qualquer divergência invalidaria a atribuição à estrutura (verificado no Gate 0-B).

---

## 10. Relação com as conclusões da tese

Se o Gate 1-B **falhar** (o caso provável), a Approach B **fecha** o argumento do
teto: as Soluções 1–3 mostraram que o *sinal* satura; a Solução 4 mostrou que a
*formulação de classificação* é a que melhor o extrai; e a Approach B mostra que a
*estrutura conjunta* também não abre regime novo — **o teto é informacional, não
artefato de decisões independentes**. Se **passar**, abre-se o Stage 2 (a única
rota que restaria para furar a fronteira). Em ambos os casos é o **teste de
fechamento** da investigação.
