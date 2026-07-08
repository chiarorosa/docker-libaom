# Plano de Hipóteses e Experimentos — Poda de Particionamento AV1 guiada por ML

**Objetivo.** Reduzir o custo da busca recursiva de particionamento do codificador
AV1 (modo All-Intra, conteúdo 4K) por meio de uma heurística de poda aprendida,
mantendo a perda de qualidade (taxa BD) desprezível. A poda deve ser barata o
suficiente para que o ganho de tempo supere seu próprio custo de inferência.

**Metodologia.** Um **modelo substituto** (ConvNeXt sobre luminância) estabelece o
**limite superior de desempenho** predizível a partir dos pixels; um **modelo
estudante** leve (rede de multicamadas sobre atributos calculados manualmente),
obtido por **destilação de conhecimento**, é o que efetivamente roda no
codificador (via `av1_nn_predict`). A avaliação intermediária usa uma **simulação
oráculo** (redução de busca vs. risco de taxa-distorção) sobre uma **sequência de
validação independente (Jockey)**; a avaliação final é a **taxa BD e o tempo de
codificação reais**.

---

## 1. Hipóteses avaliadas (concluídas)

### H1 — A decisão de particionamento é predizível a partir da luminância
- **Teste:** treinar o modelo substituto e medir na simulação oráculo.
- **Resultado — PARCIALMENTE CONFIRMADA.** Existe sinal, porém **limitado**:
  limite superior de ~13–18% de redução da busca a ~1% de partições verdadeiras
  suprimidas. A predição é possível, mas fraca.

### H2 — O nível 8×8 é um ponto de decisão relevante
- **Teste:** distribuição empírica dos rótulos por nível.
- **Resultado — REFUTADA.** No regime 4K All-Intra, o 8×8 é **folha terminal,
  100% NONE** (sub-8×8 não é explorado). Foi **removido do modelo** (níveis
  64/32/16), mas **mantido na árvore** para a contabilidade da redução de busca
  (podar um 16×16 elimina a avaliação dos seus quatro filhos 8×8).

### H3 — O subajuste do substituto decorre de otimização/regularização/arquitetura
- **Testes:** aquecimento da taxa de aprendizado; balanceamento da perda por
  nível; redução de regularização (stochastic depth → 0, weight decay 0,05 →
  0,01); atributos locais de alta resolução nas cabeças finas; **sobreajuste em
  amostra reduzida**; **entropia dos rótulos por nível**.
- **Resultado — REFUTADA.** Nenhuma alteração deslocou o limite. Evidências
  convergentes: (i) o modelo **sobreajusta amostra reduzida** (logo, capacidade e
  gradiente estão corretos); (ii) no conjunto completo a perda **estagna no limite
  inferior dado pela entropia da distribuição a priori** (não atinge o mínimo de
  ajuste perfeito ~0,5); (iii) três configurações distintas convergem à mesma
  perda até a 4ª casa decimal. → O limite é de **informação da entrada**, não do
  modelo. A decisão de particionamento é predominantemente guiada por
  taxa-distorção (taxa, contexto, vizinhança) e não determinada pela textura do
  bloco.

### H4 — A inicialização pré-treinada (ImageNet) melhora a extração de sinal
- **Teste:** treinar com backbone pré-treinado (stem adaptado por média
  RGB→luminância) vs. treino do zero.
- **Resultado — REFUTADA.** Perda de treino e simulação oráculo **idênticas** ao
  treino do zero. **Decisão:** o deployment usa o modelo **treinado do zero**
  (autocontido e mais simples de justificar); o pré-treino permanece apenas como
  evidência de que o limite é de informação.

### H5 — Atributos manuais mais ricos aproximam o estudante do limite do substituto
- **Teste:** 10 → 18 atributos (heterogeneidade entre sub-blocos, perfis de
  linha/coluna, densidade de bordas), re-destilar, simulação oráculo.
- **Resultado — PARCIALMENTE REFUTADA.** Ganho marginal. O gargalo do estudante
  não são apenas os atributos, mas também a **calibração/capacidade descritiva**.

### H6 — A calibração da destilação melhora o estudante
- **Teste:** destilar sem ponderação de classe e com temperatura 1 (predições
  mais nítidas), com varredura fina de τ.
- **Resultado — PARCIALMENTE CONFIRMADA.** Criou um ponto operacional seguro real
  (**8,5% de redução a ~0,5% de risco**), inexistente antes. Persiste uma
  **transição abrupta** na curva de compromisso: o estudante não cobre a faixa
  intermediária de risco (1–5%) que o substituto cobre.

**Síntese das conclusões.** Há **dois limites empilhados**: (a) o **limite de
informação da luminância** (~13–18%), que só sobe alterando a *entrada*; e (b) o
**limite dos atributos manuais** (~8,5%), abaixo do substituto, atribuível à
destilação e à capacidade descritiva dos atributos.

---

## 2. Em andamento

### A — Extração de dataset ampliado (4 sequências adicionais)
- CityAlley, ReadySetGo, ShakeNDry, YachtRide → `results/dataset_new/`
  (4 QPs × 5 quadros, cpu-used=0). **Estado:** 2 de 4 concluídas (CityAlley,
  ReadySetGo); ShakeNDry e YachtRide pendentes. Processo retomável.
- **Finalidade:** ampliar de 4 para 8 sequências e permitir partição
  treino/validação mais robusta (6/2).

---

## 3. Experimentos futuros

### H7 — A redução de ~8,5% da busca produz ganho de tempo útil com taxa BD desprezível
- **Teste (próximo passo):** sincronizar os 18 atributos no código C
  (`partition_strategy.c` ↔ `features.py`), gerar os pesos do estudante,
  recompilar o codificador com a poda habilitada, verificar equivalência
  bit-a-bit com a poda desabilitada e decodificação válida, e medir **taxa BD +
  tempo de codificação** em τ=0,84 na sequência de validação independente.
- **Decide:** se o resultado atual já constitui uma contribuição válida (fecha a
  tese) ou se é necessário elevar o limite.

### H8 — O limite do substituto (~13–18%) justifica um estudante convolucional
- **Teste (opcional):** medir a taxa BD do **substituto** diretamente, via
  *External Partition API* do libaom, sem destilação (estudo de limite superior).
- **Decide:** se vale escrever inferência convolucional nova em C para capturar a
  diferença entre o estudante de atributos manuais e o substituto.

### H9 — Enriquecer a ENTRADA com contexto de taxa-distorção eleva o limite de informação
- **Justificativa:** é a **única alavanca** para superar o limite da luminância
  (H1/H3). A decisão é guiada por RD; fornecer ao modelo o que a decisão usa
  (vizinhança entre superblocos, posição no quadro, pré-análise barata de custo)
  pode elevar o limite superior.
- **Teste:** re-instrumentar a extração para registrar o contexto, re-treinar o
  substituto, comparar o limite na simulação oráculo. **Esforço maior**
  (redesenho da extração e do dataset).

### H10 — Mais dados (8 sequências, split 6/2) melhoram generalização e extração de sinal fraco
- **Teste:** unir `dataset_new` ao `dataset`, re-treinar com partição 6/2,
  comparar métricas de validação e limite da simulação oráculo. Depende da
  conclusão da extração em andamento.

---

## 4. Infraestrutura consolidada (pronta e validada)

- **Extração de rótulos:** instrumentação em C (`LOG_PARTITION_DATA`) +
  `build_dataset.py` (retomável) → `.pkl` com `sample_id` para reagrupar amostras
  em superblocos por quadro.
- **Modelo substituto:** `partition_model/{model,train,data}.py` — ConvNeXt
  multinível (64/32/16) com fusão de atributos locais + globais e perda
  balanceada por nível.
- **Destilação e exportação:** `{features,student,distill,simulate_pruning,
  export_weights}.py` — estudante por tamanho de bloco reutilizando
  `av1_nn_predict`; simulação oráculo como critério de decisão; geração do
  cabeçalho C de pesos.
- **Integração no codificador:** `partition_strategy.c`, sob guarda de compilação
  `PARTITION_ML_STUDENT` (compilação padrão bit-a-bit idêntica), com limiares por
  variável de ambiente.
- **Avaliação:** `benchmark/{run_benchmark,bd_rate}.py` — taxa BD (Bjøntegaard) e
  tempo de codificação, com PSNR externo (sem depender de compilação com
  estatísticas internas).
