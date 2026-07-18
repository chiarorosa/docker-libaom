# Andamento da tese — status e próximo passo

**Documento vivo.** Onde a tese está, o que foi decidido, e o próximo passo
concreto. Atualizado em 2026-07-10 (branch `ml-partition-dev`). Índice de
artefatos: `RASTREABILIDADE.md`. Planos: `PLANO_hipoteses_experimentos.md`,
`PLANO_H9_contribuicao_tese.md`. Protocolo congelado: `PROTOCOLO_avaliacao.md`.

---

## 1. Arco da tese (o que estabelecemos, em ordem)

1. **Infraestrutura** — instrumentação C do libaom, extração de dataset, ConvNeXt
   substituto, destilação → estudante MLP embarcado (`av1_nn_predict`), harness de
   benchmark (taxa BD + tempo). *(H1–E, concluído.)*
2. **Exploração H1–H6** — sinal da luminância parecia limitado (teto ~13–18%).
   *(Depois invalidado — ver item 4.)*
3. **Pré-H7** — descoberta de que a alavanca não era o modelo, e sim o **espaço de
   ações da política** (poda de retangulares via P(REST)) e a **métrica** (custo,
   não nós). Implementado.
4. **Bug crítico da luma-em-branco** — o dataset guardava luma `float32 [0,1]` e os
   consumidores assumiam `uint8`, treinando os modelos **sobre imagem em branco**.
   Corrigido; toda a cadeia re-treinada sobre pixels reais. **Invalidou as
   conclusões H1–H6** (eram sobre entrada vazia).
5. **H7/H8 (dado limpo)** — curva de operação real: **~7–29% de speedup a 0,4–1,6%
   de taxa BD**; teto do substituto (H8) ≈ de graça (−0,11% BD).
6. **Ablação de atribuição** — resultado **negativo**: no lever NONE-commit, um
   limiar de variância trivial **empata/supera** o estudante de pixels. Ou seja,
   os pixels saturam na variância; o ganho não era atribuível ao ML.
7. **Pivô H9** — hipótese: **contexto de taxa-distorção barato** supera o teto de
   pixels. Protocolo congelado (Fase 0), instrumentação e re-extração (Fase 1),
   **Gate 2 offline PASSOU** (Fase 2): o contexto RD grátis (H9a) supera pixels
   ~50% relativo no lever NONE-commit em risco casado.

**Tese, em uma frase (estado atual):** demonstra-se que o particionamento
All-Intra satura, no domínio de pixels, numa estatística trivial (variância) — por
ablação rigorosa — e que **contexto de taxa-distorção barato (vizinhança + quant +
posição) é necessário e suficiente para superar esse teto**, produzindo uma poda
aprendida com ganho de tempo atribuível ao ML. Falta confirmar no encoder real
(Gate 5).

### 1.1 Por que o `dataset_h9` existe — de um resultado negativo forte a uma direção positiva

O `dataset_h9` não é "mais um dataset": é o experimento que converte um resultado
negativo numa contribuição positiva. A sequência:

1. **Resultado negativo (ablação de atribuição).** No domínio de pixels, um
   baseline trivial — apenas a variância do bloco — empata ou supera o modelo de
   ML em todo ponto de speedup comparável. Ou seja, o ganho **não era atribuível
   ao aprendizado**: nenhum dos 24 atributos de pixel fazia o modelo bater a
   variância isolada.
2. **Por que esse negativo tem valor.** Ele caracteriza, com rigor, que a
   informação de particionamento **presente nos pixels satura numa estatística
   trivial**. É um resultado negativo forte: mostra, com evidência, que aumentar a
   capacidade do modelo no domínio de pixels não leva a lugar nenhum.
3. **A virada positiva (hipótese H9).** O particionamento do AV1 é, por definição,
   uma decisão de **taxa-distorção (RD)**. Logo, o sinal capaz de elevar o limite é
   o **contexto RD** que a própria decisão usa — e boa parte dele é **barata e já
   está residente na memória do encoder**: tamanho dos blocos vizinhos (acima/
   esquerda), força de quantização (qindex) e posição no quadro. É informação que
   **não está nos pixels**.
4. **`dataset_h9` materializa a hipótese.** Re-extração das 16 sequências
   adicionando os campos B (vizinhança), C (quantização/posição) e E (custo RD real
   do `PARTITION_NONE`, registrado apenas como referência de limite superior).
5. **Evidência positiva até aqui (Gate 2, offline).** MLPs por tamanho de bloco
   sobre o conjunto **H9a** (pixels + vizinhança + quant/posição, tudo de custo
   zero) superam a variância em ~50% relativo, de forma consistente. A confirmação
   definitiva no encoder real (Gate 5) permanece pendente.

### 1.2 O que roda no encoder — o professor (ConvNeXt) nunca foi a heurística

Há dois artefatos com papéis distintos, e é fácil confundi-los:

- **ConvNeXt (o "substituto"/professor).** Modelo convolucional de ~107 MB que
  observa o superbloco 64×64 inteiro. Usado **apenas em tempo de treino** — como
  professor numa **destilação de conhecimento** — e como **referência de limite
  superior** (experimento H8: quanto um modelo convolucional forte alcançaria,
  medido por replay no encoder). É pesado demais para inferência a cada nó de
  particionamento; **nunca foi embarcado no encoder**.
- **Estudante (o que de fato roda).** Uma **MLP pequena por tamanho de bloco**,
  executada pela função nativa `av1_nn_predict` do libaom sobre atributos manuais
  do bloco. É esse artefato que produz o melhor resultado atual — a curva de
  **~7–29% de speedup a 0,4–1,6% de BD-rate** (`h7h8_real`). O ConvNeXt destilou
  conhecimento para ele em treino, mas não participa da inferência.

**Consequência para o enquadramento da tese.** O objetivo original ("aplicar
convoluções para reduzir o tempo de codificação") realizou-se, na prática, como
convoluções no papel de **professor + referência de limite superior**, não como o
mecanismo embarcado. Mais: a ablação de atribuição e o Gate 2 indicam que o **ganho
implantável é tabular** (atributos manuais + contexto RD barato), **não
convolucional**. Por isso a Fase 3 recomendada treina o estudante diretamente sobre
H9a, em vez de reintroduzir o ConvNeXt.

### 1.3 O papel do ConvNeXt na escrita da tese (e a tensão do objetivo declarado)

O trabalho com o ConvNeXt não é descartável — é peça central da narrativa
metodológica, mesmo não estando no pruner embarcado.

**Onde ele entra na escrita:**
- **Metodologia:** a arquitetura do substituto ConvNeXt multinível (luma+qindex,
  cabeças por nível espelhando a hierarquia de particionamento do AV1), a
  destilação de conhecimento substituto → estudante, e a infraestrutura de
  avaliação (instrumentação em C, extração do ground truth em cpu-used=0, harness
  de BD-rate + speedup, simulação oráculo, ablação de atribuição).
- **Resultados/Discussão:** a correção do bug da luma (por que H1–H6 foram
  refeitos), a curva operacional H7/H8, e a ablação de atribuição — o **resultado
  negativo** (a variância trivial empata/supera o ML em speedup comparável).

**Por que o ConvNeXt é indispensável, não opcional:** ele é o que dá
**credibilidade ao resultado negativo**. A afirmação "os pixels saturam na
variância" só é rigorosa porque o modelo de pixels testado **não era fraco**. Se
houvesse apenas uma MLP pequena falhando em superar a variância, caberia a objeção
"o modelo tinha capacidade insuficiente". Ter um modelo convolucional de alta
capacidade, com o experimento de limite superior (H8), e ainda assim **não separar
da variância** em speedup comparável, fecha essa objeção — transforma "o modelo não
conseguiu" em "**o domínio de pixels não contém o sinal**".

**Elo com a solução final:** é esse aprendizado que **motiva e justifica** a virada
para o contexto RD (H9a). Sem o ConvNeXt, a virada seria um palpite; com ele, é uma
conclusão baseada em evidência. A técnica de "medir o teto para saber quanto ainda
resta" também migra para o lado RD: o H9c (`none_rdcost`) é o análogo do que o
ConvNeXt/H8 foi para os pixels.

**Tensão do objetivo declarado (decisão de narrativa a tomar).** Se o objetivo da
tese permanecer "aplicar convoluções para reduzir o tempo de codificação", há uma
tensão: a **solução implantada não é convolucional**. O enquadramento mais fiel aos
dados é: *caracterizar com rigor o que a decisão de particionamento precisa — e o
que ela não precisa (mais capacidade no domínio de pixels) — e entregar um ganho de
tempo atribuível a partir de contexto RD barato*. Nesse enquadramento, as
convoluções são o **instrumento de diagnóstico e a referência de limite superior**,
protagonistas da investigação, não do mecanismo embarcado. Recomenda-se revisar
título/resumo/objetivos com essa informação explícita.

---

## 2. Status por fase

| Fase | Descrição | Estado | Evidência |
|---|---|---|---|
| Infra (H1–E) | instrumentação, dataset, surrogate, destilação, benchmark | ✅ | commits até `71097b1` |
| Pré-H7 | poda de rect + métrica de custo + τ por nível + contexto hierárquico | ✅ | `834639e`, `PREH7_analise_alavancas.md` |
| Bug luma-branco | diagnóstico + correção + guarda | ✅ | `cb9d407`, `63f299c` |
| H7/H8 | curva speedup/BD real + teto do substituto | ✅ | `172e669`, `benchmark/h7h8_real*` |
| Ablação atribuição | ml vs variância vs aleatório (resultado negativo) | ✅ | `2380e91`, `benchmark/ablation_matched.csv` |
| H9 Fase 0 | protocolo congelado (test/val/train, QPs, métricas) | ✅ | `aabbbee`, `PROTOCOLO_avaliacao.md` |
| H9 Fase 1 | instrumentação RD (B/C/E) + re-extração 16 seqs | ✅ | `a17c525`, `a6748c5`, `dataset_h9/` (64 pkl) |
| H9 Fase 2 | Gate 2 offline (sinal do contexto RD) | ✅ **PASSOU** | `8866757`, `models/gate2_final.csv` |
| H9 Fase 3 | estudante tabular direto sobre H9a; Gate 3 (val) | ✅ **PASSOU** | `173aa8f`, `models/student_h9a/` |
| H9 Fase 4 | features B/C em C; paridade C↔Python; no-op byte-idêntico | ✅ **PASSOU** | `b3cd3c1`..`ecf436b`, `models/student_h9a/gate4_evidence.txt` |
| H9 Fase 5 | benchmark no teste held-out + ablação de atribuição | ✅ **CONCLUÍDA** (veredito matizado) | `docs/RESULTADOS_fase5.md`, `benchmark/h9_test/` |
| **H9 Fase 6** | validação universal (seqs CTC) vs presets nativos cpu-used 1/2 | ⏳ **PRÓXIMO** | ver §4.2 |

Legenda: ✅ concluído · ⏳ próximo · 🔄 em andamento · ⬜ pendente.

---

## 3. Veredito do Gate 2 (base da decisão atual)

`models/gate2_final.csv` (10 seqs treino, 60k superblocos, ensemble). No lever
**NONE-commit** (relevante para tempo), redução de custo em risco casado de
SPLIT-lost {0,5/1/2}%:

| subset | 0,5% | 1% | 2% |
|---|---:|---:|---:|
| variância | 0 | 0 | 0 |
| pixels24 | 10,1 | 15,3 | 18,9 |
| **H9a (contexto RD grátis)** | **15,7** | **20,1** | **24,9** |
| H9c (teto, none_rdcost) | 33,0 | 33,0 | 39,7 |

**Decisão: cenário (a)** — seguir com o modelo **pré-busca H9a** (pixels +
vizinhança + quant + posição; SATD do bloco D não agrega, descartado). O teto H9c
(`none_rdcost`, pós-NONE) fica documentado como headroom para uma extensão futura
(poda pós-NONE aprendida).

**Ressalva carregada até o fim:** a simulação oráculo superestima o tempo real
(~5×: 35% custo → 7% wall-clock). O que vale é a **margem relativa** de ~50% do
H9a sobre pixels; o árbitro final é o **Gate 5** (benchmark de tempo real no
teste held-out).

---

## 4. Fases 3–4 CONCLUÍDAS e próximo passo (Fase 5)

**O que foi feito (decisão pela evidência).** A questão em aberto — ConvNeXt+ramo
RD vs tabular — foi resolvida pelo dado: como o ganho H9a é **tabular e não-pixel**,
o ConvNeXt pixel-only não pode ensiná-lo, e o Gate 2 já provou a MLP-por-tamanho.
Logo, treinou-se o **estudante implantável diretamente** sobre H9a (36 features =
A pixels + B vizinhança + C quant/pos), CE de rótulo duro, **sem professor**, na
partição congelada. O ConvNeXt (`surrogate_real`) permanece como teto H8 de pixels;
o H9c segue como teto RD. Planos: `docs/superpowers/{specs,plans}/2026-07-11-*`.

**Gate 3 (validação HoneyBee/FlowerPan/Lips), alavanca NONE-commit, risco casado
(SPLIT-lost ≤ 1%), redução de custo do oráculo:**

| modelo | cost_red% @ SL≤1% | menor SPLIT-lost alcançável |
|---|---:|---:|
| variância | ~0 | 22,6% (não opera em risco baixo) |
| pixels24 | ~26–35 | ~0,3% |
| **H9a** | **~55–58** | 0,19% |

O **H9a dobra os pixels e domina a variância** (que nem chega a risco baixo),
reforçando o Gate 2. SPLIT-recall (métrica de segurança) 0,98/0,82 em 64/32px vs
0,73/0,52 dos pixels. **Ressalva mantida:** o oráculo superestima o tempo real
(~5×); vale a margem relativa; o **Gate 5** (encoder real) é o árbitro.
Artefatos: `results/models/student_h9a/` (`students.pt`, `oracle_sim_*.csv`, header
de 36 features exportado — o `student_real` implantado ficou intocado).

**Fase 4 — CONCLUÍDA (integração em C).** As features B/C foram sincronizadas em
`student_node_features` (`partition_strategy.c`), espelhando `node_features_h9a`;
o estudante H9a (36 features) é agora o **implantado**. Gate 4 passou:
- **paridade C↔Python** verde — 36/36 features a `0.0e+00`, probs <1e-3 (588 nós);
- **no-op byte-idêntico** ao `aom_baseline` com a flag desligada (md5 igual, em
  quadro texturizado);
- decode válido + testes AllIntra. Evidência: `models/student_h9a/gate4_evidence.txt`.

Ressalva (do review final): a paridade prova a **aritmética** das features; o
**sourcing** do ctx (vizinhança/quant lidos em runtime) é validado por revisão de
código contra o idioma nativo e, fim-a-fim, pelo benchmark da Fase 5.

**PRÓXIMO PASSO (Fase 5 — benchmark de tese).** No conjunto de **teste** held-out
(Jockey/RaceNight/RiverBank, ≥10 quadros): curva taxa BD × speedup do estudante H9a
com `libaom_perf` (rebuild com o código de 36 features) vs `libaom_perf_anchor`;
**ablação de atribuição** em speedup casado (H9a vs pixels vs variância vs aleatório,
via `ablation_attrib.py`/`analyze_ablation.py`); comparação com o
`intra_cnn_based_part_prune` nativo. Gate 5 (sucesso da tese): H9a domina a variância
em taxa BD a speedup casado, por margem além do ruído, em ≥2 das 3 seqs de teste.

---

## 4.1 Fase 5 — resultados PARCIAIS (em andamento, 2/3 seqs)

**Estado:** benchmark real detached rodando; Jockey e RaceNight concluídos,
RiverBank em execução. `RESULTADOS_fase5.md` (final) ainda não escrito. Resultados
em `results/benchmark/h9_test/<seq>/{curve_safe,curve_aggr,ablation}`. Encoder de
teste `libaom_perf` (36 features, flag on) vs âncora `libaom_perf_anchor` (libaom
cru); cpu-used=0, single-thread, 10 quadros, cq {20,32,43,55}.

**Calibração (validação, HoneyBee) — grid de τ:** com grid alargado
`{0.95..0.50}`, ml e variância **sobrepuseram** e o ml **dominou** (a tempo casado:
@1,5× ml 0,66% vs var 3,53%; @1,75× 1,41% vs 4,92%). Congelou o grid.

**Pilar 1 — redução de tempo (curva H9a vs baseline): FORTE e consistente.**

| seq | melhor ponto barato | ponto agressivo |
|---|---|---|
| Jockey | 0,19% BD @ TS 21,6% (1,28×) | 2,03% BD @ TS 57,2% (2,34×) |
| RaceNight | 0,27% BD @ TS 18,1% (1,22×) | 1,72% BD @ TS 50,7% (2,03×) |

**Pilar 2 — atribuição ao aprendizado (ml vs aleatório): CLARO.** A tempo casado,
o ml domina o aleatório nas duas seqs (ex. RaceNight @1,3×: ml 0,90% vs random
6,01%). O ganho é do modelo, não de poda aleatória.

**Pilar 3 — vs variância (a barra difícil): SEM head-to-head limpo no teste.**
Em Jockey e RaceNight (2/2), as faixas de speedup de ml e variância **NÃO se
sobrepõem**: ml opera em baixo speedup e baixo BD (≤1,46–1,59×, <1,7% BD); a
variância só opera agressivo (≥1,89–2,13×, ≥1,76–3,96% BD). São **regimes
disjuntos** — a variância, mesmo no ajuste mais conservador (τ=0,95), já pula para
alto speedup/alto BD e **não alcança a região implantável de baixo-BD** onde o ml
vive.

- **Leitura honesta:** a dominação limpa "ml vence a variância a tempo casado" só
  se materializou na **validação** (onde o grid sobrepôs). No **teste**, o resultado
  é diferente e mais matizado: ml e variância servem a operating points distintos;
  o ml ocupa a fronteira controlável e barata que a variância grosseira não
  acessa; a variância domina apenas em regime agressivo (que custa BD alto).
- **Causa da não-sobreposição (nota metodológica):** o grid foi congelado na
  calibração do HoneyBee, onde a variância τ=0,95 dava 1,34× (sobrepunha). O
  conteúdo de teste poda mais agressivo sob variância, então τ=0,95 já dá 1,9–2,1×.
  Faltou o lado da variância descer a τ=0,97/0,99. **Não estendido** — seria mexer
  em config vendo dado de teste (viola o congelamento anti-cherry-picking). Fica
  como limitação documentada / trabalho futuro.

**Cruzamento a BD casado (dual do speedup casado).** Como as faixas não sobrepõem
em speedup, testou-se o eixo BD (mesma qualidade → quem economiza mais tempo).
Resultado (`results/benchmark/matched_bd.py`): com o ml **implantado (rect-off)**,
há sobreposição em BD só no **Jockey** (1,76–2,03%), onde o ml ganha — @BD 1,76%:
ml **2,17× (TS 53,4%)** vs variância 1,89× (TS 47%). No **RaceNight** é disjunto até
em BD (variância mínima 3,96% > ml máx 1,72%). Ressalva: usa a política rica do ml
(rect-off) contra a variância sem — é comparação **sistema-vs-sistema**, não
atribuição pura.

**Atribuição LIMPA (política casada) — o argumento mais forte.** O rect-off é ação
**secundária** do ml (3ª saída, P(REST)); a alavanca **primária** é o NONE-commit.
A variância **estruturalmente não faz rect-off** (fixa P(REST)=1). Logo, a
comparação justa usa o **modo primário do ml (NONE-commit puro) vs variância,
política idêntica** — e aí há uma afirmação de atribuição **pura** que dispensa
sobreposição de speedup:

> Sob a **mesma política** (NONE-commit, sem rect-off para ninguém), variando só a
> **fonte de escore**, o escore do ml **alcança pontos de baixo-BD que o da
> variância não alcança**: ml de **0,09–0,10%** BD para cima; variância **nunca
> abaixo de 1,76% (Jockey) / 3,96% (RaceNight)**. Tudo idêntico exceto o escore →
> o ganho é do modelo. O escore RD é **mais discriminativo** (commita NONE só
> quando é o caso, BD ~0%); a variância commita em qualquer bloco liso, inclusive
> nos que deveriam dividir (daí o piso de BD alto).

Isto é matched-*policy* e caracteriza a **região alcançável** — não precisa de
sobreposição de speedup, e vale nas 2 seqs de teste. **O valor central do ml não
depende do rect-off.**

**FULL CONCLUÍDO (3/3 seqs) — 2026-07-13. Ver `docs/RESULTADOS_fase5.md`.**
RiverBank confirmou o padrão: sem sobreposição de speedup (3/3), mas o escore do
ml alcança BD mínimo 0,008% vs 0,75% da variância (razão 94×; Jockey 11×,
RaceNight 44×). Veredito Gate 5: a **forma estrita** ("ml domina variância a
speedup casado em ≥2/3") **não** é atingida no teste por não-sobreposição; mas a
contribuição está sustentada — (i) redução de tempo forte vs baseline (TS ~30–48%
a 0,6–1,4% BD, 3/3); (ii) ml ≫ aleatório (3/3); (iii) atribuição a política casada
(escore do ml alcança baixo-BD inacessível à variância, 3/3); (iv) validação com
dominação direta. Limitações + trabalho futuro (grid da variância a τ=0,97/0,99 na
validação; variância com rect-off; SOTA nativo) em `RESULTADOS_fase5.md` §5.

**Implicação para a escrita:** o Gate 5, como enquadrado ("ml domina variância a
speedup casado em ≥2/3 seqs"), **não** é atingido no teste por não-sobreposição.
Mas a tese afirma com honestidade e rigor: (i) redução de tempo forte vs baseline;
(ii) ganho atribuível ao aprendizado (vs aleatório); (iii) **atribuição a política
casada** — o escore do ml alcança a fronteira implantável de baixo-BD que o da
variância é incapaz de tocar (a barra difícil, agora respondida de forma limpa);
(iv) na validação, dominação direta a speedup casado. RiverBank (3ª seq) pode ou
não alterar o quadro; o veredito final sai quando o full fechar.

---

## 4.2 PRÓXIMO PASSO — Fase 6: validação universal (CTC) vs presets nativos

**Motivação (validação externa real).** As 16 seqs do split 10/3/3 (treino/val/
teste, todas UVG 4K) são o **universo do próprio ML** — mesmo as de teste, embora
held-out do treino, são da mesma fonte/distribuição. As sequências da **CTC**
(*Common Test Conditions*) são **balizadores universais entre codificadores de
vídeo** (padrão da comunidade), fora do universo do ML. Rodá-las é (i) uma
generalização externa genuína e (ii) resultados **comparáveis à literatura**.

**Comparação (o que de fato importa na prática):** posicionar o pruner ML **contra
o próprio botão de velocidade do AV1**. Codificar **N quadros × 4 cq** em cada seq
CTC, **pareando o tempo** de quatro configurações:
1. **LIBAOM original** — cpu-used=0, busca completa (âncora de qualidade).
2. **LIBAOM + ML (ponto equilibrado, levemente conservador em BD)** — cpu-used=0 +
   poda H9a no operating point de **bom equilíbrio BD-rate × TS**, puxado para o lado
   **conservador em BD**. Seleção a partir dos **dados de teste** (Fase 5). Candidato:
   **~P_rect** (τ_none=0,95 + rect-off τ_rest=0,20) → médias das 3 seqs **~0,46% BD a
   TS ~26,5% (1,36×)** — BD menor que P_ref/A1 (~0,6%) mantendo TS relevante.
3. **LIBAOM + ML (ponto agressivo)** — cpu-used=0 + poda H9a no ponto de **máxima TS
   com BD-rate ainda implantável e justificável**. Candidato: ~A3 (**~1,4–2% BD a
   TS ~48–57%, 2,0–2,3×**). **Justificativa via variância + aleatório:** nesse
   speedup (~2×) — região onde a variância *opera* — a variância custa ~2,4–4% BD e o
   aleatório ~6–12%, enquanto o ML mantém ~1,4–2%; logo o ponto agressivo é
   defensável (o ML preserva BD onde as heurísticas triviais explodem).
4. **preset nativo cpu-used=1** — o modo nativo de acelerar.
5. **preset nativo cpu-used=2** — idem, mais agressivo. *(cpu 1 e 2 a decidir.)*

**Dois pontos de ML** (não um): (i) o **equilibrado** posiciona o ML no regime
"quase de graça" contra os presets; (ii) o **agressivo** mede o ML no regime de
alta economia, com o BD justificado pela comparação com variância/aleatório no
mesmo speedup.

**A pergunta-chave:** o **ML (cpu0 + poda) fica numa fronteira taxa-BD × tempo
melhor que os presets nativos**? Se o ML+cpu0 entrega BD-rate menor a speedup
casado que cpu1/cpu2, a poda aprendida **agrega valor além do que o encoder já faz
de graça** com seus presets — é a comparação que um profissional realmente faz
("por que não só usar cpu-used=1?"). Esta é a validação decisiva da utilidade
prática da contribuição, distinta da ablação de atribuição (que compara *fontes de
escore* sob mesma política).

*Nota:* isto **substitui** como próximo passo o "trabalho futuro" do
`RESULTADOS_fase5.md` §5 (grid da variância / variância+rect-off na validação),
que fica secundário.

**Papel na tese — dois conjuntos, dois capítulos (justificativa estrutural).** Há
uma separação clara de função entre os dois cenários experimentais:

- **Universo do ML (Fases 2–5): VALIDAÇÃO da solução proposta.** Os experimentos
  nas sequências do aprendizado de máquina (split 10/3/3), com **ml + variância +
  aleatório e todas as combinações** de política/atribuição (Gates, ablações,
  speedup/BD/política casada, oráculo), são **parte fundamental da validação** — eles
  provam *por que* a solução funciona e que o ganho é **atribuível** ao modelo e ao
  contexto RD. Pertencem à **Metodologia/validação** da tese.
- **CTC (Fase 6): os RESULTADOS FINAIS.** As sequências da CTC, com **libaom original
  + 2 presets nativos vs o ML final** (versão única e ótima), são os **números de
  manchete** da tese — validação externa universal, comparável à literatura, e o
  posicionamento prático contra o botão de velocidade nativo. Pertencem ao capítulo
  de **Resultados**.

Ou seja: **validação (ML-set) ≠ resultado final (CTC)**. Os dois são necessários — o
primeiro sustenta e caracteriza a contribuição; o segundo a mede como produto, em
condições universais.

---

## 5. Decisões em aberto

- ~~**Arquitetura do substituto H9a** (ConvNeXt+ramo RD vs tabular)~~ — **RESOLVIDO
  na Fase 3**: estudante tabular direto sobre H9a (sem professor); ver §4.
- **Publicação do dataset** — `docs/ZENODO_datasheet.md` pronto; conversão para
  `.npz` uint8 (`pkl_to_npz.py`) **não executada** (backup bruto bin+pkl indo para
  Google Drive primeiro).
- **Extensão pós-NONE (H9c)** — opcional, documentada como headroom; só se o
  Gate 5 do H9a pré-busca ficar aquém.
- **Fase 6 (CTC) — parâmetros a decidir:** (a) **quais sequências CTC** (o usuário
  disponibilizará); (b) **N quadros** por seq; (d) **quais presets nativos**
  (cpu-used=1 e 2, ou outro par); (e) métrica de pareamento (BD-rate a speedup
  casado, os três pilares). **(c) RESOLVIDO — dois operating points de ML**
  (§4.2): equilibrado **conservador em BD** (~P_rect, ~0,46% BD a TS ~26%, escolhido
  nos dados de teste) e agressivo (~A3, máxima TS com BD implantável, justificado
  vs variância/aleatório).
  Os τ exatos dos dois pontos serão fixados a partir das curvas de teste da Fase 5
  antes de rodar a CTC.
- **Microbenchmark de inferência isolada do pruner — CONCLUÍDO (2026-07-17).**
  Medição direta (instrumentação `AV1_PRUNER_TIMING` em `partition_strategy.c`,
  encode real cpu1; `docs/RESULTADOS_microbench_pruner.md`): como algoritmo de
  decisão isolado, a **inferência do MLP é ~50× mais barata por chamada** que a da
  CNN nativa (~486 ns vs ~24.700 ns) — cerca de uma ordem e meia de grandeza. As
  três MLPs (24/36/39) têm custo quase idêntico (ocultas `[64,32]`). Escopo: mede
  a inferência (passagem direta); a extração de features do MLP (preprocessamento,
  otimizável) e a frequência de invocação são de integração, fora do escopo do
  algoritmo isolado.

---

## 6. Riscos vivos

- Gate 5 pode não confirmar a margem do Gate 2 (oráculo superestima) → mitigação:
  a decisão final é sempre o tempo de parede; se a margem não sobreviver, a
  contribuição recai na caracterização do teto informacional + o estudo H9c.
- Custo de inferência das features B/C é ~zero (dados residentes), então não há
  risco de "a poda não se pagar" no regime H9a (grátis).

---

## 7. H9c — teto de contexto RD pós-NONE (2026-07-15): implementado, testado, não sobrevive ao piloto real

Investigação completa do H9c como aposta de resultado adicional (plano
`docs/superpowers/plans/2026-07-15-h9c-teto-rd-pos-none.md`, 8 de 9 tarefas
executadas). Vetor de 39 features (A+B+C+E: o vetor H9a de 36 mais o
rate/dist/rdcost real do `PARTITION_NONE`, disponível só depois que o encoder já
avaliou essa partição). Estudante MLP separado, integrado em C como um segundo
hook — `av1_prune_after_none`, dentro de `av1_rd_pick_partition`
(`partition_search.c`), logo após `none_partition_search()` — decidindo se vale
a pena continuar buscando SPLIT/retangular/AB/4-way, ou parar ali (reaproveita
`av1_disable_all_splits`). Gate por `AV1_STUDENT_H9C_ENABLE` (default off).

**Gate 3 (oráculo, HoneyBee/FlowerPan/Lips) — PASSOU, com folga:** H9c chega a
61,2% de redução de custo a apenas 0,20% de split-lost — 5× menos risco que o
teto de 1% permitido — já superando o H9a (~55-58% @ SL≤1%) nesse mesmo
critério.

**Gate C (paridade + no-op) — PASSOU:** features 0-35 bit-exatas contra
`node_features_h9c` (features 36-38, o bloco E, são verificadas por construção,
não reconstruídas independentemente — o C aplica `log1p` ao `RD_STATS` real que
o próprio encoder acabou de calcular). Flag off byte-idêntico ao `aom_baseline`;
H9c desabilitado (flag on, env unset) não perturba o H9a.

**Piloto de tempo real (Jockey, 2 quadros, encoder de verdade) — NÃO PASSOU.**
Comparando no TS% mais próximo entre os dois:

| ponto | BD-Rate | TS% | Speedup |
|---|--:|--:|--:|
| H9c (τ=0,95) | 0,264% | 19,43% | 1,241× |
| H9a (P0, já medido, mesma seq) | 0,194% | 21,58% | 1,275× |

O H9a domina nos dois eixos ao mesmo tempo (TS maior **e** BD-Rate menor). O
H9c não bate o H9a em tempo real, apesar de superá-lo no oráculo (61,2% vs.
~55-58%) — o mesmo padrão já documentado neste projeto (H7/H8, Fase 5): a
simulação oráculo superestima o ganho real (~5× historicamente) e essa margem
não sobrevive ao encoder de verdade.

*Nota metodológica:* as duas linhas vêm de execuções diferentes (H9c: piloto
novo de 2 quadros, âncora própria ~178s@cq20; H9a: `curve_safe` anterior, mais
quadros, âncora própria ~894s@cq20) — BD-Rate/TS% são cada um normalizado à sua
própria âncora, então a comparação é válida, mas não é o mesmo harness de
execução. O viés dessa diferença é conservador a favor do H9c (2 quadros =
maior fração intra, favorece o TS agregado do hook), e mesmo assim o H9c perde
nos dois eixos — a decisão de parar é robusta a essa diferença.

**Decisão (regra de parada pré-registrada no próprio plano, seguida à risca):**
não prosseguir para o benchmark CTC (Task 9). O H9c fica **implementado,
testado, e desligado por padrão** (`AV1_STUDENT_H9C_ENABLE` unset) — inerte em
produção, disponível como base para uma futura tentativa com outro ângulo de
implantação (ex.: mais quadros/τ no piloto antes de descartar de vez, ou a
extensão H9c mencionada no §5 como possível "poda pós-NONE aprendida"), mas não
compõe os resultados finais da tese. **Leitura para a tese:** reforça a
caracterização honesta do teto de contexto RD — mesmo o teto mais informativo
testado (rdcost real do NONE) não se traduz em vantagem de tempo de parede
sobre o já otimizado pruner nativo, fechando com evidência (não suposição) a
pergunta que motivou o H9c.

---

## 8. H9c revisitado na CTC (2026-07-16): confound do H9a e resultado limpo do swap

Após o §7, testou-se o H9c em sequências CTC (cpu-used=0, 4 QP, varredura de τ)
e, num segundo momento, como substituto direto da CNN nativa (swap, cpu 1/2/3).
Duas descobertas, uma corretiva e uma positiva.

### 8.1 CONFOUND: o H9a rodava por baixo dos testes pré-busca do H9c

**Prova concreta.** O build `libaom_perf` roda o estudante **H9a sempre** em
quadros intra: `student_prune_partition` é chamado por `av1_prune_partitions_
before_search` sob o gate puramente geométrico `try_student_prune`
(`partition_strategy.c:2306-2311`) — **sem** flag de habilitação (ao contrário do
H9c, que tem `student_h9c_enabled()`/`AV1_STUDENT_H9C_ENABLE`,
`partition_strategy.c:2137`). Os scripts de teste do H9c (`encode_h9c_cq20.py`)
setaram **apenas** `AV1_STUDENT_H9C_ENABLE`/`_TAU`, sem neutralizar os τ do H9a,
que rodou nos seus defaults compilados (`tau_none=tau_split=0.9`,
`student_get_taus`). Logo, as linhas `h9c_tau*` mediram **H9a@0,9 + H9c
empilhados**, não H9c isolado.

**Quantificação (Neon1224 cpu0; re-execução `h9ciso_*` com H9a neutralizado via
τ=2/2/−1, única variável alterada):**

| config | BD-Rate | TS% | |
|---|--:|--:|---|
| h9c_tau95 (H9a@0,9 + H9c) | 0,267% | 17,15% | medido antes |
| h9ciso_tau95 (H9c isolado) | 0,037% | **2,96%** | |
| h9c_tau90 (H9a@0,9 + H9c) | 0,270% | 17,36% | |
| h9ciso_tau90 (H9c isolado) | 0,037% | **4,23%** | |
| h9c_tau60 (H9a@0,9 + H9c) | 0,386% | 20,53% | |
| h9ciso_tau60 (H9c isolado) | 0,100% | **9,31%** | |

O H9c **isolado poda muito pouco** (~3–9% TS); **82–96% do TS** que fora
atribuído ao H9c vinha, na verdade, do H9a. **RETRATAÇÃO:** as afirmações de que
"o H9c é 2–4× mais eficiente que o H9a" e "supera o nativo em eficiência" (leitura
das tabelas pré-busca confundidas) **são artefato do confound e ficam retiradas**.

### 8.2 SWAP (limpo): H9c ≈ CNN nativa como pruner intra a cpu1/2

O experimento de swap (`encode_swap_h9c.py`) **sempre** neutralizou o H9a (τ=2/2/−1)
e desligou a CNN nativa (`AV1_DISABLE_NATIVE_CNN=1`), medindo H9c como **único**
pruner intra no lugar da CNN nativa, a cpu fixo. Média de 3 seqs (Boxing/
FoodMarket2/Tango), vs âncora cpu0:

| cpu | config | BD-Rate | TS% | speedup |
|:--:|---|--:|--:|--:|
| 1 | native (CNN) | 0,368% | 28,60% | 1,402× |
| 1 | h9c_tau95 | 0,370% | 27,72% | 1,386× |
| 1 | h9c_tau90 | 0,392% | 29,58% | 1,424× |
| 1 | h9a_bal (swap) | 1,074% | 38,06% | 1,621× |
| 2 | native (CNN) | 0,407% | 38,18% | 1,619× |
| 2 | h9c_tau95 | 0,431% | 37,45% | 1,602× |
| 2 | h9c_tau90 | 0,442% | 39,05% | 1,646× |
| 3 | native (CNN) | 2,796% | 66,41% | 2,986× |
| 3 | h9c_tau90 | 3,511% | 70,50% | 3,408× |

**Leitura:** como substituto da CNN nativa, o H9c fica **praticamente empatado
com ela em cpu1 e cpu2** (fronteira taxa-BD × tempo quase idêntica — a cpu1,
native 0,368%/28,6% vs h9c_tau95 0,370%/27,7%), e é **muito mais competitivo que
o H9a era no swap** (h9a_bal custava 1,07% BD @ cpu1, ~3× o BD do H9c ao mesmo
regime). Só a **cpu3** (preset agressivo) o H9c perde para a nativa (3,5% vs 2,8%
BD). Ou seja: o H9c não é o "super-eficiente" que o número confundido sugeria,
mas é um **pruner de partição intra equivalente à CNN nativa nos presets
práticos (cpu1/2)** — alternativa aprendida, de custo computacional muito menor,
que iguala o SOTA embarcado nesse regime. Esta é a leitura defensável.

Artefatos: `results/benchmark/fase6_swap_h9c/` (swap), linhas `h9ciso_*` em
`results/benchmark/fase6/raw_results.csv` (isolação). Scripts:
`src/scripts/fase6/{encode_swap_h9c.py, encode_h9c_iso.py, encode_h9c_cq20.py}`.

### 8.3 Decomposição completa (Neon1224) e análise de fronteira

**Decomposição aditiva (Neon1224 cpu0 vs âncora), com o baseline que faltava
(`h9adef` = H9a@0,9 sozinho, H9c off — via `encode_h9adef.py`):**

| config | BD-Rate | TS% | TS/BD |
|---|--:|--:|--:|
| âncora libaom | 0,000% | 0,00% | — |
| H9a@0,9 sozinho (`h9adef`) | 0,312% | 17,10% | 54,8 |
| H9a@0,9 + H9c (`h9c_tau90`) | 0,270% | 17,36% | 64,3 |
| H9c sozinho (`h9ciso_tau90`) | 0,037% | 4,23% | 114,7 |
| H9a bal (implantado) | 0,639% | 22,58% | 35,3 |
| H9a aggr (implantado) | 1,412% | 34,49% | 24,4 |

Leitura: (i) **o H9a carrega o peso** em qualquer variante; (ii) **o H9c sozinho
quase não sai da âncora** (4% TS); (iii) empilhar H9c sobre o H9a **não adiciona
TS** (+0,26pp) mas **baixa o BD** ao mesmo TS (0,312→0,270; TS/BD 54,8→64,3) — o
H9c parece corrigir levemente decisões do H9a via rdcost real pós-NONE.
RESSALVA: 1 seq, deltas pequenos (0,04pp BD), pode ser ruído — precisa
confirmação (ver §8.4).

**Fronteira Pareto global (BD × TS, média de 3 seqs Boxing/FoodMarket2/Tango,
todos os níveis cpu):** pontos não-dominados, do menor BD ao maior:
`h9c_tau95`(0,213%/13,9%) → `h9c_tau90`(0,227%/15,0%) → `native_cpu1`(0,368%/
28,6%) → `h9c_tau90_cpu1`(0,392%/29,6%) → `native_cpu2`(0,407%/38,2%, TS/BD 94 =
**pico de eficiência**) → `h9c_tau90_cpu2`(0,442%/39,1%) → `h9a_bal_cpu2`(1,125%/
47,1%) → `h9a_aggr_cpu1/2`(1,9-2,0%/52-60%) → `native_cpu3`(2,796%/66,4%) →
`h9c_*_cpu3`(3,5%/70%) → `h9a_*_cpu3`(4,2-4,8%/73-78%).

**Três conclusões:**
1. **Ninguém DOMINA a CNN nativa** — nenhum ponto ML é estritamente melhor (mais
   TS a ≤ BD). A cpu1/2 o H9c-swap fica colado (empate técnico); a nativa mantém
   o pico de eficiência (TS/BD 78-94).
2. **O H9c É dono do extremo de baixo BD** que a nativa NÃO alcança: 0,21-0,23%
   BD / 14-15% TS. A escada discreta da nativa pula de cpu0 (0% TS) para cpu1
   (~28% TS), deixando todo o regime 0-28% TS descoberto — o ML preenche
   continuamente. **Valor real = granularidade fina em baixo speedup**, não
   superar o pico.
3. **Levers não se somam** (teto informacional): H9a (pixels+contexto), H9c
   (rdcost pós-NONE) e a CNN nativa exploram o mesmo sinal correlacionado
   ("blocos fáceis"). Prova: H9c sobre H9a = +0,26pp TS.

### 8.4 Experimentos em andamento (2026-07-16)

- **Completar swap H9c nas 5 seqs faltantes** (Crosswalk, Neon1224,
  NocturneDance, PierSeaSide, TimeLapse) → `results/benchmark/fase6_swap_h9c/`,
  para média de 8 seqs comparável ao swap H9a. RODANDO.
- **Frontier-check combinado** (`encode_swap_combo.py`): H9a **conservador**
  (τ_none 0,98/0,95, sem rect-off) + H9c, como substituto da CNN nativa a
  cpu1/2/3, em Tango → `results/benchmark/fase6_swap_combo/`. Testa a única
  combinação não explorada (motivada pela dica de eficiência do §8.3-ii). PRIOR:
  não fura a fronteira (levers correlacionados). ENFILEIRADO (dispara ao fim do
  swap das 5 seqs). Se confirmar o prior, fecha a caracterização do H9c.
  **CONCLUÍDO (2026-07-17): NÃO fura a fronteira.** Tango, vs âncora cpu0 — o
  combinado fica entre H9c-swap e H9a-swap, com eficiência decrescente (TS/BD @
  cpu1: nativa 81,9 > H9c 77,2 > comb0,98 67,9 > comb0,95 65,2 > H9a_bal 29,9); a
  nativa permanece no topo, o combinado não a domina. Confirma a Conclusão 3
  (levers correlacionados / teto informacional). Artefato:
  `results/benchmark/fase6_swap_combo/`.

Swap H9c COMPLETO nas 8 seqs (2026-07-17): H9c ≈ CNN nativa a cpu1/2 confirmado
em 8 seqs (não só 3). Síntese consolidada em `docs/SINTESE_resultados_metodologia.md`.

Scripts adicionais desta rodada: `src/scripts/fase6/{encode_h9adef.py,
encode_swap_combo.py}`.

---

## Solução 4 — regressão de *regret* (2026-07-17): RESULTADO NEGATIVO

Explorou-se uma proposta extra sobre os dados já coletados (`dataset_h9`, sem
re-extração): reformular a poda de **classificação do rótulo** para **regressão do
custo RD de podar** (*regret*), com NN regressora (MLP por tamanho, cabeça única)
sobre as mesmas 36 features H9a. Design/plano em
`docs/superpowers/{specs,plans}/2026-07-17-solucao4-regret-regression*`.

- **Gate 0 PASSOU** (viabilidade), com ressalva: *regret* fortemente zero-inflado
  (dim16 98% de zeros).
- **Regret naïve → degenerado** (Gate 3 val: 0% de redução a risco casado; piso de
  SPLIT-lost 12,4%; colapso no preditor trivial ≈0, previsto pela zero-inflação).
- **Correção anti-zero-inflação** (Huber ponderado, `--balance`, peso 8–46× nos
  nós não-nulos) → aprende a cauda mas **continua sem alcançar risco baixo**
  (Gate 3 val: 0% a risco casado, piso ~11,4% em r0 ∈ {0,05;0,1;0,2}).
- **Conclusão:** a regressão de *regret* **ranqueia a poda pior** que o
  classificador H9a sobre entradas idênticas → a decisão de poda é uma
  **classificação**, não uma regressão de custo; a magnitude do *regret* não é
  ranqueável a partir de features pré-busca baratas. Gate 5 (benchmark real)
  **pulado por regra de parada de gate** (oráculo já rejeita; ele superestima o
  mérito). Resultado detalhado em `docs/RESULTADOS_solucao4.md`; entra no Capítulo
  de Resultados como **resultado negativo de valor metodológico** (reforça por que
  a formulação de classificação da Solução 2 é a correta).

Artefatos: `results/models/regret/` (naïve + gates), `results/models/regret_balanced/`.
Scripts: `src/scripts/partition_model/{regret.py,build_regret_targets.py,
gate0_regret.py,train_regret.py}` + modo `--regret-bundle` em `simulate_pruning.py`.

---

## Approach B — decisão estruturada por GNN (2026-07-18): RESULTADO NEGATIVO

Última alavanca não testada: decisão CONJUNTA do quadtree (GNN de message-passing,
PyTorch Geometric) vs nós independentes (H9a). Ablação controlada `n_layers=0` (MLP)
vs `n_layers≥1` (GNN), mesmas features/dados/split. Specs/plano em
`docs/superpowers/{specs,plans}/2026-07-17-approachB-gnn-estrutural*`. Detalhe em
`docs/RESULTADOS_approachB.md`.

- **Oráculo:** estrutura fura o teto — GNN não-causal +28pp; versão **deployable
  pixel-only** (bloco A+C, pré-passe por superbloco como a CNN nativa) recupera
  ~93–100% e supera o H9a em +20–25pp. Parecia o salvamento.
- **Ablação causal:** o ganho evaporava sob restrição estrita — mas era estrita
  demais (baniu a agregação bottom-up de PIXELS, que é deployable). A versão
  pixel-only corrigiu isso e recuperou o ganho no oráculo.
- **Benchmark REAL (replay H8, fiel às decisões):** NÃO sobrevive. Fronteira real
  (Jockey 5fr, cada modelo no seu melhor τ): H9a domina o GNN por **~2×** em BD em
  todo o sweep (GNN ~1,5% vs H9a ~0,75–0,94% a TS casado). **O oráculo inverteu o
  ranking.** Fronteira do GNN plana → qualidade das decisões (não o τ) é o limite;
  C faria as mesmas decisões (replay fiel) → não salva; longe do nativo (~0,45%@32,6%).
- **Raiz:** acurácia por-nó + custo do oráculo são maus proxies do BD×tempo real
  (custo dominado por poucas podas erradas caras). Um modelo que vence o oráculo
  perde no real — alerta metodológico mais forte que "o oráculo superestima".
- **Não perseguido:** reordenar candidatos + early-term (exige RD por-candidato
  declinado + re-extração + novo hook C, sem gate offline, contra heurísticas
  nativas dominantes; EV baixo). Trabalho futuro condicional.

Pipeline (offline, revisado, commitado): `src/scripts/partition_model/{graph_data,
gnn_model,train_gnn,gate1_gnn,gnn_replay}.py` + `src/scripts/benchmark/{gnn_replay_bench,
gnn_frontier_bench}.py`. Sem mudanças em C (tudo via replay). Approach B ENCERRADA.
