# CLAUDE.md — Artigo ISCAS 2027 (pasta de trabalho)

Precedência: `CLAUDE.md` da raiz → `../CLAUDE.md` (normas IEEE, Partes A–D) → este arquivo.
Este arquivo **não repete** as normas de `../CLAUDE.md`; registra apenas o que é específico deste
artigo: escopo, fontes, restrições vinculantes, protocolo de trabalho e estado.

---

## 1. Objeto do artigo

**Artigo carro-chefe da tese**: agrega as melhores soluções medidas (H9a pré-busca, H9c pós-NONE,
H9d partições estendidas e suas composições) sob uma única régua CTC A1. O eixo de comparação de
mesmo escopo é a **substituição direta** da CNN nativa de poda de partição do libaom com o
*preset* (`cpu-used`) fixado: dentro de um nível, só o podador muda. Recorte final: ver §5 (Estado).

**Template LaTeX**: `../LASCAS/PAPER_LASCAS_2027_SNP-AV1_SUBMIT.tex` é o molde (preâmbulo,
autores, bloco de afiliação, estilo de bibliografia manual, equações (1)–(2) de TS e E, formato das
tabelas e legendas). Partir de uma cópia dele e substituir o conteúdo.

Fontes canônicas: `R2_h9a.md` (§2.5 substituição do H9a), `R3_h9c.md` (§3.4 substituição do H9c,
§3.5 interação), `R4_h9d.md`, `R6_analise_integrada.md` (fronteira e três conclusões). Apoio:
`M1`–`M6`, `R6_analise_integrada.md`, `R7_ameacas_e_escopo.md`, `A1_INDICE_evidencias.md`,
`T_RESULTADOS_tabelas.md`. **Nunca** tirar número de `docs/` sem conferir contra `results/thesis/`.

## 2. Restrições vinculantes (não negociáveis)

1. **Taxa BD e redução de tempo só da grade CTC A1** (8 seq. 4K 10 bits, 15 quadros, cq 20/32/43/55,
   âncora `cpu-used=0`). UVG só para construção do dataset e estatística offline, rotulado como tal.
2. **Artigos irmãos não se citam** (LASCAS = SNP-AV1, H9a+H9d; ICASSP = atribuição offline da
   informação pré-busca). O H9a não pode aparecer como "nosso podador publicado em [N]". Texto
   descritivo comum (BC/NC/CSP, dataset) deve ser **reescrito**, nunca copiado do LASCAS.
3. **Razão de inferência "~50×" (ou 32×) proibida como vantagem** (`docs/RESULTADOS_microbench_pruner.md`
   §6): o número honesto é o custo implantado ≤0,32% do encode — e esse foi medido para o H9a, não
   para o H9c. Custo implantado do caminho H9c = **lacuna**; declarar, não estimar.
4. **Retratações (R3 §3.3) não reaparecem**: H9c "2–4× mais eficiente que o H9a"; H9c "supera a CNN
   nativa em eficiência"; "H9c não sobrevive ao piloto". A afirmação defensável é **paridade** na
   grade completa (p1, p2), vantagem significativa só no regime de alta qualidade (cq 20/32), derrota
   no p3.
4b. **Contraste H9d "+1,02 vs +0,26 do H9c / quatro vezes" é inválido** (auditoria
   `docs/RESULTADOS_auditoria_artigos_lascas_iscas.md` §4), embora ainda conste de `R6` §6.4. Usar o
   contraste de **sinal**: interação −1,9 pp (H9a+H9c, 4 seq., limiares padrão) vs marginal +1,02 pp
   (H9a bal.+H9d, 8 seq.), declarando a assimetria de base e de cobertura.
5. **τ=0,90 vs τ=0,95 é ruído** (R3 §3.4): não apresentar como dois pontos de fronteira distintos.
6. Redução de tempo **só na definição canônica** (média por QP → média por sequência; M3 §3.4).
   Resolução pareada ≈0,46 pp; diferenças menores não são citadas como positivas.
7. Ganho sempre com custo: a vantagem de BD-BR em alta qualidade custa ~4,3 pp de TS no p1.
8. Figuras: EN, paleta sóbria, **variante cinza** no artigo, scripts no estilo de
   `src/scripts/benchmark/plot_*_fig*.py` (dicionário `PALETAS {"cor","cinza"}`, `--variante`,
   `--out-dir`), gerados no contêiner `av1_bench` com `build/venv-ml`. Fonte mono convertida para
   TTF (PDF eXpress).
9. Colocações do Porto (revisão do LASCAS) vencem B.12; sem ±desvio no corpo do texto.

## 3. Protocolo de trabalho

1. **Planejar** cada bloco (o que entra, de que fonte, que número) → **apresentar** ao usuário →
   **escrever** só após aprovação.
2. **Blocos pequenos**: uma subseção (ou um parágrafo longo) por vez; compilar após cada bloco.
3. Todo número escrito é conferido contra a fonte canônica no momento da escrita; lacuna vira
   `[completar: ...]`, nunca estimativa.
4. Estilo de prosa: espelhar `../LASCAS/PAPER_LASCAS_2027_SNP-AV1_SUBMIT.tex` (frases longas
   encadeadas, ganho→custo, "The main strategy of this solution is to ...", declaração de escopo
   no fim da fundamentação, declaração de conformidade com a especificação AV1).
4b. **Nunca passar LaTeX por heredoc do Git Bash nem por `py -c`**: o shell colapsa `\\` em `\`
   (quebrou as linhas das Tabelas I e II em 2026-10-04). Texto LaTeX entra no `.tex` só pelas
   ferramentas Edit/Write ou por script Python salvo em arquivo. Após cada inserção, conferir a
   ordem da bibliografia contra a primeira citação.
5. Ao fechar cada etapa: compilar sem `Overfull`/referência indefinida, commit + push (sem atribuição
   de IA na mensagem).

## 3b. Perfil estilístico do molde (LASCAS SNP-AV1 SUBMIT) — recurso vinculante

Medido em 2026-10-03 sobre o corpo do LASCAS (Introdução → Conclusões, 93 frases, 2.886 palavras)
com `src/scripts/paper/style_profile.py` (corrigido no mesmo dia: o filtro de comentário cortava o
texto após `\%`). O ISCAS **clona** este estilo (léxico, sintaxe, voz, tom); o conteúdo e as frases
são novos. Após cada bloco, rodar os dois scripts no ISCAS e comparar.

**Contrato lexical (vinculante, após o episódio "rung" de 2026-10-03).** Toda palavra do ISCAS —
corpo, legendas **e rótulos de figura** — tem de estar no LASCAS, ou ser: (a) palavra gramatical
comum ou flexão de palavra do LASCAS; (b) fato técnico novo e inevitável (nomes de codec, de função
do libaom, de teste estatístico); (c) o nome do método. **Termo cunhado, metáfora, expressão
idiomática ou qualificador fora do LASCAS só entra com aprovação explícita do usuário.** Verificação
obrigatória após cada bloco: `py src/scripts/paper/lexicon_diff.py <ISCAS.tex> <LASCAS.tex>
[rótulos.txt]`, revisando cada palavra listada. **Lado oposto do contrato — frases novas:**
`py src/scripts/paper/ngram_overlap.py <ISCAS.tex> <LASCAS.tex>` lista as 6-gramas em comum; só
ficam as de preâmbulo/autores/agradecimento, nomes de padrões, colocações canônicas de B.12 e
enumerações técnicas. Qualquer outra passagem é reescrita.
**Nomenclatura fixada:** *pre-search pruner* (H9a), *post-NONE pruner* (H9c, encerra o nó),
*extended pruner* (H9d); *operating point* (nunca "rung"); *neural* (nunca "learned");
*shallow* (nunca "lightweight"). Pontos do pré-busca: **efficiency-first** (0,95/0,90/0,20) e
**time-first** (0,60/0,85/0,40) — nunca "balanced/aggressive" (crítica do orientador, 2026-10-04: não
dizia se o sentido era aceitar perda de BD-BR ou buscar TS); o III-C explica o eixo (limiar mais
exigente → menos poda → menos perda e menos TS). Terceiro ponto (0,90/0,90, τ_rest desligado) fica
sem nome: são os limiares compilados, sem intenção de projeto.

**Sintaxe**
- Frase longa e encadeada: média **31 palavras**, mediana 27, **45%** acima de 30, 14% abaixo de 15.
  Por seção (alvo de paridade): Introdução do LASCAS = 16 frases, 418 palavras, média 26,1, mediana 23.
  Encadeamento por `, so` (14), `since`/`because` (8), `, which ...`, `while`, `whenever`.
- **Dois-pontos como dobradiça** (22 em 89 frases): afirmação `:` razão ou consequência
  ("One limit follows, measured in Section V: a node the first stage has committed never reaches the second").
  Ponto e vírgula raro (8), só em enumeração longa.
- **Precisão por negação**: `X, not Y` / `X and not Y` / `X rather than Y` / `reported rather than omitted`,
  `verified rather than assumed` (8 ocorrências).
- Parágrafo fecha com a consequência, tipicamente `, so ...` ou `, which is why ...`.

**Voz e agente**
- Passiva no método e no protocolo (40 construções *be*+particípio); nunca `we`/`our` (0).
- Sujeitos inanimados ativos nos resultados: "Table I opens the two deployed points", "The search is
  exhaustive by construction", "The coupling dissolves at the aggressive point".
- Abertura de frase por sintagma nominal definido: `The first ...`, `The second ...`, `The two ...`,
  `The cost ...`; `This paper presents ...`; `The main strategy of this solution is to ...`.
- Componentes nomeados pelo papel, não pela sigla interna: "the first stage", "the unpartitioned node",
  "the extended partitions" (nunca H9a/H9c/H9d no artigo).

**Tom**
- Declarativo, contábil, sem adjetivo valorativo. **Hedging quase nulo** (1 ocorrência): só onde não
  há medida. **Zero `However`** — contraste via `but`, `while`, `instead`, `X, not Y`.
- Verbos de ação concretos do domínio: `commits`, `settles`, `opens`, `removes`, `yields`, `cedes`,
  `reaches`, `lifts`, `keeps`, `adds`.
- Limite do próprio método enunciado como fato medido ("One limit follows, measured in ...").

**Números**
- TS com 2 casas, BD-BR com 3 casas no corpo e 2 no abstract; faixas como "from A on Seq1 to B on Seq2";
  incremento em "percentage points"; razões como "about 3.4 times".

**A evitar** (marcas que não são do molde): metáfora e personificação ("answered this demand"),
`However` em início de frase, `furthermore`/`moreover`, frases curtas em série, enumeração de
contribuições em lista, adjetivos como "significant" fora do sentido estatístico.

## 4. Arquivos desta pasta

| arquivo | função |
|---|---|
| `CLAUDE.md` | este guia |
| `PAPER_ISCAS_2027.tex` | manuscrito (a criar após aprovação da proposta) |
| `fig*_*.pdf` | figuras copiadas de `results/thesis/figuras/` (variante cinza) |

## 5. Estado e decisões

- 2026-10-03 — pasta criada; proposta apresentada.
- 2026-10-03 — decisões: **cenário (A)+(D2 enxuto)** — nenhum número do LASCAS é reapresentado;
  o H9d entra só como componente de experimento novo (descrição reescrita, sem citar o LASCAS na
  submissão; se o LASCAS for aceito antes da versão final, citá-lo como origem do H9d). Pontos em
  `cpu-used=0` liberados ao lado dos presets. Limite **4 páginas técnicas + 1 de referências**.
  Acrônimo em aberto (candidatos: DNP-AV1, NPR-AV1, TNP-AV1, NPL-AV1).
- 2026-10-03 — viabilidade do D2 enxuto verificada: binário `build/libaom_perf_h9d` (22/07, flags
  idênticas a `libaom_perf`) expõe `AV1_DISABLE_NATIVE_CNN`, `AV1_STUDENT_TAU_*` e
  `AV1_STUDENT_H9D_ENABLE`; gancho `av1_prune_after_none` sem guarda de preset; integridade com H9d
  desligado byte-idêntica à linha `fase6_swap` (BoxingPractice cq32 cpu1: 1.572.268 B, PSNR-Y
  40,9600; tempo 288,8 s vs 287,4 s em julho). Script: `src/scripts/fase6/encode_swap_h9d.py`.

- 2026-10-03 14:14 (-0300) — **campanha D2 enxuta lançada** após o commit do pré-registro
  (`28f9145`, 14:12). Contêiner `av1_bench`, PID 122, log
  `results/benchmark/fase6_swap_h9d/run.log`, 192 codificações, ≈ 11 h. Tempo de parede como métrica
  (consistente com todas as campanhas anteriores, nenhuma tem *user time*); `user_s`/`sys_s` gravados
  como colunas extras (`5c88035`), sem âncora remedida em *user time* — por decisão do usuário, não
  usados no artigo. Ao terminar: `encode_swap_h9d.py --check-integrity` (P0) antes de qualquer leitura.

- 2026-10-03 — acrônimo **NPL-AV1 (Neural Pruner Ladder for AV1)**. Bloco 1 escrito e aprovado:
  preâmbulo (cópia do LASCAS SUBMIT, `IEEEtran.cls` local como no LASCAS), título, Abstract,
  Index Terms. Compilação no contêiner `latex_build` (monta `results/` em `/work`): 0 warnings,
  fontes Type 1 embutidas. Pendente no Abstract: frase do H9d empilhado (após D2).

### Exceções às normas autorizadas pelo usuário
- 2026-10-04 — **B.15 (ganho sem custo) no Abstract**: a faixa "time savings from 40.20% to 77.30%"
  do pré-busca entra sem a faixa de BD-BR (0,92% a 4,35%), e o incremento "up to 0.87 percentage
  points" do podador de estendidas sem o seu custo (< 0,04 pp). Motivo: evitar que 4,35% de BD-BR
  isolado afaste o revisor na primeira leitura; o custo do ponto destacado (1,07% a 50,99%) fica na
  mesma frase e todos os custos estão no corpo. Não reverter em revisão sem consultar o usuário.
- 2026-10-04 — **B.2, movimento 7 (ineditismo) fora do Abstract**: a frase "To the best of the
  authors' knowledge…" fica só nas Conclusões, por decisão do usuário.

### Pendências no artigo
- 2026-10-04 — **rascunho completo**: Abstract reescrito (246 palavras, três podadores, "Ten of the
  thirteen"; opção (a), sem a ressalva de não dominância, que fica no corpo) e Seção V escrita.
  Sem Fig. 2, por decisão do usuário.
- 2026-10-04 — **revisão final concluída**: auditoria de números
  (`src/scripts/paper/audit_numbers_iscas.py`, 201/206 casados; os 5 restantes são parâmetros de
  layout e um falso positivo de arredondamento); corrigido 4,47 → 4,46 (arredondamento duplo);
  siglas BD-BR, PSNR, TS e RD definidas no corpo; frase de 97 palavras das Conclusões dividida
  (máx. agora 71); `\IEEEtriggeratref{16}` equilibra a página 5; Underfull da página 1 resolvido com
  quatro cortes de redundância na Introdução. Frase do treino (AdamW/PyTorch) retirada a pedido.
  Fontes STIX TrueType na Fig. 1, como no LASCAS aceito.

### Pendências fora do artigo
- ✔ 2026-10-04 (commit `ce0b4a5`): `M3` — a pendência de `--threads` era equívoco (o M3 descreve
  a grade UVG, `--threads=1`); explicitado o ladrilhamento 4K da CTC (`--threads=2`) e declarado o
  tempo de parede (razão user/parede medida 1,72). `R3` — "menos da metade" corrigido (p2 = 2/3),
  ressalva das redes AB/4-way, paridade só de BD-BR, retirada a razão ~50×. `R4`/`R6` — contraste
  "+0,26, quatro vezes" trocado pelo contraste de sinal; R6 ganhou a D2 como terceira prova.
- **LASCAS** (corrigir na versão final): (i) "Motion Picture Expert Group" → "Moving Picture Experts
  Group"; (ii) ref. Corrêa 2020 com autores errados ("B. Zatt" não é autor; os 6 autores do Xplore,
  DOI 10.1109/TCSI.2020.2973031, são M. M. Corrêa, B. H. Waskow, J. W. Goebel, D. M. Palomino,
  G. R. Corrêa, L. V. Agostini → "M. M. Corrêa et al."); (iii) "L. Netto" → "L. Neto" e
  "intraframe" no título (DOI 10.1109/MDAT.2022.3146083); (iv) "One of the largest individual costs
  of AV1 intra-frame coding is the block partitioning decision [bender2023]" extrapola a fonte —
  Bender perfila o libaom em geral (inter domina, 76,98%) e só afirma que "partition tree processing
  ... has a significant impact on the overall computational cost". Já corrigidos no ISCAS;
  (v) trocar "balanced/aggressive" por "efficiency-first/time-first", com a frase de mecanismo do
  III-C do ISCAS, para que os dois artigos usem a mesma terminologia.
- ✔ (resolvido em `ce0b4a5`) `R3` §3.4 ("nicho nativamente vazio"): ressalvar que, em AI, as DNNs nativas
  `av1_ml_prune_ab_partition` e `av1_ml_prune_4_partition` **continuam ativas** depois do NONE
  (`ml_prune_partition=1` em todos os presets AI, `speed_features.c:339`; a checagem de
  `frame_is_intra_only` dentro delas só controla a gravação de atributos). Só as três decisões
  estruturais (breakout, early-term after split, prune rect) estão desligadas. Enunciado correto,
  usado no ISCAS: "no learned model of the native encoder decides, in AI, whether the search of a
  node continues once its unpartitioned cost is known".
- CNN nativa: nenhuma publicação revisada por pares a descreve (busca de 2026-10-03: Han 2021 e
  Bender 2023 não a mencionam; histórico do googlesource inacessível). Fonte = libaom v3.10.0,
  *speed feature* `intra_cnn_based_part_prune` (`speed_features.h:689`, `partition_strategy.c:189`).
- Desvio tempo de parede vs *user time* (CTC §5.7): decidir se/como declarar (afeta também o LASCAS).

- 2026-10-04 — **campanha D2 concluída** às 01:46 (-0300): 192/192 linhas, 0 erros; **P0 cumprida**,
  96/96 linhas-base byte-idênticas a `fase6_swap`. O watcher por `kill -0` não disparou (processo
  zumbi sob PID 1 `sleep infinity`); encerrado manualmente. P1–P3 ainda não lidas.
- 2026-10-04 — III-C escrito: só τ_stop = 0,95 no artigo (0,90 equivalente no ruído, R3 §3.4);
  fronteira sem τ = 0,90 = 9 de 12 pontos não dominados aprendidos (antes da D2); número do Abstract
  fica `[completar]` até recalcular a fronteira com a D2.

- 2026-10-04 — **leitura da D2: P0–P3 cumpridas** (`docs/RESULTADOS_fase6_swap_h9d.md`).
  H9d sobre H9a bal. substituto: ΔTS +0,87 pp (p1) e +0,83 pp (p2), 8/8 seq., p = 0,001; ΔBD +0,032 e
  +0,039 pp; p3: ΔTS −0,14 pp (0/8), ΔBD +0,002. Fronteira (sem τ=0,90 e sem pontos de p0 do LASCAS):
  **13 não dominados, 10 do NPL-AV1**; novo ponto não dominado H9a bal.+H9d p2 (1,069% / 50,99%).

- 2026-10-04 — **custo implantado medido para todos os podadores**
  (`docs/RESULTADOS_overhead_podadores_iscas.md`, commit `9cf1899`): pós-NONE 0,26–0,36%, pré-busca
  efficiency-first 0,41–0,48%, estendidas 0,37–0,42% (pré-busca + estendidas 0,78–0,90%), CNN nativa
  0,13–0,19% do tempo de codificação (3 quadros, p1, `--threads=1`). Defeito corrigido na
  instrumentação (inferência do H9d somada no acumulador do H9c). Extração duplicada pelo podador de
  estendidas declarada.

- 2026-10-04 — **passada de legibilidade contra a narrativa do LASCAS** (blocos 1–4, commits
  `81930e5`, `5dd67f8`, `750ec7a` e o do bloco 4): corpo de 31,0/27 (LASCAS) e 34,1/33 (ISCAS antes)
  para **24,9 de média e 25 de mediana**, máx. 47 palavras. Decisões: (i) a Introdução nomeia os três
  podadores e o argumento do preset vai para o §4, como justificativa da avaliação; (ii) a "main
  strategy" do III é a mesma da Introdução; (iii) a regra de composição é **enunciada no III** com os
  dois limites ("Two limits follow, measured in Section IV") e só confirmada no IV; (iv) o terceiro
  ponto (0,90/0,90, τ_rest desligado) fica só na nota b da Tabela I; (v) Resultados citam faixas
  (5,15 a 7,61 pp, derivadas da Tabela I), não triplas. Correções de rigor: o p3 não é onde o libaom
  "começa" a podar estendidas (as redes AB/4-way estão ativas em todos os presets); o que entra no p3
  é a poda pelo resultado do SPLIT (`prune_ext_part_using_split_info`) → "libaom adds its own
  pruning"; o pós-NONE só fica "do lado da eficiência" em p1 e p2 (em p3 tem BD-BR maior que a CNN).

- 2026-10-04 — **Tabelas I e II refeitas para leitura isolada** (crítica do orientador: notas a/b da
  Tab. I e a coluna de preset "1, 1, 1" da Tab. II). Tab. I sem notas remissivas: grupo "NPL-AV1:",
  linha Δ do podador de estendidas (+0,032/+0,87, +0,039/+0,83, +0,002/−0,14) no lugar da linha
  absoluta (que não fechava a conta: 50,05 + 0,83 ≠ 50,99, base recodificada), uma linha "lower/higher
  is better". O ponto de `cpu-used=0` e o absoluto 50,16% → 50,99% (1,069%) foram para o texto; a
  contagem 16/13/10 diz agora de onde vêm as 16. Tab. II com presets como linhas de grupo, BD-BR e TS
  agrupados, p < 0,05 em negrito. Palavras aprovadas fora do molde: *better, minus, bold, added*.

## 5b. Conformidade com a CTC (CWG-G082 v9, `src/samples/aomctc_test_set/`), conferida em 2026-10-03

Conforme: sequências A1, 15 quadros (`--limit=15`), flags do §4.1, ladrilhamento 4K do §4
(`--tile-rows=0 --tile-columns=1 --threads=2 --row-mt=0`), âncora em `--cpu-used=0`.
Desvios a declarar no artigo: (i) `--cq-level` 20/32/43/55 em vez de `--qp` da Tabela 12 e sem
`--use-fixed-qp-offsets=1` (inexistentes no aomenc v3.10.0); (ii) tempo de **parede** em vez do
*user time* de `/usr/bin/time` exigido no §5.7 (vale também para o LASCAS); (iii) **novo no ISCAS**:
os braços de teste rodam em `cpu-used` 1–3 — a CTC fixa `cpu-used=0` para comparar codificadores;
aqui o preset é a variável controlada e só a âncora segue a regra. Nota: `M3` §3.3 diz
`--threads=1`, mas o código (`encode_ctc.encode`) usa `--threads=2`, que é o que a CTC manda para 4K —
corrigir o texto do M3.

## 6. Pré-registro da campanha D2 enxuta (registrado em 2026-10-03, ANTES da 1ª codificação)

**Desenho.** CTC A1, 8 seq. × cq {20,32,43,55} × `cpu-used` {1,2,3}, 15 quadros, protocolo de
`encode_ctc.encode` (inalterado). Dois braços por (seq, cq, preset), executados em sequência no
mesmo binário `libaom_perf_h9d`, CNN nativa desligada: `h9a_bal_cpuN` (base, H9d off) e
`h9a_bal_h9d_cpuN` (base + H9d PL10, limiares compilados 0,0910/0,1031/0,0144). 192 codificações.
Âncora: `anchor` `cpu-used=0` de `results/benchmark/fase6/raw_results.csv`.

**Grandezas.** ΔTS(N) = TS(h9a_bal_h9d_cpuN) − TS(h9a_bal_cpuN), TS na definição canônica (M3 §3.4)
contra a âncora; ΔBD(N) = BD-BR(h9d) − BD-BR(base), ambos contra a âncora, PSNR-Y. Resolução de
tempo: 0,46 pp (2σ pareado).

**Previsões** (fundamento: `speed_features.c`, `set_allintra_speed_features_framesize_independent`):

| id | previsão | fundamento |
|---|---|---|
| P0 | as 96 linhas-base reproduzem byte a byte as linhas `h9a_bal_cpuN` de `fase6_swap` | H9d desligado é inerte (1 ponto já verificado) |
| P1 | em p1 e em p2: **0,46 pp < ΔTS < 1,02 pp** | resíduo de partições estendidas preservado (nenhuma poda nativa nova de estendidas até p2), mas `reuse_best_prediction_for_part_ab=1` (p≥1) barateia a avaliação AB |
| P2 | em p1, p2 e p3: **ΔBD ≤ +0,05 pp** | o H9d só remove candidatos estendidos e custou +0,018 pp em p0 |
| P3 | **ΔTS(p3) < min(ΔTS(p1), ΔTS(p2))** | `prune_ext_part_using_split_info=1` (p≥3): o codificador passa a podar as estendidas — sobreposição de ação prevista pela regra de composição |

**Regras de leitura, fixadas antes da medição.**
1. Nenhuma recalibração após ver os dados: nada de rodar PL20 ou outro limiar e reportar o melhor.
2. ΔTS ≤ 0,46 pp é reportado como **não resolvido**, nunca como positivo.
3. Previsão falhada é reportada como falhada, com o valor obtido. Se P1 falhar por ΔTS baixo, o
   texto diz que o ganho do H9d se restringe ao `cpu-used=0`; se P3 falhar, a explicação por
   sobreposição com a poda nativa de estendidas **não** é sustentada e não entra no artigo.
4. Se P0 falhar em qualquer linha, a campanha é invalidada até diagnóstico.
