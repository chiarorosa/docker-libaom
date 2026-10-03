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
5. Ao fechar cada etapa: compilar sem `Overfull`/referência indefinida, commit + push (sem atribuição
   de IA na mensagem).

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

### Pendências fora do artigo
- `M3` §3.3: corrigir `--threads=1` → `--threads=2` (o código e a CTC usam 2 para 4K).
- `R3` §3.4: "menos da metade da taxa BD da nativa nos dois presets" só vale no p1
  (0,065 vs 0,153); no p2 é 0,173 vs 0,259 (67%). Corrigir o texto.
- Desvio tempo de parede vs *user time* (CTC §5.7): decidir se/como declarar (afeta também o LASCAS).

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
