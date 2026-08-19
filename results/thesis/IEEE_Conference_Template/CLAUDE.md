# CLAUDE.md — Normas IEEE e padrão de escrita ViTech

Escopo: este arquivo governa **todo texto de artigo/tese produzido neste diretório e todo texto
destinado a submissão IEEE**. As regras da Parte A são normativas (violá-las causa rejeição
editorial); as da Parte B são o padrão de escrita do Video Technology Research Group (ViTech/UFPel),
destilado dos artigos em `writing_style_ViTech(...)/` e de cumprimento obrigatório.

Precedência: `CLAUDE.md` da raiz do projeto → este arquivo → convenções gerais. Onde a *call for
papers* de uma conferência específica divergir do template genérico (p. ex. ICIP numera seções em
arábico, SBCCI rotula "Keywords" em vez de "Index Terms"), **a instrução da conferência vence** — e
só ela; nada mais é negociável.

Fontes desta especificação:
- `conference_101719.tex` + `IEEEtran.cls` — template oficial de conferência IEEE.
- `IEEEtran_HOWTO.pdf` (Michael Shell, v1.8b+) — manual normativo da classe.
- `writing_style_ViTech(Video_Technology_Research_Group_UFPel)/` — corpus de estilo:
  - **GM-RF** (ICIP 2022) — conferência 5 páginas, ML + AV1 intra.
  - **Direction-Based Fast Mode Decision** (SBCCI 2022) — conferência, heurística + hardware.
  - **DM-FIFS** (IEEE TCSI-I 2026) — periódico, ML + AV1 IFS.

> O corpus ViTech é **fonte de forma, não de conteúdo**. Nenhum número, sequência de vídeo,
> resultado ou afirmação desses artigos pode ser reaproveitado como dado. Conteúdo derivado da tese
> sai exclusivamente de `results/thesis/`.

---

# PARTE A — Normas IEEE (inegociáveis)

## A.1 Classe e preâmbulo

```latex
\documentclass[conference]{IEEEtran}
```

- Opções por categoria (o default está em negrito no HOWTO): tamanho (**10pt**), modo
  (**conference** | journal | technote | peerreview), papel (**letterpaper** | a4paper),
  colunas (**twocolumn**), lados (**oneside**/**twoside**). Categorias são ortogonais.
- `10pt` para a esmagadora maioria dos artigos; `9pt` só em technote; `11pt` só se a conferência pedir.
- `\IEEEoverridecommandlockouts` **somente** se precisar de `\thanks` (rodapé de financiamento) em
  modo conferência — o modo conferência desabilita `\thanks`, `\IEEEPARstart`, `\IEEEbiography`,
  `\IEEEmembership`, `\IEEEpubid`. Se não houver financiamento a declarar, comente a linha.
- Pacotes permitidos e suficientes: `cite`, `amsmath,amssymb,amsfonts`, `algorithmic`, `graphicx`,
  `textcomp`, `xcolor`. Adicione `url` se houver URLs.
- **Proibido**: `geometry`, `pslatex`, `mathptm`, `fullpage`, qualquer pacote que altere margens,
  fontes, espaçamentos ou estilo de heading. Proibido `cuted.sty`/`midfloat.sty` (material atravessando
  o meio das duas colunas — a IEEE não faz isso). Proibido `algorithm.sty`/`algorithm2e.sty` como
  ambiente flutuante: os únicos floats da IEEE são `figure` e `table`.
- Não ajustar manualmente margem, entrelinha, tamanho de papel ou espaçamento de seção. "Muitos erros
  com IEEEtran consistem em fazer demais, não de menos" (HOWTO, Apêndice D).

## A.2 Título

- Capitalização de título: capitalize tudo, exceto `a, an, and, as, at, but, by, for, in, nor, of,
  on, or, the, to, up` — que só sobem para maiúscula se forem a primeira ou a última palavra.
- **Sem** matemática, símbolos especiais, notas de rodapé ou caracteres não-ASCII no título.
- Subtítulos não são indexados pelo Xplore — não usar.
- `using` vs `Using`: se "that uses" puder substituir "using" com o mesmo sentido, capitalize; senão,
  mantenha minúsculo.
- `\\` é permitido apenas para equilibrar o comprimento visual das linhas do título.

## A.3 Autores e afiliações (modo conferência)

- Classe desenhada para até **seis** autores; mínimo um. Ordem da esquerda para a direita, depois
  descendo — essa é a ordem que indexadores e citações futuras usarão.
- Até três afiliações → formato multicoluna com `\IEEEauthorblockN{}` (nomes) +
  `\IEEEauthorblockA{}` (afiliação), separados por `\and`.
- Mais de três autores ou texto largo demais → formato longo com `\IEEEauthorrefmark{n}` ligando nome
  a afiliação.
- Não listar autores em colunas por afiliação nem agrupar por instituição. Afiliação o mais sucinta
  possível — não diferenciar departamentos da mesma organização.
- Boilerplate de agradecimento do grupo (conferir a vigência com o orientador antes de submeter):
  > This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível
  > Superior – Brasil (CAPES) – Finance Code 001, and also by the Brazilian research support agencies
  > CNPq and FAPERGS.
  Em conferência, vai em `\section*{Acknowledgment}`; financiamento específico do artigo vai no
  rodapé sem número da primeira página (`\thanks`).

## A.4 Abstract e Index Terms

```latex
\begin{abstract} ... \end{abstract}
\begin{IEEEkeywords} ... \end{IEEEkeywords}
```

- **CRÍTICO** (texto do próprio template): não usar símbolos, caracteres especiais, notas de rodapé
  nem matemática no título ou no abstract.
- Citações no abstract: evitar. Se a conferência exigir, o número deve ser autocontido.
- Siglas definidas no abstract **precisam ser redefinidas na primeira ocorrência no corpo do texto**.
- Index terms: sem matemática, sem símbolos especiais. Lista de termos válidos por e-mail em
  `keywords@ieee.org`.
- A classe emite o rótulo correto ("Index Terms—" ou "Keywords—") conforme o modo; nunca digitar o
  rótulo à mão.

## A.5 Seções e headings

- `\section`, `\subsection`, `\subsubsection`, `\paragraph`. Numeração automática: romano maiúsculo,
  letra maiúscula, arábico, letra minúscula (modos não-compsoc). **Nunca numerar heading à mão.**
- Se não houver ao menos **dois** subtópicos, não introduza subheading algum.
- Seções não numeradas (`Acknowledgment`, `References`) via `\section*{}`.
- Referência cruzada sempre por `\ref`/`\eqref` — nunca número literal.

## A.6 Citações e referências

- `\usepackage{cite}` — ordena e comprime automaticamente números adjacentes no estilo IEEE. Para
  funcionar, múltiplas citações adjacentes **devem** estar em um único `\cite{a,b,c}`.
- Numeração consecutiva entre colchetes; a pontuação da frase vem **depois** do colchete: `... no AV1 [3].`
- Referir-se apenas pelo número: `as shown in [3]` — nunca "Ref. [3]" ou "reference [3]", exceto em
  início de frase: `Reference [3] was the first ...`.
- Formato de entrada: dar **todos** os nomes de autores, a menos que sejam seis ou mais — só então
  `et al.` (sem ponto em "et"). No título do artigo citado, capitalizar apenas a primeira palavra,
  exceto nomes próprios e símbolos de elementos.
- Estados especiais: `unpublished` (submetido mas não publicado), `in press` (aceito).
- Preferir gerar por BibTeX (`\bibliographystyle{IEEEtran}`); ao enviar o fonte a terceiros, colar o
  `.bbl` dentro do `.tex`. Formatar referência à mão é a última opção (propenso a erro).
- Notas de rodapé numeradas separadamente em sobrescrito, ao pé da coluna em que foram citadas.
  **Nunca** nota de rodapé no abstract ou na lista de referências; em tabela, usar letras.

## A.7 Equações

- `\begin{equation} ... \label{eqn_x} \end{equation}`; numeração consecutiva; `displaymath` se não
  quiser número.
- Referir-se por `\eqref{eqn_x}` → "(1)", **sem** a palavra "equation" — exceto em início de frase:
  "Equation (1) is ...".
- Pontuar a equação como parte da frase (vírgula ou ponto ao final).
- Definir todos os símbolos imediatamente antes ou depois da equação.
- Itálico para símbolos romanos de grandezas e variáveis; **não** italicizar símbolos gregos. Sinal de
  menos é travessão de menos, não hífen.
- **Proibido `eqnarray`** — usar `align` (amsmath) ou `IEEEeqnarray`. Com amsmath, adicionar
  `\interdisplaylinepenalty=2500` para restaurar quebras automáticas.
- `\nonumber` nunca dentro de `{array}`. `{subequations}` incrementa o contador principal mesmo sem
  exibir número — atenção a saltos na numeração.
- É responsabilidade do autor quebrar equações longas para caberem na coluna. Usar subfunções é
  válido; **reduzir o tamanho da fonte matemática não é**. Equação em duas colunas: evitar (raríssima
  na IEEE, exige manipulação manual do contador).

## A.8 Figuras

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{fig}
  \caption{Descrição.}
  \label{fig_x}
\end{figure}
```

- **Legenda abaixo da figura.** `\label` **depois** (ou dentro) de `\caption` — colocar antes é o erro
  mais comum do LaTeX e faz a referência apontar para o número da seção.
- `\centering`, nunca o ambiente `center` (adiciona espaço vertical indesejado).
- Posicionar no **topo** (`[!t]`) ou base de coluna; evitar o meio. Nunca na primeira coluna da
  primeira página. Figura larga → `figure*` (duas colunas), definida **na página anterior** àquela em
  que deve aparecer.
- Inserir a figura **depois** de ela ser citada no texto.
- Chamada no texto: **"Fig. 1"**, inclusive em início de frase (modo conferência tradicional).
- Formato: vetorial EPS/PDF para desenhos, gráficos e diagramas; bitmap (PNG/TIFF/EPS/PDF) só para
  fotos. **Inaceitáveis**: BMP, EMF, VSD, GIF. Line art bitmapizada é erro listado no HOWTO.
- Rótulos de eixo com **palavras**, não símbolos: "Magnetization (A/m)", não "M" nem "A/m" sozinho.
  Unidade entre parênteses; nunca rotular eixo só com unidade nem com razão grandeza/unidade
  ("Temperature (K)", não "Temperature/K"). Rótulos em 8 pt.
- Subfiguras: `subfig` com `caption=false,font=footnotesize`; a prática ViTech/IEEE dominante é
  descrever (a), (b), ... dentro da legenda principal em vez de dar legenda a cada subfigura.
- Fontes embutidas e subsetadas, apenas Type 1 (vetorial).

## A.9 Tabelas

```latex
\begin{table}[!t]
  \renewcommand{\arraystretch}{1.3}
  \caption{Table Type Styles}
  \label{tab_x}
  \centering
  \begin{tabular}{...} ... \end{tabular}
\end{table}
```

- **Legenda ACIMA da tabela** (ao contrário da figura), capitalizada como título (mesma regra de
  capitalização do A.2). Unidades e letras matemáticas dentro da legenda vão em `\upshape` para não
  serem versaletadas.
- Texto interno em `footnotesize` (default da classe); `\arraystretch` ≈ 1.3 para abrir as linhas.
- Notas de tabela com **letras** sobrescritas, ao pé da própria tabela (`threeparttable` ou
  `\multicolumn` na última linha), nunca `\footnote`.
- Chamada no texto: **"Table I"**, por extenso e com numeral romano gerado pela classe.

## A.10 Unidades, números e abreviações

- SI (MKS) como unidade primária; unidades inglesas apenas entre parênteses como secundárias.
- Não misturar SI e CGS. Não misturar grafia por extenso e abreviação: "Wb/m²" ou "webers per square
  meter", nunca "webers/m²". Por extenso quando a unidade aparece isolada no texto.
- Zero antes do ponto decimal: `0.25`, nunca `.25`. `cm³`, nunca `cc`.
- Definir toda abreviação/sigla na primeira ocorrência no texto, **mesmo já definida no abstract**.
  Não precisam de definição: IEEE, SI, MKS, CGS, ac, dc, rms. Evitar siglas no título e nos headings.

## A.11 Erros que reprovam (checar sempre)

- Texto de instrução do template deixado no artigo → **pode impedir a publicação** (aviso em vermelho
  no `conference_101719.tex`). Remover tudo.
- `\label` antes de `\caption`.
- Margens/fontes/espaçamentos alterados manualmente ou por pacote.
- Line art em bitmap; fontes não embutidas.
- Equação estourando a largura da coluna.
- Referências formatadas à mão com desvios do estilo IEEE.
- Estouro do limite de páginas da conferência.
- Última página com colunas desbalanceadas: equalizar manualmente com `\newpage`,
  `\enlargethispage{-X.Yin}` ou `\IEEEtriggeratref{n}`. Não usar `balance.sty`/`flushend.sty`
  (não confiáveis).

## A.12 Miscelânea de estilo IEEE (do template)

- "data" é **plural**.
- Pontuação dentro das aspas apenas quando se cita um pensamento completo ou um título; quando as
  aspas apenas destacam um termo, a pontuação fica **fora**.
- Frase parentética no fim de período: pontuação fora do parêntese. Período inteiro entre parênteses:
  pontuação dentro.
- "inset" (não "insert"); "alternatively" (não "alternately"); nunca "essentially" no sentido de
  "approximately"/"effectively".
- Vigiar homófonos: affect/effect, complement/compliment, discreet/discrete, principal/principle;
  imply ≠ infer; i.e. = "that is", e.g. = "for example".
- Prefixo "non" colado à palavra, em geral sem hífen.
- Grafia americana: "acknowledgment" (sem "e" depois do "g").

---

# PARTE B — Padrão de escrita ViTech

## B.0 Princípio

Toda afirmação é ancorada em evidência (número medido, citação ou dedução explicitada). Todo ganho é
reportado junto do seu custo. Toda comparação declara se é justa; quando não é, a limitação vem
**antes** do número. O texto não persuade por adjetivo, persuade por contabilidade.

## B.1 Arquitetura do artigo

Conferência (4–6 páginas):
```
I. Introduction → II. <Ferramenta> in AV1 (fundamentação) → III. Proposed <ACRÔNIMO> Solution
→ IV. Results and Comparisons → V. Conclusions → Acknowledgment → References
```
Periódico: acrescenta seção própria de *Related Works*, uma de *Assessment and Ablation Experiments*
antes do método, e subseções de treinamento/validação dentro do método.

Regras: a seção de fundamentação é nomeada pela ferramenta atacada, não "Background". A seção do
método carrega o acrônimo da solução no título. A de resultados é "Results and Comparisons" /
"Results and Discussion" — nunca só "Results".

## B.2 Abstract — sequência obrigatória de movimentos

1. **Enquadramento**: o que é o objeto e por que existe (uma frase).
2. **Problema**: a ferramenta eleva o esforço computacional (uma frase).
3. **Apresentação**: `This paper presents the <Nome por Extenso> (<ACRÔNIMO>), a <tipo de solução>
   for <alvo> applying <técnica>.`
4. **Mecanismo**: como a solução opera, em 1–2 frases.
5. **Resultado**: `Experimental results show that <ACRÔNIMO> achieves an average time savings of
   X.XX%, with a BD-BR of Y.YY%.` — números concretos, sempre par ganho/custo.
6. **Comparação**: posição frente aos trabalhos relacionados, com o custo admitido
   (`at a cost of a higher BD-BR`).
7. **Ineditismo**: `To the best of the authors' knowledge, this is the first solution in the
   literature ...`

Sem citações, sem matemática, sem sigla não expandida. 150–250 palavras.

## B.3 Index Terms

4–6 termos, separados por vírgula, do geral para o específico, começando pelo domínio:
`Video coding, AV1, computational effort reduction, fractional motion estimation, machine learning.`

## B.4 Introdução — sequência de movimentos (M1–M8)

- **M1 Macrocontexto**: demanda por vídeo / tráfego / royalties, com fonte citável (Cisco, Statista).
- **M2 Panorama de codecs**: padrões (HEVC, VVC, via ISO/IEC MPEG + ITU-T VCEG) *versus* codecs
  abertos (VP8/VP9/AV1); fundação da AOMedia, origem em VP9/Thor/Daala, lançamento em 2018.
- **M3 Estrutura do AV1**: fluxo híbrido — intra, inter, transformadas, quantização, filtros in-loop,
  codificação de entropia — e a nota de que o AV1 introduziu novidades em cada etapa.
- **M4 O custo**: eficiência de codificação superior *ao preço* de esforço computacional, quantificado
  contra um competidor e citado ("twice as long", "up to 58 times more run time").
- **M5 Estreitamento**: da etapa alvo (intra / FME / IFS / partição), por que ela é cara.
- **M6 Lacuna**: enumerar os poucos trabalhos existentes, **classificá-los** (eficiência de
  codificação / hardware / heurística / outro codec) e fechar com a lacuna explícita —
  `none of these works explore ...`.
- **M7 Contribuição**: `This paper presents <ACRÔNIMO>, a ... The main strategy of this solution is
  to ...`. Em periódico, as contribuições viram lista com marcadores (3–4 itens, cada um começando
  por substantivo ou verbo no infinitivo).
- **M8 Fecho**: em conferência, prévia do resultado principal; em periódico, sempre
  `The rest of this paper is organized as follows. Section II presents ...`.

## B.5 Seção de fundamentação

Presente do indicativo, alta densidade de citação, zero opinião. Descreve a ferramenta no nível de
detalhe **exatamente** necessário para o método proposto ser compreensível — nem mais. Encerra com um
**escopo explícito**, delimitando o que o trabalho toca e o que não toca:

> *"This work is focused on reducing the number of modes evaluated inside this loop. Some intra-frame
> prediction modes are not included in this loop, such as CfL, ... and therefore these four modes are
> not focused on this work."*

Esse fechamento de escopo é obrigatório. É ele que protege o artigo na revisão.

## B.6 Trabalhos relacionados

Um parágrafo por trabalho (ou por família de trabalhos do mesmo grupo), na fórmula:

> `The work [N] proposes <X>. The authors introduce <Y>. Results show <Z>.` + **veredito de
> comparabilidade**.

O veredito é obrigatório e vem em três sabores:
- comparável → `allowing a comparison with our work`;
- não comparável, com a razão → `it focuses on different inter-prediction tools, which prevents a
  direct and fair comparison`;
- adaptável → `their concept can all be adapted to work with the AV1 coding tools`.

Trabalho anterior do próprio grupo é tratado como qualquer outro, identificado como tal
(`Our previous work is presented in [14]`), e é o baseline contra o qual o avanço é medido.

## B.7 Método proposto

1. Nomear a solução, expandir o acrônimo, declarar a **estratégia principal em uma frase**.
2. Justificar cada decisão de projeto com razão técnica explícita (por que este agrupamento, por que
   este modelo, por que dois modelos e não um — trade-off viés/variância, custo do inferidor).
3. Figura de fluxograma + caminhada textual passo a passo por ela.
4. **Declaração de conformidade**: `fully compliant with the AV1 specification` /
   `does not affect bitstream syntax`. Obrigatória em solução que mexe no encoder.
5. Declarar os limites estruturais do próprio método (`this initial interpolation ... represents a
   limiting factor in achieving a full reduction`), antecipando a leitura dos resultados.

## B.8 Metodologia experimental e treinamento

Itens obrigatórios, cada um em frase própria e verificável:
- versão exata do libaom;
- configuração do encoder (All-Intra / Random Access) e **norma seguida** (CTC / VCTQM), citada;
- valores de CQ (tipicamente 20, 32, 43, 55) e número de frames;
- sequências, resolução, bit depth, subamostragem de croma, origem (XIPH), e diversidade demonstrada
  (dispersão SI × TI conforme ITU-T P.910, quando aplicável);
- partição treino/validação/teste e a **declaração antivazamento**: `no video sequence was reused
  across the training, testing, or evaluation datasets, thus avoiding data leakage and overfitting`;
- features: origem, seleção, e `all selected features are natively available during encoding, ensuring
  zero computational overhead for feature extraction`;
- ferramentas (scikit-learn / XGBoost / m2cgen) e o porquê da exportação para C
  (`ensuring that the models execute natively within the encoding pipeline`);
- métricas de classificação reportadas nas três fases (treino, validação, teste).

## B.9 Resultados e discussão

- Abre reafirmando o setup em uma frase (`The developed solution was evaluated using the libaom X.Y
  AV1 reference software, following the CTC and under AI encoder configuration.`).
- Define cada métrica antes de usá-la, com equação numerada quando não for padrão, e cita BD-BR
  (Bjøntegaard) na primeira aparição.
- Tabela por sequência, com médias por resolução e média geral. O texto **caminha pela tabela**: média
  ± desvio-padrão, depois a razão do comportamento.
- Toda observação de comportamento é seguida de explicação causal ou de hedge honesto:
  `This fact partly explains ...`, `This may be attributed to ...`, `likely attributed to ...`.
- Toda comparação com a literatura declara antes a (in)comparabilidade do setup:
  `it employs a different experimental setup, including an outdated version of libaom, a distinct set
  of video sequences, and different CQ parameters. Consequently, a direct comparison ... is not fair.
  Therefore, highlighting this limitation, the results in [18] reported ...`
- Overhead do próprio método é medido e reportado (`the two models spent 4.17% of the IFS execution
  time`), nunca omitido.
- Quando houver, reportar significância estatística com o teste nomeado e o p-valor — inclusive
  quando o resultado **não** é significativo, explicando por quê.

## B.10 Conclusão

Fórmula fixa, sem número novo que não esteja nos Resultados:

1. `This paper presented <ACRÔNIMO>, <o que é>, reaching <resultado principal>.`
2. Como funciona, em 2–3 frases (recapitulação mecânica, não repetição literal do método).
3. Onde foi implementado e como foi avaliado.
4. Comparação com o original e com os relacionados, com números.
5. Reafirmação do ineditismo (`Besides, to the best of the authors' knowledge, ...`).
6. Em periódico, `Future research directions include ...` — 3–4 direções concretas e executáveis.

## B.11 Regras de sentença

| Dimensão | Regra |
|---|---|
| Tempo verbal | Presente para descrever codec, ferramenta e verdades gerais; **passado** para o que foi feito neste trabalho ("the models were trained", "was implemented"); presente para o que os resultados mostram ("results show", "the results demonstrate"). |
| Voz | Passiva domina em método e experimentos. Ativa em 1ª pessoa do plural apenas em movimentos de reivindicação e comparação: `our work reached`, `To the best of our knowledge`, `We explored`. |
| Agente | `This work` / `This paper` / `The proposed method` é o sujeito canônico. **Nunca 1ª pessoa do singular.** |
| Hedging | Calibrado ao grau de evidência: `may be attributed to`, `likely attributed to`, `is expected`, `partly explains`. Nunca hedge onde há medida; nunca afirmação categórica onde há só correlação. |
| Concessão | Quase todo parágrafo de resultado tem estrutura ganho→custo: `However`, `On the other hand`, `Nevertheless`, `Conversely`, `at a cost of`, `as expected`. |
| Parágrafo | 3–7 frases; uma ideia por parágrafo; primeira frase enuncia a ideia, última fecha a consequência. |
| Frase | Declarativa, ordem direta. Sem pergunta retórica, sem exclamação, sem coloquialismo, sem metáfora. |
| Enumeração inline | `(i) ..., (ii) ..., and (iii) ...` para itens curtos dentro de parágrafo. |

## B.12 Léxico e colocações recorrentes (usar em inglês, verbatim)

**Reivindicação e ineditismo**
`To the best of the authors' knowledge, ...` · `this is the first solution in the literature ...` ·
`the most advanced ... reported to date`

**Introdução de resultado**
`Experimental results show that ...` · `The results demonstrate ...` · `Observing the results in
Fig. N, it is possible to conclude that ...` · `The main conclusion when observing Table N is that ...`

**Análise**
`one can conclude that ...` · `Another important observation is that ...` · `This fact also helps to
explain why ...` · `It is important to emphasize that ...` · `It is important to highlight that ...` ·
`as expected` · `In fact, ...`

**Comparação**
`at a cost of a higher BD-BR` · `X times higher/lower than [N]` · `with negligible impact on coding
efficiency` · `a direct and fair comparison is not possible` · `Therefore, highlighting this
limitation, ...` · `a better trade-off between computational effort and coding efficiency`

**Estrutura**
`The rest of this paper is organized as follows.` · `This work is focused on ...` ·
`The main strategy of this solution is to ...` · `The main contributions of this article are:` ·
`Future research directions include ...`

**Domínio (grafia canônica do grupo)**
coding efficiency · computational effort · encoding time reduction · time savings (TS) · BD-BR ·
RD-Cost · rate-distortion · reference software (libaom) · Common Test Conditions (CTC) ·
All-Intra (AI) / Random Access (RA) · Super Block (SB) · partition tree · trade-off ·
fast decision algorithm · skip / prune

## B.13 Números e métricas

- BD-BR e TS com **duas casas decimais** no corpo e nas tabelas (`50.19%`, `7.41%`, `22.56%`); no
  abstract e na conclusão é aceitável arredondar para leitura (`an average BDBR of 7% with an encoding
  time reduction of 50%`) desde que o valor exato conste dos Resultados.
- Ganhos relativos como razão: `10 times higher`, `7.3 times lower`, `nearly six times higher`.
- Média sempre acompanhada de desvio-padrão ou IC 95% quando houver dispersão relevante.
- Sinal declarado: BD-BR negativo = ganho de eficiência (dizer isso explicitamente na primeira vez).
- Nenhum número aparece no texto sem estar na tabela/figura correspondente, e vice-versa.

## B.14 Legendas e chamadas

- Figura: frase nominal curta, sem ponto final quando é rótulo puro
  (`Fig. 1. GM-RF Flowchart.` / `Fig. 4. Percentage of filter pairs usage by CQ and resolution.`).
  Se a figura for adaptada, creditar: `Modified from [14].`
- Tabela: título nominal capitalizado (`TABLE II — GROUPS OF INTRA-FRAME PREDICTION MODES`).
- Toda figura e toda tabela é citada no texto **antes** de aparecer, e tem pelo menos um parágrafo que
  a interpreta. Figura sem leitura no texto é figura a ser removida.

## B.15 Proibições

- Reportar ganho sem o custo correspondente.
- Comparar com a literatura sem declarar a (in)comparabilidade do setup.
- Número na conclusão ou no abstract que não esteja nos Resultados.
- Superlativo sem medida ("excelente", "muito superior", "dramático").
- 1ª pessoa do singular; anglicismo em texto PT; tradução literal EN→PT.
- Termo "professor/teacher" para modelos ou componentes de ML.
- Citar artigos irmãos da mesma tese entre si — a aprovação dos pares não é garantida.
- Reaproveitar dados, sequências ou resultados do corpus ViTech como se fossem deste trabalho.

---

# PARTE C — Checklist pré-submissão

**Normativo**
- [ ] Todo texto-guia do template removido.
- [ ] Limite de páginas da conferência respeitado.
- [ ] `\label` sempre após `\caption`.
- [ ] Legendas: figura abaixo, tabela acima.
- [ ] Nenhum pacote de margem/fonte carregado; nenhum espaçamento ajustado à mão.
- [ ] Figuras vetoriais para line art; fontes Type 1 embutidas e subsetadas.
- [ ] Equações dentro da largura da coluna; nenhuma via `eqnarray`.
- [ ] Citações agrupadas em `\cite{a,b,c}`; pontuação depois do colchete.
- [ ] Referências no estilo IEEE; `et al.` só com 6+ autores.
- [ ] Sem símbolo/matemática/nota de rodapé em título, abstract e index terms.
- [ ] Siglas redefinidas no corpo mesmo se já definidas no abstract.
- [ ] Colunas da última página equalizadas manualmente.
- [ ] "Fig. N" e "Table N" nas chamadas; `(N)` para equações.

**Estilístico**
- [ ] Abstract cumpre os sete movimentos de B.2.
- [ ] Introdução cumpre M1–M8; a lacuna está enunciada explicitamente.
- [ ] Fundamentação encerra com delimitação de escopo.
- [ ] Cada trabalho relacionado tem veredito de comparabilidade.
- [ ] Método declara conformidade com a especificação AV1 e seus próprios limites.
- [ ] Metodologia lista libaom, configuração, CTC, CQs, sequências, split e antivazamento.
- [ ] Todo ganho vem com custo; toda comparação injusta vem sinalizada antes do número.
- [ ] Overhead do método medido e reportado.
- [ ] Conclusão sem número inédito; direções futuras concretas (periódico).
- [ ] Toda figura/tabela citada e interpretada no texto.

---

# PARTE D — Interface com a tese

- Conteúdo (números, hipóteses, resultados, retratações) vem **apenas** de `results/thesis/`
  (M1–M6, R1–R7, A3 é vinculante). Nunca de `docs/`.
- Figuras de artigo: sempre em inglês, paleta acadêmica sóbria já validada; geradas no contêiner
  `av1_bench` com `venv-ml`.
- Este arquivo define **como escrever**; não define o que é verdade sobre o trabalho.
