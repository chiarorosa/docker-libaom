# Plano dos capítulos de Metodologia e Resultados

> **Documento-espinha.** Organiza a escrita dos dois capítulos centrais da tese a
> partir, e apenas a partir, do material produzido neste projeto. Cada documento
> deste diretório é redigido em prosa de tese e traz, ao fim de cada seção, a
> **procedência** da afirmação (documento-fonte, artefato numérico, script de
> reprodução e commit). Criado em **2026-07-29**, branch `ml-partition-dev`.

---

## 1. Objetivo declarado da tese

O objetivo declarado, fixado pelo orientando em 2026-07-29 e adotado por todos os
documentos deste diretório, é:

> **Estudo de aprendizado de máquina (do inglês *machine learning* – ML), com foco
> em redes neurais, para propor soluções e heurísticas aplicadas ao
> particionamento de blocos do AV1 em predição intraquadro, buscando redução do
> tempo de codificação com o menor impacto possível na eficiência de codificação.**

Este enquadramento é o que sustenta a arquitetura dos dois capítulos. Ele acomoda,
sem tensão, o conjunto completo do que foi investigado: o modelo substituto
convolucional (ConvNeXt), os perceptrones de múltiplas camadas por tamanho de
bloco (H9a, H9c, H9d), a rede de grafos do Approach B e a reformulação por
regressão de *regret* — todas são redes neurais propostas como heurística de poda
para a mesma decisão de particionamento intraquadro. As soluções que não
sobreviveram aos critérios de decisão não são descarte: são o resultado do
estudo, e entram nos capítulos com essa função.

O enquadramento também dissolve a tensão registrada em `docs/ANDAMENTO_tese.md`
§1.3 — a de que a solução implantada não é convolucional. Sob o objetivo acima, a
questão não é qual família de rede foi embarcada, e sim **que informação e que
ação a decisão de particionamento exige**, pergunta que o estudo responde por
medição.

---

## 2. A tese em uma frase, no estado em que os dados a sustentam

Demonstra-se, por uma hierarquia medida sobre o mesmo conjunto reservado —
variância < ConvNeXt < `pixels24` < H9a —, que **nenhuma via baseada em pixels
compete com o contexto de taxa-distorção barato** (vizinhança de particionamento,
quantização e posição) na decisão de particionamento intraquadro do AV1, e que
esse contexto é **suficiente para superar todas elas sob política casada**. A
afirmação **não** se estende a superar o podador nativo na média da grade CTC:
ali o valor prático medido é a **granularidade fina em baixo regime de aceleração**
que a escada discreta dos presets nativos não oferece.

Duas soluções positivas foram implantadas em C e medidas sob protocolo CTC — o
**H9a** (poda pré-busca) e o **H9d** (poda seletiva das partições estendidas, que
se soma ao H9a por **+1,02 pp** de redução de tempo ao custo de **+0,018 pp** de
taxa BD). Cinco vias foram encerradas nos seus critérios de decisão, cada uma com
resultado negativo medido.

---

## 3. Estrutura do Capítulo de Metodologia

| § | Título da seção | Documento deste diretório |
|---|---|---|
| 1 | Objeto, escopo e formulação do problema | `M1_objeto_e_formulacao.md` |
| 2 | Instrumentação do codificador e geração do conjunto de dados | `M2_instrumentacao_e_dataset.md` |
| 3 | Protocolo de avaliação congelado e critérios de decisão em cascata | `M3_protocolo_avaliacao.md` |
| 4 | Projeto de atributos e política de poda | `M4_atributos_e_politica.md` |
| 5 | Arquitetura de software, paridade e garantia de inércia | `M5_arquitetura_software.md` |
| 6 | Arquiteturas de rede neural e metodologia de atribuição | `M6_modelos_e_atribuicao.md` |

O fio condutor do capítulo é a **auditabilidade**: cada decisão de projeto foi
tomada contra um critério quantitativo fixado antes da medição, e o custo caro —
integração em C e codificação real — só foi pago depois que o sinal se provou
fora do codificador.

---

## 4. Estrutura do Capítulo de Resultados

| § | Título da seção | Documento deste diretório |
|---|---|---|
| 1 | O domínio de pixels: diagnóstico, limite superior e os cinco negativos | `R1_dominio_pixels.md` |
| 2 | H9a — poda pré-busca por contexto de taxa-distorção | `R2_h9a.md` |
| 3 | H9c — refinamento pós-NONE, o fator de confusão e o resultado limpo | `R3_h9c.md` |
| 4 | H9d — poda seletiva das partições estendidas | `R4_h9d.md` |
| 5 | Reformulações do problema: resultados negativos de valor metodológico | `R5_resultados_negativos.md` |
| 6 | Análise integrada: fronteira de compromisso e as três conclusões | `R6_analise_integrada.md` |
| 7 | Ameaças à validade, limitações e decisões de escopo | `R7_ameacas_e_escopo.md` |

A separação entre os dois cenários experimentais é estrutural e deve ser mantida
no texto: os experimentos no universo do próprio aprendizado de máquina (partição
10/3/3 sobre as 16 sequências UVG 4K) **validam e caracterizam** a contribuição,
provando que o ganho é atribuível ao modelo; as oito sequências da CTC (*Common
Test Conditions*) produzem os **resultados finais**, comparáveis à literatura e
medidos contra o próprio botão de velocidade do codificador.

---

## 5. Aparato de apoio

| documento | função |
|---|---|
| `A1_INDICE_evidencias.md` | Todo número destinado aos capítulos, com documento-fonte, artefato numérico, script de reprodução e commit. É a prova de cada afirmação e a defesa contra alegação de seleção *a posteriori*. |
| `A2_TABELAS_E_FIGURAS.md` | Plano das tabelas e figuras dos dois capítulos: legenda, dados de origem e script sugerido. Nenhuma figura existe ainda no projeto. |
| `A3_RETRATACOES_E_LACUNAS.md` | Afirmações retiradas ou corrigidas ao longo da investigação, que o texto final **não pode** repetir; e as lacunas conhecidas, com o que cada uma custaria para fechar. |

---

## 6. Regras editoriais válidas para todos os documentos

1. **Estilo.** Prosa acadêmica conforme o perfil da *skill* `escrita-tese`:
   impessoalidade total, passiva analítica, períodos de 25 a 35 palavras,
   conectivos "Então,", "Além disso,", "Deste modo,", "Por outro lado,", "Neste
   caso,", "pois", "uma vez que", "ou seja,", "Por fim,". São proibidos "No
   entanto", "Entretanto", "Contudo", "Todavia", "Dessa forma", "Ademais".
2. **Terminologia.** Português acadêmico rigoroso. "Modelo substituto" e "modelo
   estudante (destilado)", nunca "professor"; "custo computacional", nunca
   "complexidade", na comparação com a rede convolucional nativa; "cota superior"
   ou "limite superior de desempenho", nunca "teto"; "conjunto de teste
   reservado", nunca *held-out*; "atributos", nunca *features*; "critério de
   decisão" ou "etapa de validação", nunca *gate*; "sobreajuste", nunca *overfit*;
   "transição abrupta na curva", nunca "penhasco". Estrangeirismos legítimos em
   itálico; siglas expandidas na primeira ocorrência.
3. **Nenhum dado inventado.** Todo número provém de um documento ou artefato deste
   projeto. Lacuna se marca com `[completar: descrição do que falta]`; nunca se
   preenche por estimativa.
4. **Procedência obrigatória.** Cada seção fecha com uma nota de procedência
   listando documento-fonte, artefato numérico em `results/`, script de
   reprodução e, quando registrado, o commit.
5. **Retratações respeitadas.** As afirmações listadas em
   `A3_RETRATACOES_E_LACUNAS.md` foram retiradas por medição posterior e não podem
   reaparecer em nenhum documento.
6. **Honestidade dos vereditos.** Critério de decisão não atingido se declara como
   não atingido, com o que de fato se obteve; resultado negativo se apresenta como
   resultado, com o valor metodológico que carrega.

---

## 7. Fontes primárias do projeto

- `docs/SINTESE_resultados_metodologia.md` — consolidação de metodologia e
  resultados; estrutura de capítulos proposta em §7.
- `docs/ANDAMENTO_tese.md` — histórico de decisões, retratações e arco da
  investigação.
- `docs/INVENTARIO_solucoes.md` — todas as soluções e todas as configurações
  testadas, por família, com cobertura experimental.
- `docs/RASTREABILIDADE.md` — grafo de proveniência, inventário de scripts e
  pipeline de reprodução comando a comando.
- `docs/PROTOCOLO_avaliacao.md`, `docs/DECISOES_escopo.md`,
  `docs/METODOLOGIA_pipeline_ML.md`, `docs/ARQUITETURA_pruner_implantado.md`.
- Os vinte e cinco documentos `docs/RESULTADOS_*.md`, um por experimento.
- Planos e especificações em `docs/superpowers/{plans,specs}/`.
- Artefatos numéricos em `results/benchmark/`, `results/models/` e
  `results/dataset_h9/`.
