# Auditoria numérica dos artigos LASCAS 2027 e ISCAS 2027

Data: 2026-08-13. Objeto: verificar se as tabelas de `results/thesis/PAPER_LASCAS_2027.md`
e `results/thesis/PAPER_ISCAS_2027.md` reproduzem fielmente a fonte canônica, e se todos
os resultados de taxa BD e de redução de tempo apresentados nos dois artigos provêm
exclusivamente da grade das condições comuns de teste (CTC) Classe A1.

Motivação: restrição editorial fixada nesta data — os dois artigos só podem apresentar
taxa BD e redução de tempo medidas sobre a CTC A1; o conjunto UVG pode ser citado quanto
à construção do conjunto de dados e a estatísticas de distribuição, jamais quanto a taxa
BD ou redução de tempo.

## 1. Método de verificação

A redução de tempo foi **recalculada do zero** a partir dos tempos brutos de
`results/benchmark/fase6/raw_results.csv`, na definição canônica declarada em
`results/thesis/M3_protocolo_avaliacao.md` §3.4: para cada ponto de quantização,
`1 − t_configuração / t_âncora`; média sobre os quatro pontos; média sobre as sequências.
Nenhum valor foi lido de tabela intermediária.

Comando de reprodução (PowerShell, host ou contêiner `av1_bench`):

```powershell
$rows = Import-Csv results/benchmark/fase6/raw_results.csv
$anchor = @{}
foreach ($r in $rows | Where-Object {$_.config -eq 'anchor'}) { $anchor["$($r.seq)|$($r.cq)"] = [double]$r.time_s }
foreach ($g in $rows | Where-Object {$_.config -ne 'anchor'} | Group-Object config) {
  $perSeq = foreach ($s in $g.Group | Group-Object seq) {
    $ts = $s.Group | ForEach-Object { 1 - ([double]$_.time_s / $anchor["$($_.seq)|$($_.cq)"]) }
    (($ts | Measure-Object -Average).Average)
  }
  "{0,-18} {1} seqs  {2,6:N2}%" -f $g.Name, $perSeq.Count, ((($perSeq | Measure-Object -Average).Average)*100)
}
```

## 2. Conferência das tabelas — nenhuma divergência

| Tabela | Fonte canônica | Artefato | Resultado |
|---|---|---|---|
| LASCAS II (6 linhas) | `R4_h9d.md` §4.6 | `fase6/bdrate_average.csv` | idêntica |
| LASCAS III (4 preços) | `R4_h9d.md` §4.7 | `fase6/raw_results.csv` | idêntica |
| ISCAS I (fator de confusão) | `R3_h9c.md` §3.3 | `fase6/raw_results.csv` | idêntica |
| ISCAS II (9 linhas) | `R3_h9c.md` §3.4 | `fase6_swap_h9c/swap_average.csv` | idêntica |
| ISCAS III (testes pareados) | `R3_h9c.md` §3.4 | `fase6_analysis/paired_tests.csv` | idêntica |
| ISCAS IV (decomposição) | `R3_h9c.md` §3.5 | `fase6/raw_results.csv` | idêntica |

Redução de tempo recalculada, contra o valor publicado: `ml_balanced` 17,72 (17,72);
`ml_bal_h9d` 18,74 (18,74); `ml_aggr` 31,51 (31,51); `ml_aggr_h9d` 31,68; `ml_bal_h9d_pl20`
19,81; `ml_aggr_h9d_pl20` 32,16; `native_cpu1/2/3` 32,59 / 42,72 / 67,94 (idem);
`h9c_tau45` 21,35 (21,4); `h9ciso_tau90` 5,61 (5,6); `h9adef` 10,88 (10,9);
`h9c_tau90` 13,59 (13,6); `h9c_tau95` 12,61 (12,6).

Os dois estimadores do preço do botão de limiar — 0,063 pp/pp por interpolação por
sequência e 0,0606 pp/pp pela média das médias — foram conferidos contra a nota de
`R4_h9d.md` §4.6 e não se misturam em tabela alguma do artigo.

## 3. Conformidade com a restrição CTC A1

### 3.1 ISCAS — conforme

Todas as campanhas citadas foram verificadas sequência a sequência em
`fase6/raw_results.csv` e `fase6_swap_h9c/`. A cobertura é exclusivamente CTC A1
(BoxingPractice, Crosswalk, FoodMarket2, Neon1224, NocturneDance, PierSeaSide, Tango,
TimeLapse). O subconjunto de três sequências da varredura de τ da Seção IX é
Neon1224, PierSeaSide e TimeLapse, também CTC. O único número medido fora da CTC é o
critério de decisão offline (61,2% de redução de custo de busca simulado a 0,20% de perda
de `SPLIT`), que é simulação sobre o conjunto de dados anotado, e não taxa BD nem redução
de tempo.

### 3.2 LASCAS — quatro não conformidades

As campanhas `results/benchmark/c5_finetau/` e `results/benchmark/h9d_ub/` foram medidas
sobre **Jockey, RaceNight e RiverBank**, que são o conjunto de teste reservado do UVG, e
não sequências da CTC. Verificação: nomes dos arquivos `.obu` nos dois diretórios e coluna
`seq` de `c5_finetau/raw.csv` e `h9d_ub/raw.csv`.

| Passagem do artigo | Número apresentado | Campanha | Conjunto |
|---|---|---|---|
| §IV-C, cota superior do desligamento | +0,89% de taxa BD a 1,431×; marginal 1,293× a +0,798% | `h9d_ub` | UVG |
| §IV-B, densidade da curva de limiares | maior vão de aceleração de 0,15× em 21 vizinhanças | `c5_finetau` | UVG |
| §V, cascata de atribuição | faixas de aceleração disjuntas; taxa BD 11, 44 e 94 vezes menor que a da variância | E5/curva de τ | UVG |
| Figura 2 | curva de limiares do H9a | `c5_finetau` | UVG, **sobreposta** a pontos e *presets* de CTC |

Permanecem admissíveis, por serem distribuição e não taxa BD nem redução de tempo: a
Tabela I e a Figura 1 (decomposição do custo de busca sobre 875.317 nós), a calibração da
Seção IV-B (erro esperado de 0,0112; precisão de 95,6% e 96,5%) e a área sob a curva ROC
de 0,902 da Seção IV-C. Todas exigem, por outro lado, rótulo explícito de que provêm do
conjunto de dados construído sobre o UVG, o que o artigo hoje não declara.

### 3.3 Consequência sobre a alegação de granularidade contínua

Retirada a curva de τ, restam em CTC apenas **dois** pontos de operação do H9a: 17,72% e
31,51% de redução de tempo. A faixa de "12% a 22%" citada no resumo e na Seção VI é a
**dispersão entre sequências** do ponto equilibrado, e não uma faixa varrida pelo limiar.
Valores recalculados por sequência do ponto equilibrado: FoodMarket2 12,35; NocturneDance
12,59; PierSeaSide 16,78; TimeLapse 17,31; BoxingPractice 19,39; Crosswalk 20,18; Tango
21,26; Neon1224 21,93. A afirmação de continuidade precisa ser reescrita, ou sustentada
por uma varredura de τ sobre a grade CTC, cujo custo é de aproximadamente 160 codificações.

### 3.4 Estado — não conformidades corrigidas

As quatro não conformidades de §3.2 e a alegação de continuidade de §3.3 foram corrigidas
no texto do LASCAS na mesma data desta auditoria: os números medidos sobre o UVG foram
removidos do corpo, a restrição de escopo foi declarada nas Seções V e VIII, a Figura 2 foi
respecificada sobre dados exclusivamente de CTC, e a Tabela I, a Figura 1 e as estatísticas
offline das Seções III e IV passaram a identificar o conjunto de dados de origem. O ISCAS
recebeu a mesma declaração de escopo nas Seções II e V, sem remoção de resultado algum.

Permanece pendente o alinhamento do resumo em inglês do LASCAS, que ainda cita a faixa de
12%–22% como intervalo contínuo.

## 4. Achado independente da restrição: o marginal de +0,26 pp do H9c

O contraste central do LASCAS — o H9d soma +1,02 ponto percentual onde o H9c somara
+0,26 — aparece no resumo, na Seção I, na Seção VII e na Conclusão do artigo, e provém de
`docs/SINTESE_resultados_metodologia.md` §5-quater e de `docs/ANDAMENTO_tese.md` §8.
O rastreio do valor mostra que ele é o marginal do H9c medido **na sequência Neon1224
apenas**, sobre a base do H9a nos limiares compilados por padrão.

O mesmo marginal, recalculado nas quatro sequências em que a decomposição existe:

| Sequência | H9a só (`h9adef`) | H9a + H9c (`h9c_tau90`) | Marginal |
|---|--:|--:|--:|
| Neon1224 | 16,75% | 17,07% | +0,32 pp |
| PierSeaSide | 10,43% | 12,84% | +2,41 pp |
| Tango | 7,16% | 17,06% | +9,90 pp |
| TimeLapse | 9,17% | 11,49% | +2,33 pp |
| **Média** | **10,88%** | **14,62%** | **+3,74 pp** |

O valor de +0,26 pp é, deste modo, a sequência de **menor** marginal das quatro medidas, ao
passo que o +1,02 pp do H9d é média sobre **oito** sequências e sobre uma base distinta — o
ponto equilibrado implantado, e não os limiares compilados por padrão. Na média de quatro
sequências, o H9c soma +3,74 pp, valor **superior** ao do H9d, e o contraste de "quatro
vezes" se inverte.

Isto não invalida a tese composicional, que se apoia igualmente na interação medida de
−1,9 ponto percentual (`R3_h9c.md` §3.5) e no colapso do ganho do H9d sobre a base
agressiva (`R4_h9d.md` §4.7). Invalida o **par de números** com que a tese é apresentada.
A correção exige uma de três vias: comparar o H9d contra o H9c sobre a mesma base e a
mesma cobertura de sequências, o que não está medido; apresentar o contraste com a média
de quatro sequências, o que o inverte; ou substituir o argumento numérico pela interação
medida e pelo colapso sobre a base agressiva, ambos já disponíveis.

### 4.1 Estado — contraste corrigido

A terceira via foi adotada no texto do LASCAS na mesma data desta auditoria. A razão entre
magnitudes foi retirada do resumo e das Seções I, VII e IX, e substituída pelo **contraste
de sinal** entre as duas composições que consomem entrada idêntica: interação medida de
−1,9 ponto percentual, na composição do podador binário com o pré-busca, contra marginal de
+1,02 ponto percentual do podador seletivo. A assimetria entre as duas medições — quatro
sequências sobre a base de limiares padrão, contra oito sequências sobre o ponto equilibrado
implantado — passou a ser declarada explicitamente na Seção VII do artigo, com o registro de
que uma comparação pareada entre os dois podadores não foi executada. O valor de 0,26 ponto
percentual não aparece mais em passagem alguma do artigo, à exceção do apêndice interno de
conformidade, que documenta a retratação.

Permanece não medido, e é a via que fecharia a questão de forma definitiva: o marginal do
podador binário e o do podador seletivo sobre a **mesma** base do H9a e as **mesmas** oito
sequências da grade CTC.

## 5. Limitações desta auditoria

1. A taxa BD **não** foi recalculada; foi conferida contra `bdrate_average.csv` e
   `swap_average.csv`, que são saída do mesmo `src/scripts/benchmark/bd_rate.py` que
   produziu os números publicados. Um erro dentro desse script não seria detectado aqui.
2. A verificação de conjunto de sequências cobre as campanhas efetivamente citadas nos
   dois artigos, e não a totalidade dos diretórios de `results/benchmark/`.
3. O rastreio do valor de +0,26 pp chegou a `docs/ANDAMENTO_tese.md`; o valor exato de
   0,26 difere do recálculo direto da Neon1224, que dá 0,32 pp, provavelmente por os dois
   provirem de agregações distintas do mesmo par de execuções. A conclusão — sequência
   única, base distinta — não depende desta diferença.
4. Os artefatos de `results/benchmark/` não são versionados; esta auditoria vale para o
   estado local desses arquivos na data indicada.

## 6. Adendo de 2026-09-04 — retirada da comparação com a escada de presets do LASCAS

Restrição editorial acrescentada nesta data, por orientação: o artigo LASCAS **não pode
apresentar comparação com os presets nativos** (`cpu-used` 1, 2 e 3). A justificativa é de
escopo, e é a mesma que o próprio resumo do artigo declara: o SNP-AV1 introduz duas redes
neurais em dois estágios da busca de partição e não toca em etapa alguma fora dela,
enquanto qualquer preset nativo reconfigura dezenas de heurísticas espalhadas por todas as
etapas do codificador. Pôr os dois lado a lado ordena arranjos de escopo distinto, e não
técnicas de busca de partição.

### 6.1 O que saiu do artigo

| Passagem removida | Conteúdo |
|---|---|
| Resumo | a admissão de não dominância dos presets e a alegação de preencher o vão da escada |
| Seção I | a formulação "não é mais um preset de velocidade" |
| Seção VI (resultados) | os dois parágrafos de qualificação da comparação e a distinção arquitetural em relação à rede convolucional nativa |
| Seção VI | o parágrafo do valor de `cpu-used=1` (32,59% a 0,449%) e da granularidade não coberta |
| Seção VI | a menção ao custo do caminho da rede nativa no parágrafo de sobrecarga |
| Conclusão | a ressalva de não dominância da escada |
| Figura 3 | as três séries de preset nativo e a faixa sombreada de vão não coberto |

Permanece uma única menção, de duas sentenças, no primeiro parágrafo de comparação da
Seção VI: a declaração de escopo que justifica a ausência da comparação, sem número algum
de preset. Ela existe para antecipar a pergunta do revisor, e não para comparar.

Permanecem legítimas, por não serem comparação de desempenho, as menções a `cpu-used=0`
(âncora e regime de extração do conjunto de dados) e a `cpu-used=6` (verificação de que
um preset rápido já poda heuristicamente e por isso não serve para rotular).

### 6.2 O que entrou no lugar

O eixo de comparação passou a ser interno, e o argumento central do artigo passou a ser a
**taxa de câmbio** entre tempo e taxa BD das duas alavancas que a solução implantada
oferece ao seu usuário, ambas medidas na mesma grade CTC A1 e visíveis na Figura 3:

| Alavanca | Segmento medido | Preço (pp de taxa BD por pp de tempo) |
|---|---|--:|
| botão de limiar do 1º estágio | `ml_balanced` → `ml_aggr` | 0,0606 |
| 2º estágio, calibração implantada | `ml_balanced` → `ml_bal_h9d` | 0,0179 |
| 2º estágio, calibração forte | `ml_balanced` → `ml_bal_h9d_pl20` | 0,0399 |

A razão publicada é, por conseguinte, **3,4×** (0,0606 / 0,0179 = 3,38), e não os 3,5× da
versão anterior. Os dois valores são o mesmo resultado sob estimadores distintos, já
registrados em `results/thesis/R4_h9d.md` §4.6: 3,5× pelo estimador de interpolação por
sequência (0,063) e 3,38× pela média das médias (0,0606). O artigo passou a usar **apenas
o estimador de média das médias**, que é o único derivável dos pontos que a própria
Figura 3 plota, e o script da figura recalcula os três preços a cada execução e os imprime
para auditoria. O estimador de interpolação por sequência continua sendo a fonte do teste
de não dominância sequência a sequência (6 de 8, duas por dominância de Pareto estrita),
que é contagem e não preço, e os dois não se misturam em tabela alguma.

Comando de reprodução (contêiner `av1_bench`):

```bash
build/venv-ml/bin/python src/scripts/benchmark/plot_operating_space_fig3.py \
    --out-dir results/thesis/figuras
```

### 6.3 Consequências para o restante do texto

O enunciado corrigido da Conclusão 3 da tese — dois podadores se somam na medida em que os
conjuntos de candidatos que retiram são disjuntos, no ponto de operação em que efetivamente
rodam (`R6_analise_integrada.md` §6.4) — passou a ser a espinha argumentativa do artigo, na
Seção V (regra do empilhamento) e na Seção VI (a inércia do 2º estágio sobre a base
agressiva, +0,17 pp, abaixo da resolução de ~0,46 pp). A prova por sobreposição entre o
H9a e o podador binário **não** entra, por pertencer ao artigo irmão.

### 6.4 Limitações deste adendo

1. A retirada é editorial e não altera medição alguma: nenhum número foi recalculado, com a
   exceção declarada dos três preços da tabela acima, que saem do mesmo
   `bdrate_average.csv` já auditado na Seção 2.
2. A tese continua apresentando a fronteira global com os presets nativos, como manda o seu
   próprio escopo (`M1_objeto_e_formulacao.md` §1.5). A restrição vale para o artigo LASCAS.
