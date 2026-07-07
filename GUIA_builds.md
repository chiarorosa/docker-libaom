# Guia de Experimentos: Otimização da libaom (AV1)

Este documento descreve o fluxo de trabalho científico para o desenvolvimento, depuração e avaliação de desempenho de otimizações no codec de referência **AV1 (libaom v3.10.0)** utilizando um ambiente isolado via Docker.

A arquitetura foi projetada para blindar os experimentos contra contaminações, isolando o código original (**Baseline**) das suas modificações (**Experimento**).

## Docker diariamente:
```bash
cd C:\dev\av1-docker

docker compose run --rm research_env /bin/bash
```

---

## Mapa de Estrutura do Laboratório

* **`/workspace/src/aom_baseline`**: Código-fonte original e intocado da Alliance for Open Media (AOMedia).
* **`/workspace/src/aom`**: Código-fonte de trabalho onde você implementará suas otimizações (editável via Windows/VS Code).
* **`/workspace/build/libaom_perf_baseline`**: Executável purista de referência (`Release` + Assembly).
* **`/workspace/build/libaom_dev_generic`**: Executável de desenvolvimento (`RelWithDebInfo` + C Puro) para validação lógica.
* **`/workspace/build/libaom_perf`**: Executável otimizado com suas modificações (`Release` + Assembly) para coleta de dados.

---

## FASE 1: Preparação do Baseline (Executar Apenas Uma Vez)

Esta fase cria a "testemunha cega" do seu projeto. O binário gerado aqui servirá como o grupo de controle absoluto para os benchmarks da sua tese.

### 1. Clonar o Fonte Puro do Baseline
Clona a tag estável `v3.10.0` em um diretório isolado dedicado:
```bash
git clone --depth 1 --branch v3.10.0 [https://aomedia.googlesource.com/aom](https://aomedia.googlesource.com/aom) /workspace/src/aom_baseline

```

### 2. Configurar o Ambiente de Performance Purista

Prepara o build system utilizando o gerador **Ninja** focado em máxima otimização por hardware (`Release` e `NASM=ON`):

```bash
cmake -S /workspace/src/aom_baseline -B /workspace/build/libaom_perf_baseline \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CCACHE=1 \
  -DENABLE_NASM=ON \
  -DENABLE_EXAMPLES=ON \
  -DENABLE_TESTS=OFF \
  -DENABLE_DOCS=OFF \
  -DCONFIG_INTERNAL_STATS=1

```

### 3. Compilar o Baseline Absoluto

Compila focando estritamente no alvo do codificador, enviando os logs direto para a pasta compartilhada:

```bash
cmake --build /workspace/build/libaom_perf_baseline --target aomenc -j"$(nproc)" 2>&1 | tee /workspace/logs/build-libaom_perf_baseline.log

```

---

## FASE 2: Loop Diário de Desenvolvimento Lógico (Seu Fluxo de Trabalho)

Aqui você passará 90% do tempo. O objetivo é testar hipóteses, depurar lógica e garantir a conformidade do fluxo de vídeo.

> **Fluxo de Trabalho:** Você edita os arquivos em `C:\dev\av1-docker\src\aom` usando o VS Code no Windows e roda os comandos abaixo no terminal do container Linux. 
> não esqueça de criar uma nova branch para testar novas otimizações:
> ```bash
> git checkout -b nome-da-branch
> ```

### 1. Configurar o Ambiente Genérico (C Puro) (não repetir ao compilar)

Força o codec a rodar estritamente em **C puro** (`generic`), impedindo que o Assembly original ignore as suas modificações. Ativa também os símbolos de depuração (`RelWithDebInfo`):

```bash
cmake -S /workspace/src/aom -B /workspace/build/libaom_dev_generic \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DAOM_TARGET_CPU=generic \
  -DENABLE_CCACHE=1 \
  -DENABLE_EXAMPLES=ON \
  -DENABLE_TESTS=ON \
  -DENABLE_DOCS=OFF \
  -DCONFIG_INTERNAL_STATS=1

```

### 2. Compilação Incremental do Ecossistema de Dev (repetir ao modificar o código)

Compila todo o ecossistema (necessário para mapear os testes unitários da AOMedia). Devido ao **ccache** e ao **Ninja**, as re-compilações após suas alterações levarão apenas alguns segundos:

```bash
cmake --build /workspace/build/libaom_dev_generic -j"$(nproc)" 2>&1 | tee /workspace/logs/build-libaom_dev_generic.log

```

#### Comando Bônus: Testes Unitários Oficiais

Para garantir que suas alterações não quebraram as regras estruturais e matemáticas internacionais do AV1:

```bash
ninja -C /workspace/build/libaom_dev_generic test

```

### 3. Sanity Check: Teste Rápido com Vídeo Sintético

Garante que o codec gera um arquivo válido (`.ivf`) e exibe a tabela de PSNR/Métricas sem travar o terminal.

```bash
# A) Cria um arquivo YUV de teste rápido (10 frames pretos em baixa resolução)
dd if=/dev/zero bs=115200 count=10 of=/tmp/quick_test.yuv

# B) Executa o seu codificador modificado em modo rápido de 1 passo
/workspace/build/libaom_dev_generic/aomenc \
  -w 320 -h 240 \
  --fps=30/1 \
  --limit=10 \
  --passes=1 \
  --psnr \
  -o /workspace/results/test_dev.ivf \
  /tmp/quick_test.yuv

```

---

## FASE 3: Experimentos de Performance (Benchmarks Finais)

Utilize esta fase **apenas quando seu código estiver totalmente validado e livre de bugs lógicos na Fase 2**. Aqui, sua lógica em C será compilada em conjunto com as otimizações vetorizadas globais do codec para competir de forma justa contra o Baseline.

### 1. Configurar o Ambiente de Alta Performance com as SUAS Modificações

```bash
cmake -S /workspace/src/aom -B /workspace/build/libaom_perf \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CCACHE=1 \
  -DENABLE_NASM=ON \
  -DENABLE_EXAMPLES=ON \
  -DENABLE_TESTS=OFF \
  -DENABLE_DOCS=OFF \
  -DCONFIG_INTERNAL_STATS=1

```

### 2. Compilar o Seu Codificador Otimizado

Gera o executável final de alta performance que será submetido às sequências de teste pesadas (UHD/4K) da sua pesquisa:

```bash
cmake --build /workspace/build/libaom_perf --target aomenc -j"$(nproc)" 2>&1 | tee /workspace/logs/build-libaom_perf.log

```

---

## Próximos Passos (Coleta de Dados)

Ao final deste fluxo, você terá dois binários de produção idênticos em infraestrutura, diferindo apenas na sua inteligência algorítmica:

1. **Controle:** `/workspace/build/libaom_perf_baseline/aomenc`
2. **Seu Experimento:** `/workspace/build/libaom_perf/aomenc`

Ambos estão configurados com `-DCONFIG_INTERNAL_STATS=1`, o que significa que seus scripts de automação de testes em Python/Bash poderão ler as métricas de tempo, taxas de compressão (bitrate) e qualidade visual (PSNR/SSIM) diretamente impressas nos terminais e salvas nos arquivos de log.