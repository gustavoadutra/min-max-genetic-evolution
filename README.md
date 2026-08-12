# Min-Max Genetic Evolution

Uma implementação em Python de um agente de Connect‑4 (tabuleiro 5x5) que combina um agente Min‑Max com uma evolução de pesos via Algoritmo Genético. O projeto permite evoluir parâmetros da função de avaliação do agente e comparar o agente evoluído com um agente baseline.

Principais arquivos
- `main.py` — ponto de entrada interativo. Gera/usa os pesos evoluídos, executa a evolução e oferece um menu para jogar ou comparar agentes.
- `genetic_algorithm.py` — implementação do algoritmo genético (inicialização, seleção por torneio, crossover, mutação, avaliação de fitness, evolução e plotagem de histórico).
- `minmax_agent.py` — agente Min‑Max com poda alfa‑beta e função de avaliação baseada em três features: open_lines, three_in_a_row, center_control.
- `connect4game.py` — implementação do jogo Connect‑4 em um tabuleiro 5x5, incluindo checagem de vitória, clonagem de estado e utilitários de impressão.
- `requirements.txt` — dependências necessárias para rodar (principalmente numpy e matplotlib).

Stack
- Linguagem: Python 3.8+
- Bibliotecas notáveis: numpy, matplotlib

Instalação rápida
1. Clone o repositório:

   git clone https://github.com/gustavoadutra/min-max-genetic-evolution.git
   cd min-max-genetic-evolution

2. (Opcional) Crie e ative um ambiente virtual:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\\.venv\\Scripts\\activate  # Windows

3. Instale dependências:

   pip install -r requirements.txt

Executando
- Executar o menu interativo (ponto de entrada):

  python main.py

  O script fará o seguinte fluxo (resumido):
  - Se existir `best_weights.pkl`, pergunta se deseja rodar uma nova evolução. Caso contrário, executa a evolução com os parâmetros definidos em `main.py`.
  - Salva os melhores pesos em `best_weights.pkl`.
  - Plota o histórico de fitness em `fitness_evolution.png`.
  - Disponibiliza um menu para:
    1) Jogar contra o agente evoluído
    2) Jogar contra o agente baseline (pesos padrão)
    3) Ver um jogo entre agentes (exibindo tabuleiro)
    4) Comparar agentes (jogar 100 partidas por padrão)
    5) Gerar um heatmap de avaliação de jogadas (`move_heatmap.png`)
    6) Sair

Parâmetros do Algoritmo Genético
Os valores padrão definidos em `main.py` (e usados por `GeneticAlgorithm`) incluem:
- population_size (tamanho da população): 20
- num_generations (número de gerações): 20
- tournament_size (tamanho do torneio): 3
- crossover_rate: 0.8
- mutation_rate: 0.2
- mutation_scale: 0.5

Como personalizar
- Edite os parâmetros no dicionário `ga_params` em `main.py` ou inicialize `GeneticAlgorithm(...)` com os valores desejados num script separado.
- Se você quiser executar apenas a evolução sem o menu interativo, importe `GeneticAlgorithm` e chame `ga = GeneticAlgorithm(...); best = ga.run()` em um script novo.

Arquivos gerados
- `best_weights.pkl` — pickle com os pesos encontrados pela evolução.
- `fitness_evolution.png` — plot do histórico de fitness por geração.
- `move_heatmap.png` — heatmap de avaliação de jogadas para um estado de tabuleiro.

Dependências (extraídas de requirements.txt)
- contourpy==1.3.2
- cycler==0.12.1
- fonttools==4.57.0
- kiwisolver==1.4.8
- matplotlib==3.10.1
- numpy==2.2.5
- packaging==25.0
- pillow==11.2.1
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- six==1.17.0

Notas de execução
- O projeto usa processamento paralelo (ProcessPoolExecutor) para avaliar fitness de indivíduos em `genetic_algorithm.py`. Em alguns ambientes (Windows, ambientes restritos) isso pode requerer ajustes ou causar problemas se executado em notebooks; prefira rodar `python main.py` em um terminal.
- A função de avaliação e o jogo assumem um tabuleiro de tamanho 5x5 e objetivo de conectar 4 peças em linha.

Possíveis melhorias (sugestões)
- Tornar os parâmetros configuráveis por linha de comando (argparse) ou por arquivo de configuração JSON/YAML.
- Adicionar testes automatizados em `tests/` (unitários para contagem de features, lógica de vitória e operadores genéticos).
- Separar a lógica de experimentos em scripts dentro de `examples/` para replicabilidade.
- Adicionar CI (GitHub Actions) que execute linting e testes.

Contribuição
1. Faça fork do repositório.
2. Crie uma branch com sua feature: `git checkout -b feature/descricao`.
3. Abra um pull request descrevendo as mudanças e os motivos.

Licença
- Nenhuma licença foi adicionada ao repositório. Se desejar, posso criar um arquivo `LICENSE` (ex.: MIT). Indique qual licença prefere.

Contatos
- Autor: Gustavo Adutra (GitHub: @gustavoadutra)

---

Se quiser, atualizo o README com instruções de execução não‑interativas (ex.: comandos para rodar evolução em lote, executar comparação em N jogos) ou adiciono exemplos prontos em `examples/`. Também posso criar o arquivo LICENSE se você indicar a licença preferida.