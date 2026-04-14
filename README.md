# genetic-knapsack-solver

Учебный проект по решению задачи `subset sum` с помощью генетического алгоритма.

Программа:
- генерирует список предметов, цены, скрытый бинарный вектор и целевую сумму;
- поддерживает два режима генерации: `random` и `superincreasing_disguised`;
- ищет бинарный вектор, сумма которого совпадает с `target_sum` или максимально к нему близка;
- поддерживает три режима алгоритма: базовый ГА, `NGA-1` и `NGA-2`;
- умеет один раз вызывать `NGA` по лимиту повторов без улучшения и затем продолжать обычный ГА;
- останавливается по точному совпадению, стагнации или лимиту поколений.

## Запуск через uv и PyPy

```powershell
uv python pin pypy3.11
uv sync --python pypy3.11
uv run --python pypy3.11 python main.py
```

Пример запуска с параметрами:

```powershell
uv run --python pypy3.11 python main.py --items 12 --generation-mode random --population-size 100 --generations 200 --stagnation 30 --repeat-limit 30 --nga-mode two_point --seed 42
```

Режимы `nga`:
- `none` — базовый ГА без вмешательства;
- `two_point` — одноразовое поколение с двухточечным кроссовером и двухточечной мутацией;
- `elite_heavy_mutation` — сохраняется лучшая особь, остальные сильно мутируются по доле `--nga-mutation-fraction`.

## Benchmark

Бенчмарк сравнивает три режима: `none`, `two_point`, `elite_heavy_mutation`. По умолчанию он использует генератор `superincreasing_disguised`.

Значения по умолчанию:
- `33` независимые задачи на конфигурацию;
- `n = 25,26`;
- `generation_mode = superincreasing_disguised`;
- `mutation_rate = 0.9`;
- `crossover_rate = 0.95`;
- `population_stop_pairs = 500:500,1000:1000,2000:2000,3000:3000,5000:5000`;
- `algorithm_modes = none,two_point,elite_heavy_mutation`;
- `generations = 500000`.

Запуск:

```powershell
uv run --python pypy3.11 python benchmark.py
```

Пример с явными параметрами:

```powershell
uv run --python pypy3.11 python benchmark.py --repeats 33 --items 25,26 --generation-mode superincreasing_disguised --generations 500000 --mutation-rate 0.9 --crossover-rate 0.95 --population-stop-pairs 500:500,1000:1000,2000:2000,3000:3000,5000:5000 --algorithm-modes none,two_point,elite_heavy_mutation
```

После запуска создаётся папка `benchmark_results/YYYY-MM-DD_HH-MM-SS_mmmmmm`, внутри:
- `benchmark_state.json` с промежуточным состоянием для `--resume`;
- `results.csv` с данными по каждому прогону;
- `summary.md` со средними метриками, режимом генерации, режимом алгоритма, настройками запуска и точным временем выполнения.

Продолжение прерванного прогона:

```powershell
uv run --python pypy3.11 python benchmark.py --resume --output-dir benchmark_results\2026-04-13_12-00-00_000000
```

## Тесты

```powershell
uv run --python pypy3.11 python -m unittest discover -s tests
```

## Что выводит программа

- режим генерации;
- список предметов и их цены;
- целевую сумму и скрытый бинарный вектор;
- режим `NGA` и лимит его срабатывания;
- лучшее найденное решение;
- найденную сумму;
- fitness и абсолютную разницу;
- число поколений;
- информацию о точном совпадении;
- информацию о применении `NGA`;
- причину остановки.
