# genetic-knapsack-solver

Учебный проект по решению задачи `subset sum` с помощью генетического алгоритма.

Программа:
- генерирует список предметов, цены, скрытый бинарный вектор и целевую сумму;
- поддерживает два режима генерации: `random` и `superincreasing_disguised`;
- ищет бинарный вектор, сумма которого совпадает с `target_sum` или максимально к нему близка;
- останавливается по точному совпадению, стагнации или лимиту поколений.

## Запуск через uv и PyPy

```powershell
uv python pin pypy3.11
uv sync --python pypy3.11
uv run --python pypy3.11 python main.py
```

Пример запуска с параметрами:

```powershell
uv run --python pypy3.11 python main.py --items 12 --generation-mode random --population-size 100 --generations 200 --stagnation 30 --seed 42
```

## Benchmark без NGA

Бенчмарк сейчас запускает только базовый ГА, без `nga`-логики. По умолчанию он использует генератор `superincreasing_disguised`.

Значения по умолчанию:
- `25` задач;
- `n = 25`;
- `generation_mode = superincreasing_disguised`;
- `mutation_rate = 0.9`;
- `crossover_rate = 0.95`;
- `population_stop_pairs = 500:500,1000:1000,2000:2000,3000:3000`;
- `generations = 500000`.

Запуск:

```powershell
uv run --python pypy3.11 python benchmark.py
```

Пример с явными параметрами:

```powershell
uv run --python pypy3.11 python benchmark.py --tasks 25 --items 25,26 --generation-mode superincreasing_disguised --generations 500000 --mutation-rate 0.9 --crossover-rate 0.95 --population-stop-pairs 3000:3000,5000:5000,6000:6000
```

После запуска создаётся папка `benchmark_results/YYYY-MM-DD_HH-MM-SS_mmmmmm`, внутри:
- `results.csv` с данными по каждому прогону;
- `summary.md` со средними метриками, режимом генерации, настройками запуска и точным временем выполнения.

## Тесты

```powershell
uv run --python pypy3.11 python -m unittest discover -s tests
```

## Что выводит программа

- режим генерации;
- список предметов и их цены;
- целевую сумму и скрытый бинарный вектор;
- лучшее найденное решение;
- найденную сумму;
- fitness и абсолютную разницу;
- число поколений;
- информацию о точном совпадении;
- причину остановки.
