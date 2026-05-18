# genetic-knapsack-solver

Учебный проект по решению задачи `subset sum` генетическим алгоритмом.

Программа:

- генерирует список предметов, цены, скрытый бинарный вектор и целевую сумму;
- поддерживает генерацию задач в режимах `random` и `superincreasing_disguised`;
- ищет бинарный вектор, сумма которого равна `target_sum` или максимально близка к нему;
- поддерживает несколько solver-режимов: `classic`, `restart_rescue`, `two_stage_restart`, `five_stage_restart`;
- поддерживает NGA-вмешательства внутри `classic`: `none`, `two_point`, `elite_heavy_mutation`, `staged_hypermutation`;
- умеет запускать benchmark-матрицы и сохранять промежуточное состояние для `--resume`.

## Запуск через PyPy и uv

Основной способ запуска в примерах - через `uv` с выбранным интерпретатором PyPy. Так команды не зависят от прямого пути к `.venv\Scripts\pypy.exe`.

Базовый запуск:

```powershell
uv run --python pypy3.11 python main.py
```

Если PyPy-окружение нужно пересобрать или зафиксировать для проекта:

```powershell
uv python pin pypy3.11
uv sync --python pypy3.11
uv run --python pypy3.11 python main.py
```

Дальше в примерах используется этот шаблон:

```powershell
uv run --python pypy3.11 python main.py ...
uv run --python pypy3.11 python benchmark.py ...
```

## Режимы генерации задачи

`--generation-mode random`

Генерирует случайные уникальные цены, скрытый бинарный вектор и целевую сумму.

`--generation-mode superincreasing_disguised`

Генерирует более сложную задачу через замаскированную суперпоследовательность. Это основной режим для benchmark.

## Solver-Режимы

### `five_stage_restart`

Новый многоэтапный solver с пятью прогрессивными стадиями возмущения.

Логика:

- Стадии 1–2: детерминированные структурные операции (`split_reverse`, `swap_halves`), сохраняющие лучшую особь.
- Стадии 3–5: случайная мутация с нарастающей силой:
  - Стадия 3: стандартная `restart_mutation_fraction` (по умолчанию 0.4);
  - Стадия 4: `--multistage-stage4-fraction` (по умолчанию 0.60);
  - Стадия 5: `--multistage-stage5-fraction` (по умолчанию 0.80).
- Использует настраиваемые кроссовер, мутацию и схему потомков (`four_children_select_two`).
- Поддерживает перенос нескольких элит через `--multistage-elite-count`.

Параметры:

- `--multistage-crossover-type one_point|two_point` - тип кроссовера.
- `--multistage-mutation-type one_point|two_point` - тип мутации.
- `--multistage-offspring-mode two_children|four_children_select_two` - схема потомков.
- `--multistage-elite-count 3` - количество элитных особей, переносимых между стадиями.
- `--multistage-stage4-fraction 0.50` - сила мутации на стадии 4.
- `--multistage-stage5-fraction 0.70` - сила мутации на стадии 5.

Лучшая конфигурация для `n=28, p=5000` (93.94% точных решений):

```powershell
uv run --python pypy3.11 python main.py --solver-mode five_stage_restart --items 28 --population-size 5000 --generations 500000 --stagnation 5000 --mutation-rate 0.9 --crossover-rate 0.95 --multistage-crossover-type two_point --multistage-mutation-type two_point --multistage-offspring-mode four_children_select_two --multistage-elite-count 3 --multistage-stage4-fraction 0.50 --multistage-stage5-fraction 0.70 --seed 42
```

### `classic`

Базовый solver. Запускает один проход ГА.

Операторы по умолчанию:

- одноточечный кроссовер;
- одноточечная мутация;
- турнирная селекция;
- перенос элиты в новое поколение.

В этом solver доступны `NGA`-режимы через `--nga-mode`.

Пример обычного ГА без NGA:

```powershell
uv run --python pypy3.11 python main.py --solver-mode classic --nga-mode none --items 12 --population-size 100 --generations 200 --stagnation 30 --seed 42
```

### `classic + two_point`

Одноразовое NGA-вмешательство. Когда число поколений без улучшения достигает `--repeat-limit`, строится специальное поколение с:

- двухточечным кроссовером;
- двухточечной мутацией.

После этого solver возвращается к обычному `classic`.

Пример:

```powershell
uv run --python pypy3.11 python main.py --solver-mode classic --nga-mode two_point --repeat-limit 30 --items 12 --population-size 100 --generations 200 --stagnation 30 --seed 42
```

### `classic + elite_heavy_mutation`

Одноразовое NGA-вмешательство. При достижении `--repeat-limit`:

- лучшая особь сохраняется без изменений;
- остальные особи сильно мутируются;
- сила мутации задаётся через `--nga-mutation-fraction`.

Пример:

```powershell
uv run --python pypy3.11 python main.py --solver-mode classic --nga-mode elite_heavy_mutation --repeat-limit 30 --nga-mutation-fraction 0.4 --items 12 --population-size 100 --generations 200 --stagnation 30 --seed 42
```

### `classic + staged_hypermutation`

Многошаговое NGA-вмешательство внутри одного запуска `classic`.

Срабатывает в точках стагнации из `--nga-trigger-points`. В каждой точке:

- лучшая особь сохраняется;
- остальные особи сильно мутируются;
- сила мутации задаётся через `--nga-mutate-points`.

Для этого режима `--repeat-limit` не используется. Все `nga_trigger_points` должны быть меньше `--stagnation`.

Пример:

```powershell
uv run --python pypy3.11 python main.py --solver-mode classic --nga-mode staged_hypermutation --nga-trigger-points 50,100,150,200 --nga-mutate-points 40 --items 26 --population-size 3000 --generations 500000 --stagnation 3000 --mutation-rate 0.9 --crossover-rate 0.95 --seed 42
```

Пример с разной силой мутации по точкам:

```powershell
uv run --python pypy3.11 python main.py --solver-mode classic --nga-mode staged_hypermutation --nga-trigger-points 50,100,150 --nga-mutate-points 40,50,60 --items 26 --population-size 3000 --generations 500000 --stagnation 3000
```

### `restart_rescue`

Отдельный solver-режим с полными рестартами и rescue-этапом.

Логика:

- запускается обычный ГА;
- при стагнации запускаются новые полные рестарты;
- пока новый рестарт улучшает `difference`, solver продолжает рестарты;
- при первом неулучшающем рестарте выполняется один rescue-этап;
- если rescue не помогает, возвращается лучший накопленный результат.

Для этого режима `--nga-mode` должен оставаться `none`, а `--repeat-limit` не используется.

Пример:

```powershell
uv run --python pypy3.11 python main.py --solver-mode restart_rescue --items 26 --population-size 3000 --generations 500000 --stagnation 3000 --mutation-rate 0.9 --crossover-rate 0.95 --restart-max-count 5 --rescue-min-mutated-bits-ratio 0.4 --seed 42
```

### `two_stage_restart`

Новый двухэтапный solver.

Этап 1:

- запускает ГА с одноточечным кроссовером и одноточечной мутацией;
- при стагнации сохраняет лучшую особь запуска;
- строит следующий запуск из финальной популяции прошлого запуска: элита переносится, остальные особи сильно мутируются;
- переходит к этапу 2, когда лучший `difference` текущего запуска равен лучшему `difference` предыдущего запуска.

Этап 2:

- стартует от последней популяции этапа 1;
- использует настраиваемый тип кроссовера и мутации;
- сравнивает `difference` только с предыдущим запуском этапа 2;
- завершается, когда на этапе 2 снова выполнено условие равенства `difference`.

Для этого режима:

- `--nga-mode` должен оставаться `none`;
- `--repeat-limit` не используется;
- `--generations` должен быть не меньше `--stagnation`;
- способ построения restart-популяции пока один: `elite_from_last_population`;
- сила restart-мутации задаётся через `--restart-mutation-fraction`;
- операторы второго этапа задаются через `--stage2-crossover-type` и `--stage2-mutation-type`.

Пример (лучшая конфигурация с `four_children_select_two` на обоих этапах):

```powershell
uv run --python pypy3.11 python main.py --solver-mode two_stage_restart --items 28 --population-size 5000 --generations 500000 --stagnation 5000 --mutation-rate 0.9 --crossover-rate 0.95 --stage1-crossover-type two_point --stage1-mutation-type two_point --stage1-offspring-mode four_children_select_two --stage2-restart-fraction 0.60 --stage2-crossover-type two_point --stage2-mutation-type two_point --stage2-offspring-mode four_children_select_two --seed 42
```

## Параметры `main.py`

Общие параметры:

- `--items 12` - количество предметов в задаче.
- `--generation-mode random|superincreasing_disguised` - режим генерации задачи.
- `--solver-mode classic|restart_rescue|two_stage_restart|five_stage_restart` - общий solver-режим.
- `--population-size 100` - размер популяции.
- `--generations 200` - максимальное число поколений внутри одного запуска.
- `--stagnation 30` - лимит поколений без улучшения.
- `--crossover-rate 0.8` - вероятность кроссовера.
- `--mutation-rate 0.05` - вероятность обычной мутации.
- `--tournament-size 3` - размер турнира при селекции.
- `--seed 42` - seed для воспроизводимости.

Параметры `classic + NGA`:

- `--nga-mode none|two_point|elite_heavy_mutation|staged_hypermutation` - тип NGA-вмешательства.
- `--repeat-limit 30` - лимит без улучшения для одноразовых `two_point` и `elite_heavy_mutation`.
- `--nga-mutation-fraction 0.4` - доля битов для сильной мутации в `elite_heavy_mutation`.
- `--nga-trigger-points 50,100,150` - точки стагнации для `staged_hypermutation`.
- `--nga-mutate-points 40` или `--nga-mutate-points 40,50,60` - проценты сильной мутации для `staged_hypermutation`.

Параметры `restart_rescue`:

- `--restart-max-count 5` - максимальное число полных рестартов; если не задано, лимит не применяется.
- `--rescue-min-mutated-bits-ratio 0.4` - доля битов для rescue-мутации.

Параметры `two_stage_restart`:

- `--restart-population-mode elite_from_last_population` - стратегия построения новой популяции между запусками.
- `--restart-mutation-fraction 0.4` - доля битов для сильной мутации между запусками.
- `--stage1-crossover-type one_point|two_point` - тип кроссовера на первом этапе (по умолчанию `one_point`).
- `--stage1-mutation-type one_point|two_point|reverse` - тип мутации на первом этапе (по умолчанию `one_point`).
- `--stage1-offspring-mode two_children|four_children_select_two` - схема потомков первого этапа (по умолчанию `two_children`).
- `--stage2-crossover-type one_point|two_point` - тип кроссовера на втором этапе.
- `--stage2-mutation-type one_point|two_point|reverse` - тип мутации на втором этапе; `reverse` разворачивает вектор как `vector[::-1]`.
- `--stage2-restart-fraction 0.40` - доля мутируемых битов при переходе к этапу 2 (по умолчанию 0.40).

Параметры `five_stage_restart`:

- `--multistage-crossover-type one_point|two_point` - тип кроссовера (по умолчанию `one_point`).
- `--multistage-mutation-type one_point|two_point` - тип мутации (по умолчанию `one_point`).
- `--multistage-offspring-mode two_children|four_children_select_two` - схема потомков (по умолчанию `two_children`).
- `--multistage-elite-count 1` - число элитных особей (по умолчанию 1).
- `--multistage-stage4-fraction 0.60` - сила мутации на стадии 4 (по умолчанию 0.60).
- `--multistage-stage5-fraction 0.80` - сила мутации на стадии 5 (по умолчанию 0.80).

## Benchmark

Benchmark запускается через точку входа `genetic-knapsack-benchmark` и сохраняет три файла:

- `benchmark_state.json` - промежуточное состояние и данные для `--resume`;
- `results.csv` - все отдельные прогоны;
- `summary.md` - сводная таблица по группам.

Основной запуск через PyPy (рекомендуемый способ):

```powershell
uv run genetic-knapsack-benchmark
```

Или напрямую через `python main.py` для отдельных прогонов:

```powershell
uv run --python pypy3.11 python main.py
```

Benchmark-режимы задаются через `--algorithm-modes`.

Доступные значения:

- `none` - `classic` без NGA.
- `two_point` - `classic + two_point`.
- `elite_heavy_mutation` - `classic + elite_heavy_mutation`.
- `staged_hypermutation` - `classic + staged_hypermutation`.
- `restart_rescue` - solver `restart_rescue`.
- `two_stage_restart` - solver `two_stage_restart` (Rust core).
- `five_stage_restart` - solver `five_stage_restart` (Rust core).

Все реализованные solver-режимы можно запускать через benchmark: `classic` представлен режимами `none`, `two_point`, `elite_heavy_mutation`, `staged_hypermutation`, отдельные solver-режимы — `restart_rescue`, `two_stage_restart` и `five_stage_restart`.

## Параметры benchmark

- `--items 25,26` - список размерностей `n` через запятую.
- `--repeats 33` или `--tasks 33` - число независимых задач на каждую конфигурацию.
- `--generation-mode random|superincreasing_disguised` - режим генерации задач.
- `--generations 500000` - максимум поколений для одного запуска.
- `--mutation-rate 0.9` - вероятность обычной мутации.
- `--crossover-rate 0.95` - вероятность кроссовера.
- `--tournament-size 3` - размер турнира.
- `--population-stop-pairs 3000:3000,5000:5000` - пары `population_size:stagnation`.
- `--algorithm-modes none,two_point,elite_heavy_mutation,staged_hypermutation,restart_rescue,two_stage_restart,five_stage_restart` - список режимов.
- `--nga-mutation-fraction 0.4` - доля сильной мутации для `elite_heavy_mutation`.
- `--nga-trigger-points 50,100,150` - точки стагнации для `staged_hypermutation`.
- `--nga-mutate-points 40` или `--nga-mutate-points 40,50,60` - сила мутации для `staged_hypermutation`.
- `--restart-max-count 5` - максимальное число полных рестартов для `restart_rescue`; если не задано, лимит не применяется.
- `--rescue-min-mutated-bits-ratio 0.4` - минимальная доля мутируемых битов для rescue-этапа `restart_rescue`.
- `--restart-population-mode elite_from_last_population` - стратегия restart-популяции.
- `--restart-mutation-fraction 0.4` - сила restart-мутации.
- `--stage1-crossover-type one_point|two_point` - кроссовер первого этапа для `two_stage_restart`.
- `--stage1-mutation-type one_point|two_point|reverse` - мутация первого этапа для `two_stage_restart`.
- `--stage1-offspring-mode two_children|four_children_select_two` - схема потомков первого этапа для `two_stage_restart`.
- `--stage2-crossover-type one_point|two_point` - кроссовер второго этапа для `two_stage_restart`.
- `--stage2-mutation-type one_point|two_point|reverse` - мутация второго этапа для `two_stage_restart`; пример `reverse`: `[1,0,1,1,1,1,1,0,0,0] -> [0,0,0,1,1,1,1,1,0,1]`.
- `--stage2-restart-fraction 0.40` - доля мутируемых битов при переходе к этапу 2 для `two_stage_restart`.
- `--multistage-crossover-type one_point|two_point` - кроссовер для `five_stage_restart`.
- `--multistage-mutation-type one_point|two_point` - мутация для `five_stage_restart`.
- `--multistage-offspring-mode two_children|four_children_select_two` - схема потомков для `five_stage_restart`.
- `--multistage-elite-count 1` - число элитных особей для `five_stage_restart`.
- `--multistage-stage4-fraction 0.60` - сила мутации на стадии 4 для `five_stage_restart`.
- `--multistage-stage5-fraction 0.80` - сила мутации на стадии 5 для `five_stage_restart`.
- `--seed 42` - базовый seed.
- `--output-root benchmark_results` - корневая папка для автоматически создаваемых каталогов.
- `--output-dir benchmark_results\my_run` - явный каталог результата.
- `--resume` - продолжить прерванный benchmark из `--output-dir`.

## Benchmark-Примеры

Быстрый smoke benchmark по всем режимам:

```powershell
uv run genetic-knapsack-benchmark --repeats 1 --items 6 --generations 12 --mutation-rate 0.4 --crossover-rate 0.6 --population-stop-pairs 10:3 --algorithm-modes none,two_point,elite_heavy_mutation,staged_hypermutation,restart_rescue,two_stage_restart,five_stage_restart --nga-trigger-points 1 --nga-mutate-points 40 --restart-max-count 1 --output-dir benchmark_results\smoke_run
```

Benchmark лучшей конфигурации `five_stage_restart` для `n=28,29` (рекомендуется):

```powershell
uv run genetic-knapsack-benchmark --repeats 33 --items 28,29 --generation-mode superincreasing_disguised --generations 500000 --mutation-rate 0.9 --crossover-rate 0.95 --population-stop-pairs 3000:3000,5000:5000 --algorithm-modes five_stage_restart --multistage-crossover-type two_point --multistage-mutation-type two_point --multistage-offspring-mode four_children_select_two --multistage-elite-count 3 --multistage-stage4-fraction 0.50 --multistage-stage5-fraction 0.70 --output-dir benchmark_results\five_stage_best_n28_n29
```

Benchmark лучшей конфигурации `two_stage_restart` (stage1=2p/2p/4→2) для `n=28,29`:

```powershell
uv run genetic-knapsack-benchmark --repeats 33 --items 28,29 --generation-mode superincreasing_disguised --generations 500000 --mutation-rate 0.9 --crossover-rate 0.95 --population-stop-pairs 3000:3000,5000:5000 --algorithm-modes two_stage_restart --stage1-crossover-type two_point --stage1-mutation-type two_point --stage1-offspring-mode four_children_select_two --stage2-restart-fraction 0.60 --stage2-crossover-type two_point --stage2-mutation-type two_point --stage2-offspring-mode four_children_select_two --output-dir benchmark_results\two_stage_best_n28_n29
```

Benchmark только `restart_rescue`:

```powershell
uv run genetic-knapsack-benchmark --items 26,27,28 --population-stop-pairs 3000:3000,5000:5000 --algorithm-modes restart_rescue --restart-max-count 5 --rescue-min-mutated-bits-ratio 0.4 --output-dir benchmark_results\restart_rescue_n26_n28
```

Полная матрица всех benchmark-режимов:

```powershell
uv run genetic-knapsack-benchmark --repeats 33 --items 25,26 --generation-mode superincreasing_disguised --generations 500000 --mutation-rate 0.9 --crossover-rate 0.95 --population-stop-pairs 500:500,1000:1000,2000:2000,3000:3000,5000:5000 --algorithm-modes none,two_point,elite_heavy_mutation,staged_hypermutation,restart_rescue,two_stage_restart,five_stage_restart --nga-trigger-points 50,100,150 --nga-mutate-points 40 --restart-max-count 5
```

Продолжить прерванный прогон:

```powershell
uv run genetic-knapsack-benchmark --resume --output-dir benchmark_results\five_stage_best_n28_n29
```

## Тесты

Через CPython:

```powershell
uv run python -m pytest -q
```

Через PyPy:

```powershell
uv run --python pypy3.11 python -m unittest discover -s tests -q
```

Если в PyPy-окружении установлен `pytest`, можно запускать:

```powershell
uv run --python pypy3.11 python -m pytest -q
```

## Дополнительная документация

- [Итоговые тесты и анализ](benchmark_results/2026-04-21_filtered_analysis_pop2000plus.md) — сравнительная таблица всех методов для `n=25..29`, включая результаты `five_stage_restart` и `two_stage_restart` с настроенными параметрами.
