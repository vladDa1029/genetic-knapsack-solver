# ТЗ: режим `staged_hypermutation` для `classic` solver

## Статус

Этот документ описывает **планируемый** режим.

На момент написания режим **ещё не реализован в коде**. Документ фиксирует согласованное поведение до начала реализации.

## Назначение режима

Режим нужен для случаев, когда:

- обычный `classic` ГА долго стоит на плато;
- одноразового `NGA`-вмешательства недостаточно;
- полный `restart_rescue` слишком дорогой или слишком агрессивный.

Идея режима:

- внутри одного запуска `classic` ГА отслеживается стагнация;
- в нескольких заранее заданных точках стагнации популяция резко "встряхивается";
- лучшая особь сохраняется;
- остальные особи сильно мутируются;
- счётчик стагнации сбрасывается **только если после вмешательства появилось реальное улучшение**.

## Место режима в проекте

Это **не новый `solver_mode`**.

Режим должен быть встроен в существующий:

- `solver_mode = classic`
- `nga_mode = staged_hypermutation`

Старые режимы:

- `none`
- `two_point`
- `elite_heavy_mutation`

должны остаться без изменения логики.

## Входные данные

Алгоритм получает:

- `prices`
- `target_sum`
- обычные параметры `classic` ГА:
  - `population_size`
  - `generations`
  - `stagnation`
  - `crossover_rate`
  - `mutation_rate`
  - `tournament_size`
- новые параметры режима:
  - `nga_mode = staged_hypermutation`
  - `nga_trigger_points: list[int]`
  - `nga_mutate_points: int | list[int]`
- `rng`

## Основная метрика прогресса

Прогресс определяется только по:

- `difference = abs(target_sum - current_sum)`

Правило улучшения:

- улучшение есть только если `new_difference < old_difference`
- если `new_difference == old_difference`, улучшения нет
- если `new_difference > old_difference`, улучшения нет

### Почему сравнение идёт по `difference`, а не по `fitness`

В текущем проекте `fitness` численно совпадает с `difference`.

Но для этого режима правило всё равно фиксируется через `difference`, потому что:

- это прямая целевая метрика;
- это сохраняет корректный смысл стагнации, даже если позже `fitness` станет более сложным;
- это согласуется с `restart_rescue`, где глобальный прогресс тоже лучше оценивать не по "внутреннему" fitness, а по близости к цели.

## Новые параметры режима

### `nga_trigger_points`

Список точек стагнации, в которых должен срабатывать режим.

Пример:

- `nga_trigger_points = [50, 100, 150, 200]`

Смысл:

- если счётчик стагнации достиг одного из этих значений, запускается staged hypermutation.

### `nga_mutate_points`

Проценты сильной мутации для разных точек стагнации.

Примеры:

- `40`
- `[40, 50, 60]`
- `[40, 40, 50, 60]`

Смысл:

- это пользовательские проценты, а не доли;
- внутри кода они должны преобразовываться в доли:
  - `40 -> 0.40`
  - `50 -> 0.50`
  - `60 -> 0.60`

## Правило сопоставления `nga_trigger_points` и `nga_mutate_points`

Пусть:

- `trigger_points = [t1, t2, ..., tk]`

Тогда разрешены два варианта задания мутаций:

### Вариант 1. Один процент на все точки

- `nga_mutate_points = 40`

Тогда для всех trigger-point используется одна и та же сила мутации:

- `[40, 40, ..., 40]`

### Вариант 2. Отдельный процент на каждую точку

- `nga_mutate_points = [40, 50, 60]`

Тогда длина списка должна совпадать с длиной `nga_trigger_points`.

Пример:

- `nga_trigger_points = [50, 100, 150]`
- `nga_mutate_points = [40, 50, 60]`

Тогда:

- при `50` мутируем `40%`
- при `100` мутируем `50%`
- при `150` мутируем `60%`

### Невалидные случаи

Конфигурация должна считаться невалидной, если:

- `nga_trigger_points` пустой
- в `nga_trigger_points` есть значения `< 1`
- в `nga_trigger_points` есть значения `>= stagnation`
- `nga_trigger_points` не отсортирован строго по возрастанию
- длина списка `nga_mutate_points` не равна `1` и не равна длине `nga_trigger_points`
- в `nga_mutate_points` есть значения вне диапазона `0..100`

## Важное согласованное правило

Во всех `NGA`-режимах после вмешательства мы должны сразу проверять:

- появилось ли улучшение;
- нужно ли сбрасывать счётчик стагнации.

Это правило уже логически ожидается и для старых режимов, и для нового режима должно быть таким же.

## Состояние алгоритма

Внутри запуска нужно хранить:

- `best_vector`
- `best_sum`
- `best_difference`
- `stagnation_counter`
- `nga_trigger_generations: list[int]`

### Что не нужно хранить

Отдельный `fired_triggers` в базовом варианте не нужен.

Причина:

- `stagnation_counter` растёт по одному;
- конкретная точка стагнации внутри одного плато может встретиться только один раз;
- после улучшения счётчик всё равно сбрасывается на `0`.

Если позже потребуется периодическое повторное срабатывание на одной и той же точке без сброса, это уже будет другой режим.

## Основной цикл обновления

После построения нового поколения и оценки его лучшей особи:

```text
if current_best_difference < best_difference:
    best_vector = current_best_vector
    best_sum = current_best_sum
    best_difference = current_best_difference
    stagnation_counter = 0
else:
    stagnation_counter += 1
```

Это правило обязательно.

## Условие срабатывания staged hypermutation

Если после обновления:

```text
stagnation_counter in nga_trigger_points
```

то запускается staged hypermutation.

### Повторное достижение той же точки стагнации

Если после какого-то вмешательства позже произошло улучшение, счётчик сбрасывается в `0`.

После этого при новом выходе на ту же точку стагнации:

- staged hypermutation должен сработать снова.

Пример:

- `nga_trigger_points = [100]`
- дошли до `100` -> режим сработал
- позже появилось улучшение -> счётчик сбросился
- снова дошли до `100` -> режим снова сработал

Это согласованное поведение.

## Действие staged hypermutation

При срабатывании очередной точки:

1. Сохраняется лучшая особь без изменений.
2. Формируется новая популяция той же длины.
3. Первая особь новой популяции:
   - точная копия `best_vector`
4. Все остальные особи:
   - берутся из текущей популяции;
   - подвергаются сильной мутации с процентом, соответствующим текущей trigger-point.

## Правило сильной мутации

Для каждой особи, кроме сохранённой лучшей:

1. Пусть длина хромосомы равна `n`.
2. Берётся текущий процент мутации `p` из `nga_mutate_points`.
3. Переводим его в долю:
   - `ratio = p / 100`
4. Вычисляем:
   - `k = ceil(ratio * n)`
5. Выбираем `k` различных индексов без повторов.
6. Для каждого выбранного индекса инвертируем бит:
   - `0 -> 1`
   - `1 -> 0`

Итоговая формула:

- `k = min(n, max(1, ceil((p / 100) * n)))`

## Обязательная немедленная переоценка после staged hypermutation

После формирования новой популяции staged hypermutation:

- популяция должна быть немедленно переоценена;
- должен быть найден её новый лучший кандидат;
- этот кандидат должен быть сравнён с текущим `best_difference`.

### Если после staged hypermutation есть улучшение

Если:

- `new_best_difference < best_difference`

то:

- обновляется лучший результат;
- `stagnation_counter = 0`

### Если после staged hypermutation улучшения нет

Если:

- `new_best_difference >= best_difference`

то:

- лучший результат не меняется;
- `stagnation_counter` не сбрасывается.

Это согласованное правило.

## Точное совпадение после staged hypermutation

Если после немедленной переоценки staged hypermutation:

- `best_difference == 0`

то алгоритм должен немедленно завершиться с:

- `stop_reason = exact_match`

## Критерий остановки

После возможного staged hypermutation остаётся обычное правило:

```text
if stagnation_counter >= stagnation_limit:
    stop
```

Причина остановки:

- `stagnation`

То есть staged hypermutation:

- не отменяет обычную стагнацию;
- не превращает режим в бесконечный поиск;
- не заменяет `restart_rescue`.

## Полная схема алгоритма

```text
initialize population
evaluate population
set best_*
stagnation_counter = 0
nga_trigger_generations = []

for generation in 1..generations:
    build next population using usual GA operators
    evaluate population
    find current best

    if current_best_difference < best_difference:
        update global best
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    if best_difference == 0:
        stop with exact_match

    if stagnation_counter in nga_trigger_points:
        apply staged hypermutation with corresponding mutation percent
        evaluate mutated population immediately
        append generation to nga_trigger_generations

        if post_nga_best_difference < best_difference:
            update global best
            stagnation_counter = 0

        if best_difference == 0:
            stop with exact_match

    if stagnation_counter >= stagnation_limit:
        stop with stagnation

stop with generation_limit
```

## Новые поля конфигурации

При внедрении режима в код должны появиться:

- `nga_mode = "staged_hypermutation"`
- `nga_trigger_points: list[int]`
- `nga_mutate_points: int | list[int]`

## Новые поля результата

Для этого режима в результат нужно добавить:

- `nga_trigger_generations: list[int]`

При этом:

- `nga_used = True`, если хотя бы одно staged hypermutation реально сработало

Текущее поле:

- `nga_trigger_generation`

для этого режима уже недостаточно и должно быть либо заменено, либо оставлено только для обратной совместимости.

## Практический стартовый пресет

Для первых экспериментов:

- `population_size = 5000`
- `stagnation = 250`
- `nga_trigger_points = [50, 100, 150, 200]`
- `nga_mutate_points = 40`
- `mutation_rate = 0.9`
- `crossover_rate = 0.95`

Более агрессивный вариант:

- `nga_trigger_points = [50, 100, 150]`
- `nga_mutate_points = [40, 50, 60]`

## Будущие улучшения

Пока в режим не включается:

- `elite_keep_count`
- список `elite_keep_count` по trigger-point

Но это нужно учитывать как следующее возможное расширение.

Идея будущего улучшения:

- вместо сохранения только одной лучшей особи сохранять несколько лучших;
- либо задавать это числом:
  - `elite_keep_count = 3`
- либо задавать списком по trigger-point:
  - `[1, 2, 3]`

Сейчас это **не входит** в первую реализацию.

## Краткий смысл режима

`staged_hypermutation` делает следующее:

- работает как обычный `classic` ГА;
- отслеживает стагнацию по `difference`;
- в нескольких заранее заданных точках резко увеличивает разнообразие популяции;
- после каждого вмешательства сразу проверяет, было ли реальное улучшение;
- сбрасывает стагнацию только при реальном прогрессе;
- если прогресса так и нет, завершает поиск по обычному лимиту стагнации.
