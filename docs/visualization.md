# Visualization

## Текущее состояние

Полноценный frontend UI ещё не реализован.

Вместо этого сейчас есть генератор статического HTML-отчёта:

- `scripts/render_visual_report.py`

## Что входит в отчёт

HTML-репорт содержит:

- карточки с метриками;
- график распределения target;
- график feature importance;
- график `actual vs predicted`;
- график top factors из explainability;
- sample prediction/explanation blocks;
- sample DSS recommendations;
- raw metadata по модели и схеме.

## Где сохраняется отчёт

По умолчанию:

- `storage/artifacts/<project-id>/visual-report.html`

Пример для текущего `demo` проекта:

- `storage/artifacts/demo/visual-report.html`

## Генерация

```bash
python scripts/render_visual_report.py \
  --project-id demo \
  --sample-size 40
```

## Почему sample-size важен

Explainability внутри отчёта использует SHAP.

Из-за этого:

- маленький `sample-size` собирается быстро;
- большой `sample-size` заметно замедляет генерацию;
- для повседневной проверки разумно использовать `40`;
- для более насыщенного отчёта можно использовать `120` и выше.

## Следующий шаг

Логичное развитие текущей визуализации:

1. перенести HTML-report в полноценный frontend;
2. сделать отдельные страницы для:
   - training result
   - prediction result
   - explanation result
   - DSS recommendation result
3. добавить интерактивные графики вместо статических PNG.
