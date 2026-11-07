# HW3 — Wine Quality

Автор: Шкундин Глеб. Домашняя работа №3: логистическая регрессия, метрики, дисбаланс классов и приложение Streamlit.

## Файлы

- `HW3_MMDA_Shkundin.ipynb` — выполненный ноутбук с анализом и обучением модели.
- `wine_quality.csv` — исходные данные.
- `requirements_notebook.txt` — зависимости ноутбука.
- `wine_app/` — приложение, обученная модель, конфигурация и тестовые данные.

## Запуск приложения

Используйте Python **3.13**. Из корня репозитория:

```bash
python -m pip install -r wine_app/requirements.txt
python -m streamlit run wine_app/app.py
```

Модель уже обучена; выполнять ноутбук перед запуском не требуется.

## Деплой на Streamlit Community Cloud

Откройте [Streamlit Community Cloud](https://share.streamlit.io/) и создайте приложение из GitHub со следующими параметрами:

| Параметр | Значение |
| --- | --- |
| Repository | `ShkundinG/HW3_MMAD` |
| Branch | `main` |
| Main file path | `wine_app/app.py` |
| Python version (Advanced settings) | `3.13` |

Нажмите **Deploy**. Streamlit установит зависимости из `wine_app/requirements.txt`; модель, конфигурация и CSV загружаются из папки приложения независимо от текущей рабочей директории.

Подробности о приложении: [wine_app/README.md](wine_app/README.md).
