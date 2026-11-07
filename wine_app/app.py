from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title='Wine Quality | Рекомендации вин', page_icon='🍷', layout='wide')

@st.cache_resource(show_spinner='Загружаю модель...')
def load_model():
    return joblib.load(ROOT / 'model.joblib')

@st.cache_data(show_spinner=False)
def load_test_data():
    return pd.read_csv(ROOT / 'test.csv')

@st.cache_data(show_spinner=False)
def load_config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))

@st.cache_data(show_spinner=False)
def get_test_probabilities(_model, test_features):
    return _model.predict_proba(test_features)[:, 1]

model = load_model()
test = load_test_data()
config = load_config()
features = [name for name in test.columns if name != 'target']
y_test = test['target'].to_numpy().astype(int)
X_test = test.loc[:, features]
probas = get_test_probabilities(model, X_test)
threshold_default = float(config['threshold'])

st.sidebar.title('🍷 Wine Quality')
st.sidebar.write('Порог определяет, насколько уверенной должна быть модель, чтобы рекомендовать вино.')
threshold = st.sidebar.slider('Порог рекомендации', min_value=0.0, max_value=1.0, value=threshold_default, step=0.0005, key='decision_threshold')
st.sidebar.caption(f'Порог из валидации: {threshold_default:.4f}. При снижении порога рекомендаций больше; при повышении — меньше.')

st.title('🍷 Подбор вин по лабораторным характеристикам')
st.write('Модель оценивает вероятность того, что экспертная оценка вина будет не ниже 7 из 10. Решение обновляется при изменении порога, без повторного обучения модели.')

tab_one, tab_metrics, tab_batch, tab_coeffs = st.tabs(['Оценка одного вина', 'Модельные метрики', 'Пакетный прогноз', 'Что выучила модель'])

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь, %',
    'is_red': 'Тип вина'
}

with tab_one:
    st.subheader('Лабораторный профиль вина')
    inputs = {}
    cols = st.columns(3)
    for index, feature in enumerate(features):
        with cols[index % 3]:
            if feature == 'is_red':
                inputs[feature] = st.radio('Тип вина', [0, 1], index=int(round(X_test[feature].median())), format_func=lambda value: 'Красное' if value == 1 else 'Белое', horizontal=True)
            else:
                low = float(X_test[feature].min())
                high = float(X_test[feature].max())
                default = float(X_test[feature].median())
                increment = max((high - low) / 1000, 1e-5)
                inputs[feature] = st.number_input(labels.get(feature, feature), min_value=low, max_value=high, value=default, step=increment, format='%.5f', key='single_' + feature)
    single = pd.DataFrame([inputs], columns=features)
    probability = float(model.predict_proba(single)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric('Вероятность хорошего вина', f'{probability:.1%}')
    c2.metric('Текущий порог', f'{threshold:.1%}')
    if probability >= threshold:
        st.success('Рекомендовать: модель считает это вино хорошим при выбранном пороге.')
    else:
        st.warning('Не рекомендовать: оценка модели ниже выбранного порога.')

with tab_metrics:
    st.subheader('Качество модели на отложенной тестовой выборке')
    predictions = (probas >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    roc_auc = metrics.roc_auc_score(y_test, probas)
    average_precision = metrics.average_precision_score(y_test, probas)
    for column, (label, value) in zip(st.columns(5), [('Precision', precision), ('Recall', recall), ('F1', f1), ('ROC-AUC', roc_auc), ('PR-AUC (AP)', average_precision)]):
        column.metric(label, f'{value:.3f}')
    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Не хорошее', 'Хорошее']).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title('Матрица ошибок')
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with right:
        st.write('**Что означает порог**')
        st.write('Увеличение порога обычно уменьшает число рекомендованных вин и повышает осторожность отбора, но может увеличить число пропущенных хороших вин. Снижение порога обычно повышает полноту и расширяет подборку, но увеличивает риск рекомендовать посредственное вино.')
        st.write(f'В тесте: **{int(predictions.sum())}** рекомендаций из **{len(predictions)}** вин. Верных рекомендаций: **{int(cm[1, 1])}**. Пропущенных хороших вин: **{int(cm[1, 0])}**.')
    fpr, tpr, _ = metrics.roc_curve(y_test, probas)
    p_curve, r_curve, _ = metrics.precision_recall_curve(y_test, probas)
    fp = int(cm[0, 1])
    tn = int(cm[0, 0])
    roc_point_x = fp / (fp + tn) if fp + tn else 0.0
    curve1, curve2 = st.columns(2)
    with curve1:
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], ls='--', color='gray')
        ax.scatter([roc_point_x], [recall], color='darkred', s=75, zorder=5, label=f'Порог {threshold:.4f}')
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC-кривая')
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with curve2:
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        ax.plot(r_curve, p_curve, label=f'PR-AUC (AP) = {average_precision:.3f}')
        ax.axhline(y_test.mean(), ls='--', color='gray', label='Случайный отбор')
        ax.scatter([recall], [precision], color='darkred', s=75, zorder=5, label=f'Порог {threshold:.4f}')
        ax.set(xlabel='Recall', ylabel='Precision', title='Precision–Recall-кривая')
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

with tab_batch:
    st.subheader('Прогноз для списка вин')
    st.write('Загрузите CSV с 12 признаками и без обязательного столбца `target`. Названия столбцов должны совпадать с названиями обучающих признаков.')
    sample_csv = X_test.head(5).to_csv(index=False).encode('utf-8')
    st.download_button('Скачать образец CSV', sample_csv, file_name='wine_example.csv', mime='text/csv')
    uploaded_file = st.file_uploader('Выберите CSV-файл', type=['csv'])
    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)
            missing = [name for name in features if name not in batch.columns]
            if missing:
                st.error('В CSV отсутствуют необходимые столбцы: ' + ', '.join(missing))
            elif batch.empty:
                st.error('CSV не содержит строк с данными.')
            else:
                batch_features = batch.loc[:, features].apply(pd.to_numeric, errors='coerce')
                bad = batch_features.isna().any(axis=1)
                if bool(bad.any()) or not np.isfinite(batch_features.to_numpy(dtype=float)).all():
                    st.error(f'В файле есть пропуски или нечисловые значения в признаках. Строк с пропусками: {int(bad.sum())}.')
                else:
                    batch_probas = model.predict_proba(batch_features)[:, 1]
                    result = batch.copy()
                    result['probability_good'] = batch_probas
                    result['recommend'] = (batch_probas >= threshold).astype(int)
                    st.success(f'Обработано {len(result)} вин; рекомендовано {int(result["recommend"].sum())}.')
                    st.dataframe(result, use_container_width=True)
                    st.download_button('Скачать результаты CSV', data=result.to_csv(index=False).encode('utf-8-sig'), file_name='wine_predictions.csv', mime='text/csv')
        except (pd.errors.ParserError, UnicodeError, ValueError) as error:
            st.error('Не удалось прочитать или обработать CSV: ' + str(error))

with tab_coeffs:
    st.subheader('Влияние стандартизованных признаков')
    regression = model.named_steps['logisticregression']
    coef = pd.Series(regression.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(coef.values >= 0, '#267a5e', '#a34848')
    ax.barh([labels.get(feature, feature) for feature in coef.index], coef.values, color=colors)
    ax.axvline(0.0, color='black', linewidth=1)
    ax.set(xlabel='Коэффициент при изменении признака на 1 стандартное отклонение', title='Коэффициенты логистической регрессии')
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.write(f'**Сильнее за хорошее вино:** {", ".join(labels.get(name, name) for name in coef.nlargest(3).index)}.')
    st.write(f'**Сильнее против:** {", ".join(labels.get(name, name) for name in coef.nsmallest(3).index)}.')
    st.caption('Коэффициенты отражают связь признаков с прогнозом модели после стандартизации, но не доказывают причинно-следственное влияние характеристик на качество вина.')
