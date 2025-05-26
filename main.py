import os
import pandas as pd
import numpy as np
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import hdbscan
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
import warnings
from itertools import combinations
from karateclub import Graph2Vec
import networkx as nx

os.makedirs('results', exist_ok=True)

warnings.filterwarnings('ignore')

# сохранение графиков
def save_figure(fig, filename):
    path = os.path.join('results', filename)
    fig.savefig(path)
    plt.close(fig)

# сохранение текста
def save_text(text, filename):
    path = os.path.join('results', filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

# подготовка данных
def load_and_prepare_data(file_path):
    log_csv = pd.read_csv(file_path, sep=',', engine='python')
    log_csv.columns = log_csv.columns.str.strip()

    if 'DateTime' not in log_csv.columns:
        raise ValueError(f"Столбец DateTime не найден. Доступные столбцы: {log_csv.columns.tolist()}")

    log_csv['DateTime'] = pd.to_datetime(log_csv['DateTime'])
    log_csv = log_csv.sort_values('DateTime')

    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'patient'}
    event_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    return event_log, log_csv

# векторизация
def vectorize_traces(event_log, method='bow'):
    traces = []
    for trace in event_log:
        trace_actions = [event['action'] for event in trace]
        traces.append(trace_actions)

    if method == 'bow':
        vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, binary=True)
        bow_matrix = vectorizer.fit_transform([' '.join(trace) for trace in traces])
        return bow_matrix.toarray(), traces

    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform([' '.join(trace) for trace in traces])
        return tfidf_matrix.toarray(), traces

    elif method == 'act2vec':
        model = Word2Vec(traces, vector_size=10, window=5, min_count=1, workers=4)
        act2vec_vectors = []
        for trace in traces:
            if len(trace) == 0:
                act2vec_vectors.append(np.zeros(10))
            else:
                vecs = [model.wv[action] for action in trace if action in model.wv]
                if len(vecs) > 0:
                    act2vec_vectors.append(np.mean(vecs, axis=0))
                else:
                    act2vec_vectors.append(np.zeros(10))
        return np.array(act2vec_vectors), traces


    elif method == 'graph2vec':

        all_actions = sorted({action for trace in traces for action in trace})

        action_to_id = {action: i for i, action in enumerate(all_actions)}

        graphs = []

        for trace in traces:

            G = nx.DiGraph()

            for i in range(len(trace)):
                G.add_node(i, label=action_to_id[trace[i]])  # индекс узла — позиция в трейсе, атрибут — id действия

            for i in range(len(trace) - 1):
                G.add_edge(i, i + 1)

            graphs.append(G)

        model = Graph2Vec(dimensions=20, wl_iterations=2)

        model.fit(graphs)

        graph2vec_vectors = model.get_embedding()

        return graph2vec_vectors, traces


    else:
        raise ValueError("Unknown vectorization method")


def cluster_traces(vectors, traces):
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2, cluster_selection_epsilon=0.7, cluster_selection_method='eom')
    return clusterer.fit_predict(scaled_vectors)

# визуализация и сохранение кластеров
def visualize_and_save_clusters(vectors, clusters, method_name):
    tsne = TSNE(n_components=2, random_state=42,perplexity = 5, learning_rate = 20, n_iter = 2000, early_exaggeration = 24)
    reduced_vectors = tsne.fit_transform(vectors)

    df_tsne = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
    df_tsne['cluster'] = clusters

    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='cluster', palette='viridis', s=100)
    plt.title(f'Кластеризация трасс пациентов ({method_name})')
    save_figure(fig, f'clusters_{method_name}.png')

# статистические тесты
def perform_statistical_tests(log_csv, clusters, method_name):
    unique_patients = log_csv['patient'].unique()
    patient_cluster = dict(zip(unique_patients, clusters))
    log_csv['cluster'] = log_csv['patient'].map(patient_cluster)

    # Рассчитываем продолжительность лечения для каждого пациента
    duration = log_csv.groupby('patient').apply(
        lambda x: (x['DateTime'].max() - x['DateTime'].min()).days
    ).reset_index(name='duration')
    duration['cluster'] = duration['patient'].map(patient_cluster)

    results = f"Результаты статистических тестов для метода {method_name}:\n\n"

    # 1. Chi-squared test для распределения ресурсов по кластерам
    if 'org:resource' in log_csv.columns:
        resource_cluster = pd.crosstab(log_csv['org:resource'], log_csv['cluster'])
        if not resource_cluster.empty and resource_cluster.shape[0] > 1 and resource_cluster.shape[1] > 1:
            chi2, p, dof, expected = chi2_contingency(resource_cluster)
            results += "Chi-squared test для распределения ресурсов по кластерам:\n"
            results += f"Chi2 = {chi2:.3f}, p-value = {p:.4f}\n"
            if p < 0.05:
                results += "Есть статистически значимые различия в использовании ресурсов между кластерами (p < 0.05)\n"
            else:
                results += "Нет статистически значимых различий в использовании ресурсов между кластерами (p >= 0.05)\n"
            results += "\n"
        else:
            results += "Недостаточно данных для выполнения chi-squared test по ресурсам\n\n"

    # 2. T-tests и ANOVA для продолжительности лечения
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) > 1 and 'duration' in duration.columns:
        results += "T-tests для продолжительности лечения между кластерами:\n"
        cluster_durations = [duration[duration['cluster'] == c]['duration'] for c in unique_clusters]

        # Попарные сравнения
        for (i, j) in combinations(unique_clusters, 2):
            group1 = duration[duration['cluster'] == i]['duration']
            group2 = duration[duration['cluster'] == j]['duration']

            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                results += f"Кластер {i} (n={len(group1)}, mean={group1.mean():.2f}) vs Кластер {j} (n={len(group2)}, mean={group2.mean():.2f}):\n"
                results += f"t = {t_stat:.3f}, p = {p_val:.4f}\n"
                if p_val < 0.05:
                    results += "Есть статистически значимые различия (p < 0.05)\n"
                else:
                    results += "Нет статистически значимых различий (p >= 0.05)\n"
                results += "\n"

        # ANOVA для всех кластеров
        if len(unique_clusters) > 2:
            f_stat, p_val = f_oneway(*cluster_durations)
            results += f"ANOVA для всех кластеров:\nF = {f_stat:.3f}, p = {p_val:.4f}\n"
            if p_val < 0.05:
                results += "Есть статистически значимые различия между хотя бы одной парой кластеров (p < 0.05)\n"
            else:
                results += "Нет статистически значимых различий между кластерами (p >= 0.05)\n"
    else:
        results += "Недостаточно данных для выполнения t-tests по продолжительности лечения\n"

    save_text(results, f'statistical_tests_{method_name}.txt')

def analyze_and_save_clusters(log_csv, clusters, method_name):
    unique_patients = log_csv['patient'].unique()
    patient_cluster = dict(zip(unique_patients, clusters))
    log_csv['cluster'] = log_csv['patient'].map(patient_cluster)

    # распределение действий
    action_cluster = pd.crosstab(log_csv['action'], log_csv['cluster'])
    if not action_cluster.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        action_cluster.plot(kind='bar', stacked=True, ax=ax)
        plt.title(f'Распределение действий по кластерам ({method_name})')
        plt.ylabel('Количество случаев')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, f'actions_distribution_{method_name}.png')

    # распределение ресурсов
    if 'org:resource' in log_csv.columns:
        resource_cluster = pd.crosstab(log_csv['org:resource'], log_csv['cluster'])
        if not resource_cluster.empty:
            fig, ax = plt.subplots(figsize=(max(12, len(resource_cluster) * 0.8), 8))

            colors = sns.color_palette("husl", len(resource_cluster.columns))

            # график
            resource_cluster.plot(
                kind='bar',
                stacked=True,
                ax=ax,
                width=0.8,
                color=colors
            )

            plt.title(f'Распределение ресурсов по кластерам ({method_name})')
            plt.ylabel('Количество случаев')
            plt.xlabel('Медицинский персонал/ресурсы')

            rotation = 45 if len(resource_cluster) > 5 else 0
            ha = 'right' if len(resource_cluster) > 5 else 'center'
            plt.xticks(rotation=rotation, ha=ha)

            for container in ax.containers:
                ax.bar_label(container, label_type='center', fmt='%d', padding=2)

            plt.tight_layout()
            save_figure(fig, f'resources_distribution_{method_name}.png')
        else:
            print(f"Предупреждение: Нет данных для resources_distribution_{method_name}")

    # анализ продолжительности лечения
    log_csv['DateTime'] = pd.to_datetime(log_csv['DateTime'])
    duration = log_csv.groupby('patient').apply(
        lambda x: (x['DateTime'].max() - x['DateTime'].min()).days
    ).reset_index(name='duration')
    duration['cluster'] = duration['patient'].map(patient_cluster)

    if not duration.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=duration, x='cluster', y='duration', ax=ax)
        plt.title('Продолжительность лечения по кластерам (дни)')
        save_figure(fig, f'treatment_duration_{method_name}.png')

    return log_csv


def save_interpretation(log_csv, clusters, traces, method_name):
    cluster_info = {}
    unique_patients = log_csv['patient'].unique()

    for patient, cluster in zip(unique_patients, clusters):
        if cluster not in cluster_info:
            cluster_info[cluster] = []
        cluster_info[cluster].append(patient)

    interpretation = "Результаты кластеризации:\n"

    for cluster, patients in cluster_info.items():
        interpretation += f"\nКластер {cluster} ({len(patients)} пациентов):\n"
        cluster_data = log_csv[log_csv['patient'].isin(patients)]

        top_actions = cluster_data['action'].value_counts().nlargest(10)
        interpretation += "\nТоп действий:\n" + top_actions.to_string() + "\n"

        duration = cluster_data.groupby('patient').apply(
            lambda x: (x['DateTime'].max() - x['DateTime'].min()).days
        ).mean()
        interpretation += f"\nСредняя продолжительность лечения: {duration:.1f} дней\n"

        trace_strs = [' -> '.join(traces[i]) for i, c in enumerate(clusters) if c == cluster]
        most_common_trace = max(set(trace_strs), key=trace_strs.count) if trace_strs else "Нет данных"
        interpretation += "\nТипичная трасса:\n" + most_common_trace + "\n"

    interpretation += "\nВыводы:\n"
    interpretation += "1. Кластеры различаются по составу медицинских действий\n"
    interpretation += "2. Кластеры могут различаться по распределению медицинских ресурсов\n"
    interpretation += "3. Разные кластеры имеют разную продолжительность лечения\n"
    interpretation += "4. Рекомендуется анализировать типичные трассы для оптимизации процессов\n"

    save_text(interpretation, f'interpretation_{method_name}.txt')


def main():
    file_path = 'ArtificialPatientTreatment1.csv'
    methods = ['bow', 'tfidf', 'act2vec', 'graph2vec']

    try:
        event_log, log_csv = load_and_prepare_data(file_path)

        for method in methods:
            try:
                vectors, traces = vectorize_traces(event_log, method=method)
                clusters = cluster_traces(vectors, traces)
                if method == 'graph2vec':
                    clusters = np.where(clusters == 0, -1, clusters)
                visualize_and_save_clusters(vectors, clusters, method)
                log_csv = analyze_and_save_clusters(log_csv, clusters, method)
                perform_statistical_tests(log_csv, clusters, method)
                save_interpretation(log_csv, clusters, traces, method)

                # Сохраняем данные с кластерами
                log_csv.to_csv(os.path.join('results', f'clustered_data_{method}.csv'), index=False)

            except Exception as e:
                error_msg = f"Ошибка при обработке методом {method}: {str(e)}"
                save_text(error_msg, f'error_{method}.txt')
                continue

    except Exception as e:
        save_text(f"Ошибка при загрузке данных: {str(e)}", 'loading_error.txt')


if __name__ == "__main__":
    main()
