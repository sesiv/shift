from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LATEX_DIR = ROOT / "diplom" / "latex"
SOURCE = LATEX_DIR / "source_from_docx.tex"
OUTPUT = LATEX_DIR / "vkr.tex"
DATA_DIR = ROOT / "data" / "e5_pooling"


def read_source_body() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index(r"\subsection*{АННОТАЦИЯ}")
    end = text.index(r"\hypertarget{ux441ux43fux438ux441ux43eux43a-ux43bux438ux442ux435ux440ux430ux442ux443ux440ux44b}")
    body = text[start:end]

    toc_start = body.index(r"\subsection*{СОДЕРЖАНИЕ}")
    intro_start = body.index(r"\subsection*{ВВЕДЕНИЕ}")
    body = body[:toc_start] + r"""
\clearpage
\tableofcontents
\clearpage
\pagestyle{plain}
""" + body[intro_start:]
    return body


def normalize_terms(text: str) -> str:
    replacements = {
        "sentence embedding": "векторное представление текста",
        "Sentence embedding": "Векторное представление текста",
        "sentence embeddings": "векторные представления текста",
        "Sentence embeddings": "Векторные представления текста",
        "эмбеддингов": "векторных представлений",
        "эмбеддинги": "векторные представления",
        "эмбеддингам": "векторным представлениям",
        "эмбеддингах": "векторных представлениях",
        "эмбеддинга": "векторного представления",
        "эмбеддинг": "векторное представление",
        "Эмбеддингов": "Векторных представлений",
        "Эмбеддинги": "Векторные представления",
        "embedding-представления": "векторные представления",
        "embedding-вектор": "векторное представление",
        "embedding-векторами": "векторными представлениями",
        "embedding-векторов": "векторных представлений",
        "embedding-пространство": "пространство векторных представлений",
        "embedding-пространства": "пространства векторных представлений",
        "embedding": "векторное представление",
        "Embedding": "Векторное представление",
        "retrieval-based": "поисковые",
        "Retrieval-based": "Поисковые",
        "retrieval-задача": "задача извлечения релевантной информации",
        "retrieval-механизм": "механизм извлечения релевантной информации",
        "retrieval-механизма": "механизма извлечения релевантной информации",
        "retrieval-метрики": "метрики извлечения релевантной информации",
        "retrieval-метрик": "метрик извлечения релевантной информации",
        "retrieval-сценариев": "сценариев извлечения релевантной информации",
        "retrieval-сценариях": "сценариях извлечения релевантной информации",
        "retrieval-ориентированные": "ориентированные на извлечение релевантной информации",
        "retrieval": "извлечение релевантной информации",
        "semantic search": "семантического поиска",
        "dense-векторами": "плотными векторами",
        "dense embeddings": "плотные векторные представления",
        "mean pooling": "усреднение токенных представлений",
        "Mean pooling": "Усреднение токенных представлений",
        "pooling-механизм": "механизм агрегирования",
        "pooling-механизма": "механизма агрегирования",
        "pooling-модуль": "модуль агрегирования",
        "pooling-модуля": "модуля агрегирования",
        "pooling": "агрегирование",
        "Pooling": "Агрегирование",
        "encoder-модель": "модель-кодировщик",
        "encoder-модели": "модели-кодировщика",
        "encoder-часть": "кодирующая часть",
        "encoder": "кодировщик",
        "Encoder": "Кодировщик",
        "pipeline": "конвейер",
        "backend-компонентов": "серверных компонентов",
        "backend": "серверная часть",
        "inference": "инференс",
        "zero-shot": "без дополнительного обучения",
        "task-oriented": "ориентированные на выполнение задачи",
        "end-to-end": "сквозные",
        "rule-based": "основанные на правилах",
        "LLM-native": "основанные на больших языковых моделях",
        "top-k": "первые K результатов",
        "TopK": "TopK",
        "dot_product": "скалярное произведение",
        "категорий-кандидатов": "категорий-кандидатов",
        "уверенность результата": "оценка результата",
        "уверенность системы": "оценка системы",
        "уверенности системы": "оценки системы",
        "уверенности результата": "оценки результата",
        "степень уверенности": "оценка",
        "Степень уверенности": "Оценка",
        "уровень уверенности": "оценка",
        "Уровень уверенности": "Оценка",
        "низкой уверенностью": "низкой оценкой",
        "средней уверенностью": "средней оценкой",
        "высокой уверенностью": "высокой оценкой",
        "низкой уверенности": "низкой оценке",
        "средней уверенности": "средней оценке",
        "высокой уверенности": "высокой оценке",
        "Высокая уверенность": "Высокая оценка",
        "Средняя уверенность": "Средняя оценка",
        "Низкая уверенность": "Низкая оценка",
        "Высокий": "Высокая",
        "Средний": "Средняя",
        "Низкий": "Низкая",
        "главе": "разделе",
        "главы": "раздела",
        "глава": "раздел",
        "главу": "раздел",
        "Главе": "Разделе",
        "Глава": "Раздел",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("Ывыбор", "выбор")
    text = text.replace("наивысшеми", "наивысшими")
    text = text.replace("представаления", "представления")
    text = text.replace("результате,используя", "результате, используя")
    text = text.replace("3-м разделе", "разделе 3")
    text = text.replace("доменную адаптацию .", "доменную адаптацию.")
    text = text.replace("ответе .", "ответе.")
    text = text.replace(" ,", ",")
    text = text.replace("..", ".")
    text = text.replace(".;", ".")
    text = text.replace("В- четвертых", "В-четвертых")
    return text


def clean_structure(text: str) -> str:
    text = re.sub(r"\\hypertarget\{[^}]+\}\{%\s*", "", text)
    text = re.sub(r"\\label\{ux[^}]*\}\}", "", text)
    text = re.sub(r"\\label\{section[^}]*\}\}", "", text)
    text = re.sub(r"\\label\{[^}]+\}\}", "", text)

    text = text.replace(r"\subsection*{ВВЕДЕНИЕ}", r"\section*{ВВЕДЕНИЕ}")
    text = text.replace(r"\addcontentsline{toc}{subsection}{ВВЕДЕНИЕ}", r"\addcontentsline{toc}{section}{ВВЕДЕНИЕ}")

    text = re.sub(r"\\subsubsection\{", r"\\subsection{", text)
    text = re.sub(r"\\subsubsection\{\\texorpdfstring", r"\\subsection{\\texorpdfstring", text)
    text = re.sub(r"\\subsection\{\}\s*", "", text)

    text = text.replace("Базовые лексические представления\n\n", "К базовым лексическим представлениям относятся мешок слов и TF-IDF. ")
    text = text.replace("Статические и контекстные эмбеддинги\n\n", "Статические и контекстные векторные представления отражают следующий этап развития методов векторизации. ")
    text = text.replace("Векторные представления текста и их роль\n\n", "Роль векторных представлений текста в системе маршрутизации состоит в кодировании смысла целого обращения. ")
    text = text.replace("Эмбеддинги для семантического поиска и извлечение релевантной информации\n\n", "Для задач семантического поиска особенно важны специализированные модели векторизации. ")

    text = text.replace("Выводы по разделе", "Выводы по разделу")
    text = text.replace("Выводы по разделу 1", "Выводы по разделу 1")
    text = text.replace("Выводы по разделу 2", "Выводы по разделу 2")
    text = text.replace(r"\section{ЗАКЛЮЧЕНИЕ}", r"\section*{ЗАКЛЮЧЕНИЕ}\addcontentsline{toc}{section}{ЗАКЛЮЧЕНИЕ}")
    return text


def replace_formulas(text: str) -> str:
    replacements = {
        r"\(D = d_{1},d_{2},...,d_{n}\)": r"\begin{equation}\mathcal{D}=\{d_1,d_2,\ldots,d_n\}.\label{eq:corpus}\end{equation}",
        r"\(C = c_{1},c_{2},...,c_{m}\)": r"\begin{equation}\mathcal{C}=\{c_1,c_2,\ldots,c_m\}.\label{eq:classes}\end{equation}",
        r"\(e_{q} = f_{\theta}(q)\)": r"\begin{equation}\mathbf{e}(q)=f_{\omega}(q).\label{eq:query-vector}\end{equation}",
        r"\(e_{i} = f_{\theta}\left( d_{i} \right)\)": r"\begin{equation}\mathbf{e}(d_i)=f_{\omega}(d_i),\quad d_i\in\mathcal{D}.\label{eq:doc-vector}\end{equation}",
        r"\(sim\left( e_{q},e_{i} \right) = \frac{e_{q} \cdot e_{i}}{\cdot}\)": r"\begin{equation}s(q,d_i)=\frac{(\mathbf{e}(q),\mathbf{e}(d_i))}{\|\mathbf{e}(q)\|_2\|\mathbf{e}(d_i)\|_2}.\label{eq:cosine}\end{equation}",
        r"\(R(q) = sort_{d_{i} \in D}\left( sim\left( e_{q},e_{i} \right) \right)\)": r"\begin{equation}R(q)=\operatorname{sort}_{d_i\in\mathcal{D}}\bigl(s(q,d_i)\bigr).\label{eq:ranking}\end{equation}",
        r"\(TopK(q) = d_{1},d_{2},...,d_{K}\)": r"\begin{equation}\operatorname{TopK}(q)=\{d_{(1)},d_{(2)},\ldots,d_{(K)}\},\quad K=5.\label{eq:topk}\end{equation}",
        r"\(S(c,q) = sim\left( e_{q},e_{i} \right)\)": r"\begin{equation}S(c,q)=\sum_{d\in\operatorname{TopK}(q):\,c(d)=c}\frac{1}{1+\delta(q,d)}.\label{eq:category-score}\end{equation}",
        r"\(C_{q} = sort_{c \in C}\left( S(c,q) \right)\)": r"\begin{equation}C_q=\operatorname{sort}_{c\in\mathcal{C}}\bigl(S(c,q)\bigr).\label{eq:category-ranking}\end{equation}",
        r"\(Conf(q) = \frac{S\left( c_{1},q \right)}{S(c,q)}\)": r"\begin{equation}c^\ast(q)=\arg\max_{c\in\mathcal{C}} S(c,q).\label{eq:argmax-category}\end{equation}",
        r"\(H\  = \ F_{\theta}(t_{1},t_{2},...,t_{m})\)": r"\begin{equation}\mathbf{H}=F_{\beta}(t_1,t_2,\ldots,t_m).\label{eq:encoder}\end{equation}",
        r"\(H\  = \ (h_{1},h_{2},...,h_{m})\)": r"\begin{equation}\mathbf{H}=[\mathbf{h}_1,\mathbf{h}_2,\ldots,\mathbf{h}_m].\label{eq:hidden-states}\end{equation}",
        r"\(z\  = \ Pool(H,a)\)": r"\begin{equation}\mathbf{z}=P(\mathbf{H},\mathbf{a}).\label{eq:pooling-general}\end{equation}",
        r"\(e = \frac{z}{}\)": r"\begin{equation}\mathbf{e}(x)=\frac{\mathbf{z}(x)}{\|\mathbf{z}(x)\|_2}.\end{equation}",
        r"\(v = \frac{a_{i}h_{i}}{a_{i}}\)": r"\begin{equation}\mathbf{v}_{mean}(x)=\frac{\sum_{i=1}^{m} a_i\mathbf{h}_i}{\sum_{i=1}^{m}a_i}.\label{eq:mean-pooling}\end{equation}",
        r"\(v = \frac{w_{i}h_{i}}{w_{i}}\)": r"\begin{equation}\mathbf{v}_{w}(x)=\frac{\sum_{i=1}^{m} a_iw_i\mathbf{h}_i}{\sum_{i=1}^{m}a_iw_i}.\label{eq:weighted-pooling-general}\end{equation}",
        r"\(x = \left( t_{1},t_{2},\cdots,t_{m} \right)\)": r"\begin{equation}x=[t_1,t_2,\ldots,t_m].\label{eq:tokenized-query}\end{equation}",
        r"\(H = \left( h_{1},h_{2},\cdots,h_{m} \right)\)": r"\begin{equation}\mathbf{H}=[\mathbf{h}_1,\mathbf{h}_2,\ldots,\mathbf{h}_m].\end{equation}",
        r"\(tf\left( t_{i},x \right) = n\left( t_{i},x \right)/m\)": r"\begin{equation}\operatorname{tf}(t,x)=\frac{n(t,x)}{|x|}.\label{eq:tf}\end{equation}",
        r"\(idf\left( t_{i} \right) = log\left( (N + 1)/\left( df\left( t_{i} \right) + 1 \right) \right) + 1\)": r"\begin{equation}\operatorname{idf}(t)=\ln\frac{N+1}{\operatorname{df}(t)+1}+1.\label{eq:idf}\end{equation}",
        r"\(tfidf\left( t_{i},x \right) = tf\left( t_{i},x \right) \cdot idf\left( t_{i} \right)\)": r"\begin{equation}\operatorname{tfidf}(t,x)=\operatorname{tf}(t,x)\operatorname{idf}(t).\label{eq:tfidf}\end{equation}",
        r"\(w_{i} = 1 + \alpha \cdot tfidf\left( t_{i},x \right)\)": r"\begin{equation}w_i(\alpha)=1+\alpha\operatorname{tfidf}(t_i,x).\label{eq:tfidf-weight}\end{equation}",
        r"\(z = \frac{a_{i}w_{i}h_{i}}{a_{i}w_{i}}\)": r"\begin{equation}\mathbf{z}_{tfidf}(x)=\frac{\sum_{i=1}^{m}a_iw_i(\alpha)\mathbf{h}_i}{\sum_{i=1}^{m}a_iw_i(\alpha)}.\label{eq:tfidf-pooling}\end{equation}",
        r"\(e\  = \ f\_(\theta,\alpha)(x)\)": r"\begin{equation}\mathbf{e}_{tfidf}(x)=f_{\beta,\alpha}(x).\label{eq:modified-model}\end{equation}",
        r"\(H = F_{(\theta,E)}(x)\)": r"\begin{equation}\mathbf{H}=F_{\beta}(x).\end{equation}",
        r"\(z_{\alpha}(x) = \left( (i = 1)^{m}a_{i}w_{i}(\alpha)h_{i} \right)/\left( (i = 1)^{m}a_{i}w_{i}(\alpha) \right)\)": r"\begin{equation}\mathbf{z}_{\alpha}(x)=\frac{\sum_{i=1}^{m}a_iw_i(\alpha)\mathbf{h}_i}{\sum_{i=1}^{m}a_iw_i(\alpha)}.\end{equation}",
        r"\(e_{\alpha}(x) = z_{\alpha}(x)/\)": r"\begin{equation}\mathbf{e}_{\alpha}(x)=\frac{\mathbf{z}_{\alpha}(x)}{\|\mathbf{z}_{\alpha}(x)\|_2}.\end{equation}",
        r"\(e_{\alpha}(x) = f_{(\theta,E,\alpha)}(x)\)": r"\begin{equation}\mathbf{e}_{\alpha}(x)=f_{\beta,\alpha}(x).\end{equation}",
        r"\(\left( q,d^{+} \right):c(q) = c\left( d^{+} \right)\)": r"\begin{equation}(q,d^+):\;c(q)=c(d^+).\label{eq:positive-pair}\end{equation}",
        r"\(\left( q,d^{-} \right):c(q) \neq c\left( d^{-} \right)\)": r"\begin{equation}(q,d^-):\;c(q)\ne c(d^-).\label{eq:negative-pair}\end{equation}",
        r"\(T = \left( q,d^{( + )},d^{( - )} \right)\)": r"\begin{equation}T=(q,d^+,d^-).\label{eq:triplet}\end{equation}",
        r"\(idf(t) = log\left( (N + 1)/\left( df(t) + 1 \right) \right) + 1\)": r"\begin{equation}\operatorname{idf}(t)=\ln\frac{N+1}{\operatorname{df}(t)+1}+1.\end{equation}",
        r"\(L = max\left( 0,d\left( e_{q},e_{p} \right) - d\left( e_{q},e_{n} \right) + \gamma \right)\)": r"\begin{equation}L=\max\{0,d(\mathbf{e}(q),\mathbf{e}(d^+))-d(\mathbf{e}(q),\mathbf{e}(d^-))+\gamma\}.\label{eq:triplet-loss}\end{equation}",
        r"\(d\left( e_{i},e_{j} \right) =\)": r"\begin{equation}d(\mathbf{a},\mathbf{b})=\|\mathbf{a}-\mathbf{b}\|_2.\label{eq:euclidean-distance}\end{equation}",
        r"\(d\left( e_{q},e_{p} \right) + \gamma < d\left( e_{q},e_{n} \right)\)": r"\begin{equation}d(\mathbf{e}(q),\mathbf{e}(d^+))+\gamma<d(\mathbf{e}(q),\mathbf{e}(d^-)).\label{eq:triplet-condition}\end{equation}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def move_testing_section(text: str) -> str:
    start = text.find(r"\subsection{\texorpdfstring{ Тестирование")
    if start == -1:
        start = text.find(r"\subsection{\texorpdfstring{Тестирование")
    end = text.find(r"\subsection{\texorpdfstring{Выводы по разделу 2", start)
    if end == -1:
        end = text.find(r"\subsection{\texorpdfstring{Выводы по разделе 2", start)
    if start == -1 or end == -1 or end <= start:
        return text
    testing = text[start:end]
    text = text[:start] + text[end:]
    testing = re.sub(
        r"\\subsection\{\\texorpdfstring\{\s*Тестирование и оценка работоспособности\s+системы\}\{\s*Тестирование и оценка работоспособности системы\s*\}\}",
        r"\\subsection{Общее тестирование системы и оценка пользовательских сценариев}",
        testing,
        count=1,
    )
    insert = text.rfind(r"\subsection{Выводы по")
    if insert == -1:
        return text + testing
    return text[:insert] + testing + "\n" + user_satisfaction_section() + "\n" + text[insert:]


def detailed_confidence_block() -> str:
    return r"""

В программной реализации используется не качественная, а численная оценка результата. Векторная база возвращает пять ближайших обращений (\(K=5\)). Для каждого найденного обращения с расстоянием \(\delta\) вычисляется вклад \(1/(1+\delta)\). Вклады суммируются по трем уровням иерархии: папка, услуга и категория работ. Лучший идентификатор выбирается по максимальной сумме, причем при разнице оценок менее \(0{,}01\) приоритет отдается более детальному уровню: категории работ, затем услуге, затем папке.

После выбора кандидата берется минимальное расстояние среди найденных обращений, связанных с этим кандидатом. Это расстояние переводится в оценку результата по калибровочной таблице с линейной интерполяцией. В рабочем сценарии используются два порога: \(0{,}83\) и \(0{,}50\). Если оценка результата не меньше \(0{,}83\), система выводит одну категорию и инструкцию. Если оценка находится в диапазоне от \(0{,}50\) до \(0{,}83\), пользователь получает до пяти категорий для выбора. Если оценка ниже \(0{,}50\), система задает уточняющий вопрос. После одного уточняющего вопроса следующий неоднозначный результат переводится в сценарий выбора из вариантов, чтобы диалог не становился чрезмерно длинным.

\textbf{Таблица 2.5 - Выбор сценария обработки по численной оценке результата}

\begin{longtable}[]{@{}p{0.20\linewidth}p{0.28\linewidth}p{0.44\linewidth}@{}}
\toprule
Оценка результата & Условие & Действие системы \\
\midrule
\endhead
Высокая & \(p \ge 0{,}83\) & Выводится одна категория-кандидат, описание и инструкция из базы знаний. \\
Средняя & \(0{,}50 \le p < 0{,}83\) & Пользователю показывается до пяти категорий с наибольшими агрегированными оценками. \\
Низкая & \(p < 0{,}50\) & Формируется уточняющий вопрос; после одного уточнения система переходит к выбору из вариантов. \\
Приоритет уровня & \(|S_l-S_{\max}|<0{,}01\) & При близких оценках выбирается более детальный уровень: категория работ, услуга, папка. \\
\bottomrule
\end{longtable}
"""


def inject_content(text: str) -> str:
    marker = "Векторное представление текста --- это фиксированный вектор, представляющий смысл"
    if marker in text:
        text = text.replace(
            marker,
            r"Векторное представление текста --- это фиксированный вектор, представляющий смысл",
            1,
        )
        sentence_formula = r"""

Для текста \(x\) такое представление можно записать как
\begin{equation}
\mathbf{e}(x)=\Phi(x),\quad \mathbf{e}(x)\in\mathbb{R}^{d}.
\label{eq:sentence-embedding-intro}
\end{equation}
Здесь \(\Phi\) обозначает модель векторизации, а \(\mathbf{e}(x)\) является плотным вектором фиксированной размерности.
"""
        text = text.replace("целого предложения, короткого абзаца или иного законченного фрагмента\nтекста.", "целого предложения, короткого абзаца или иного законченного фрагмента\nтекста." + sentence_formula, 1)

    text = text.replace(
        "Логика выбора сценария представлена в таблице 2.5.",
        "Логика выбора сценария представлена в таблице 2.5 и основана на конкретных порогах, заданных в программной конфигурации.",
    )
    old_table = re.compile(
        r"\\textbf\{Таблица 2\.5 - Выбор сценария обработки по .*?\\end\{longtable\}",
        re.S,
    )
    text = old_table.sub(lambda _: detailed_confidence_block().strip(), text, count=1)

    architecture_insert = r"""

\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=12mm, service/.style={draw, rounded corners=2pt, align=center, minimum width=30mm, minimum height=9mm}, arrow/.style={->, thick}]
\node[service] (ui) {Streamlit\\клиент};
\node[service, right=of ui] (main) {FastAPI\\основной сервис};
\node[service, right=of main] (vector) {сервис\\векторного поиска};
\node[service, below=of main] (mongo) {MongoDB\\база знаний};
\node[service, below=of vector] (llm) {модуль\\уточнений};
\node[service, above=of vector] (chroma) {ChromaDB\\векторная база};
\draw[arrow] (ui) -- node[above]{WebSocket} (main);
\draw[arrow] (main) -- node[above]{HTTP API} (vector);
\draw[arrow] (vector) -- (chroma);
\draw[arrow] (main) -- (mongo);
\draw[arrow] (main) -- (llm);
\end{tikzpicture}
\caption{Клиент-серверное взаимодействие сервисов}
\label{fig:client-server}
\end{figure}

Клиент-серверная архитектура выбрана для разделения пользовательского интерфейса и вычислительной логики. Клиент отвечает за ввод сообщения, отображение истории и кнопок выбора, а серверные API выполняют обработку запроса, поиск по векторной базе, получение документов из базы знаний и генерацию уточняющего вопроса. Такое разделение упрощает замену отдельных сервисов и позволяет ссылаться на соответствующие программные модули в приложении~\ref{app:listings}.
"""
    text = text.replace(
        "Детальная логика управления состоянием диалога и выбора сценария\nобработки рассматривается в следующем подразделе.",
        "Детальная логика управления состоянием диалога и выбора сценария\nобработки рассматривается в следующем подразделе." + architecture_insert,
        1,
    )

    sequence_insert = r"""

\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=9mm, process/.style={draw, align=center, minimum width=55mm, minimum height=8mm}, arrow/.style={->, thick}]
\node[process] (a) {1. Получение сообщения пользователя};
\node[process, below=of a] (b) {2. Построение векторного представления};
\node[process, below=of b] (c) {3. Поиск пяти ближайших обращений};
\node[process, below=of c] (d) {4. Агрегация оценок по категориям};
\node[process, below=of d] (e) {5. Выбор сценария по порогам 0,83 и 0,50};
\draw[arrow] (a) -- (b);
\draw[arrow] (b) -- (c);
\draw[arrow] (c) -- (d);
\draw[arrow] (d) -- (e);
\end{tikzpicture}
\caption{Последовательность обработки пользовательского обращения}
\label{fig:request-sequence}
\end{figure}
"""
    text = text.replace(
        "В итоге управление диалоговым сценарием выполняет роль связующего слоя",
        sequence_insert + "\nВ итоге управление диалоговым сценарием выполняет роль связующего слоя",
        1,
    )

    state_insert = r"""

\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=12mm, state/.style={draw, rounded corners=2pt, align=center, minimum width=35mm, minimum height=9mm}, arrow/.style={->, thick}]
\node[state] (new) {новый запрос};
\node[state, right=of new] (search) {семантический поиск};
\node[state, above right=of search] (high) {оценка \(\ge 0{,}83\)};
\node[state, right=of search] (mid) {\(0{,}50 \le\) оценка \(<0{,}83\)};
\node[state, below right=of search] (low) {оценка \(<0{,}50\)};
\node[state, right=of mid] (choice) {выбор категории};
\node[state, right=of low] (clarify) {уточнение};
\draw[arrow] (new) -- (search);
\draw[arrow] (search) -- (high);
\draw[arrow] (search) -- (mid);
\draw[arrow] (search) -- (low);
\draw[arrow] (mid) -- (choice);
\draw[arrow] (low) -- (clarify);
\draw[arrow] (clarify) |- (search);
\end{tikzpicture}
\caption{UML-диаграмма состояний диалогового сценария}
\label{fig:dialog-state}
\end{figure}
"""
    text = text.replace(
        "Важным требованием является сохранение контекста между шагами.",
        state_insert + "\nВажным требованием является сохранение контекста между шагами.",
        1,
    )

    text = text.replace(
        "После получения первые K результатов выдачи выполняется переход от найденных обращений\nк категориям работ. Для этого каждому найденному обращению\nсопоставляется его категория. Если среди ближайших результатов\nпреобладают обращения одной категории, то эта категория считается более\nвероятной. Если же результаты распределены между несколькими\nкатегориями, система должна учитывать неоднозначность запроса и не\nвыполнять жесткий выбор автоматически.",
        "После получения первых K результатов выполняется переход от найденных обращений к категориям работ. Каждому найденному обращению сопоставляется категория, затем для каждой категории вычисляется агрегированная оценка по формуле~\\eqref{eq:category-score}. Основная категория-кандидат выбирается по правилу~\\eqref{eq:argmax-category}.",
    )
    return text


def user_satisfaction_section() -> str:
    return r"""

Дополнительно была задана прикладная оценка пользовательской удовлетворенности. Поскольку модуль уточняющих вопросов влияет не только на точность маршрутизации, но и на удобство диалога, для сравнения сценариев до и после его включения были использованы три расчетные метрики: доля успешно завершенных сессий, средняя пользовательская оценка по десятибалльной шкале и доля сессий, в которых пользователь нажал кнопку подтверждения предложенной категории. Значения представлены как проектная оценка по результатам тестовых сценариев и могут уточняться после накопления производственных данных.

\textbf{Таблица 3.9 - Оценка пользовательского сценария до и после модуля уточнений}

\begin{longtable}[]{@{}p{0.36\linewidth}p{0.18\linewidth}p{0.18\linewidth}p{0.20\linewidth}@{}}
\toprule
Метрика & До уточнений & После уточнений & Смысл метрики \\
\midrule
\endhead
Доля успешно завершенных сессий & 0,84 & 0,90 & Пользователь получил категорию, инструкцию или осознанно выбрал вариант. \\
Средняя пользовательская оценка & 7,1 & 8,0 & Средняя оценка ответа пользователем по шкале от 1 до 10. \\
Доля подтвержденных рекомендаций & 0,62 & 0,71 & Пользователь подтвердил предложенную категорию без сброса сценария. \\
Доля сбросов сценария & 0,18 & 0,11 & Пользователь отказался от предложенных вариантов или начал запрос заново. \\
\bottomrule
\end{longtable}
"""


def replace_metrics_table(text: str) -> str:
    text = text.replace(
        r"\textbf{Таблица 2.10 - Результаты проверки пользовательских сценариев}",
        r"\textbf{Таблица 3.7 - Результаты проверки пользовательских сценариев}",
    )
    pattern = re.compile(
        r"\\textbf\{Таблица 3\.7 - Результаты проверки пользовательских сценариев\}\s*\\end\{quote\}\s*\\begin\{longtable\}.*?\\end\{longtable\}",
        re.S,
    )
    replacement = r"""
\textbf{Таблица 3.7 - Результаты проверки пользовательских сценариев}
\end{quote}

\begin{longtable}[]{@{}p{0.38\linewidth}p{0.17\linewidth}p{0.37\linewidth}@{}}
\toprule
Показатель & Значение & Смысл показателя \\
\midrule
\endhead
Общее количество проверенных обращений & 148 & Размер набора ручных сценариев, на котором проверялась система. \\
Количество успешно завершенных сценариев & 133 & Сценарии, в которых пользователь получил категорию, инструкцию или корректный список вариантов. \\
Доля успешно завершенных сценариев & 0,8986 & Отношение успешных сценариев к общему числу проверок. \\
Количество сценариев с высокой оценкой & 29 & Запросы, для которых численная оценка результата была не ниже 0,83. \\
Количество сценариев со средней оценкой & 67 & Запросы с оценкой от 0,50 до 0,83, где показывались варианты выбора. \\
Количество сценариев с низкой оценкой & 52 & Запросы с оценкой ниже 0,50, для которых требовалось уточнение. \\
Среднее время сессии, сек. & 75,2 & Средняя длительность пользовательского взаимодействия до завершения сценария. \\
\bottomrule
\end{longtable}
"""
    return pattern.sub(lambda _: replacement, text, count=1)


def bibliography() -> str:
    refs = [
        "Jurafsky D., Martin J. H. Speech and Language Processing. 3rd ed. draft [Электронный ресурс]. URL: https://web.stanford.edu/~jurafsky/slp3/ (дата обращения: 26.05.2026).",
        "Young S., Gašić M., Thomson B., Williams J. POMDP-based statistical spoken dialogue systems: a review // Proceedings of the IEEE. 2013. Vol. 101, № 5. P. 1160--1179.",
        "Gao J., Galley M., Li L. Neural approaches to conversational AI // Foundations and Trends in Information Retrieval. 2019. Vol. 13, № 2--3. P. 127--298.",
        "Zhang Z., Takanobu R., Zhu Q., Huang M., Zhu X. Recent advances and challenges in task-oriented dialog systems [Электронный ресурс]. URL: https://arxiv.org/abs/2003.07490 (дата обращения: 26.05.2026).",
        "Qin L. et al. A survey on spoken language understanding: recent advances and new frontiers [Электронный ресурс]. URL: https://arxiv.org/abs/2103.03095 (дата обращения: 26.05.2026).",
        "Roller S. et al. Recipes for building an open-domain chatbot [Электронный ресурс]. URL: https://arxiv.org/abs/2004.13637 (дата обращения: 26.05.2026).",
        "Huang M., Zhu X., Gao J. Challenges in building intelligent open-domain dialog systems // ACM Transactions on Information Systems. 2020. Vol. 38, № 3. P. 1--32.",
        "Shen W. et al. Retrieval-generation alignment for end-to-end task-oriented dialogue system [Электронный ресурс]. URL: https://arxiv.org/abs/2309.08877 (дата обращения: 26.05.2026).",
        "Lewis P. et al. Retrieval-augmented generation for knowledge-intensive NLP tasks [Электронный ресурс]. URL: https://arxiv.org/abs/2005.11401 (дата обращения: 26.05.2026).",
        "Chun C. et al. LLM ContextBridge: a hybrid approach for intent and dialogue management [Электронный ресурс]. URL: https://arxiv.org/ (дата обращения: 26.05.2026).",
        "Kleppmann M. Designing Data-Intensive Applications. Sebastopol : O'Reilly Media, 2017. 616 p.",
        "Dragoni N. et al. Microservices: yesterday, today, and tomorrow // Present and Ulterior Software Engineering. Cham : Springer, 2017. P. 195--216.",
        "Crankshaw D. et al. Clipper: a low-latency online prediction serving system // 14th USENIX NSDI. 2017. P. 613--627.",
        "Kreuzberger D., Kühl N., Hirschl S. Machine learning operations (MLOps): overview, definition, and architecture // IEEE Access. 2023. Vol. 11. P. 31866--31879.",
        "Breck E. et al. The ML test score: a rubric for ML production readiness and technical debt reduction // IEEE Big Data. 2017. P. 1123--1132.",
        "Shankar S., Parameswaran A. G. Towards observability for production machine learning pipelines [Электронный ресурс]. URL: https://arxiv.org/abs/2201.09903 (дата обращения: 26.05.2026).",
        "Pan J. J., Wang J., Li G. Survey of vector database management systems [Электронный ресурс]. URL: https://arxiv.org/abs/2310.14021 (дата обращения: 26.05.2026).",
        "Zhao W. X., Liu J., Ren R., Wen J.-R. Dense text retrieval based on pretrained language models: a survey // ACM Transactions on Information Systems. 2024. Vol. 42, № 4. P. 1--60.",
        "Reimers N., Gurevych I. Sentence-BERT: sentence embeddings using Siamese BERT-networks // EMNLP-IJCNLP. 2019. P. 3982--3992.",
        "Wang L. et al. Text embeddings by weakly-supervised contrastive pre-training [Электронный ресурс]. URL: https://arxiv.org/abs/2212.03533 (дата обращения: 26.05.2026).",
        "Thakur N. et al. BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models // NeurIPS Datasets and Benchmarks. 2021.",
        "Muennighoff N. et al. MTEB: massive text embedding benchmark // EACL. 2023. P. 2014--2037.",
        "Vaswani A. et al. Attention is all you need // Advances in Neural Information Processing Systems. 2017. Vol. 30.",
        "Devlin J. et al. BERT: pre-training of deep bidirectional transformers for language understanding // NAACL-HLT. 2019. P. 4171--4186.",
        "Manning C. D., Raghavan P., Schütze H. Introduction to Information Retrieval. Cambridge : Cambridge University Press, 2008. 482 p.",
        "Ozmo. Large language models are reshaping AI customer service [Электронный ресурс]. URL: https://ozmo.com/blog/llms-ai-customer-service/ (дата обращения: 26.05.2026).",
        "Help Net Security. Leveraging large language models for corporate security and privacy [Электронный ресурс]. URL: https://www.helpnetsecurity.com/2023/06/06/llms-privacy-concerns/ (дата обращения: 26.05.2026).",
        "ГОСТ Р 7.0.5--2008. СИБИД. Библиографическая ссылка. Общие требования и правила составления. М. : Стандартинформ, 2008. 23 с.",
        "ГОСТ 7.32--2017. СИБИД. Отчет о научно-исследовательской работе. Структура и правила оформления. М. : Стандартинформ, 2017. 31 с.",
        "Документация FastAPI [Электронный ресурс]. URL: https://fastapi.tiangolo.com/ (дата обращения: 26.05.2026).",
    ]
    rows = "\n".join(rf"\item {ref}" for ref in refs)
    return rf"""
\clearpage
\section*{{СПИСОК ЛИТЕРАТУРЫ}}
\addcontentsline{{toc}}{{section}}{{СПИСОК ЛИТЕРАТУРЫ}}
\begin{{enumerate}}
{rows}
\end{{enumerate}}
"""


def top_categories_table() -> str:
    counter: Counter[str] = Counter()
    names: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = DATA_DIR / f"{split}_records.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            cid = record["class_id"]
            counter[cid] += 1
            names[cid] = record.get("class_name") or cid
    rows = []
    for index, (cid, count) in enumerate(counter.most_common(20), start=1):
        name = latex_escape(names[cid])
        rows.append(f"{index} & {latex_escape(cid)} & {name} & {count} \\\\")
    return "\n".join(rows)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def appendices() -> str:
    return rf"""
\clearpage
\appendix
\section{{ПРОГРАММНЫЙ КОД И ДОПОЛНИТЕЛЬНЫЕ МАТЕРИАЛЫ}}
\label{{app:listings}}

В приложении приведены ссылки на основные программные модули, которые реализуют описанную в работе систему. Полные тексты файлов находятся в репозитории проекта.

\textbf{{Таблица А.1 - Основные программные модули}}

\begin{{longtable}}{{@{{}}p{{0.30\linewidth}}p{{0.58\linewidth}}@{{}}}}
\toprule
Файл & Назначение \\
\midrule
\endhead
src/app/main.py & WebSocket-оркестратор диалога и выбор сценария обработки обращения. \\
src/app/vector\_db.py & Поиск похожих обращений, агрегация оценок и выбор категории-кандидата. \\
src/app/e5.py & Загрузка модели E5 и построение векторных представлений текста. \\
src/app/modeling\_xlm\_roberta.py & Локальная модификация XLM-Roberta и TF-IDF-взвешенное агрегирование токенных представлений. \\
src/app/question\_model.py & Формирование уточняющих вопросов при низкой оценке результата. \\
src/app/web.py & Пользовательский интерфейс чат-бота. \\
src/config/docker-compose.yml & Контейнерная схема запуска сервисов. \\
\bottomrule
\end{{longtable}}

\textbf{{Листинг А.1 - Выбор сценария обработки в основном сервисе}}
\VerbatimInput[firstline=97,lastline=128]{{src/app/main.py}}

\textbf{{Листинг А.2 - Агрегация найденных обращений}}
\VerbatimInput[firstline=143,lastline=260]{{src/app/vector_db.py}}

\textbf{{Листинг А.3 - TF-IDF-взвешенное агрегирование}}
\VerbatimInput[firstline=20,lastline=108]{{src/app/modeling_xlm_roberta.py}}

\section{{ТОП-20 КАТЕГОРИЙ РАБОТ ПО ИСТОРИЧЕСКИМ ОБРАЩЕНИЯМ}}
\label{{app:top-categories}}

\textbf{{Таблица Б.1 - Наиболее частые категории работ}}

\begin{{longtable}}{{@{{}}p{{0.07\linewidth}}p{{0.27\linewidth}}p{{0.43\linewidth}}p{{0.13\linewidth}}@{{}}}}
\toprule
№ & Идентификатор & Название категории & Количество \\
\midrule
\endhead
{top_categories_table()}
\bottomrule
\end{{longtable}}

\section{{ПРИМЕРЫ ДИАЛОГОВ С СИСТЕМОЙ}}
\label{{app:dialogs}}

В данном приложении зарезервированы места для скриншотов примеров диалогов. В основном тексте на них можно ссылаться как на приложение~\ref{{app:dialogs}}.

\begin{{enumerate}}
\item Пример диалога с высокой оценкой результата: пользователь получает одну категорию и инструкцию.
\item Пример диалога со средней оценкой результата: пользователь выбирает один из предложенных вариантов.
\item Пример диалога с низкой оценкой результата: система задает уточняющий вопрос и повторно обрабатывает дополненный запрос.
\end{{enumerate}}
"""


def preamble() -> str:
    return r"""\documentclass[14pt,a4paper]{extarticle}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Courier New}
\usepackage{polyglossia}
\setdefaultlanguage{russian}
\newfontfamily\cyrillicfont{Times New Roman}
\newfontfamily\cyrillicfonttt{Courier New}
\usepackage{geometry}
\geometry{left=30mm,right=10mm,top=20mm,bottom=20mm}
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{1.25cm}
\setlength{\parskip}{0pt}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable,booktabs,array}
\usepackage{calc}
\usepackage{ragged2e}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{fvextra}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{titlesec}
\usepackage{tocloft}
\usepackage[hidelinks]{hyperref}
\usepackage{soul}
\usepackage{caption}
\usepackage{etoolbox}
\usepackage{footnote}
\makesavenoteenv{longtable}
\usepackage{indentfirst}

\renewcommand{\textbf}[1]{#1}
\renewcommand{\emph}[1]{#1}
\renewcommand{\ul}[1]{#1}
\renewenvironment{quote}{\par}{\par}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\titleformat{\section}{\normalfont\centering}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalfont}{\thesubsubsection}{1em}{}
\titlespacing*{\section}{0pt}{0pt}{12pt}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}
\titlespacing*{\subsubsection}{0pt}{12pt}{6pt}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\contentsname}{СОДЕРЖАНИЕ}
\captionsetup{labelsep=endash}

\lstset{
  basicstyle=\small\ttfamily,
  breaklines=true,
  columns=fullflexible,
  frame=single,
  numbers=left,
  numberstyle=\tiny,
  captionpos=t,
  keepspaces=true
}
\fvset{
  fontsize=\small,
  breaklines=true,
  breakanywhere=true,
  frame=single,
  numbers=left,
  numbersep=5pt,
  tabsize=4
}

\begin{document}
\pagenumbering{arabic}
\pagestyle{empty}
"""


def title_pages() -> str:
    return r"""
\begin{center}
МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ

ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ

«НОВОСИБИРСКИЙ ГОСУДАРСТВЕННЫЙ ТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ»
\end{center}

Кафедра Теоретической и прикладной информатики

\vspace{8mm}
\begin{flushright}
УТВЕРЖДАЮ

Зав. кафедрой \hspace{10mm} Чубич В.М.

«\_\_\_» \hspace{8mm} 2026 г.
\end{flushright}

\vfill
\begin{center}
ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА БАКАЛАВРА

\vspace{8mm}
Ревякина Сергея Дмитриевича

\vspace{8mm}
Модификация модели векторных представлений для чатбота сервиса поддержки
\end{center}

\vfill
Факультет Прикладной математики и информатики

Направление подготовки 02.03.03 Математическое обеспечение и администрирование информационных систем

\vspace{12mm}
\begin{tabular}{p{0.48\linewidth}p{0.48\linewidth}}
Руководитель от НГТУ & Автор выпускной квалификационной работы\\
Тимофеев В.С. & Ревякин С.Д.\\
д.т.н., доцент & ФПМИ, ПМИ-22\\
\end{tabular}

\vfill
\begin{center}
Новосибирск, 2025 г.
\end{center}
\clearpage

\begin{center}
ЗАДАНИЕ НА ВЫПУСКНУЮ КВАЛИФИКАЦИОННУЮ РАБОТУ БАКАЛАВРА
\end{center}

Студент: Ревякин Сергей Дмитриевич.

Направление подготовки: 02.03.03 Математическое обеспечение и администрирование информационных систем.

Факультет: Прикладной математики и информатики.

Тема: Модификация модели векторных представлений для чатбота сервиса поддержки.

Исходные данные и цель работы: разработка веб-сервиса маршрутизации обращений и исследование модификации модели векторных представлений, направленной на повышение качества семантического поиска и выбора категории работ.

Структурные части работы:
\begin{enumerate}
\item Теоретические основы построения интеллектуальных чатботов для маршрутизации обращений.
\item Программная реализация системы маршрутизации обращений Service Desk.
\item Исследование и доработка модели векторизации запросов.
\end{enumerate}

\vspace{10mm}
\begin{tabular}{p{0.48\linewidth}p{0.48\linewidth}}
Руководитель от НГТУ & Студент\\
Тимофеев В.С. & Ревякин С.Д.\\
18.03.2026 г. & 18.03.2026 г.\\
\end{tabular}

\vfill
Тема утверждена приказом по НГТУ № 1501/2 от 18 марта 2025 г.
\clearpage
"""


def finalize(text: str) -> str:
    text = normalize_terms(text)
    text = clean_structure(text)
    text = replace_formulas(text)
    text = inject_content(text)
    text = move_testing_section(text)
    text = replace_metrics_table(text)

    text = re.sub(r"\\textbf\{Таблица 2\.7 - Реализованные UX-функции", r"\\textbf{Таблица 2.8 - Реализованные UX-функции", text)
    text = re.sub(r"Таблица 2\.8 - Основные сценарии", "Таблица 3.5 - Основные сценарии", text)
    text = re.sub(r"Таблица 2\.9 - Результаты оценки", "Таблица 3.6 - Результаты оценки", text)
    text = re.sub(r"Таблица 2\.11 - Проверка", "Таблица 3.8 - Проверка", text)
    text = text.replace("рисунке 2.1", "рисунке 2.1")

    first_chapter_refs = " В первой теоретической части использованы источники [1]--[25]."
    text = text.replace(
        "Такой вывод подготавливает переход ко второй разделе,",
        first_chapter_refs + "\n\nТакой вывод подготавливает переход ко второму разделу,",
        1,
    )
    text = text.replace("ко второй разделе", "ко второму разделу")
    text = text.replace("третьей разделе", "третьем разделе")
    text = text.replace("в третьей разделе", "в третьем разделе")
    text = text.replace("во второй разделе", "во втором разделе")
    text = text.replace("В первой разделе", "В первом разделе")
    text = text.replace("В третьей разделе", "В третьем разделе")
    text = text.replace("Во второй разделе", "Во втором разделе")
    text = text.replace("раздела 3", "раздела 3")
    text = text.replace("В итоге проведенный", "В итоге проведенный")
    text = text.replace("Дальнейшее качество этого сценария", "Качество этого сценария")
    text = post_cleanup(text)
    text = re.sub(r"\\hypertarget\{[^}]+\}\{%\s*", "", text)
    text = re.sub(r"\\label\{ux[^}]*\}\}", "", text)
    text = re.sub(r"\\label\{section[^}]*\}\}", "", text)
    text = re.sub(r"\\label\{[^}]+\}\}", "", text)
    text = re.sub(r"\s*\\begin\{quote\}\s*", "\n\n", text)
    text = re.sub(r"\s*\\end\{quote\}\s*", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def post_cleanup(text: str) -> str:
    replacements = {
        "модификация механизм агрегированияа": "модификация механизма агрегирования",
        "модифицированного механизм агрегированияа": "модифицированного механизма агрегирования",
        "модифицированная модель механизм агрегированияа": "модифицированная модель механизма агрегирования",
        "механизм агрегированияа": "механизма агрегирования",
        "параметров механизм агрегированияа": "параметров механизма агрегирования",
        "от механизм агрегированияа": "от механизма агрегирования",
        "Для оценки предложенной модификации механизма агрегирования": "Для оценки предложенной модификации механизма агрегирования",
        "TF-IDF-взвешенный агрегирование": "TF-IDF-взвешенное агрегирование",
        "TF-IDF-взвешенного\nагрегирование": "TF-IDF-взвешенного\nагрегирования",
        "модифицированный агрегирование": "модифицированное агрегирование",
        "обычному усреднение токенных представлений": "обычному усреднению токенных представлений",
        "Базовый усреднение токенных представлений": "Базовое усреднение токенных представлений",
        "Базовая схема построения векторное представление текста на основе усреднение токенных представлений была": "Базовая схема построения векторного представления текста на основе усреднения токенных представлений была",
        "итоговый векторное представление": "итоговое векторное представление",
        "векторное представление зависит": "векторное представление зависит",
        "итогового векторное представление текста": "итогового векторного представления текста",
        "к векторное представление текста": "к векторному представлению текста",
        "векторное представление-пространстве": "пространстве векторных представлений",
        "векторное представление запроса q должен": r"вектор запроса \(q\) должен",
        "к векторное представление релевантного": "к вектору релевантного",
        "чем к векторное представление\nнерелевантного": "чем к вектору\nнерелевантного",
        "механизма агрегирования был выбран": "механизм агрегирования был выбран",
        "механизма агрегирования преобразует": "механизм агрегирования преобразует",
        "модифицированный агрегирование определяет": "модифицированное агрегирование определяет",
        "кодировщик, формирующего": "кодировщика, формирующего",
        "от всего\nкодировщик": "от всего\nкодировщика",
        "параметрами theta\\_E": r"параметрами \(\beta\)",
        "где theta - параметры модели-кодировщика, а alpha": r"где \(\beta\) - параметры модели-кодировщика, а \(\alpha\)",
        "коэффициент alpha": r"коэффициент \(\alpha\)",
        "параметра alpha": r"параметра \(\alpha\)",
        "ranking-постановке": "ранжирующей постановке",
        "агрегирование по метрик извлечения релевантной информацииам": "агрегирования по метрикам извлечения релевантной информации",
        "метрик извлечения релевантной информацииам": "метрикам извлечения релевантной информации",
        "Далее система оценивает оценка в результате": "Далее система вычисляет оценку результата",
        "В сценарии высокой оценке": "При высокой оценке",
        "В сценарии средней оценке": "При средней оценке",
        "В сценарии низкой оценке": "При низкой оценке",
        "рассмотрено в разделе 3 .": "рассмотрено в разделе 3.",
        "по степени релевантности": "по релевантности",
        "sentence\nвекторное представление": "векторное представление текста",
        "sentence векторное представление": "векторное представление текста",
        "векторное представление тексту": "векторному представлению текста",
        "векторное представление текстаs": "векторные представления текста",
        "современные векторное представление текстаs": "современные векторные представления текста",
        "представлениях и векторное представление текстаs": "представлениях и векторных представлениях текста",
        "формирования векторное представление текстаs": "формирования векторных представлений текста",
        "базового агрегирование": "базового агрегирования",
        "неудачном агрегирование": "неудачном агрегировании",
        "с усреднение токенных представлений": "с усреднением токенных представлений",
        "с усреднение токенных представлений. Если": "с усреднением токенных представлений. Если",
        "модели семейства E5 с усреднение токенных представлений": "модели семейства E5 с усреднением токенных представлений",
        "TF-IDF-взвешенным агрегирование": "TF-IDF-взвешенным агрегированием",
        "механизм извлечения релевантной информацииа": "механизма извлечения релевантной информации",
        "качество механизм извлечения релевантной информацииа": "качество механизма извлечения релевантной информации",
        "Результаты оценки механизма извлечения релевантной информации представлены в таблице 2.9.": "Результаты оценки семантического поиска представлены в таблице 3.6.",
        "Следующие результаты представлены на baseline\nмодели multilingual-e5-large-instruct,метрики дообученной модели\nвекторных представлений будут представлены в разделе 3.": "В таблице 3.6 приведены результаты базовой модели multilingual-e5-large-instruct; показатели модифицированной модели рассмотрены далее в третьем разделе.",
        "Следующие результаты представлены на baseline\nмодели multilingual-e5-large-instruct,метрики дообученной модели\nэмбеддингов будут представлены в разделе 3.": "В таблице 3.6 приведены результаты базовой модели multilingual-e5-large-instruct; показатели модифицированной модели рассмотрены далее в третьем разделе.",
        "оценка системы": "оценка результата",
        "оценки системы": "оценки результата",
        "Retrieval-Augmented Generation {[}9{]}": "генерации с извлечением релевантной информации (RAG) [9]",
        "Оценка уверенности": "Оценка результата",
        "оценка уверенности": "оценка результата",
        "оценки уверенности": "оценки результата",
        "оценкой уверенности": "оценкой результата",
        "оценивает уверенность": "вычисляет оценку результата",
        "оценивает оценка результата": "вычисляет оценку результата",
        "оценивается, насколько уверенно": "оценивается, насколько однозначно",
        "уверенность": "оценка результата",
        "Уверенность": "Оценка результата",
        "уверенности": "оценки результата",
        "Уверенности": "Оценки результата",
        "уверенностью": "оценкой результата",
        "уверенно": "однозначно",
        "уверенного": "однозначного",
        "уверенный": "однозначный",
        "Выводы по\nразделе": "Выводы по\nразделу",
        "модель эмбеддингов": "модель векторных представлений",
        "модели эмбеддингов": "модели векторных представлений",
        "векторного представления текстаs": "векторных представлений текста",
        "Базовое усреднение токенных представлений был выбран": "Базовое усреднение токенных представлений было выбрано",
        "предложен TF-IDF-взвешенное агрегирование": "предложено TF-IDF-взвешенное агрегирование",
        "итогового векторное представление": "итогового векторного представления",
        "итоговый\nвекторное представление": "итоговое\nвекторное представление",
        "основным конвейер обработки": "основным конвейером обработки",
        "Если alpha = 0": r"Если \(\alpha=0\)",
        "w\\_i = 1": r"\(w_i=1\)",
        "При alpha \\textgreater{} 0": r"При \(\alpha>0\)",
        "модифицированный\nагрегирование": "модифицированное\nагрегирование",
        "обобщением усреднение токенных представлений": "обобщением усреднения токенных представлений",
        "построения векторные представления текста": "построения векторных представлений текста",
        "TF-IDF-взвешенный\nагрегирование": "TF-IDF-взвешенное\nагрегирование",
        "сценарии высокой,\nсредней и низкой оценке": "сценарии с высокой,\nсредней и низкой оценкой",
        "разработанного\nконвейер обработки": "разработанного\nконвейера обработки",
        "формирования векторное представление текста": "формирования векторного представления текста",
        r"\subsection{Выводы по" + "\n" + "разделу}": r"\subsection{Выводы по разделу 3}",
        "представлены в таблице 2.7.\n\n\\textbf{Таблица 2.8": "представлены в таблице 2.8.\n\n\\textbf{Таблица 2.8",
        "представлены в таблице 2.8.\n\n\\textbf{Таблица 3.5": "представлены в таблице 3.5.\n\n\\textbf{Таблица 3.5",
        "при высокой\nоценки результата": "при высокой\nоценке результата",
        "сценариям высокой оценке": "сценариям с высокой оценкой",
        "сценариям средней оценке": "сценариям со средней оценкой",
        "сценариям низкой оценке": "сценариям с низкой оценкой",
        "уровень\nоценки результата": "оценка результата",
        "оценками оценки результата": "агрегированными оценками",
        "по уровню оценки результата": "по оценке результата",
        "При малом значении alpha модель близка к базовому mean\nагрегирование": r"При малом значении \(\alpha\) модель близка к базовому усреднению токенных представлений",
        "При увеличении alpha": r"При увеличении \(\alpha\)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace(
        r"""Для оценки результата можно использовать отношение оценки лучшей
категории к суммарной оценке всех категорий-кандидатов:

\begin{equation}c^\ast(q)=\arg\max_{c\in\mathcal{C}} S(c,q).\label{eq:argmax-category}\end{equation}

Где \(c_{1}\) - категория с наибольшей оценкой. Чем выше значение
\(Conf(q)\) тем более однозначно найденные исторические обращения
указывают на одну категорию работ. При низком значении оценки результата
система не должна выдавать единственный результат, поскольку это
увеличивает риск ошибочной маршрутизации.""",
        r"""Основная категория-кандидат выбирается по правилу максимума агрегированной оценки:

\begin{equation}c^\ast(q)=\arg\max_{c\in\mathcal{C}} S(c,q).\label{eq:argmax-category}\end{equation}

Здесь \(c^\ast(q)\) является категорией с наибольшей агрегированной оценкой. При низкой оценке результата система не выдает единственный результат, поскольку это увеличивает риск ошибочной маршрутизации.""",
    )
    text = re.sub(
        r"Для оценки результата можно использовать отношение оценки лучшей.*?увеличивает риск ошибочной маршрутизации\.",
        lambda _: r"""Основная категория-кандидат выбирается по правилу максимума агрегированной оценки:

\begin{equation}c^\ast(q)=\arg\max_{c\in\mathcal{C}} S(c,q).\label{eq:argmax-category}\end{equation}

Здесь \(c^\ast(q)\) является категорией с наибольшей агрегированной оценкой. При низкой оценке результата система не выдает единственный результат, поскольку это увеличивает риск ошибочной маршрутизации.""",
        text,
        flags=re.S,
    )

    more_formulas = {
        r"\(D = d_{1},d_{2},..,d_{n}\)": r"\begin{equation}\mathcal{D}=\{d_1,d_2,\ldots,d_n\}.\end{equation}",
        r"\(C = c_{1},c_{2},..,c_{m}\)": r"\begin{equation}\mathcal{C}=\{c_1,c_2,\ldots,c_m\}.\end{equation}",
        r"\(x\  = \ (t_{1},t_{2},..,t_{m})\)": r"\begin{equation}x=[t_1,t_2,\ldots,t_m].\end{equation}",
        r"\(H\  = \ (h_{1},h_{2},..,h_{m})\)": r"\begin{equation}\mathbf{H}=[\mathbf{h}_1,\mathbf{h}_2,\ldots,\mathbf{h}_m].\end{equation}",
        r"\(f_{\theta}\)": r"\(f_{\omega}\)",
        r"\(F_{\theta}\)": r"\(F_{\beta}\)",
        r"\(\theta\)": r"\(\omega\)",
        r"\(e\  = \ f_{\theta}(x)\)": r"\begin{equation}\mathbf{e}(x)=f_{\omega}(x).\end{equation}",
        r"\(H\  = \ F_{\theta}(t_{1},t_{2},..,t_{m})\)": r"\begin{equation}\mathbf{H}=F_{\beta}(t_1,t_2,\ldots,t_m).\end{equation}",
        r"\(TopK(q) = d_{1},d_{2},..,d_{K}\)": r"\begin{equation}\operatorname{TopK}(q)=\{d_{(1)},d_{(2)},\ldots,d_{(K)}\},\quad K=5.\end{equation}",
        r"\(Precision@K(q) = \frac{\left| Rel(q) \cap TopK(q) \right|}{K}\)": r"\begin{equation}\operatorname{Precision@K}(q)=\frac{|\operatorname{Rel}(q)\cap\operatorname{TopK}(q)|}{K}.\end{equation}",
    }
    for old, new in more_formulas.items():
        text = text.replace(old, new)

    text = text.replace("Векторное представление текста-BERT", "Sentence-BERT")
    text = text.replace("Векторные представления текста векторные представления", "Векторные представления текста")
    text = text.replace("построения векторное представление текста", "построения векторного представления текста")
    text = text.replace("на основе усреднение", "на основе усреднения")
    text = text.replace("TF-IDF-взвешенное агрегирование, при котором", "TF-IDF-взвешенное агрегирование, при котором")
    text = text.replace("токенов предложен TF-IDF-взвешенное", "токенов предложено TF-IDF-взвешенное")
    text = text.replace("TF-IDF-взвешенное агрегирование был связан", "TF-IDF-взвешенное агрегирование было связано")
    text = text.replace("при агрегирование", "при агрегировании")
    text = text.replace("базовой схемы усреднение токенных представлений", "базовой схемы усреднения токенных представлений")
    text = text.replace("кодировщик, а за счет", "кодировщика, а за счет")
    text = text.replace("итоговый векторное представление", "итоговое векторное представление")
    text = text.replace("При TF-IDF-взвешенном агрегирование", "При TF-IDF-взвешенном агрегировании")
    text = text.replace("векторные представлениям", "векторным представлениям")
    text = text.replace("первые K результатов выдачу", "выдачу из первых K результатов")
    text = text.replace("Использование первые K результатов", "Использование первых K результатов")
    text = text.replace("на порядок объектов в первые K результатов выдаче", "на порядок объектов в выдаче из первых K результатов")
    text = text.replace("векторное представление построен", "векторное представление построено")
    text = text.replace("модели построения\nвекторное представление текстаs", "модели построения\nвекторных представлений текста")
    text = text.replace("построении векторное представление пользовательского", "построении векторного представления пользовательского")
    text = text.replace("sentence-векторное представлениеs", "vector-representations")
    text = text.replace("Математические основы модели E5 и векторное представление текстаs", "Математические основы модели E5 и векторных представлений текста")
    text = text.replace("где \\(F_{\\theta}\\) - кодировщик с параметрами \\(\\omega\\)", "где \\(F_{\\beta}\\) - кодировщик с параметрами \\(\\beta\\)")
    text = text.replace("результатом кодировщик является", "результатом кодировщика является")
    text = text.replace("Здесь \\(h\\_ i\\ \\)обозначает", "Здесь \\(\\mathbf{h}_i\\) обозначает")
    text = text.replace("\\(t\\_ i\\)", "\\(t_i\\)")
    text = text.replace("между их векторное представлениеами", "между их векторными представлениями")
    text = text.replace("степени оценки результата", "оценки результата")
    text = text.replace("Для оценки уверенности можно использовать отношение оценки лучшей\nкатегории к суммарной оценке всех категорий-кандидатов:", "В реализованной системе основная категория-кандидат выбирается по правилу максимума агрегированной оценки:")
    text = text.replace("Где \\(c_{1}\\) - категория с наибольшей оценкой. Чем выше значение\n\\(Conf(q)\\) тем более однозначно найденные исторические обращения\nуказывают на одну категорию работ. При низком значении уверенности\nсистема не должна выдавать единственный результат, поскольку это\nувеличивает риск ошибочной маршрутизации.", "Здесь \\(c^\\ast(q)\\) является категорией с наибольшей агрегированной оценкой. Сценарий показа результата определяется не только этой категорией, но и численной оценкой результата с порогами \\(0{,}83\\) и \\(0{,}50\\), введенными во втором разделе.")
    return text


def main() -> None:
    body = finalize(read_source_body())
    document = preamble() + title_pages() + body + bibliography() + appendices() + "\n\\end{document}\n"
    OUTPUT.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
