%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  LaTeX class for preparing papers for "Upravlenie Bolshimi Sistemami"
%%  (under MikTeX 1.2c--2.4 platform)
%%  The class is based on the analogous class of "Avtomatika i Telemekhanika".
%%  charset=windows-1251
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}[1998/12/01]
\ProvidesClass{ubs}[2005/05/07 alpha v2.1
LaTeX class for preparing papers to "Upravlenie Bolshimi Sistemami.
Requires A.Lebedev's Russian fonts set "pscyr" (http://tex.imm.uran.ru/texserver/fonts/pscyr/pscyr4c/)."]
\ClassWarningNoLine{ubs}{This class requires MikTeX 1.2c--2.4}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions
\LoadClass[a5paper,11pt,twoside]{article} % стандартный класс
\RequirePackage{cmap} % распознавание кодировки для copy-paste
\RequirePackage[T1,T2A]{fontenc} % шрифты
%%%\RequirePackage[cp1251]{inputenc} % в кодировке Windows
\RequirePackage[utf8]{inputenc} % в кодировке Windows
\RequirePackage[english,russian]{babel} % русификация
\RequirePackage{pscyr} % !!! набор русских шрифтов А. Лебедева (http://tex.imm.uran.ru/texserver/fonts/pscyr/pscyr4c/). Нужен для отображения заголовков в BOLD ARIAL, BOLD ITALIC ARIAL. Этот пакет необходимо установить, если хотите смотреть Сборник УБС "as it is".
\RequirePackage[leqno]{amsmath} % расширенные возможности для задания формул
\RequirePackage{amsfonts} % то же самое
\RequirePackage{amssymb} % то же самое
\RequirePackage{ifthen} % расширенные возможности логических операций
\RequirePackage{calc} % расширенные возможности арифметических операций
\RequirePackage{myruss} % !!! подправленный стиль Львовского (ставит точки номеров разделов)
\RequirePackage{ruslist} % !!! перечни
\RequirePackage{verbatim} % для возможности закомментировать кусок текста \begin{comment} ... \end{comment}
\RequirePackage{108-170} % !!! размеры полей
\RequirePackage{ccaption} % расширенные возможности работы с подписями к таблицам и рисункам.
%%  !!! --- так помечены нестандартные пакеты

%% Шрифты
\renewcommand{\sfdefault}{far}
\renewcommand{\rmdefault}{ftm}

%%  Сколь угодно жидкие строки
\sloppy

% Интервалы
\setlength{\parindent}{7.5 mm} % абзацный отступ
\setlength\parskip{0\p@ \@plus \p@} % интервал между абзацами
\setlength\baselineskip{12pt plus 1 pt minus 1 pt} % межстрочный интервал 12 пт

%%  Знаки нестрогих равенств (как принято у нас)
\def\le{\leqslant}
\def\leq{\leqslant}
\def\ge{\geqslant}
\def\geq{\geqslant}

%%  Аннотация
\renewenvironment{abstract}%
{\vspace{0pt}
\begin{center}}%
{\end{center}\vspace{6pt}}

%%  Список литературы
\def\@biblabel#1{#1.\hfill}
\addto\extrasrussian{%
\def\bibname{\center\normalsize\textsf\textbf\textit{Литература}}
\def\refname{\center\normalsize\textsf\textbf\textit{Литература}}}

%% The following definition is based on OL.sty VERSION: 1.1 of Optical Society of America
%% to normalize parskib bewteen bibitems and the vertical indent before the bibliography
\renewenvironment{thebibliography}[1]
     {\vspace{-24pt}\section*{\refname
        \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
      \list{\@arabic\c@enumiv .}
           {\settowidth\labelwidth{0in}%
            \leftmargin\labelwidth
  \setlength{\parsep}{0pc}
  \setlength{\labelsep}{12 pt}
  \setlength{\itemsep}{0.05pc}%{-0.06in}
  \setlength{\listparindent}{0in}
  \setlength{\itemindent}{0in}
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m \normalsize}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist \vskip.2in}
\let\@openbib@code\@empty
\newcommand\OSAJNLReferences{\thebibliography{}}
\newcommand\OSAJNLReflinks{\thebibliography{}}
\newcommand\References{\thebibliography{}}
\let\endOSAJNLReferences\endthebibliography
\let\endOSAJNLlinks\endthebibliography
\let\endReferences\endthebibliography

%%  Приложение
\newcounter{isappendix}
\setcounter{isappendix}{0}
\def\appendix%
{\section*{Приложение}%
\refstepcounter{isappendix}}
\@addtoreset{equation}{isappendix}
\def\theequation%
{{\ifnum \arabic{isappendix} > 0 {\mbox{П.}\arabic{equation}}%
\else{\arabic{equation}}\fi}}

%%  Макросы для определения заголовка
\def\udk#1{\gdef\@udk{\small#1\\}} %% код УДК
\def\bbk#1{\gdef\@bbk{\small#1\\}} %% код ББК
%%\def\bbk#1{\gdef\@bbk{#1}} %% код ББК
\def\title#1{\gdef\@title{#1}} %% название статьи
\def\engtitle#1{\gdef\@engtitle{#1}} %% название статьи на английском
\def\thanks#1{\gdef\@thanks{#1}} %% благодарности и ссылки на финансирование
\def\authors#1{\gdef\@authors{#1}} %% авторы
\def\engauthors#1{\gdef\@engauthors{#1}} %% авторы на английском
\def\organization#1{\gdef\@organization{{\large(\textit{#1})}}} %% место работы
\def\abstract#1{\gdef\@abstract{\parbox{108mm}{ \textit{\footnotesize #1} }}} %% текст аннотации
\def\engabstract#1{\gdef\@engabstract{\parbox{108mm}{\textit{\footnotesize Abstract: #1}}}} %% текст аннотации на английском
\def\emails#1{\gdef\@emails{#1}} %% адреса электронной почты
\def\rubric#1{\gdef\@rubric{#1}} %% рубрика
\def\issue#1{\gdef\@issue{#1}} %% выпуск Сборника
\def\keywords#1{\gdef\@keywords{#1}} %% ключевые слова
\def\engkeywords#1{\gdef\@engkeywords{#1}} %% ключевые слова на английском
\def\presented#1{\gdef\@presented{#1}} %% представляет статью
\def\received#1{\gdef\@received{#1}} %% статья получена
\def\published#1{\gdef\@published{#1}} %% дата опубликования

%Колонтитулы%
\renewcommand{\@evenhead}{\textrm{\textit{Управление большими системами. Выпуск \@issue\hfil}}}
\renewcommand{\@oddhead}{\textrm{\textit{\hfil\@rubric}}}
\renewcommand{\@evenfoot}{\textrm{\thepage\hfil}}
\renewcommand{\@oddfoot}{\textrm{\hfil\thepage}}

%% Сноски
\renewcommand{\footnoterule}{\vspace*{-6pt}%
     \hrule width .4\columnwidth%
     \vspace*{5.6pt}}
\renewcommand{\@makefnmark}{\noindent\hbox{\mathsurround=1pt%
$^{\@thefnmark}$}}
\renewcommand{\@makefntext}[1]{\parindent=1em\noindent%
\hbox to 1.8em{\hss\@makefnmark}\footnotesize\textit{#1}}

%%  Заголовок
\renewcommand{\maketitle}{%
\begin{center}
\large\bfseries\MakeUppercase{\textar{\textbf{\@title}}}\@thanks%
\\[6pt] %
\normalsize\normalfont
\@authors\\[6pt]
\@abstract\\[6pt]
\end{center}
{\small Ключевые слова: \@keywords.}}

%%  Информация о статье на английском языке
\newcommand{\makeenginfo}{%
\begin{flushleft}
\large\bfseries\MakeUppercase{\textbf{\@engtitle}}\\[6pt]
\normalsize\normalfont
\@engauthors\\[6pt]
\@engabstract\\[6pt]
\end{flushleft}
{\small Keywords: \@engkeywords.}
\begin{flushleft}
\@udk%
\@bbk%
\end{flushleft}
}

%%  Дополнительная информация в конце статьи
\newcommand{\makeauxinfo}{%
\vspace{6pt}\begin{flushright}
\emph{Статья представлена к публикации\\членом редакционной коллегии \@presented\\[6pt]Поступила в редакцию \@received\\Дата опубликования \@published}
\end{flushright}
}

%%  Точки после номеров разделов
\def\@aftersepkern{\kern-.5em}
\def\postsection{.\@aftersepkern}
\def\postsubsection{.\@aftersepkern}
\def\postsubsubsection{.\@aftersepkern}
\def\postparagraph{.\@aftersepkern}
\def\postsubparagraph{.\@aftersepkern}
\def\presection{}
\def\presubsection{}
\def\presubsubsection{}
\def\preparagraph{}
\def\presubparagraph{}
\def\sectioncenterstyle{\centering}
\def\subsectioncenterstyle{\centering}

%%  Точки после номеров теорем
\def\afterthmseparator{.}
\def\@begintheorem#1#2{%
\setlength{\topsep}{1pt}%
\setlength{\partopsep}{0mm}%
\trivlist\item[\hspace{\parindent}\hskip\labelsep{\underline{\rm
#1\ \textup{#2}\unskip\afterthmseparator}}]\itshape}
\def\@opargbegintheorem#1#2#3{%
\setlength{\topsep}{1pt}%
\setlength{\partopsep}{0mm}%
\trivlist\item[\hspace{\parindent}\hskip\labelsep{\underline{\rm
#1\ \textup{#2}\ {\rm{#3}}\afterthmseparator}}]\itshape}

%%  Раздел
\renewcommand\section{\@startsection{section}{1}{\z@}%
{12\p@ \@plus 1\p@ \@minus 1\p@}%
{12\p@ \@plus 1\p@ \@minus 1\p@}%
{\sffamily\normalsize\flushleft\bfseries\itshape}}

%%  Подраздел
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
{12\p@ \@plus 1\p@ \@minus 4\p@}%
{1\p@ \@plus 1\p@ \@minus 1\p@}%
{\rmfamily\normalsize\flushleft\slshape}}

%%  Подписи к рисункам, с использованием пакета ccaption
\captiondelim{.~} % после номера рисунка или таблицы ставится точка, а не двоеточие
\captionnamefont{\itshape}
\captiontitlefont{\itshape}

%%  Рисунок
\renewenvironment{figure}[1][\htbp@figure]{
    \vspace{6 pt}
    \@float{figure}[#1]%
    \setlength{\abovecaptionskip}{0 pt}%
    \setlength{\belowcaptionskip}{-12 pt}%
    \captionstyle{\centerlastline\centering}%
    \begin{center}
    }
    {\end{center}\end@float}

%%  Таблица
\renewenvironment{table}[1][\htbp@table]{
    \@float{table}[#1]%
    \setlength{\abovecaptionskip}{6 pt}%
    \setlength{\belowcaptionskip}{0 pt}%
    \captionstyle{\raggedright}%
    }
    {\end@float}

%%  Теорема
\newtheorem{theorem}{\bfseries Теорема}

%%  Следствие
\newtheorem{corollary}{\bfseries\itshape Следствие}

%%  Лемма
\newtheorem{lemma}{\bfseries\itshape Лемма}

%%  Утверждение
\newtheorem{statement}{\bfseries\itshape Утверждение}

%%  Предложение
\newtheorem{proposition}{\bfseries\itshape Предложение}

%%  Алгоритм
%\newtheorem{algorithm}{\bfseries Алгоритм}

%%  Определение
\newtheorem{definition}{\itshape Определение}

%%  Задача
\newtheorem{problem}{\bfseries Задача}

%%  Доказательство
\newenvironment{proof}
{\vspace{0pt}\par{\textbf%
{{Доказательство.\ }}}}%
{\noindent\vspace{0pt}}

%%  Доказательство теоремы
\newenvironment{proofoftheorem}[1]
{\vspace{0pt}\par{\textbf%
{{Доказательство теоремы #1.}}}}%
{\noindent\vspace{0pt}}

%%  Доказательство следствия
\newenvironment{proofofcorollary}[1]
{\vspace{0pt}\par{\textbf%
{{Доказательство следствия #1.}}}}%
{\noindent\vspace{0pt}}

%%  Доказательство леммы
\newenvironment{proofoflemma}[1]
{\vspace{0pt}\par{\textbf%
{{Доказательство леммы #1.}}}}%
{\noindent\vspace{0pt}}

%%  Доказательство утверждения
\newenvironment{proofofstatement}[1]
{\vspace{0pt}\par{\textbf%
{{Доказательство утверждения #1.}}}}%
{\noindent\vspace{0pt}}

%%  Доказательство предложения
\newenvironment{proofofproposition}[1]
{\vspace{0pt}\par{\textbf%
{{Доказательство предложения #1.}}}}%
{\noindent\vspace{0pt}}

%%  Алгоритм
\newtheorem{alg}{\bfseries Алгоритм}%
\newenvironment{algorithm}[1][\unskip]{\begin{alg}[#1]\em}{\end{alg}}%
\def\thealgorithm{\thealg}

%%  Замечание
\newtheorem{rem}{\bfseries\itshape Замечание}%
\newenvironment{remark}[1][\unskip]{\begin{rem}[#1]\em}{\end{rem}}%
\def\theremark{\therem}

%%  Пример
\newtheorem{exmpl}{\bfseries Пример}%
\newenvironment{example}[1][\unskip]{\begin{exmpl}[#1]\rm}{\protect$\bullet$\end{exmpl}}%
\def\theexample{\theexmpl}

%%  Ссылка на формулу
\def\formref#1{{\rm(\ref{#1})}}

%%  Ссылка на элемент перечня
\def\itemref#1{{\rm\ref{#1})}}

%%  В окружении multline сдвиг=0
\multlinegap=0pt

%% Интервалы перед отключными формулами и после них сделаны существенно меньше стандартных.
%% Интервалы переопределены для equation, gather и multline.

\setlength{\abovedisplayskip}{1 pt plus 4 pt minus 2 pt}
\setlength{\abovedisplayshortskip}{1 pt plus 2 pt minus 2 pt}
\setlength{\belowdisplayskip}{1 pt plus 4 pt minus 2 pt}
\setlength{\belowdisplayshortskip}{1 pt plus 2 pt minus 2 pt}

\let\mygather\gather
\let\endmygather\endgather
\renewenvironment{gather}
    {\abovedisplayskip=1 pt plus 4 pt minus 2 pt%
    \abovedisplayshortskip=1 pt plus 2 pt minus 2 pt%
    \belowdisplayskip=1 pt plus 4 pt minus 2 pt%
    \belowdisplayshortskip=1 pt plus 2 pt minus 2 pt%
    \mygather
    }{\endmygather}

\let\myeq\equation
\let\endmyeq\endequation
\renewenvironment{equation}
    {\abovedisplayskip=1 pt plus 4 pt minus 2 pt%
    \abovedisplayshortskip=1 pt plus 2 pt minus 2 pt%
    \belowdisplayskip=1 pt plus 4 pt minus 2 pt%
    \belowdisplayshortskip=1 pt plus 2 pt minus 2 pt%
    \myeq
    }{\endmyeq}

\let\myml\multline
\let\endmyml\endmultline
\renewenvironment{multline}
    {\abovedisplayskip=1 pt plus 4 pt minus 2 pt%
    \abovedisplayshortskip=1 pt plus 2 pt minus 2 pt%
    \belowdisplayskip=1 pt plus 4pt minus 2 pt%
    \belowdisplayshortskip=1 pt plus 2 pt minus 2 pt%
    \myml
    }{\endmyml}

% более сильные штрафы за висячие строки
\widowpenalty 250
\clubpenalty 250

% игнорировать пустое место внизу страницы - не натягивать на него за счет интервалов
% эмуляция поведения MS Word для совместимости шаблонов
\raggedbottom

\endinput
