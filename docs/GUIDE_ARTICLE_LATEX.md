# Guide LaTeX : écrire l'article scientifique du projet

Ce guide couvre la rédaction d'un article scientifique de bout en bout :
l'outil (LaTeX via Overleaf), la structure attendue d'un article, la syntaxe
LaTeX nécessaire section par section, la bibliographie, et une trame complète
prête à compiler pour ce projet.

Sommaire :

1. [Pourquoi LaTeX, et avec quel outil](#1-pourquoi-latex)
2. [Anatomie d'un article scientifique (IMRaD)](#2-anatomie-dun-article)
3. [Squelette minimal d'un document LaTeX](#3-squelette-minimal)
4. [Le préambule expliqué (français inclus)](#4-le-préambule)
5. [Texte, sections, listes](#5-texte-sections-listes)
6. [Mathématiques : écrire les métriques](#6-mathématiques)
7. [Figures : les images de détection](#7-figures)
8. [Tableaux : le tableau comparatif](#8-tableaux)
9. [Références croisées](#9-références-croisées)
10. [Bibliographie avec BibTeX](#10-bibliographie)
11. [Trame complète de l'article du projet](#11-trame-complète)
12. [Conseils de rédaction scientifique](#12-conseils-de-rédaction)
13. [Collaboration à trois sur Overleaf](#13-collaboration)
14. [Check-list avant de rendre](#14-check-list)
15. [LaTeX en général : au-delà de l'article](#15-latex-en-général)

---

## 1. Pourquoi LaTeX

LaTeX est le standard des publications scientifiques : numérotation
automatique des sections, figures, tableaux et équations ; références croisées
qui restent justes quand le document bouge ; bibliographie générée ; qualité
typographique constante. On décrit la **structure** (« ceci est une section »,
« ceci est une figure ») et LaTeX s'occupe de la mise en forme.

**Outil recommandé : [Overleaf](https://www.overleaf.com)** (gratuit). Rien à
installer, compilation dans le navigateur, et surtout **édition à plusieurs en
temps réel**, idéal pour un groupe de trois. L'alternative locale (TeX Live +
VS Code ou TeXstudio) fonctionne mais n'apporte rien ici ; la collaboration y
est plus laborieuse.

Sur Overleaf : *New Project → Blank Project*, puis coller la trame de la
section 11. Compilateur conseillé : `pdfLaTeX` (menu du bouton *Recompile*).

## 2. Anatomie d'un article

La structure quasi universelle est **IMRaD** : Introduction, Méthodes,
Résultats, Discussion. Pour ce projet :

| Section | Contenu | Correspondance projet |
|---|---|---|
| **Titre + auteurs** | Précis, informatif | « Détection des véhicules éloignés… : amélioration basée sur YOLOv8 » |
| **Résumé (abstract)** | 150–250 mots : contexte, problème, méthode, résultat chiffré, conclusion | À écrire en dernier |
| **1. Introduction** | Contexte, problème, limites de l'existant, contributions | Trafic béninois, faiblesse de YOLO sur petits objets |
| **2. Travaux connexes** | Ce que d'autres ont fait, en quoi on s'en distingue | YOLO, détection de petits objets, datasets de trafic |
| **3. Méthodologie** | Reproductible par un tiers : données, modèle, protocole | BMD-45, remappage 4 classes, fine-tuning, protocole d'évaluation |
| **4. Résultats** | Chiffres et figures, sans interprétation | Tableau comparatif, rappel par taille, images côte à côte |
| **5. Discussion** | Interprétation, limites | Les 3 lectures possibles du README §10 ; limite « pas de données béninoises » |
| **6. Conclusion** | Rappel de l'apport, perspectives | |
| **Références** | Tout ce qui est cité | |

Deux principes : les **Résultats** rapportent, la **Discussion** interprète :
ne pas mélanger ; et les **limites** s'annoncent explicitement (le README le
dit déjà très bien pour l'absence de jeu de données béninois).

## 3. Squelette minimal

```latex
\documentclass[a4paper,11pt]{article}

% ---------- préambule : les packages se chargent ici ----------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}

\title{Mon titre}
\author{Première Autrice \and Deuxième Auteur}
\date{Août 2026}

% ---------- le document proprement dit ----------
\begin{document}
\maketitle

\begin{abstract}
Le résumé.
\end{abstract}

\section{Introduction}
Le texte...

\end{document}
```

À savoir : les commandes commencent par `\`, les blocs sont des
`\begin{...}...\end{...}` (« environnements »), `%` commence un commentaire,
et les caractères `% & # _ { }` doivent être échappés dans le texte
(`\%`, `\&`, `\_`, etc.).

## 4. Le préambule

Préambule recommandé pour un article en français :

```latex
\documentclass[a4paper,11pt]{article}

% --- Encodage et langue
\usepackage[utf8]{inputenc}     % accents directement dans le source
\usepackage[T1]{fontenc}        % césure correcte des mots accentués
\usepackage[french]{babel}      % typographie française (espaces avant : ; ! ?)
\usepackage{lmodern}            % police vectorielle propre

% --- Mise en page
\usepackage[margin=2.5cm]{geometry}
\usepackage{microtype}          % micro-ajustements typographiques

% --- Sciences
\usepackage{amsmath, amssymb}   % équations
\usepackage{siunitx}            % nombres et unités : \num{0.612}, \SI{45}{fps}
\usepackage{graphicx}           % \includegraphics
\usepackage{booktabs}           % tableaux professionnels (\toprule...)
\usepackage{subcaption}         % sous-figures (a) (b) côte à côte

% --- Liens et références (hyperref toujours en dernier ou presque)
\usepackage[hidelinks]{hyperref}
\usepackage[french,nameinlink]{cleveref}  % \cref{...} : « figure 2 » automatique
```

Note `babel` français : les deux-points, point-virgules et guillemets suivent
automatiquement la typographie française. Utiliser `\og texte \fg{}` pour les
guillemets « français ».

## 5. Texte, sections, listes

```latex
\section{Méthodologie}          % numérotée : 3
\subsection{Jeu de données}     % 3.1
\subsubsection{Prétraitement}   % 3.1.1
\section*{Remerciements}        % étoile = non numérotée

\textbf{gras} \textit{italique} \texttt{code ou nom de fichier}
\emph{mise en valeur}           % préférer \emph à \textit dans le corps

\begin{itemize}
  \item premier point ;
  \item second point.
\end{itemize}

\begin{enumerate}
  \item première étape ;
  \item deuxième étape.
\end{enumerate}
```

Un saut de ligne vide crée un nouveau paragraphe. Ne jamais forcer la mise en
page avec des `\\` en fin de paragraphe ni des `\vspace` : si le résultat
semble l'exiger, c'est presque toujours la structure qu'il faut revoir.

## 6. Mathématiques

En ligne dans une phrase : `$mAP@0{,}5 = 0{,}612$`. En bloc numéroté :

```latex
La précision et le rappel se définissent comme suit :
\begin{equation}
  \text{Précision} = \frac{VP}{VP + FP}, \qquad
  \text{Rappel} = \frac{VP}{VP + FN},
  \label{eq:precision-rappel}
\end{equation}
où $VP$, $FP$ et $FN$ désignent les vrais positifs, faux positifs et faux
négatifs. Le F1-score est leur moyenne harmonique :
\begin{equation}
  F_1 = 2 \cdot \frac{\text{Précision} \times \text{Rappel}}
                     {\text{Précision} + \text{Rappel}}.
\end{equation}
```

L'IoU (au cœur du protocole d'évaluation du projet) :

```latex
\begin{equation}
  \mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
\end{equation}
```

Et la mAP :

```latex
\begin{equation}
  \mathrm{mAP} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{AP}_i
\end{equation}
```

Symboles utiles : `\times`, `\cdot`, `\frac{a}{b}`, `\sum_{i=1}^{N}`,
`\geq`, `\leq`, `\in`, indices `x_i`, exposants `x^2` (accolades si plusieurs
caractères : `x_{ij}`).

## 7. Figures

Le cas d'usage principal ici : la même scène d'embouteillage vue par les deux
modèles, côte à côte.

```latex
\begin{figure}[tb]
  \centering
  \begin{subfigure}{0.48\linewidth}
    \includegraphics[width=\linewidth]{figures/embouteillage1_base.jpg}
    \caption{YOLOv8 pré-entraîné.}
    \label{fig:demo-base}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.48\linewidth}
    \includegraphics[width=\linewidth]{figures/embouteillage1_finetune.jpg}
    \caption{YOLOv8 fine-tuné.}
    \label{fig:demo-finetune}
  \end{subfigure}
  \caption{Détections sur une même scène d'embouteillage à Cotonou. Le modèle
  fine-tuné retrouve les motos partiellement masquées à l'arrière-plan.}
  \label{fig:demo}
\end{figure}
```

À savoir :

- `[tb]` laisse LaTeX placer la figure en haut ou en bas d'une page : les
  figures « flottent », c'est normal, on les référence par `\cref{fig:demo}`
  au lieu de dire « ci-dessous » ;
- créer un dossier `figures/` dans le projet Overleaf et y déposer les images
  (celles de `data/outputs/`, **plaques et visages floutés**) ;
- la légende doit rendre la figure compréhensible seule, sans lire le corps du
  texte ;
- pour les courbes d'entraînement, exporter en PDF depuis matplotlib
  (`plt.savefig("courbe.pdf")`) : vectoriel, net à toute échelle.

## 8. Tableaux

Le tableau comparatif, livrable central du projet, avec `booktabs` (jamais
de barres verticales, des filets horizontaux seulement) :

```latex
\begin{table}[tb]
  \centering
  \caption{Comparaison sur le jeu de test entre YOLOv8n pré-entraîné (COCO) et
  fine-tuné. Le rappel par taille suit la convention COCO, aires ramenées à la
  résolution d'entrée du réseau ($640$ px).}
  \label{tab:comparaison}
  \begin{tabular}{lccc}
    \toprule
    Métrique & Pré-entraîné & Fine-tuné & Écart \\
    \midrule
    mAP@0,5          & 0,512 & 0,612 & $+19{,}5\,\%$ \\
    mAP@0,5:0,95     & 0,318 & 0,401 & $+26{,}1\,\%$ \\
    Précision        & 0,671 & 0,712 & $+6{,}1\,\%$  \\
    Rappel           & 0,540 & 0,633 & $+17{,}2\,\%$ \\
    F1-score         & 0,598 & 0,670 & $+12{,}0\,\%$ \\
    \midrule
    Rappel objets petits & 0,201 & 0,342 & $\mathbf{+70{,}1\,\%}$ \\
    Rappel objets moyens & 0,486 & 0,571 & $+17{,}5\,\%$ \\
    Rappel objets grands & 0,742 & 0,760 & $+2{,}4\,\%$  \\
    \bottomrule
  \end{tabular}
\end{table}
```

(Chiffres fictifs à remplacer par ceux de `results/comparaison.csv` ; le
script `compare_models.py` produit exactement ces lignes.) Mettre en gras la
ligne qui répond à la problématique, ici le rappel des petits objets.

En français, la virgule décimale s'obtient par `0{,}5` en mode mathématique
(les accolades évitent l'espace parasite après la virgule), ou automatiquement
avec `siunitx` : `\num{0.512}` s'affiche « 0,512 » grâce à babel.

## 9. Références croisées

```latex
\section{Méthodologie}\label{sec:methode}
...
\begin{equation}...\label{eq:iou}\end{equation}
...

Comme décrit en \cref{sec:methode}, ... le \cref{tab:comparaison} montre...
la \cref{fig:demo} illustre... selon l'\cref{eq:iou}...
```

`cleveref` (`\cref`) ajoute lui-même le mot « figure », « tableau »,
« section » : cohérent et toujours à jour. Compiler **deux fois** si des `??`
apparaissent : la première passe collecte les numéros, la seconde les insère
(Overleaf gère cela seul en général).

## 10. Bibliographie

Créer un fichier `references.bib` dans le projet Overleaf :

```bibtex
@misc{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title  = {Ultralytics {YOLOv8}},
  year   = {2023},
  url    = {https://github.com/ultralytics/ultralytics},
}

@article{redmon2016yolo,
  author  = {Redmon, Joseph and Divvala, Santosh and Girshick, Ross and
             Farhadi, Ali},
  title   = {You Only Look Once: Unified, Real-Time Object Detection},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition
             (CVPR)},
  year    = {2016},
}

@article{lin2014coco,
  author  = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  title   = {Microsoft {COCO}: Common Objects in Context},
  journal = {European Conference on Computer Vision (ECCV)},
  year    = {2014},
}
```

(Vérifier et compléter l'entrée du jeu de données retenu : auteurs et licence
figurent sur sa page Hugging Face ; le README §11 rappelle que la licence doit
être citée.)

Dans le document :

```latex
Nous utilisons YOLOv8~\cite{yolov8_ultralytics}, héritier de l'architecture
YOLO~\cite{redmon2016yolo}, pré-entraîné sur COCO~\cite{lin2014coco}.
...
\bibliographystyle{plain}      % ou abbrv ; en fin de document
\bibliography{references}
```

Le `~` avant `\cite` est une espace insécable : la référence ne part jamais
seule à la ligne. Règle d'or : **toute affirmation non triviale qui ne vient
pas de vos expériences porte une citation.**

## 11. Trame complète

Trame prête à coller dans Overleaf (`main.tex`), à compléter :

```latex
\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{microtype}
\usepackage{amsmath, amssymb}
\usepackage{siunitx}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage[hidelinks]{hyperref}
\usepackage[french,nameinlink]{cleveref}

\title{Détection des véhicules éloignés, de petite taille et dans les
       embouteillages : amélioration de YOLOv8 appliquée au contexte béninois}
\author{Aïchatou Traore \and Benoît Djossou \and Andréa Afouda\\[4pt]
        \small AMA, Cohorte 2, Projet Intégrateur 1, Groupe 5}
\date{Août 2026}

\begin{document}
\maketitle

\begin{abstract}
% 150-250 mots, à écrire EN DERNIER : contexte (1-2 phrases), problème,
% méthode, résultat principal chiffré, conclusion.
\end{abstract}

\section{Introduction}\label{sec:intro}
% Contexte : trafic urbain au Bénin, enjeux de la vidéoprotection.
% Problème : YOLO perd en rappel sur les véhicules petits/éloignés/masqués.
% Contributions : (1) protocole de comparaison à classes alignées COCO,
% (2) fine-tuning sur trafic urbain dense, (3) évaluation ventilée par taille.

\section{Travaux connexes}\label{sec:related}
% YOLO et ses versions ~\cite{redmon2016yolo, yolov8_ultralytics}.
% Détection de petits objets. Jeux de données de trafic existants et
% absence de jeu béninois (limite assumée, cf. README §3).

\section{Méthodologie}\label{sec:methode}
\subsection{Jeu de données}
% Source, licence, effectifs par split et par classe (tableau produit par
% import_dataset.py), remappage vers 4 classes alignées COCO.
\subsection{Modèle et entraînement}
% YOLOv8n, hyperparamètres (reprendre configs/entrainement.yaml),
% augmentation orientée petits objets (mosaic, scale).
\subsection{Protocole d'évaluation}
% Même jeu de test pour les deux modèles, réindexation des classes,
% métriques (\cref{eq:precision-rappel}), rappel par taille avec aires
% ramenées à la résolution d'entrée (640 px).
\begin{equation}
  \text{Précision} = \frac{VP}{VP+FP}, \qquad
  \text{Rappel} = \frac{VP}{VP+FN}
  \label{eq:precision-rappel}
\end{equation}

\section{Résultats}\label{sec:resultats}
% \cref{tab:comparaison} : tableau de compare_models.py.
% Figure côte à côte (\cref{fig:demo}).
% AUCUNE interprétation ici : les faits chiffrés seulement.

\section{Discussion}\label{sec:discussion}
% Interprétation : laquelle des trois lectures (README §10) s'applique ?
% Limites : domaine indien vs béninois, taille du sous-ensemble, yolov8n.

\section{Conclusion}\label{sec:conclusion}
% Apport en une phrase, résultat clé, perspectives (données locales,
% modèles plus grands, suivi multi-objets).

\bibliographystyle{plain}
\bibliography{references}
\end{document}
```

## 12. Conseils de rédaction

- **Écrire l'article dans cet ordre** : Méthodologie (pendant les
  expériences), Résultats, Discussion, Introduction, Résumé, Titre. Le résumé
  en dernier, toujours.
- **Une idée par paragraphe**, annoncée par sa première phrase.
- **Chiffrer chaque affirmation** : pas « le modèle s'améliore nettement »
  mais « le rappel des objets petits passe de 0,20 à 0,34 (+70 %) ».
- **Reproductibilité** : la méthodologie doit permettre à un lecteur de
  refaire l'expérience : versions (YOLOv8n, ultralytics 8.4), graine (42),
  hyperparamètres, effectifs des splits. Le projet est bien outillé pour ça :
  tout est dans les configs versionnées.
- **Assumer les limites** : un jeu indien faute de jeu béninois, ce n'est pas
  un défaut caché mais une limite documentée + une perspective. Un résultat
  mesuré et expliqué vaut mieux qu'un résultat impressionnant sans protocole
  (README §10 ; cette phrase peut presque aller telle quelle en discussion).
- **Temps et voix** : présent pour ce qui est vrai en général, passé composé
  pour ce qui a été fait ; limiter le « nous avons » répétitif en variant les
  tournures.
- Ne jamais coller de capture d'écran de tableau ou d'équation : tout en
  LaTeX natif.

## 13. Collaboration

Sur Overleaf, *Share* → inviter les deux coéquipiers par e-mail (l'offre
gratuite limite le nombre d'éditeurs simultanés ; sinon, partager le « link
sharing » en édition).

Organisation efficace à trois :

- un fichier par section (`sections/intro.tex`, `sections/methode.tex`, …)
  inclus dans `main.tex` par `\input{sections/intro}` : chacun travaille dans
  son fichier, aucun conflit ;
- les commentaires `%` servent de TODO (`% TODO(Benoît) : chiffres finaux`) ;
- l'historique Overleaf (*History*) permet de revenir en arrière ;
- relecture croisée : chaque section est relue par quelqu'un qui ne l'a pas
  écrite.

## 14. Check-list

- [ ] Le résumé contient un résultat **chiffré**.
- [ ] Chaque figure et chaque tableau est référencé dans le texte (`\cref`),
      a une légende autosuffisante, et aucun n'est orphelin.
- [ ] Le tableau comparatif reprend **exactement** `results/comparaison.csv`.
- [ ] Plaques et visages floutés sur toutes les images.
- [ ] Licence du jeu de données citée (README §11).
- [ ] Toutes les citations compilent sans `??` ni avertissement.
- [ ] Graine, versions et hyperparamètres indiqués dans la méthodologie.
- [ ] Les limites sont énoncées dans la discussion.
- [ ] Compilation propre : zéro erreur, avertissements examinés.
- [ ] Relecture complète par les trois membres.

## 15. LaTeX en général

L'article n'est qu'un des documents que LaTeX sait produire. Cette section
est un cours d'usage général : rapports longs, mémoires, présentations,
code source, schémas, et de quoi travailler en dehors d'Overleaf. De quoi
réutiliser l'outil pendant toute la formation et après.

### 15.1 Les classes de documents

La première ligne du fichier décide de tout :

| Classe | Usage | Particularités |
|---|---|---|
| `article` | Articles, comptes rendus, devoirs | Pas de chapitres, le plus courant |
| `report` | Rapports de stage, mémoires | `\chapter{}`, page de titre séparée |
| `book` | Livres, thèses longues | Recto-verso, `\frontmatter`/`\mainmatter` |
| `beamer` | Présentations (diapositives) | Voir 15.3 |
| `letter` | Courriers | Adresses, formules de politesse |

Presque tout ce qui est vu dans ce guide (préambule, figures, tableaux,
bibliographie) fonctionne à l'identique dans toutes les classes.

### 15.2 Un document long : rapport de stage ou mémoire

Deux nouveautés par rapport à l'article : les chapitres et le découpage en
fichiers.

```latex
\documentclass[a4paper,12pt]{report}
% ... même préambule que l'article ...

\begin{document}

\begin{titlepage}
  \centering
  {\LARGE Université d'Abomey-Calavi\par}
  \vspace{3cm}
  {\Huge\bfseries Titre du mémoire\par}
  \vspace{2cm}
  {\Large Présenté par Prénom NOM\par}
  \vfill
  {\large Année académique 2026-2027\par}
\end{titlepage}

\tableofcontents      % table des matières, générée automatiquement
\listoffigures        % table des figures (optionnelle)
\listoftables         % table des tableaux (optionnelle)

\input{chapitres/introduction}
\input{chapitres/etat_de_l_art}
\input{chapitres/methodologie}
\input{chapitres/resultats}
\input{chapitres/conclusion}

\bibliographystyle{plain}
\bibliography{references}
\appendix             % les \chapter suivants deviennent Annexe A, B...
\input{chapitres/annexes}
\end{document}
```

Chaque fichier de `chapitres/` commence par son `\chapter{Titre}` et ne
contient ni préambule ni `\begin{document}` : `\input` insère son contenu
tel quel. Mêmes bénéfices que pour l'article à trois : un fichier par
chapitre, aucun conflit d'édition.

La hiérarchie complète : `\chapter` puis `\section`, `\subsection`,
`\subsubsection`. La table des matières se met à jour seule à la
compilation.

### 15.3 Présentations avec Beamer

Beamer produit des diapositives PDF : mêmes équations, mêmes références,
même bibliographie que l'article, ce qui en fait l'outil naturel pour la
soutenance d'un travail écrit en LaTeX.

```latex
\documentclass{beamer}
\usetheme{Madrid}               % essayer aussi Berlin, Frankfurt, metropolis
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}

\title{Détection des véhicules éloignés et de petite taille}
\author{Groupe 5}
\institute{AMA, Cohorte 2}
\date{Août 2026}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{Sommaire}
  \tableofcontents
\end{frame}

\section{Problématique}
\begin{frame}{Pourquoi YOLO échoue sur les petits véhicules}
  \begin{itemize}
    \item les véhicules éloignés occupent peu de pixels ;
    \item les embouteillages créent des chevauchements ;
    \item<2-> ce point n'apparaît qu'au deuxième clic.
  \end{itemize}
\end{frame}

\begin{frame}{Résultats}
  \begin{columns}
    \column{0.5\linewidth}
      \includegraphics[width=\linewidth]{figures/avant.jpg}
    \column{0.5\linewidth}
      \includegraphics[width=\linewidth]{figures/apres.jpg}
  \end{columns}
\end{frame}

\end{document}
```

L'unité est la `frame` (une diapositive). La notation `<2->` fait apparaître
un élément progressivement. Conseil sobriété : un message par diapositive,
peu de texte, et résister aux animations.

### 15.4 Insérer du code source

Pour un rapport technique contenant du Python, le package `listings` :

```latex
\usepackage{listings}
\usepackage{xcolor}
\lstset{
  language=Python,
  basicstyle=\ttfamily\small,
  keywordstyle=\color{blue},
  commentstyle=\color{gray},
  numbers=left, numberstyle=\tiny,
  frame=single, breaklines=true,
  showstringspaces=false,
}

\begin{lstlisting}[caption={Chargement du modèle}, label={lst:chargement}]
from ultralytics import YOLO
modele = YOLO("models/yolov8n.pt")
\end{lstlisting}
```

Alternative plus jolie : `minted` (coloration par Pygments), qui demande
d'activer une option de compilation ; sur Overleaf, il fonctionne
directement. Pour quelques lignes dans une phrase, `\verb|code|` ou
`\texttt{code}` suffisent.

### 15.5 Théorèmes et définitions

Utile dans tout document mathématique (package `amsthm`) :

```latex
\usepackage{amsthm}
\newtheorem{theoreme}{Théorème}[section]   % numéroté par section
\newtheorem{definition}[theoreme]{Définition}
\newtheorem{proposition}[theoreme]{Proposition}

\begin{definition}[IoU]\label{def:iou}
  Soient $A$ et $B$ deux régions du plan. L'intersection sur union est
  $\mathrm{IoU}(A,B) = |A \cap B| / |A \cup B|$.
\end{definition}

\begin{proof}
  Le raisonnement... le carré de fin de preuve est automatique.
\end{proof}
```

### 15.6 Schémas avec TikZ

TikZ dessine des schémas vectoriels directement dans le document :
architectures de réseaux, organigrammes, chaînes de traitement. Un exemple
minimal, une chaîne de traitement :

```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}

\begin{tikzpicture}[
  bloc/.style={rectangle, draw, rounded corners, minimum height=2.5em,
               minimum width=6em, align=center},
  fleche/.style={-{Stealth}, thick},
]
  \node[bloc] (donnees)  {Jeu de\\données};
  \node[bloc, right=of donnees] (import) {Import et\\remappage};
  \node[bloc, right=of import]  (train)  {Fine-tuning\\YOLOv8};
  \node[bloc, right=of train]   (eval)   {Évaluation\\comparée};
  \draw[fleche] (donnees) -- (import);
  \draw[fleche] (import) -- (train);
  \draw[fleche] (train) -- (eval);
\end{tikzpicture}
```

TikZ est profond (il existe des livres entiers) ; le bon réflexe est de
partir d'un exemple proche sur <https://texample.net> et de l'adapter.
Alternative pragmatique : dessiner ailleurs (draw.io, Inkscape) et exporter
en PDF pour `\includegraphics`.

### 15.7 Travailler en local, comprendre les erreurs

Sans Overleaf : installer une distribution TeX ([TeX Live](https://www.tug.org/texlive/)
sur Windows et Linux, MacTeX sur macOS), puis un éditeur (VS Code avec
l'extension *LaTeX Workshop*, ou TeXstudio). La commande qui compile tout
dans le bon ordre (LaTeX, bibliographie, références, autant de passes que
nécessaire) :

```bash
latexmk -pdf main.tex
```

Les erreurs les plus courantes, dans tout environnement :

| Message | Cause probable | Remède |
|---|---|---|
| `Undefined control sequence` | Commande inconnue ou package manquant | Vérifier l'orthographe, charger le package |
| `Missing $ inserted` | Symbole mathématique hors mode math | Entourer de `$...$` |
| `File not found` | Image ou fichier mal nommé | Vérifier chemin et extension |
| Référence affichée `??` | Labels pas encore résolus | Compiler une seconde fois |
| `Overfull \hbox` | Ligne qui dépasse la marge (avertissement) | Souvent bénin ; reformuler ou couper un mot long |
| `! LaTeX Error: \begin{itemize} ended by \end{document}` | Environnement non fermé | Chercher le `\end{...}` manquant |

Méthode de débogage : la première erreur de la liste est la vraie, les
suivantes en découlent souvent ; corriger de haut en bas. Si le document ne
compile plus après un gros ajout, commenter la moitié du texte ajouté
(`%`), compiler, et resserrer par dichotomie.

### 15.8 Ressources pour progresser

1. [Overleaf Learn](https://www.overleaf.com/learn) : le meilleur cours en
   ligne, une page claire par sujet (en anglais).
2. [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX) : référence libre
   et complète.
3. [TeX StackExchange](https://tex.stackexchange.com) : à peu près toute
   question de mise en forme y a déjà sa réponse.
4. [Detexify](https://detexify.kirelabs.org) : dessiner un symbole à la
   souris pour retrouver sa commande.
5. Les gabarits Overleaf (*Templates*) : mémoires, CV, articles IEEE ou
   ACM, posters scientifiques ; partir d'un gabarit éprouvé plutôt que
   d'une page blanche.

---

Pour aller plus loin : le [guide Overleaf](https://www.overleaf.com/learn)
(excellent, en anglais) couvre chaque commande en détail.
