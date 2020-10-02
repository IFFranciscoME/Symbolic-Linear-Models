%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Harvard Data Science Review Latex Class       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hdsr}

\newcommand{\headlinecolor}{\normalcolor}
\LoadClass[11pt, hidelinks, oneside]{amsart}

\numberwithin{equation}{section}
\linespread{1.1}


% Packages

% Tools for editing ("to do" and strike through)
\usepackage{todonotes}
\usepackage{soul}

\usepackage[bottom=1.0in]{geometry}

\usepackage{float}
\usepackage[colorlinks=true, linkcolor=black, urlcolor=blue, citecolor=blue, anchorcolor=blue]{hyperref}
\usepackage{wrapfig}
\usepackage{comment}
\usepackage{textgreek}
\usepackage{url}
% \usepackage{fullpage}

\usepackage{enumitem, comment, xifthen}
\usepackage{amsmath,amssymb,amsthm, amsfonts}
\usepackage[T1]{fontenc}
\usepackage{makecell}
\usepackage{booktabs}
% \usepackage[natbibapa]{apacite}
% \usepackage{natbib}

\usepackage[style=apa, natbib]{biblatex}
% Ensure reference font is consistent with rest of paper
\renewcommand*{\bibfont}{\small}

\usepackage{subcaption}
% \usepackage[dvipsnames]{xcolor}

\DeclareMathOperator*{\E}{\bf E}

\addbibresource{references.bib}

\makeatletter
\long\def\@makecaption#1#2{
        \vskip 0.8ex
        \setbox\@tempboxa\hbox{\small {\bf #1.} #2}
        \parindent 1.5em 
        \dimen0=\hsize
        \advance\dimen0 by -3em
        \ifdim \wd\@tempboxa >\dimen0
                \hbox to \hsize{
                        \parindent 0em
                        \hfil 
                        \parbox{\dimen0}{\def\baselinestretch{0.96}\small
                                {\bf #1.} #2
                                } 
                        \hfil}
        \else \hbox to \hsize{\hfil \box\@tempboxa \hfil}
        \fi
        }
\makeatother
\usepackage{booktabs}
\usepackage{hhline}
\usepackage{array,multirow}
\usepackage{lipsum}
\usepackage{siunitx,etoolbox}

\interfootnotelinepenalty=1000
\usepackage{graphicx}

\usepackage{tikz}
\usepackage{lipsum}


%%%%%%%%%%%%%%%%%%%%%
% Custom commands   %
%%%%%%%%%%%%%%%%%%%%%

% Title Formatting
\let\origmaketitle\maketitle
\def\maketitle{
  \begingroup
  \def\uppercasenonmath##1{} % this disables uppercase title
  \let\MakeUppercase\relax % this disables uppercase authors
  \origmaketitle
  \endgroup
}


% Affiliation markers
\newcommand{\upstairs}[1]{\textsuperscript{#1}}
\newcommand{\affilone}{\dag}
\newcommand{\affiltwo}{\ddag}
\newcommand{\affilthree}{$\diamond$}

% Header with HDSR logo, to appear in upper left hand corner
\newcommand{\volumeheader}[3]{%
\begin{tikzpicture}[remember picture,overlay]
\setlength{\fboxrule}{0pt}%
  \node[anchor=north, yshift=-10pt, line width=0pt] at (current page.north) {\fbox{\parbox{\dimexpr\textwidth-\fboxsep-\fboxrule\relax}{  
  
        \includegraphics[scale=0.1]{logo} \\
        Volume #1 Issue #2 \\
        DOI: #3 \\ 
        ISSN: 2644-2353}}};
  \end{tikzpicture}% 
  \\
  \\

  }


% Author emails, to appear directly above copyright notice
\newcommand{\emails}[1]
{
\begin{tikzpicture}[remember picture,overlay]
  \node[anchor=south,xshift=0pt, yshift=65pt] at (current page.south) {
  \setlength{\fboxrule}{0pt}
  \fbox{\parbox{\dimexpr\textwidth-\fboxsep-\fboxrule\relax}{ 
    \footnotesize #1}
    }
  };
\end{tikzpicture}%

}


% Copyright notice, to appear at bottom of first page
\newcommand{\copyrightnotice}[1]{%
\begin{tikzpicture}[remember picture,overlay]
  \node[anchor=south, xshift=0pt, yshift=5pt] at (current page.south) {\fbox{\parbox{\dimexpr\textwidth-\fboxsep-\fboxrule\relax}{\footnotesize This article is \copyright{} \the\year{} by author(s) as listed above. The article is licensed under a Creative Commons Attribution (CC BY 4.0) International license (https://creativecommons.org/licenses/by/4.0/legalcode), except where otherwise indicated with respect to particular material included in the article. The article should be attributed to the author(s) identified above.}}};
\end{tikzpicture}%
}
