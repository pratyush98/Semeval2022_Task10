# SemEval-2022: Structured Sentiment Analysis(Task 10)
The task details are mentioned in detail here https://semeval.github.io/SemEval2022/tasks

## Problem description

The task is to predict all structured sentiment graphs in a text (see the examples below). We can formalize this as finding all the opinion tuples *O* = *O*<sub>i</sub>,...,*O*<sub>n</sub> in a text. Each opinion *O*<sub>i</sub> is a tuple *(h, t, e, p)*

where *h* is a **holder** who expresses a **polarity** *p* towards a **target** *t* through a **sentiment expression** *e*, implicitly defining the relationships between the elements of a sentiment graph.

The two examples below (first in English, then in Basque) give a visual representation of these *sentiment graphs*.

![multilingual example](./multi_sent_graph.png)

## Subtask-1
### Monolingual
This track assumes that you train and test on the same language. Participants will need to submit results for seven datasets in five languages.

 The datasets can be found in the [data](./data) directory.

#### Data

| Dataset | Language | # sents | # holders | # targets | # expr. |
| --------| -------- | ------- | --------- | --------- | ------- |
| [NoReC_fine](https://aclanthology.org/2020.lrec-1.618/) | Norwegian | 11437 | 1128|8923 |11115 |
| [MultiBooked_eu](https://aclanthology.org/L18-1104/) | Basque |1521 |296 |1775 |2328 |
| [MultiBooked_ca](https://aclanthology.org/L18-1104/) | Catalan |1678 |235 |2336 |2756 |
| [OpeNER_es](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/4891) | Spanish |2057 |255 |3980 |4388 |
| [OpeNER_en](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/4891) | English |2494 |413 |3850 |4150 |
| [MPQA](http://mpqa.cs.pitt.edu/) | English | 10048 | 2279| 2452 | 2814 |
| [Darmstadt_unis](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448) | English | 2803 | 86 | 1119 | 1119 |

## Evaluation

The two subtasks will be evaluated separately. In both tasks, the evaluation will be based on [Sentiment Graph F<sub>1</sub>](https://arxiv.org/abs/2105.14504).
