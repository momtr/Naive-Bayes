# Naive-Bayes
> Classifier

## Output
The points in the figures below belong to different classes. The colored space is the prediction of the Naive Bayes model using the points as training data.

<img src="https://github.com/moritzmitterdorfer/Naive-Bayes/blob/master/img1.png" width="300">
<img src="https://github.com/moritzmitterdorfer/Naive-Bayes/blob/master/img2.png" width="300">
<br><br>

## How does it work?
Naive Bayes is an algorithm using Bayes' Theorem (see https://en.wikipedia.org/wiki/Bayes%27_theorem).
<br>
It is based on the following formula for propabilities:

```
P .. propability

P(A and B) = P(A) * P(A|B)
P(B and A) = P(B) * P(B|A)
P(A and B) == P(B and A)
P(A) * P(A|B) == P(B) * P(B|A)
P(B|A) = (P(A) * P(A|B)) / P(B)
```

Here, A and B can be exchanged by Features and Classes

```
A = { a1: 1, a2: 10 }
B = { class1, class2 }

```

Note, that the propability for a list of features is:

```
P(f1=value1, f2=value2) = P(f1=value1) * P(f2=value2)
```

Now, having the formula, we can just calculate the propabilities for a list of features being part of a certain class :)

```
P(class) ... number of all training examples / number of training examples belonging to that class
P(features|class) = P(feature_1|class) * ... * P(feature_n|class) 
P(feature_n|class) ... just use the gaussian bell curve
```

## Advantages
- Naive Bayes models do not have to be (extensively) trained on training data set
- Very fast with different columns
- Very fast at predicting data points (from the feature vector space) -> only a few calculations!
- Can handle many columns (linearly scalable!)


## Run

- Start a server and run `index.html`
- You can add points by pressing a, b and c on the keyboard and clicking on the canvas
- By pressing s on the keyboard, the algorithm starts processing
