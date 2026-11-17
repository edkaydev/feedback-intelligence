# Machine Learning From Zero — A Lesson for a Beginner

Imagine you are teaching a small robot to sort letters into three boxes:

| Box | Meaning | Example |
| --- | --- | --- |
| `Pos` | Happy customer | "Ndi musanyufu n'obuyambi bwa banka." |
| `Neg` | Unhappy customer | "ATM yamira kkaadi yange." |
| `Neutral` | Just information | "Njja kukozesa kiosk okwekenneenya ensimbi zange." |

That is the whole job. The robot reads a message and drops it in a box.
This job is called **text classification**.

---

## 1. The food for the robot: DATA

The robot learns from examples. Our examples live in a **CSV file**.
A CSV is like a spreadsheet saved as plain text. Each line is one **row**.
Each row has **columns** separated by commas.

Our main file is `datasets/Merged_All_Feedback.csv`.
One row looks like this (made smaller):

| record_id | channel | customer_text | sentiment |
| --- | --- | --- | --- |
| MF000001 | Bank Feedback | "Weebale, ensimbi zituuse." | Pos |

- A **row** is one example (one customer message).
- A **column** is one fact about that example.
- `customer_text` is the question we give the robot.
- `sentiment` is the correct answer we already know.

When the robot studies, we call the text **x** (input) and the answer **y** (output).
Programmers use x and y because of old maths habits. That is all.

---

## 2. Answers need tidy names: LABELS

Real data is messy. In the old research file, some answers were written
`pos`, some `Pos`, and some ` Pos` with a space. The robot would think
those are three different boxes!

So we clean every answer before training:

```python
str(value).strip().capitalize()   # " pos " -> "Pos"
```

- `str()` makes sure the value is text.
- `strip()` removes spaces at the start and end.
- `capitalize()` makes the first letter big.

Cleaning data is a big part of machine learning. Often bigger than the
learning itself.

---

## 3. Practice book and exam: TRAIN / TEST split

If you let the robot memorise the practice book, it will score 100% there
but fail in real life. So we split the examples:

- **Training set** (80%) = the practice book. The robot studies these.
- **Test set** (20%) = the sealed exam. The robot has never seen these.

We use `train_test_split(..., test_size=0.2)`.

Two extra tricks:

- `stratify=labels` keeps the same recipe in both parts. If the whole class
  has 50 Pos and 50 Neg, both practice and exam get roughly 50/50.
- `random_state=42` fixes the shuffle. Run the script again and you get the
  same split. This makes results repeatable, like a fair exam printed once.

**Rule: never train on the exam.** That is cheating, and it lies to you.

---

## 4. The robot cannot read words: FEATURES

The robot is made of maths. Maths needs numbers. We must turn each message
into a list of numbers. This is called **feature extraction**.

A **feature** is one measurable clue. For text, a clue might be:
"does the word `musanyufu` appear?" or "how often does the piece `bubi` appear?"

### TF-IDF, slowly

- **TF (Term Frequency)** = how many times a word appears in this message.
  If `bubi` appears twice, that is stronger than appearing once.
- **IDF (Inverse Document Frequency)** = how special the word is.
  A word in almost every message (`ku`, `ya`, `the`) is not helpful, so it
  gets a small score. A rare word (`yamira`, `sitaatimenti`) gets a big score.

TF-IDF = "how often here" × "how rare overall".

`sublinear_tf=True` changes the count to `1 + log(count)`. This stops a word
repeated 50 times from shouting too loudly.

`min_df=2` says: ignore words that appear in fewer than 2 messages.
Those are usually typos or one-off names.

---

## 5. Two pairs of eyes: WORD and CHAR n-grams

We join two feature makers with `FeatureUnion`.

**Eye 1 — word n-grams**
`ngram_range=(1, 2)` means we look at single words AND two-word pairs.
- single: `banka`, `ensimbi`
- pair: `ku ttabi`, `ya banka`

Pairs catch short phrases that carry meaning.

**Eye 2 — character n-grams**
`analyzer="char_wb"`, `ngram_range=(2, 5)` means pieces of 2–5 letters
inside each word.

Why bother? Look at Luganda:
- `yampadde` (helped me)
- `yakkirizza` (accepted for me)
- `yawulira` (felt)

They share little pieces like `ya...`. Character pieces let the robot notice
patterns even in a word it has never seen before. This is very useful for
languages that build long words by sticking parts together.

`FeatureUnion` simply puts the numbers from both eyes side by side.
More useful clues, better guessing.

---

## 6. The brain: LOGISTIC REGRESSION

Now the robot must learn from the numbers. We use **Logistic Regression**.

Think of every feature as a small voter:
- "I saw `musanyufu` — I vote Pos, strongly."
- "I saw `yamira` — I vote Neg, strongly."
- "I saw `ku` — I vote almost nothing."

The algorithm multiplies each feature by a **weight** (how trusted that
voter is), adds all the votes, and picks the box with the highest score.

- If `max_iter=2000` were smaller, the robot might stop studying too early.
- `C=2.0` controls how tightly it hugs the practice examples. Too tight and
  it memorises; too loose and it learns too little. This is called
  **regularisation**.
- `class_weight="balanced"` gives the tiny `Neutral` class a louder voice,
  so the robot does not just ignore it.

**Training** = `model.fit(x_train, y_train)` = "study and adjust the weights".
**Predicting** = `model.predict(new_text)` = "use the weights to choose a box".

---

## 7. One belt to hold it together: PIPELINE

```python
Pipeline([("features", features), ("classifier", classifier)])
```

A Pipeline chains steps in order: first make numbers, then guess the box.

Why is this lovely? When we save the Pipeline, we save the whole assembly:
the word rules, the character rules, AND the learned weights.
Later, one call to `predict()` does everything automatically. No steps to
forget, no mismatch between training and real use.

---

## 8. Marking the exam: SCORES

After the sealed exam, we compare the robot's answers with the real answers.

- **Accuracy** = right answers ÷ all answers.
  Easy to understand, but it can hide problems when classes are unequal.

- **Precision** = when the robot says "Pos", how often it is really Pos.
  (Did it cry wolf?)

- **Recall** = of all the real Pos messages, how many did it find?
  (Did it miss any?)

- **F1** = a blend of precision and recall.
- **Macro F1** = average F1 across all classes, giving each class equal say.
  We use this because `Neutral` has only 20 examples.

`classification_report` prints all of these in a neat table.

Our results:
- All 10,340 rows: accuracy about **0.96**, macro F1 about **0.83**.
- Bank feedback only (340 rows): accuracy about **0.87**.

The lower macro F1 is mostly the tiny `Neutral` class. That is a clue that
we could collect more neutral examples later.

---

## 9. Keeping the robot: SAVING

Training takes time, so we save the finished Pipeline:

```python
joblib.dump(model, "models/sentiment_model.joblib")
```

`joblib` freezes the Python object into a file. Later:

```python
joblib.load("models/sentiment_model.joblib")
```

brings it back, ready to classify without studying again.

We also save `models/sentiment_metrics.json` — a small note of the scores
and settings, so we remember what we did.

---

## 10. Running it (the actual commands)

```bash
python3 -m venv .venv            # make a private toolbox folder
. .venv/bin/activate             # step inside that folder
pip install -r requirements.txt  # put pandas + sklearn + joblib inside

python train_sentiment.py        # train on everything

python train_sentiment.py --bank-only \
  --predict "ATM ku Nakawa yamira kkaadi yange"
```

What the useful options mean:

| Option | What it does |
| --- | --- |
| `--data` | choose a different CSV |
| `--text-column` | classify a different column (try `luganda_text`) |
| `--bank-only` | drop the big research corpus |
| `--test-size 0.2` | change the exam size |
| `--seed 42` | change the random shuffle number |
| `--predict "..."` | classify one new message after training |

A virtual environment (`venv`) is just a private bag for installed packages,
so one project does not break another project.

---

## 11. Tiny glossary

| Word | Plain meaning |
| --- | --- |
| Dataset | A table of examples |
| Row / sample | One example |
| Feature | One clue turned into a number |
| Label | The correct answer |
| Train | Practise with answers |
| Test | Exam with hidden answers |
| Model | The program that learned |
| Fit | The act of learning |
| Predict | The act of guessing a label |
| TF-IDF | A way to score word importance |
| n-gram | A group of n words or letters |
| Pipeline | Several steps joined into one |
| Accuracy | Fraction of correct answers |
| Precision | Correct among those you picked |
| Recall | Found among all that existed |
| F1 | Balance of precision and recall |
| joblib | Saves and loads trained models |

---

## 12. One last idea

A model is not magic. It is a mirror of its examples.
If the examples are messy, the robot becomes messy.
If the examples are clean and varied, the robot becomes useful.

So the best machine-learning skill is not fancy maths.
It is looking carefully at your data.
