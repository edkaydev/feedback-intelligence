"""
Train a sentiment classifier on the merged Ugandan feedback dataset.

BIG PICTURE (read this first)
-----------------------------
We want the computer to read a customer message and say if it is:
  Pos    = happy / praise
  Neg    = unhappy / complaint
  Neutral = just information, no strong feeling

This is called TEXT CLASSIFICATION. "Classification" means "putting things
into named boxes". The boxes are called CLASSES or LABELS.

A computer cannot learn from raw feelings. It needs:
  1. examples (messages)  -> we call this the DATASET
  2. the right answer for each example -> we call this the LABEL
  3. a way to turn words into numbers -> we call this FEATURES
  4. a learning method -> we call this the MODEL or ALGORITHM
  5. a way to check if it learned well -> we call this EVALUATION

The steps below follow exactly that order.
"""

# "import" means: bring in someone else's ready-made code so we can use it.
# argparse  = reads options we type after the script name in the terminal.
# json      = writes a small human-readable results file.
# os        = helps create folders.
# unicodedata = makes funny-looking text characters consistent.
import argparse
import json
import os
import unicodedata

# joblib = saves a trained model to a file so we do not train again every time.
# pandas = reads CSV files into a table (like a spreadsheet) called a DataFrame.
import joblib
import pandas as pd

# scikit-learn (sklearn) = the machine-learning toolbox.
# TfidfVectorizer turns text into numbers.
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression is the learning algorithm we use.
from sklearn.linear_model import LogisticRegression

# metrics are the "exam marking" tools.
from sklearn.metrics import accuracy_score, classification_report, f1_score

# train_test_split splits our examples into practice set and exam set.
from sklearn.model_selection import train_test_split

# FeatureUnion joins two feature makers. Pipeline chains steps together.
from sklearn.pipeline import FeatureUnion, Pipeline


def normalize_label(value):
    """Clean one label so ' pos ', 'Pos', 'pos' all become 'Pos'.

    Real data is messy. The original research corpus had labels like
    ' Pos' (with a space) and 'pos' (small p). If we do not clean them,
    the computer thinks they are three different boxes.
    """
    # str() = make sure it is text. strip() = remove spaces at the ends.
    # capitalize() = first letter big, rest small: "pos" -> "Pos".
    return str(value).strip().capitalize()


def build_model(min_df):
    """Build the full learning machine, but do not train it yet.

    Think of this as assembling a toy robot from parts. The robot has
    two "eyes" (word eye and character eye), one "brain" (classifier),
    and a "belt" (Pipeline) that connects them in order.
    """

    # FEATURE EXTRACTION: the computer only understands numbers.
    # TF-IDF means "Term Frequency - Inverse Document Frequency".
    #   Term Frequency      = how often a word appears in THIS message.
    #   Inverse Document Frequency = rare words get more points than common
    #                          words like "the" or "and".
    # Example: "yamira" (swallowed) is more useful than "ku".
    features = FeatureUnion(
        [
            # EYE 1: look at whole words and pairs of words.
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",        # split text on whitespace/punctuation
                    ngram_range=(1, 2),     # single words AND two-word pairs
                    min_df=min_df,          # ignore words seen fewer than min_df times
                    sublinear_tf=True,      # use 1 + log(count) instead of raw count
                    strip_accents="unicode",  # treat é and e as the same idea
                ),
            ),
            # EYE 2: look at small pieces INSIDE words (2 to 5 letters).
            # Why? Luganda sticks parts together: "yampadde", "yakkirizza".
            # Character pieces help the robot notice shared endings like
            # "-bubi" (bad) or "musanyufu" (happy) even in new words.
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",     # char pieces inside word boundaries
                    ngram_range=(2, 5),     # pieces of 2, 3, 4 and 5 letters
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    # THE BRAIN: Logistic Regression.
    # It gives each feature a "vote weight" and adds the votes up.
    # If the total for "Pos" is biggest, the message is Pos.
    classifier = LogisticRegression(
        max_iter=2000,          # let it practice up to 2000 rounds
        class_weight="balanced",  # give rare classes (Neutral) a louder voice
        C=2.0,                  # how much we let it fit the training data
    )

    # PIPELINE: put the eyes first, then the brain, into one single object.
    # Good thing: later we save THIS ONE object and reuse it for predictions.
    return Pipeline([("features", features), ("classifier", classifier)])


def parse_args():
    """Read the options typed in the terminal.

    Example:
      python train_sentiment.py --bank-only --predict "Ndi musanyufu"

    Without options, defaults are used (see below).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="datasets/Merged_All_Feedback.csv",
        help="Path to the merged feedback CSV",
    )
    parser.add_argument(
        "--text-column",
        default="customer_text",
        help="Column containing the text to classify",
    )
    parser.add_argument("--label-column", default="sentiment")
    parser.add_argument("--test-size", type=float, default=0.2)  # 20% for the exam
    parser.add_argument("--seed", type=int, default=42)  # fixed random shuffle
    parser.add_argument(
        "--bank-only",
        action="store_true",
        help="Exclude the Makerere research corpus and train on bank feedback only",
    )
    parser.add_argument("--model-out", default="models/sentiment_model.joblib")
    parser.add_argument("--metrics-out", default="models/sentiment_metrics.json")
    parser.add_argument("--predict", help="Classify a single message after training")
    return parser.parse_args()


def main():
    """The whole lesson plan, from reading data to saving the model."""
    args = parse_args()

    # STEP 1: READ THE DATA.
    # A CSV is a plain text file with commas between columns, like a table.
    # dtype=str  -> keep everything as text (IDs like "007" must not become 7).
    # keep_default_na=False -> empty cells stay empty strings, not "NaN".
    frame = pd.read_csv(args.data, dtype=str, keep_default_na=False)

    # Check the columns we need really exist. If not, stop with a clear message.
    required = {args.text_column, args.label_column, "source_dataset"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(sorted(missing))}")

    # Optional: use only the bank feedback and drop the big research corpus.
    # "~" means NOT, so we keep rows whose source does NOT contain "Makerere".
    if args.bank_only:
        frame = frame[~frame["source_dataset"].str.contains("Makerere")]

    # STEP 2: CLEAN THE DATA.
    # map() applies a little function to every cell in the column.
    # NFC makes characters written two ways become one way.
    frame[args.text_column] = frame[args.text_column].map(
        lambda value: unicodedata.normalize("NFC", str(value).strip())
    )
    frame[args.label_column] = frame[args.label_column].map(normalize_label)

    # Throw away rows with no message or no answer. A teacher cannot learn
    # from a question with no answer, or an answer with no question.
    frame = frame[(frame[args.text_column] != "") & (frame[args.label_column] != "")]

    # Pull the two columns out as simple Python lists.
    # x = the questions (inputs). y = the answers (outputs).
    texts = frame[args.text_column].tolist()
    labels = frame[args.label_column].tolist()

    # Count how many examples of each label we have, just to display it.
    counts = pd.Series(labels).value_counts().to_dict()
    print(f"Rows used: {len(texts)}")
    print(f"Labels: {counts}")

    # STEP 3: SPLIT INTO PRACTICE AND EXAM.
    # 80% practice (training), 20% exam (testing).
    # "stratify" keeps the same mix of Pos/Neg/Neutral in both parts.
    # "random_state" makes the shuffle the same every run (repeatable science).
    stratify = labels if min(counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    # STEP 4: BUILD AND TRAIN.
    # If we have many examples, ignore words seen only once (likely typos).
    model = build_model(min_df=2 if len(x_train) >= 200 else 1)
    # fit() = "study these examples and their answers".
    model.fit(x_train, y_train)

    # STEP 5: TAKE THE EXAM.
    # The model has NEVER seen x_test before, so this is a fair test.
    predictions = model.predict(x_test)

    # Accuracy = how many answers were right, out of all answers.
    accuracy = accuracy_score(y_test, predictions)
    # Macro F1 = average score per class, treating small classes as important
    # as big ones. This matters because Neutral has very few examples.
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"\nTrain / test: {len(x_train)} / {len(x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    # classification_report shows precision, recall and F1 for each label.
    print("\n" + classification_report(y_test, predictions, zero_division=0))

    # STEP 6: SAVE THE BRAIN.
    # os.makedirs makes the "models" folder if it is not there yet.
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(model, args.model_out)  # freeze the trained model to a file

    # Also save a small report of what happened (numbers, not the model).
    metrics = {
        "data": args.data,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "bank_only": args.bank_only,
        "rows": len(texts),
        "labels": counts,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "model": args.model_out,
    }
    # json.dump writes the dictionary as readable text.
    with open(args.metrics_out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"Saved model -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")

    # STEP 7 (BONUS): CLASSIFY ONE NEW MESSAGE RIGHT NOW.
    # We load the saved model again to prove it really works from disk.
    if args.predict:
        loaded = joblib.load(args.model_out)
        # predict() gives the winning label.
        predicted = loaded.predict([args.predict])[0]
        # predict_proba() gives a confidence score for every label.
        probabilities = loaded.predict_proba([args.predict])[0]
        scores = {
            str(label): round(float(score), 4)
            for label, score in zip(loaded.classes_, probabilities)
        }
        print(f"\nInput: {args.predict}")
        print(f"Prediction: {predicted}")
        print(f"Scores: {scores}")


# In Python, this line means: "only run main() if we executed this file
# directly". It stops main() from running when another file imports us.
if __name__ == "__main__":
    main()
