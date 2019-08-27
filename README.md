# AutoML auto-sklearn

Simple introductory project to automatic machine learning with auto-sklearn.

## Installation

I recommend you use `virtualenv`. If you have it installed, create a new environment and activate it:

```bash
$> virtualenv -p python3 venv
$> source venv/bin/activate
```

After that, just run the following commands:

```bash
$> curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
$> sudo apt-get install build-essential swig
$> pip install -r requirements.txt
```

## Try It

You can fire up an automatic search like this:

```bash
python train.py
```

Keep in mind this is a heavy and long running process.
