# Melanoma Detection System using AlexNet

## Installation

**Python**
<pre>
python3 -m venv env
</pre>

**Installing python dependencies**
<pre>
pip install requirements.txt
</pre>

**Installing Elm**

[Install Elm using instructions](https://guide.elm-lang.org/install/elm.html)

## Training the model
<pre>
source env/bin/activate
python training.py
</pre>

## Building elm bundle
<pre>
bash build_elm.sh
</pre>

## Running webserver
<pre>
FLASK_APP=app.py FLASK_DEBUG=1 flask ru
</pre>

## Using the app

Go to http://localhost:5000/
