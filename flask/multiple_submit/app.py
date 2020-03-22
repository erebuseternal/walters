from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
app = Flask(__name__)

bootstrap = Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    print(request.form)
    selection = request.form.get('selection')
    submit_values = ['Bread', 'and', 'Butter']
    return render_template('base.html', submit_values=submit_values, selection=selection)
    