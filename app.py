from flask import Flask
from segnet.segnetTest import segnet

app = Flask(__name__)
app.register_blueprint(segnet, url_prefix='/segnet')


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
