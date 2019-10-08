from flask import Flask
from segnet.segnetTest import segnet
from unet.train_module import unet
app = Flask(__name__)
app.register_blueprint(segnet, url_prefix='/segnet')
app.register_blueprint(unet, url_prefix='/unet')


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
