from flask import Blueprint
from flask import request

segnet = Blueprint('segnet', __name__)


@segnet.route('/testget',methods=['GET'])
def testget():
    # param = request.form['param']
    return '调用get方法,获得参数'


@segnet.route('/testpost', methods=['POST'])
def testpost():
    data = request.get_data()
    param = request.form['param']
    print('data is ', data)
    return '调用post方法,获得参数' + param
