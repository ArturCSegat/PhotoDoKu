from flask import Flask, render_template, request, jsonify
import flask_cors
import cv2 as cv
import processer
import os

app = Flask(__name__)
flask_cors.CORS(app)

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    request.files["image"].save("img.png")
    img = cv.imread("img.png")
    # img = cv.flip(img, 1)
    cv.imwrite("img.png", img)
    img, board, squares = processer.process("img.png")
    img = processer.sudoku_to_img(img, board, squares)
    cv.imwrite("img.png", img)
    return '200'

if __name__ == '__main__':
    ssl_cert = os.path.join(os.getcwd(), 'ssl.cert')
    ssl_key = os.path.join(os.getcwd(), 'ssl.key')

    app.run(host='0.0.0.0', port=5000, ssl_context=(ssl_cert, ssl_key))

