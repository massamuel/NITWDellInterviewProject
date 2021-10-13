from flask import Flask
app = Flask(__name__)

@app.route('/')
def show_logs():
    file = open('../log.txt')
    return file.read()


if __name__ == '__main__':
    app.run(debug=True)