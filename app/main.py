# import all necessary 3rd python packages 
from flask import Flask, render_template, Response
# import modules.mysql_connection

from faceMeshProjectForFACE_OVAL import faceMeshDetection
# from app3 import faceMeshDetection

# from faceMeshProjectForFlask import faceMeshDetection_test
#
# Flask 類別 初始化時 傳入的 __name__ 參數，代表當前模組的名稱。
# 為固定用法，以便讓 Flask 知道在哪裡尋找資源。
# (例如: 模板和靜態文件)
app = Flask(__name__, '/')

# 裝飾器是告訴 Flask，哪個 URL 應該觸發我們的函式。
# 斜線代表的就是網站的根目錄，可以疊加。
# 例如: 新增一個 /hello 的位址
# @app.route("/")
# @app.route("/hello")
# def hello():
#     return "Hello, World!"
# 網站訪問首頁與/hello，呈現的同樣是 hello 函式，回傳的 Hello World 文字。
#render_template是flask的內建函數，字面上的意思大概是渲染模板，因此它需要一個『模板』來提供渲染的格式。
# 當return render_template('html文件')的時候，flask會先到專案資料夾『templates』去尋找相對應的html文件，
# 因此，你需要先做的就是在專案底下建置一個資料夾，並且命名為『templates』
@app.route('/')
def index():
    return render_template('faceDetection.html')

# feed a video stream as a source
@app.route('/video_feed')
def video_feed():
    # multipart/x-mixed-replace is an HTTP header. Your server can use it to push dynamically updated content to the web browser.
    # It works by telling the browser to keep the connection open and replace the web page or piece of media it is displaying with another when it receives a special token.
    # The header is old and widely supported. It works in legacy browsers.
    # Each yield expression is directly sent thru Response to the browser.
    videoMode = True
    filePath = "./videos/1-720p.mp4"
    drawFaceLms = True 
    drawID = False 
    drawFortuneTelling = False 
    return Response(faceMeshDetection(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    # mimetype 媒體類別 multipart/x-mixed-replace 資料傳輸格式

@app.route('/video_feed_1')
def video_feed_1():
    # multipart/x-mixed-replace is an HTTP header. Your server can use it to push dynamically updated content to the web browser.
    # It works by telling the browser to keep the connection open and replace the web page or piece of media it is displaying with another when it receives a special token.
    # The header is old and widely supported. It works in legacy browsers.
    # Each yield expression is directly sent thru Response to the browser.
    videoMode = True
    filePath = "./videos/1-720p.mp4"
    drawFaceLms = True 
    drawID = False 
    drawFortuneTelling = True     
    return Response(faceMeshDetection(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling),
                mimetype='multipart/x-mixed-replace; boundary=frame')

# 連接AZURE的mysql
@app.route('/try-mysql')
def try_mysql():
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    sql = ("SELECT * FROM new_table")
    cursor.execute(sql)
    return render_template('data_table.html', t_data=cursor.fetchall())

# @app.route('/video_feed_2/<int:style>')
# def video_feed_2(style):
#     # multipart/x-mixed-replace is an HTTP header. Your server can use it to push dynamically updated content to the web browser.
#     # It works by telling the browser to keep the connection open and replace the web page or piece of media it is displaying with another when it receives a special token.
#     # The header is old and widely supported. It works in legacy browsers.
#     # Each yield expression is directly sent thru Response to the browser.

#     videoMode = True
#     filePath = "./videos/1-720p.mp4"
#     drawFaceLms1 = True 
#     drawID = False 
#     drawFortuneTelling1 = False
#     # img1 = faceMeshDetection_test(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling)

#     videoMode = True
#     filePath = "./videos/1-720p.mp4"
#     drawFaceLms2 = False 
#     drawID = False 
#     drawFortuneTelling2 = True 
#     # img2 = faceMeshDetection_test(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling)

#     # img1, faces1, distance1, img2, faces2, distance2 = faceMeshDetection_test(videoMode, filePath, drawFaceLms1, drawFaceLms2, drawID, drawFortuneTelling1, drawFortuneTelling2)
#     img1 = faceMeshDetection_test(videoMode, filePath, drawFaceLms1, drawFaceLms2, drawID, drawFortuneTelling1, drawFortuneTelling2)
#     if style == 1:
#         return Response(img1,
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
        
#         return Response(img1,
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

'''
@app.route('/mysql')
def mysqll():
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    sql = ("SELECT * FROM address_book LIMIT 5")
    cursor.execute(sql)
    data = cursor.fetchall()
    # return str(data)
    # return jsonify(data)    
    return render_template('members.html', data = data)
'''
#
# end of program