import os
from gsv_main3 import GSVCapture
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
from flask import send_file
from flask_talisman import Talisman
from flask import Flask, request, jsonify


app = Flask(__name__, template_folder='web')

# Configure Content Security Policy
csp = {
    'default-src': '\'self\'',
    'style-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'https://cdn.jsdelivr.net',
        'https://fonts.googleapis.com',
        'https://unpkg.com'  # Allow CDN for Leaflet CSS
    ],
    'style-src-elem': [
        '\'self\'',
        '\'unsafe-inline\'',
        'https://cdn.jsdelivr.net',
        'https://fonts.googleapis.com',
        'https://unpkg.com'  # Allow CDN for Leaflet CSS
    ],
    'script-src': [
        '\'self\'',
        '\'nonce-random123\'',  # Add nonce for inline scripts
        'https://unpkg.com',   # Allow CDN for Leaflet and other resources
        'https://cdn.jsdelivr.net',  # Allow CDN for other resources
        'https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js',  # Add Axios CDN
        'https://cdn.jsdelivr.net/npm/chart.js',  # Add Chart.js CDN
        'https://cdnjs.cloudflare.com/ajax/libs/echarts/5.3.0/echarts.min.js'  # Add ECharts CDN
    ],
    'script-src-elem': [
        '\'self\'',
        '\'nonce-random123\'',  # Add nonce for inline scripts
        'https://unpkg.com',   # Allow CDN for Leaflet and other resources
        'https://cdn.jsdelivr.net',  # Allow CDN for other resources
        'https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js',  # Add Axios CDN
        'https://cdn.jsdelivr.net/npm/chart.js',  # Add Chart.js CDN
        'https://cdnjs.cloudflare.com/ajax/libs/echarts/5.3.0/echarts.min.js'  # Add ECharts CDN
    ],
    'img-src': [
        '\'self\'',
        'https://unpkg.com',
        'https://tiles.stadiamaps.com',
        'https://mt1.google.com',
        'https://server.arcgisonline.com',
        'https://api.mapbox.com',
        'data:'
    ],
    'connect-src': [
        '\'self\'',
        'https://api.waqi.info'  # Add API endpoint you want to connect to
    ]
}

talisman = Talisman(app, content_security_policy=csp)

gsv = GSVCapture()

conn_string = "dbname='gsv2svfnewnew' user='postgres' host='gsv_postgis' port='5432'  password='1234'"
conn_string1 = "dbname='gsv2svfnewnew' user='postgres' host='gsv_postgis' port='5432'  password='1234'"


@app.route("/gsv/api")
def api():
    """Simple API endpoint."""
    return jsonify({"hi": "hello"})


# หน้าปก
@app.route("/gsv/")
def homenew():
    """Render the new homepage."""
    return render_template('indexhomenew.html')

@app.route("/gsv/homenew")
def homeneww():
    """Render the new homepage."""
    return render_template('indexhomenew.html')

@app.route('/gsv/img/homenew1')
def homenew1():
    """Serve a specific image."""
    img_path = './homenew/1.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/img/homenew2')
def homenew2():
    """Serve a specific image."""
    img_path = './homenew/2.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/img/homenew3')
def homenew3():
    """Serve a specific image."""
    img_path = './homenew/3.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404
    
@app.route('/gsv/video')  # Route ที่ใช้สำหรับวิดีโอ
def homenewvido():
    """Serve a specific video file."""
    video_path = './homenew/Blue.mp4'  # เส้นทางของไฟล์วิดีโอ
    print(f"Looking for video at: {os.path.abspath(video_path)}")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return "Video not found", 404

# หน้า sumdeep
# @app.route("/gsv/sumdeep")
# def sumdeep():
#     """Render the new homepage."""
#     return render_template('indexsumdeep.html')

@app.route('/gsv/img/sumdeep1')
def sumdeep1():
    """Serve a specific image."""
    img_path = './homenew/1sumdeep.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/img/sumdeep2')
def sumdeep2():
    """Serve a specific image."""
    img_path = './homenew/2sumdeep.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/img/sumdeep3')
def sumdeep3():
    """Serve a specific image."""
    img_path = './homenew/3sumdeep.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

# หน้า sumco
@app.route("/gsv/sumco")
def sumco():
    """Render the new homepage."""
    return render_template('indexsumco.html')

# หน้า sumroute
@app.route("/gsv/sumroute")
def sumroute():
    """Render the new homepage."""
    return render_template('indexsumroute.html')

# หน้า deep SVF TVF BVF # api area
@app.route("/gsv/getByLatLong/")
def getByLatLong():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    res = gsv.getByLatLong(lat, lng)
    # print(res)
    return jsonify({"lat": lat, "lng": lng, "status": res})

@app.route('/gsv/getbylatlng/<lat>/<lng>/')
def getbylatlng(lat, lng):
    res = gsv.getByLatLong(lat, lng)
    return jsonify(res)

# SVF ขอบเขต
@app.route("/gsv/mapsvf")
def mapsvf():
    return render_template('indexmapsvf.html')

@app.route('/gsv/static1/<path:filename>')
def static1(filename):
    return send_from_directory('static1', filename)

# TVF ขอบเขต
# @app.route("/gsv/maptvf")
# def maptvf():
#     return render_template('indexmaptvf.html')

# @app.route('/gsv/static1/<path:filename>')
# def static2(filename):
#     return send_from_directory('static1', filename)

# BVF ขอบเขต
# @app.route("/gsv/mapbvf")
# def mapbvf():
#     return render_template('indexmapbvf.html')

# @app.route('/gsv/static1/<path:filename>')
# def static3(filename):
#     return send_from_directory('static1', filename)

# หน้า co SVF TVF BVF
@app.route("/gsv/svfco")
def svfco():
    return render_template('indexsvfco.html')

@app.route("/gsv/tvfco")
def tvfco():
    return render_template('indextvfco.html')

@app.route("/gsv/bvfco")
def bvfco():
    return render_template('indexbvfco.html')

# ข้อมูลเทศบาลนครเชียงใหม่
@app.route('/gsv/getstreeview/<week>/')
def getbyweek(week):
    query = f"SELECT * FROM gsv2svfnewnew{week};"
    with psycopg2.connect(conn_string) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return rows

@app.route('/gsv/imgweek/<int:week>')
def serve_image(week):
    img_folder = './homenew/imgweek'
    filename = f'week{week}.png'  # Adjust the filename format as needed
    filepath = os.path.join(img_folder, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        return f"Image for week {week} not found", 404


# หน้า route SVF TVF BVF
@app.route("/gsv/svfro")
def svfro():
    return render_template('indexsvfro.html')

@app.route("/gsv/tvfro")
def tvfro():
    return render_template('indextvfro.html')

@app.route("/gsv/bvfro")
def bvfro():
    return render_template('indexbvfro.html')

# ข้อมูลถนน4เส้น
@app.route('/gsv/getroute/<route>/')
def getbyroute(route):
    query = f"SELECT * FROM route{route};"
    with psycopg2.connect(conn_string1) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return rows

@app.route('/gsv/imgroute/<int:route>')
def serve_image1(route):
    img_folder = './homenew/imgroute'
    filename = f'route{route}.png'
    filepath = os.path.join(img_folder, filename)

    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        return f"Image for route {route} not found", 404

# หน้าโปลไฟล์
@app.route("/gsv/aboutnew")
def aboutnew():
    return render_template('indexaboutnew.html')

@app.route('/gsv/imgaboutnew/aboutnew1')
def aboutnew1():
    img_path = './homenew/earn.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/imgaboutnew/aboutnew2')
def aboutnew2():
    img_path = './homenew/earn1.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

# โลโก้
@app.route('/gsv/imggolo/logo')
def logo():
    img_path = './homenew/logo.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404
    
# manual
@app.route("/gsv/manual")
def manual():
    return render_template('indexmanual.html')

@app.route('/gsv/img1map/map1')
def map1():
    img_path = './homenew/1map.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/img1map/map2')
def map2():
    img_path = './homenew/2map.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404


@app.route('/gsv/img1map/map3')
def map3():
    img_path = './homenew/3map.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404


# aboutnew11
@app.route("/gsv/aboutnew11")
def aboutnew11():
    return render_template('indaxaboutnew1.html')

@app.route('/gsv/imgaboutnew1/aboutnew1')
def about1():
    img_path = './homenew/about1.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/gsv/imgaboutnew1/aboutnew2')
def about2():
    img_path = './homenew/about2.jpg'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
