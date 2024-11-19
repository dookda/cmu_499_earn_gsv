import requests
import json
import os
import numpy as np
import scipy.misc
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from cv2 import cv2
import key
import torch
from ultralytics import YOLO
import base64
import psycopg2

GSV_API_URL = "https://maps.googleapis.com/maps/api/streetview"


class Panorama():

    def __init__(self):
        self.id = "0"
        self.panoid = ""
        self.lon = ""
        self.lat = ""
        self.date = ""
        self.svf = -1.0
        self.tvf = -1.0
        self.bvf = -1.0
        self.initialized = False

    def fromJSON(self, str):
        try:
            root = json.loads(str)
            if root['status'] != "OK":
                return False
            location = root['location']
            self.date = root['date']
            self.panoid = root['pano_id']
            self.lat = location['lat']
            self.lon = location['lng']
            self.initialized = True
            return True
        except ValueError:
            return False
        return False

    def fromLocation(self, lat, lon):
        url = GSV_API_URL + "/metadata?location=" + \
            str(lat) + "," + str(lon) + "&key=" + key.apikey
        print(url)
        try:
            response = requests.get(url)
            if response.status_code == requests.codes.ok:
                return self.fromJSON(response.content)
        except ValueError:
            return False
        return False

    def toString(self):
        return str(self.id) + "," + self.panoid + "," + self.date + "," + str(self.lat) + "," + str(self.lon) + "," + str(self.svf) + "," + str(self.tvf) + "," + str(self.bvf)


class GSVCapture():

    def __init__(self):
        # Initialize necessary attributes
        self.input_shape = (3, 512, 512)

    def hello(self):
        print("hello")

    def checkDir(self, dir):
        if not (dir.endswith('/') or dir.endswith('\\')):
            dir = dir + '/'
        return dir

    def getImage(self, panoId, x, y, zoom, outdir):
        url = "https://" + "geo0.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&panoid=" + \
            panoId + "&output=tile&x=" + \
            str(x) + "&y=" + str(y) + "&zoom=" + str(zoom) + "&nbt&fover=2"
        outfile = outdir + "/" + str(x) + "_" + str(y) + ".jpg"
        try:
            response = requests.get(url)
            if response.status_code == requests.codes.ok:
                file = BytesIO(response.content)
                return file
        except ValueError:
            return None
        return None

    def equirectangular2fisheye(self, infile, outfile):
        img = Image.open(infile)
        width, height = img.size
        img = img.crop((0, 0, width, height // 2))
        width, height = img.size
        red, green, blue = img.split()
        red = np.asarray(red)
        green = np.asarray(green)
        blue = np.asarray(blue)
        fisheye = np.ndarray(shape=(512, 512, 3), dtype=np.uint8)
        fisheye.fill(0)
        x = np.arange(0, 512, dtype=float)
        x = x / 511.0
        x = (x - 0.5) * 2
        x = np.tile(x, (512, 1))
        y = x.transpose()
        dist2ori = np.sqrt((y * y) + (x * x))

        zenithD = dist2ori * 90.0
        zenithD[np.where(zenithD <= 0.000000001)] = 0.000000001
        zenithR = zenithD * 3.1415926 / 180.0
        x2 = np.ndarray(shape=(512, 512), dtype=float)
        x2.fill(0.0)
        y2 = np.ndarray(shape=(512, 512), dtype=float)
        y2.fill(1.0)
        cosa = (x*x2 + y*y2) / np.sqrt((x*x + y*y) * (x2*x2 + y2*y2))
        lon = np.arccos(cosa) * 180.0 / 3.1415926
        indices = np.where(x > 0)
        lon[indices] = 360.0 - lon[indices]
        lon = 360.0 - lon
        lon = 1.0 - (lon / 360.0)
        outside = np.where(dist2ori > 1)
        lat = dist2ori
        srcx = (lon*(width-1)).astype(int)
        srcy = (lat*(height-1)).astype(int)
        srcy[np.where(srcy > 255)] = 0
        indices = (srcx + srcy*width).tolist()

        red = np.take(red, np.array(indices))
        green = np.take(green, np.array(indices))
        blue = np.take(blue, np.array(indices))
        red[outside] = 0
        green[outside] = 0
        blue[outside] = 0
        fisheye = np.dstack((red, green, blue))
        Image.fromarray(fisheye).save(outfile)
        return [-1, -1, -1]

    def classifyOld(self, infile, outfile):
        bestModelPath = 'best_120_950.pt'
        bestModel = YOLO(bestModelPath)

        results = bestModel.predict(source=infile, imgsz=640)
        annotatedImage = results[0].plot()
        annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)
        plt.imshow(annotatedImageRGB)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def showMasks(self, results, class_id=2, class_name='tree', ax=None, outfile='img'):

        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data

            clss = boxes[:, 5]
            people_indices = torch.where(clss == class_id)
            people_masks = masks[people_indices]
            people_mask = torch.any(people_masks, dim=0).int() * 255

            cnt = people_mask.cpu().numpy()
            count = np.sum(cnt > 0)

            img_name = outfile+'fisheye_classified_cls_' + \
                str(class_name) + '.png'
            cv2.imwrite(img_name, people_mask.cpu().numpy())

            # pixArr = {class_name: int(count)}
            if ax is not None:
                # ax[class_id +1].imshow(people_mask.cpu().numpy(), cmap='gray')
                # ax[class_id +1].axis('off')
                # ax[class_id +1].set_title(f'{class_name}: {count} px')
                continue

        return int(count)

    def classify(self, infile, outfile):
        modelPath = 'best_120_950.pt'
        model = YOLO(modelPath)
        results = model.predict(source=infile, imgsz=640)
        annotatedImage = results[0].plot()
        annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)

        names = model.names
        fig, ax = plt.subplots(1, len(names) + 1, figsize=(10, 10))
        # ax[0].imshow(annotatedImageRGB)
        # ax[0].axis('off')

        pixCnt = {}
        for n in names:
            pixCnt[names[n]] = self.showMasks(
                results, n, names[n], ax, outfile)

        # print(pixCnt)
        # plt.tight_layout()
        # plt.show()

        Image.fromarray(annotatedImageRGB).save(
            outfile + "fisheye_classified.png")
        return pixCnt

    def getByID(self, outdir, panoid):
        if panoid == '':
            return [-1, -1, -1]
        outdir = self.checkDir(outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        tilesize = 512
        numtilesx = 4
        numtilesy = 2
        mosaicxsize = tilesize*numtilesx
        mosaicysize = tilesize*numtilesy
        mosaic = Image.new("RGB", (mosaicxsize, mosaicysize), "black")
        blkpixels = 0
        for x in range(0, numtilesx):
            for y in range(0, numtilesy):
                imageTile = self.getImage(panoid, x, y, 2, outdir)
                if imageTile == None:
                    return ""
                img = Image.open(imageTile)
                if y == 1:
                    pix_val = list(img.getdata())
                    blk1 = pix_val[tilesize*tilesize-1]
                    blk2 = pix_val[tilesize*(tilesize-1)]
                    blkpixels = blkpixels + sum(blk1) + sum(blk2)
                mosaic.paste(img, (x*tilesize, y*tilesize, x *
                             tilesize+tilesize, y*tilesize+tilesize))
        xstart = (512 - 128) / 2
        xsize = mosaicxsize - xstart * 2
        ysize = mosaicysize - (512 - 320)
        if blkpixels == 0:
            mosaic = mosaic.crop((xstart, 0, xstart+xsize, ysize))
        mosaic = mosaic.resize((1024, 512))
        mosaic.save(outdir + "mosaic.png")
        self.equirectangular2fisheye(
            outdir + "mosaic.png", outdir + "fisheye.png")

        pixCut = self.classify(outdir + "fisheye.png", outdir)

        with open(outdir + "mosaic.png", "rb") as mosaic_file:
            pixCut["mosaic"] = base64.b64encode(
                mosaic_file.read()).decode('utf-8')

        with open(outdir + "fisheye.png", "rb") as fe_file:
            pixCut["fe"] = base64.b64encode(fe_file.read()).decode('utf-8')

        with open(outdir + "fisheye_classified.png", "rb") as fecls_file:
            pixCut["fe_cls"] = base64.b64encode(
                fecls_file.read()).decode('utf-8')

        return pixCut

    def insert_data(self, panoid, lat, lng, res, date):
        conn_string = "dbname='gsv2svfnewnew' user='postgres' host='postgis' port='5432' password='1234'"
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO testgsv (panoid, lat, lng, datetime, building, tree, sky, fe64, fe_cls64) VALUES ('{panoid}', {lat}, {lng}, '{date}', {res['building']}, {res['tree']}, {res['sky']}, '{res['fe']}', '{res['fe_cls']}');")
        return 'insert ok'

    def getByLatLong(self, lat, lon):

        outdir = "img"
        pano = Panorama()
        pano.fromLocation(lat, lon)
        if not pano.initialized:
            # print("Not available")
            return ""
        outdir = self.checkDir(outdir)
        outdir = outdir + pano.panoid + '/'
        res = self.getByID(outdir, pano.panoid)
        print(pano.panoid, pano.date)
        self.insert_data(pano.panoid, lat, lon, res, pano.date)
        return res

    def countPixels(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print('size:', image.size)
        count = np.sum(image > 0)
        return count
