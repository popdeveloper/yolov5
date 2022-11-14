from fastapi import FastAPI,Request #,BackgroundTasks
import uvicorn

app = FastAPI()
from PIL import Image

from typing import Optional
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from starlette.requests import Request
import cv2 
import torch
import numpy as np
import io
def bytestoCV2(image_data):
    np_arr = np.asarray(bytearray(image_data),dtype=np.uint8)
    return cv2.imdecode(np_arr,cv2.IMREAD_COLOR)


@app.post('/analytic')
async def post_analytic(request:Request, devid:Optional[str] = "1", entry:Optional[str]="general"):
    body = await request.body()
    frame = bytestoCV2(body)
    print (body)
    try:
        img = Image.open(io.BytesIO(body))
    except Exception as ex:
        print (ex)
        pass
    
    

    results = model(img, size=640)  # reduce size=320 for faster inference
    return results.pandas().xyxy[0].to_json(orient="records")

    
  

if __name__ == '__main__':
    global_port = 8080
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    
    try:
        #uvicorn.run("processor_api:app", host="0.0.0.0", port=global_port,reload=True)
        uvicorn.run(app, host="0.0.0.0", port=global_port+1 )
    except Exception as ex:
        print (ex)
        global_port = global_port +  1
        uvicorn.run(app, host="0.0.0.0", port=global_port)
        #uvicorn.run("processor_api:app", host="0.0.0.0", port=global_port)
