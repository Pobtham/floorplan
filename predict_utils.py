def detect(results):
    cords = []
    confs = []
    labels = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        class_map = result.names
        for b in boxes:
          ## single object info below
            conf = float(b.conf[0])
            class_raw = float(b.cls[0])
            class_mapped = class_map[class_raw]
            xyxy = b.xyxy.tolist()[0]
            xywh = b.xywh.tolist()[0]
            confs.append(conf)
            cords.append(xyxy)
            labels.append(class_mapped)
    return [labels,cords,confs]
