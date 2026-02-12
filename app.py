from flask import Flask, render_template, request
from ultralytics import YOLO
import os, uuid, cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("runs/detect/a/dis_eti_clean/weights/best.pt")

INFO = {
    "saglikli": (
        "Diş etleri sağlıklı görünüyor.",
        "Günde 2 kez fırçalama ve diş ipi yeterlidir.",
        False
    ),
    "hafif_gingivitis": (
        "Hafif düzeyde diş eti iltihabı başlangıcı tespit edildi.",
        "Ağız hijyenini artırın, 2–3 hafta izleyin.",
        False
    ),
    "ileri_gingivitis": (
        "İleri seviye diş eti iltihabı tespit edildi.",
        "Profesyonel diş hekimi kontrolü gerekir.",
        True
    ),
    "periodontitis": (
        "Kemik kaybı riski olan ciddi diş eti hastalığı tespit edildi.",
        "Acil diş hekimi müdahalesi gereklidir.",
        True
    ),
    "plak": (
        "Diş yüzeyinde plak birikimi tespit edildi.",
        "Düzenli fırçalama ve diş ipi önerilir.",
        False
    ),
    "tartar": (
        "Diş taşı oluşumu gözlemlendi.",
        "Diş taşı temizliği için diş hekimine gidilmelidir.",
        True
    ),
}

def dis_numarasi_tahmin(cene, yon):
    if cene == "Üst çene":
        return "11–13" if yon == "Sağ" else "21–23"
    else:
        return "41–43" if yon == "Sağ" else "31–33"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filename = f"{uuid.uuid4().hex}.jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        image = cv2.imread(upload_path)
        h, w = image.shape[:2]

        results = model(upload_path)[0]
        detections = []
        seen = set()

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.50:
                continue

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].replace(" ", "_")

            if class_name in seen:
                continue
            seen.add(class_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            cene = "Üst çene" if cy < h / 2 else "Alt çene"
            yon = "Sol" if cx < w / 2 else "Sağ"
            konum = f"{cene} – {yon} Ön dişler"
            dis_no = dis_numarasi_tahmin(cene, yon)

            aciklama, oneriler, uyari = INFO.get(
                class_name,
                ("Açıklama yok.", "Bir uzmana danışın.", True)
            )

            detections.append({
                "name": class_name,
                "conf": round(conf * 100, 1),
                "konum": konum,
                "dis_no": dis_no,
                "risk": "Yüksek Güven" if conf >= 0.85 else "Orta Güven",
                "aciklama": aciklama,
                "oneriler": oneriler,
                "uyari": uyari
            })

            # 🔲 SADECE BEYAZ ÇERÇEVE (RENK YOK)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        result_name = f"result_{filename}"
        cv2.imwrite(os.path.join(RESULT_FOLDER, result_name), image)

        return render_template(
            "result.html",
            image=result_name,
            detections=detections
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
