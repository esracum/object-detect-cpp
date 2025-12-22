#include "objectdetect.h"
#include "./ui_objectdetect.h"
#include <QDebug>
#include <QDateTime>



ObjectDetect::ObjectDetect(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ObjectDetect)
{
    ui->setupUi(this);
    this->resize(1280, 800);
    sinifListesi = {"Vehicle", "UAP", "UAI", "Person"};
    cap.open("/home/esocan/Desktop/CPP/Object_Detect/simulasyon_videosu.mp4");
    std::string modelYolu = "/home/esocan/Desktop/CPP/Object_Detect/best.onnx";
    try {
            net = cv::dnn::readNetFromONNX(modelYolu);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            qDebug() << "Model basariyla yuklendi!";
        } catch (const cv::Exception& e) {
            qDebug() << "Model yuklenemedi! Hata:" << e.what();
        }

        if(cap.isOpened()) {
            timer = new QTimer(this);
            connect(timer, &QTimer::timeout, this, &ObjectDetect::kameraGuncelle);
            timer->start(30);
        }
}

ObjectDetect::~ObjectDetect()
{
    delete ui;
}
// --- YAPAY ZEKA TESPİT FONKSİYONU ---


void ObjectDetect::kameraGuncelle()
{
    qDebug() << "Kullanilan OpenCV Surumu:" << CV_VERSION;

    cap >> frame;

    // --- VİDEO İÇİN LOOP (DÖNGÜ) EKLEMESİ ---
    if(frame.empty()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    nesneleriTani(frame);

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    qtImage = QImage((const unsigned char*) (frame.data),
                     frame.cols, frame.rows,
                     frame.step,
                     QImage::Format_RGB888);

    ui->labelKamera->setPixmap(QPixmap::fromImage(qtImage).scaled(
                                   ui->labelKamera->size(),
                                   Qt::KeepAspectRatio,
                                   Qt::SmoothTransformation));
}
void ObjectDetect::nesneleriTani(cv::Mat &img) {
    cv::Mat blob;

    cv::dnn::blobFromImage(img, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);

    net.setInput(blob);

    std::vector<cv::Mat> outputs;

    // --- DÜZELTME: SADECE BURADA ÇAĞIRIYORUZ (TRY-CATCH İÇİNDE) ---
    try {
        // Çıktı katman isimlerini alıp forwarding yapıyoruz
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception &e) {
        qDebug() << "Net Forward Hatasi (OpenCV):" << e.what();
        return;
    } catch (const std::exception &e) {
        qDebug() << "Net Forward Hatasi (Std):" << e.what();
        return;
    }

    // Eğer çıktı boşsa işlem yapma (Crash önleyici)
    if (outputs.empty()) {
        qDebug() << "Hata: Model cikti uretmedi!";
        return;
    }

    // --- MODEL ÇIKTISINI İŞLEME ---
    cv::Mat &output = outputs[0];

    // YOLOv8 Format Kontrolü ve Dönüşümü (1 x 84 x 8400 -> 8400 x 84)
    if (output.dims > 2) {
        // 2 Boyutluya indir
        output = output.reshape(1, output.size[1]);
        // Transpose (Satır/Sütun yer değiştir)
        cv::transpose(output, output);
    }

    // Veri işaretçisi
    float *data = (float *)output.data;

    // Satır ve Sütun sayıları
    int rows = output.rows;
    int dimensions = output.cols;

    // YOLOv8 mi? (x, y, w, h + Sınıf Sayısı)
    bool isYolov8 = (dimensions == (sinifListesi.size() + 4));

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Görüntü ölçek çarpanları
    float x_factor = img.cols / 640.0;
    float y_factor = img.rows / 640.0;

    for (int i = 0; i < rows; ++i) {
        float confidence = 0;
        int class_id_index = -1;

        if (isYolov8) {
            // YOLOv8: İlk 4 veri [x,y,w,h], sonrakiler [class_scores...]
            float *classes_scores = data + 4;

            // En yüksek skoru bul
            cv::Mat scores(1, sinifListesi.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > 0.45) { // Eşik değer
                confidence = maxClassScore;
                class_id_index = class_id.x;
            }
        }
        else {
            // YOLOv5: [x, y, w, h, conf, class_scores...]
            // 5. eleman (index 4) confidence skorudur
            float factor_conf = data[4];
            if(factor_conf >= 0.45) {
                float *classes_scores = data + 5;
                cv::Mat scores(1, sinifListesi.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

                if (maxClassScore > 0.45) {
                    confidence = maxClassScore * factor_conf; // Daha hassas olması için çarpılabilir
                    class_id_index = class_id.x;
                }
            }
        }

        // Eğer geçerli bir tespit varsa kutuyu kaydet
        if (confidence >= 0.45 && class_id_index != -1) {
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * x_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
            class_ids.push_back(class_id_index);
        }

        // Bir sonraki satıra (tespite) geç
        data += dimensions;
    }

    // NMS (Çoklu kutuları eleme)
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, 0.45, 0.45, nms_result);

        // --- RENK LİSTESİ (BGR Formatında: Mavi, Yeşil, Kırmızı) ---
        std::vector<cv::Scalar> colors;
        colors.push_back(cv::Scalar(0, 255, 255)); // 0: Vehicle -> Sarı
        colors.push_back(cv::Scalar(130, 90, 255));   // 1: UAP -> Pembe
        colors.push_back(cv::Scalar(255, 0, 0));   // 2: UAI -> Mavi
        colors.push_back(cv::Scalar(0, 255, 0));   // 3: Person -> Yeşil

        for (int idx : nms_result) {
            cv::Rect box = boxes[idx];

            // Varsayılan renk
            cv::Scalar color = cv::Scalar(255, 255, 255);

            int currentClassId = -1;
            if(idx >= 0 && idx < class_ids.size()) {
                currentClassId = class_ids[idx];
                // ID'ye göre listeden renk seç (Mod alarak taşmayı önle)
                if (currentClassId >= 0) {
                    color = colors[currentClassId % colors.size()];
                }
            }

            cv::rectangle(img, box, color, 2);

            std::string className = "Bilinmeyen";
            if (currentClassId >= 0 && currentClassId < sinifListesi.size()) {
                className = sinifListesi[currentClassId];
            }

            std::string label = className + " %" + std::to_string((int)(confidences[idx]*100));

            // Etiket arka planı ve yazı
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(box.y, labelSize.height);

            // Arka planı da aynı renk yap (yazı okunması için)
            cv::rectangle(img, cv::Point(box.x, top - labelSize.height),
                          cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);

            // Yazı Rengi
            cv::putText(img, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            // --- TERMİNAL LOG SİSTEMİ ---

            // 1. Tarih ve Saati al (Örn: [14:35:12])
            QString zaman = QDateTime::currentDateTime().toString("[HH:mm:ss]");

            // 2. Mesajı Hazırla (Örn: [14:35:12] TESPIT: Personel (%85))
            QString logMesaji = zaman + " TESPIT: " + QString::fromStdString(className) + " (%" + QString::number((int)(confidences[idx]*100)) + ")";

            // 3. Listeye Ekle
            ui->listWidget->addItem(logMesaji);

            // 4. En alta otomatik kaydır (Terminal gibi aksın)
            ui->listWidget->scrollToBottom();

            // 5. RAM dolmasın diye eski kayıtları sil (Listede hep son 50 kayıt kalsın)
            if(ui->listWidget->count() > 50) {
                delete ui->listWidget->takeItem(0);
            }
        }

        // --- HUD / NİŞANGAH
        int cx = img.cols / 2;
        int cy = img.rows / 2;
        int gap = 20;
        int len = 40;

        cv::Scalar hudColor(0, 0, 255);

        int kalinlik = 5;

        cv::line(img, cv::Point(cx - len - gap, cy), cv::Point(cx - gap, cy), hudColor, kalinlik);
        cv::line(img, cv::Point(cx + gap, cy), cv::Point(cx + len + gap, cy), hudColor, kalinlik);
        cv::line(img, cv::Point(cx, cy - len - gap), cv::Point(cx, cy - gap), hudColor, kalinlik);
        cv::line(img, cv::Point(cx, cy + gap), cv::Point(cx, cy + len + gap), hudColor, kalinlik);
        cv::circle(img, cv::Point(cx, cy), kalinlik, hudColor, -1);
}

void ObjectDetect::on_labelKamera_linkActivated(const QString &link)
{

}

