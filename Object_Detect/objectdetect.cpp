#include "objectdetect.h"
#include "./ui_objectdetect.h"
#include <QDebug>
#include <QDateTime>

ObjectDetect::ObjectDetect(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ObjectDetect)
{
    ui->setupUi(this);

    // --- LİSTE GÖRÜNÜMÜNÜ DİJİTAL/TERMİNAL YAP ---
    ui->listWidget->setStyleSheet(
        "QListWidget {"
        "  background-color: #0d1117;"   // Koyu Arka Plan
        "  color: #33ff33;"              // Terminal Yeşili
        "  border: 1px solid #33ff33;"   // Yeşil Çerçeve
        "  border-radius: 6px;"
        "  font-family: 'Courier New', 'Consolas', monospace;"
        "  font-weight: bold;"
        "  font-size: 11px;"
        "  letter-spacing: 1px;"
        "}"
        "QScrollBar:vertical {"
        "  border: none;"
        "  background: #0d1117;"
        "  width: 8px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #33ff33;"
        "  min-height: 20px;"
        "  border-radius: 4px;"
        "}"
    );

    // FPS Sayacını Başlat
    fpsSayaci.start();

    // --- BAŞLANGIÇ AYARLARI ---

    // 1. Orijinal ikonu hafızaya al
    orjinalKayitIconu = ui->btnRecord->icon();

    // 2. İkon Boyutunu 60x60 Yap
    ui->btnRecord->setIconSize(QSize(60, 60));

    // 3. Kamera Çerçevesini Başlangıçta KOYU MAVİ Yap (#0000ff)
    ui->labelKamera->setStyleSheet("border: 2px solid #0000ff; background-color: #000000; border-radius: 6px;");

    // --- SİSTEM KURULUMLARI ---
    kayitTimer = new QTimer(this);
    connect(kayitTimer, &QTimer::timeout, this, &ObjectDetect::kayitAnimasyonuYap);

    // Sınıf listesini tanımla
    sinifListesi = {"Vehicle", "UAP", "UAI", "Person"};

    // Video ve Model Yolları
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

// --- ANA DÖNGÜ VE GÖRÜNTÜ GÜNCELLEME ---
void ObjectDetect::kameraGuncelle()
{
    // 1. TARİH VE SAATİ GÜNCELLE
    QDateTime anlikZaman = QDateTime::currentDateTime();
    ui->labelDate->setText(anlikZaman.toString("dd.MM.yyyy HH:mm:ss"));

    // 2. FPS HESAPLAMA VE LCD GÜNCELLEME
    qint64 gecenSure = fpsSayaci.elapsed();
    fpsSayaci.restart(); // Sayacı sıfırla

    double fps = 0.0;
    if (gecenSure > 0) {
        fps = 1000.0 / gecenSure;
    }

    // Label yerine LCD Ekrana sayıyı basıyoruz
    ui->lcdFPS->display(int(fps));

    // 3. KAMERADAN GÖRÜNTÜ AL
    cap >> frame;

    // Video bittiyse başa sar (Loop)
    if(frame.empty()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        return;
    }

    // 4. TESPİT YAP (Nesneleri Tanı ve Çiz)
    nesneleriTani(frame);

    // 5. TERMAL EFEKT (Eğer aktifse)
    if (termalModAktif) {
        cv::applyColorMap(frame, frame, cv::COLORMAP_JET);
    }

    // 6. VİDEO KAYIT MANTIĞI
    if (kayitAktifMi && videoYazici.isOpened()) {
        videoYazici.write(frame);
    }

    // 7. EKRANA BASMAK İÇİN RENK DÜZELTME (BGR -> RGB)
    cv::Mat displayFrame;
    cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2RGB);

    // 8. QImage OLUŞTUR VE GÖSTER
    qtImage = QImage((const unsigned char*) (displayFrame.data),
                     displayFrame.cols, displayFrame.rows,
                     displayFrame.step,
                     QImage::Format_RGB888);

    ui->labelKamera->setPixmap(QPixmap::fromImage(qtImage).scaled(
                                   ui->labelKamera->size(),
                                   Qt::IgnoreAspectRatio,
                                   Qt::SmoothTransformation));
}

// --- NESNE TANI VE ÇİZ ---
void ObjectDetect::nesneleriTani(cv::Mat &img) {
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    try {
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception &e) {
        qDebug() << "Net Forward Hatasi:" << e.what();
        return;
    } catch (const std::exception &e) {
        qDebug() << "Net Forward Hatasi (Std):" << e.what();
        return;
    }

    if (outputs.empty()) return;

    cv::Mat &output = outputs[0];
    if (output.dims > 2) {
        output = output.reshape(1, output.size[1]);
        cv::transpose(output, output);
    }

    float *data = (float *)output.data;
    int rows = output.rows;
    int dimensions = output.cols;
    bool isYolov8 = (dimensions == (sinifListesi.size() + 4));

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = img.cols / 640.0;
    float y_factor = img.rows / 640.0;

    for (int i = 0; i < rows; ++i) {
        float confidence = 0;
        int class_id_index = -1;

        if (isYolov8) {
            float *classes_scores = data + 4;
            cv::Mat scores(1, sinifListesi.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > 0.45) {
                confidence = maxClassScore;
                class_id_index = class_id.x;
            }
        }
        else {
            float factor_conf = data[4];
            if(factor_conf >= 0.45) {
                float *classes_scores = data + 5;
                cv::Mat scores(1, sinifListesi.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                if (maxClassScore > 0.45) {
                    confidence = maxClassScore * factor_conf;
                    class_id_index = class_id.x;
                }
            }
        }

        if (confidence >= 0.45 && class_id_index != -1) {
            float x = data[0]; float y = data[1]; float w = data[2]; float h = data[3];
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * x_factor);
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
            class_ids.push_back(class_id_index);
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.45, 0.45, nms_result);

    std::vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar(0, 255, 255)); // Sarı
    colors.push_back(cv::Scalar(0, 0, 255));   // Pembe
    colors.push_back(cv::Scalar(255, 0, 0));   // Mavi
    colors.push_back(cv::Scalar(0, 255, 0));   // Yeşil

    int anlikInsanSayisi = 0;
    int anlikAracSayisi = 0;

    for (int idx : nms_result) {
        cv::Rect box = boxes[idx];
        int currentClassId = (idx >= 0 && idx < class_ids.size()) ? class_ids[idx] : -1;

        if (currentClassId == 3) anlikInsanSayisi++;
        else if (currentClassId == 0) anlikAracSayisi++;

        cv::Scalar color = (currentClassId >= 0) ? colors[currentClassId % colors.size()] : cv::Scalar(255,255,255);
        cv::rectangle(img, box, color, 2);

        std::string className = (currentClassId >= 0 && currentClassId < sinifListesi.size()) ? sinifListesi[currentClassId] : "Bilinmeyen";
        std::string label = className + " %" + std::to_string((int)(confidences[idx]*100));

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(box.y, labelSize.height);
        cv::rectangle(img, cv::Point(box.x, top - labelSize.height), cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);
        cv::putText(img, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);

        QString logMesaji = QDateTime::currentDateTime().toString("[HH:mm:ss]") + " TESPIT: " + QString::fromStdString(className) + " (%" + QString::number((int)(confidences[idx]*100)) + ")";
        ui->listWidget->addItem(logMesaji);
        ui->listWidget->scrollToBottom();
        if(ui->listWidget->count() > 34) delete ui->listWidget->takeItem(0);
    }

    ui->lcdInsan->display(anlikInsanSayisi);
    ui->lcdArac->display(anlikAracSayisi);

    // HUD / Nişangah
    int cx = img.cols / 2; int cy = img.rows / 2;
    int gap = 20; int len = 40; cv::Scalar hudColor(0, 0, 255); int kalinlik = 5;
    cv::line(img, cv::Point(cx - len - gap, cy), cv::Point(cx - gap, cy), hudColor, kalinlik);
    cv::line(img, cv::Point(cx + gap, cy), cv::Point(cx + len + gap, cy), hudColor, kalinlik);
    cv::line(img, cv::Point(cx, cy - len - gap), cv::Point(cx, cy - gap), hudColor, kalinlik);
    cv::line(img, cv::Point(cx, cy + gap), cv::Point(cx, cy + len + gap), hudColor, kalinlik);
    cv::circle(img, cv::Point(cx, cy), kalinlik, hudColor, -1);
}

// --- KAYIT BUTONU TIKLAMA ---
void ObjectDetect::on_btnRecord_clicked()
{
    if (kayitAktifMi == false) {
        // --- KAYDI BAŞLAT ---
        if (frame.empty()) return;

        QString zamanDamgasi = QDateTime::currentDateTime().toString("dd-MM-yyyy_HH-mm-ss");
        std::string dosyaAdi = "Kayit_" + zamanDamgasi.toStdString() + ".avi";

        videoYazici.open(dosyaAdi, cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(frame.cols, frame.rows), true);

        if (videoYazici.isOpened()) {
            kayitAktifMi = true;
            qDebug() << "Kayit Basladi:" << dosyaAdi.c_str();

            // Animasyonu Başlat
            kayitTimer->start(500);

            // KAMERA ÇERÇEVESİ -> KIRMIZI YAP (#ff0000)
            ui->labelKamera->setStyleSheet("border: 3px solid #ff0000; background-color: #000000; border-radius: 6px;");

        } else {
            qDebug() << "Hata: Video yazici acilamadi!";
        }
    }
    else {
        // --- KAYDI BİTİR ---
        kayitAktifMi = false;
        videoYazici.release();
        qDebug() << "Kayit Bitti.";

        // Animasyonu Durdur
        kayitTimer->stop();

        // 1. Buton Stilini Sıfırla
        ui->btnRecord->setStyleSheet("QPushButton { background-color: transparent; border: none; padding: 5px; }");

        // 2. Orijinal İkonu Geri Yükle
        ui->btnRecord->setIcon(orjinalKayitIconu);

        // 3. İKON BOYUTUNU TEKRAR 60x60 YAP
        ui->btnRecord->setIconSize(QSize(60, 60));

        // 4. KAMERA ÇERÇEVESİ -> MAVİYE GERİ DÖN (#0000ff)
        ui->labelKamera->setStyleSheet("border: 2px solid #0000ff; background-color: #000000; border-radius: 6px;");
    }
}

// --- KAYIT ANİMASYONU ---
void ObjectDetect::kayitAnimasyonuYap()
{
    animasyonDurumu = !animasyonDurumu;

    if (animasyonDurumu) {
        // PARLAK KIRMIZI
        ui->btnRecord->setStyleSheet("QPushButton { background-color: #ff0000; border: 2px solid #ffffff; border-radius: 50%; color: white; }");
    } else {
        // KOYU KIRMIZI
        ui->btnRecord->setStyleSheet("QPushButton { background-color: #440000; border: 2px solid #ff0000; border-radius: 50%; color: white; }");
    }
}

void ObjectDetect::on_labelKamera_linkActivated(const QString &link)
{
}

// --- FOTOĞRAF (SNAPSHOT) BUTONU ---
void ObjectDetect::on_snapshot_clicked()
{
    if (frame.empty()) return;

    QString zamanDamgasi = QDateTime::currentDateTime().toString("dd-MM-yyyy_HH-mm-ss");
    std::string dosyaAdi = "Foto_" + zamanDamgasi.toStdString() + ".jpg";

    if (cv::imwrite(dosyaAdi, frame)) {
        qDebug() << "Fotograf Kaydedildi:" << dosyaAdi.c_str();

        ui->snapshot->setStyleSheet("QPushButton { background-color: #ff0000; border-radius: 6px; border: 2px solid #ffffff; }");

        QTimer::singleShot(200, this, [this](){
            ui->snapshot->setStyleSheet("QPushButton { background-color: transparent; border: none; padding: 5px; }");
        });

        ui->listWidget->addItem("SNAPSHOT ALINDI: " + zamanDamgasi);
        ui->listWidget->scrollToBottom();
    }
}

// --- TERMAL (EO/IR) BUTONU ---
void ObjectDetect::on_btnEoir_clicked()
{
    termalModAktif = !termalModAktif;

    if (termalModAktif) {
        // AKTİF - KIRMIZI
        ui->btnEoir->setStyleSheet("QPushButton { background-color: #ff0000; border: 2px solid #ffffff; border-radius: 6px; }");
    }
    else {
        // KAPALI - ŞEFFAF
        ui->btnEoir->setStyleSheet("QPushButton { background-color: transparent; border: none; padding: 5px; }");
    }
}

// --- PENCERE BOYUTLANDIRMA (RESIZE EVENT) ---
/*
void ObjectDetect::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    // [DÜZELTME] 'centralWidget' kullanarak gerçek iç alanı ölçüyoruz.
    // Böylece alt kısım kesilmiyor.
    int W = ui->centralwidget->width();
    int H = ui->centralwidget->height();

    int ustPanelYuksekligi = 120; // Üstteki butonların sığması için ayrılan pay
    int sagListeGenisligi = 320;
    int bosluk = 10;

    // NOT: Üst panel (Frame) olmadığı için buradaki butonlar
    // Qt Designer'da nereye koyduysan (Sol Üst) orada kalacaklar.
    // Onlara dokunmuyoruz.

    // 1. LİSTEYİ YERLEŞTİR (Sağa Yapıştır)
    int listX = W - sagListeGenisligi - bosluk;
    int listY = ustPanelYuksekligi + bosluk;
    int listH = H - ustPanelYuksekligi - (bosluk * 2);

    if (listH < 10) listH = 10; // Çökmemesi için güvenlik

    ui->listWidget->setGeometry(listX, listY, sagListeGenisligi, listH);

    // 2. KAMERAYI YERLEŞTİR (Sol Alt, Kalan Alan)
    int kamX = bosluk;
    int kamY = ustPanelYuksekligi + bosluk;
    int kamW = listX - (bosluk * 2);
    int kamH = listH;

    if (kamW < 10) kamW = 10;

    ui->labelKamera->setGeometry(kamX, kamY, kamW, kamH);
}
*/
