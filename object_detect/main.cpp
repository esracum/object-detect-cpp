#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

// --- AYARLAR ---
// Grafik analizimize göre en iyi güven eşiği
const float CONFIDENCE_THRESHOLD = 0.30; 
const float NMS_THRESHOLD = 0.45;
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

// Sınıflar ve Renkler (Senin projene özel)
const std::vector<std::string> CLASS_NAMES = {"Vehicle", "UAP", "UAI", "Person"};
const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 255), // Vehicle (Mor)
    cv::Scalar(255, 255, 0), // UAP (Sarı/Cyan)
    cv::Scalar(0, 0, 255),   // UAI (Kırmızı)
    cv::Scalar(0, 255, 0)    // Person (Yeşil)
};

int main() {
    // 1. DOSYA YOLLARI
    // Dosyanın bulunduğu klasöre göre burayı güncelle
    std::string modelPath = "best.onnx"; 
    std::string videoPath = "simulasyon_videosu.mp4"; // Video ismini kontrol et!

    std::cout << "--- SISTEM BASLATILIYOR ---" << std::endl;
    
    cv::dnn::Net net;
    try {
        std::cout << "Model yukleniyor: " << modelPath << std::endl;
        net = cv::dnn::readNetFromONNX(modelPath);
        
        // Mümkünse OpenCV Backend kullan (En uyumlusu budur)
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "HATA: Model yuklenemedi! Dosya yolunu kontrol et." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "HATA: Video acilamadi: " << videoPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::cout << "Simulasyon basladi. Cikis icin 'ESC' veya 'q' tusuna bas." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Video bitti, basa sariliyor..." << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Döngüsel video
            continue;
        }

        // --- PRE-PROCESSING ---
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // --- GÜVENLİ ÇIKTI İŞLEME (POINTER YÖNTEMİ) ---
        // OpenCV 4.5.4 çökmesini önleyen yöntem:
        // Matris boyutlarını sormadan direkt veriye erişiyoruz.
        
        float* data_ptr = (float*)outputs[0].data;
        
        // YOLOv8 Çıktısı: [1, 8, 8400] -> Biz [8, 8400] gibi davranacağız
        int rows = 8;    // 4 Sınıf + 4 Koordinat
        int cols = 8400; // Anchor sayısı
        
        // Veriyi kopyalamadan yeni bir 2D matris başlığı oluştur
        cv::Mat output_2d(rows, cols, CV_32F, data_ptr);
        
        // Transpose alarak [8400, 8] haline getir (Okuması kolay olsun)
        cv::Mat output_data;
        cv::transpose(output_2d, output_data);
        
        float* data = (float*)output_data.data;
        
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = (float)frame.cols / INPUT_WIDTH;
        float y_factor = (float)frame.rows / INPUT_HEIGHT;

        // Her bir tahmin kutusunu gez (8400 adet)
        for (int i = 0; i < cols; ++i) {
            float* row_ptr = data + (i * 8); 
            float* classes_scores = row_ptr + 4; // İlk 4'ü koordinat, gerisi skor

            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(cv::Mat(1, 4, CV_32F, classes_scores), 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > CONFIDENCE_THRESHOLD) {
                float x = row_ptr[0];
                float y = row_ptr[1];
                float w = row_ptr[2];
                float h = row_ptr[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back((float)max_class_score);
                class_ids.push_back(class_id_point.x);
            }
        }

        // NMS (Çakışan kutuları temizle)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        // Çizim İşlemleri
        for (int i : indices) {
            cv::Rect box = boxes[i];
            int class_id = class_ids[i];
            float conf = confidences[i];
            
            cv::Scalar color = COLORS[class_id % COLORS.size()];
            cv::rectangle(frame, box, color, 3);
            
            std::string label = CLASS_NAMES[class_id] + " " + cv::format("%.2f", conf);
            
            // Etiket arka planı
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
            cv::rectangle(frame, cv::Point(box.x, box.y - labelSize.height - 5), cv::Point(box.x + labelSize.width, box.y), color, -1);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
        }

        cv::imshow("Drone Inis Sistemi (C++)", frame);
        
        // 1 ms bekle, ESC basılırsa çık
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}