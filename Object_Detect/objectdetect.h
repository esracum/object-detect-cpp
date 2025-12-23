#ifndef OBJECTDETECT_H
#define OBJECTDETECT_H
#include <QMainWindow>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <QImage>              // Görüntü dönüşümü için
#include <QPixmap>             // Ekrana basmak için
#include <QMainWindow>
#include <QElapsedTimer>


QT_BEGIN_NAMESPACE
namespace Ui { class ObjectDetect; }
QT_END_NAMESPACE

class ObjectDetect : public QMainWindow
{
    Q_OBJECT


/*protected:
    void resizeEvent(QResizeEvent *event) override;
*/
public:
    ObjectDetect(QWidget *parent = nullptr);
    ~ObjectDetect();
private slots:
    void kameraGuncelle();

    void on_labelKamera_linkActivated(const QString &link);

    void on_btnRecord_clicked();

    void kayitAnimasyonuYap();

    void on_snapshot_clicked();

    void on_btnEoir_clicked();

private:
    Ui::ObjectDetect *ui;
    cv::VideoCapture cap; // Kamera/Video nesnesi
        QTimer *timer;        // Zamanlayıcı
        cv::Mat frame;        // Görüntü matrisi
        QImage qtImage;
        cv::dnn::Net net;
        std::vector<std::string> sinifListesi;
        void nesneleriTani(cv::Mat &img);
        cv::VideoWriter videoYazici; // Videoyu kaydeden araç
        bool kayitAktifMi = false;   // Kayıt durumunu tutan bayrak
        QTimer *kayitTimer;          // Animasyon için zamanlayıcı
        bool animasyonDurumu = false;
        QIcon orjinalKayitIconu;
        bool termalModAktif = false;
        QElapsedTimer fpsSayaci;

};
#endif // OBJECTDETECT_H
