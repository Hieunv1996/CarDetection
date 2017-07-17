#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
using namespace cv;

const int KEY_SPACE = 32;
const int KEY_ESC = 27;

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;

void detect(IplImage *img);

int main(int argc, char** argv)
{
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
  
  CvCapture *capture;
  IplImage  *frame;
  int input_resize_percent = 100;
  
  if(argc < 3)
  {
    std::cout << "Usage " << argv[0] << " cascade.xml video.avi" << std::endl;
    return 0;
  }

  if(argc == 4)
  {
    input_resize_percent = atoi(argv[3]);
    std::cout << "Resizing to: " << input_resize_percent << "%" << std::endl;
  }

  cascade = (CvHaarClassifierCascade*) cvLoad(argv[1], 0, 0, 0);
  storage = cvCreateMemStorage(0);
  capture = cvCaptureFromAVI(argv[2]);

  assert(cascade && storage && capture);

  cvNamedWindow("video", 1);

  IplImage* frame1 = cvQueryFrame(capture);
  frame = cvCreateImage(cvSize((int)((frame1->width*input_resize_percent)/100) , (int)((frame1->height*input_resize_percent)/100)), frame1->depth, frame1->nChannels);

  int key = 0;
  do
  {
    frame1 = cvQueryFrame(capture);

    if(!frame1)
      break;

    cvResize(frame1, frame);

    detect(frame);

    key = cvWaitKey(33);

    if(key == KEY_SPACE)
      key = cvWaitKey(0);

    if((char)key == KEY_ESC)
      break;

  }while(1);

  cvDestroyAllWindows();
  cvReleaseImage(&frame);
  cvReleaseCapture(&capture);
  cvReleaseHaarClassifierCascade(&cascade);
  cvReleaseMemStorage(&storage);

  return 0;
}
template<typename T>
std::string numberToString(T Number) {
    std::ostringstream ss;
    ss << Number;
    return ss.str();
}

void detect(IplImage *img)
{
  CvSize img_size = cvGetSize(img);
  CvSeq *object = cvHaarDetectObjects(
    img,
    cascade,
    storage,
    1.1, //1.1,//1.5, //-------------------SCALE FACTOR
    1, //2        //------------------MIN NEIGHBOURS
    0, //CV_HAAR_DO_CANNY_PRUNING
    cvSize(0,0),//cvSize( 30,30), // ------MINSIZE
    img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
    );
int totalCar = 0;
  for(int i = 0 ; i < ( object ? object->total : 0 ) ; i++)
  {
    CvRect *r = (CvRect*)cvGetSeqElem(object, i);
    if(r->width > 70){
        cvSetImageROI(img, *r);

        IplImage *tmp = cvCreateImage(cvGetSize(img),
                                      img->depth,
                                      img->nChannels);

        cvCopy(img, tmp, NULL);
        cvResetImageROI(img);
        time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );
        cvSaveImage(("/home/hieunv/Desktop/Cars/IMG_" + numberToString(now->tm_hour)
                     +"_"+ numberToString(now->tm_min) +"_"+ numberToString(now->tm_sec)+".JPG").c_str(),tmp);
//        cvRectangle(img,
//                    cvPoint(r->x, r->y),
//                    cvPoint(r->x + r->width, r->y + r->height),
//                    CV_RGB(255, 0, 0), 2, 8, 0);
    totalCar++;
    }
  }
  std::cout << "Total: " << totalCar << " cars detected." << std::endl;
//  cvShowImage("video", img);


}