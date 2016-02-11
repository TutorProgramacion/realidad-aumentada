#include <iostream>

#include <aruco\aruco.h>
#include <aruco\cvdrawingutils.h>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\opencv.hpp>

#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

using namespace cv;
using namespace aruco;
using namespace std;

MarkerDetector PPDetector;
vector<Marker> TheMarkers;
Mat TheInputImage, TheUndInputImage;
CameraParameters TheCameraParams;
Size TheGlWindowSize;

float TheMarkerSize = 0.05f;

void drawCubeModel()
{
	static const GLfloat LightAmbient[] = { 0.25f, 0.25f, 0.25f, 1.0f };  // Ambient Light Values
	static const GLfloat LightDiffuse[] = { 0.1f, 0.1f, 0.1f, 1.0f };     // Diffuse Light Values
	static const GLfloat LightPosition[] = { 0.0f, 0.0f, 2.0f, 1.0f };    // Light Position

	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, LightPosition);
	glEnable(GL_COLOR_MATERIAL);

	glScalef(0.02f, 0.02f, 0.02f);
	glTranslatef(0, 0, 1);

	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	const static GLfloat vertices[] = {
			 1.0f, -1.0f, -1.0f,
			 1.0f, -1.0f,  1.0f,
			-1.0f, -1.0f,  1.0f,
			-1.0f, -1.0f, -1.0f,
			 1.0f,  1.0f, -1.0f,
			 1.0f,  1.0f,  1.0f,
			-1.0f,  1.0f,  1.0f,
			-1.0f,  1.0f, -1.0f
	};

	const static GLfloat normals[] = {
		  0.0f, -1.0f, -0.0f,
		  0.0f,  1.0f,  0.0f,
		  1.0f,  0.0f,  0.0f,
		 -0.0f,  0.0f,  1.0f,
		 -1.0f ,-0.0f, -0.0f,
		  0.0f,  0.0f, -1.0f
	};

	const static GLubyte indices[] = {
		1 - 1, 2 - 1, 3 - 1, 4 - 1,
		5 - 1, 8 - 1, 7 - 1, 6 - 1,
		1 - 1, 5 - 1, 6 - 1, 2 - 1,
		2 - 1, 6 - 1, 7 - 1, 3 - 1,
		3 - 1, 7 - 1, 8 - 1, 4 - 1,
		5 - 1, 1 - 1, 4 - 1, 8 - 1
	};

	GLfloat normales[24];
	size_t index = 0;

	for (size_t i = 0; i < 24; i += 4)
	{
		normales[i + 0] = normals[index];
		normales[i + 1] = normals[index];
		normales[i + 2] = normals[index];
		normales[i + 3] = normals[index];

		index = index >= 18 ? 0 : index++;
	}

	glNormalPointer(GL_FLOAT, 0, normales);
	glVertexPointer(3, GL_FLOAT, 0, vertices);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor4f(0.2f, 0.35f, 0.3f, 0.75f);
	glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, indices);

	glLineWidth(2.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor4f(0.2f, 0.65f, 0.3f, 0.35f);
	glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, indices);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}

//dibujar la ecena 3D
void drawAugmentedScene()
{
	double proj_matrix[16];
	double modelview_matrix[16];

	TheCameraParams.glGetProjectionMatrix(TheInputImage.size(), TheGlWindowSize, proj_matrix, 0.05, 10);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glLoadMatrixd(proj_matrix);

	for (unsigned int m = 0; m < TheMarkers.size(); m++)
	{
		TheMarkers[m].glGetModelViewMatrix(modelview_matrix);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glLoadMatrixd(modelview_matrix);

		drawCubeModel();
	}
}

//dibujar la imagen 2D proveniente de la camara o video
void drawCameraFrame()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glOrtho(0, TheGlWindowSize.width, 0, TheGlWindowSize.height, -1.0, 1.0);
	glViewport(0, 0, TheGlWindowSize.width, TheGlWindowSize.height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glPixelZoom(1, -1);
	glRasterPos2i(0, TheGlWindowSize.height);
	glDrawPixels(TheGlWindowSize.width, TheGlWindowSize.height, GL_BGR_EXT, GL_UNSIGNED_BYTE, TheInputImage.data);
}

//actualizar los graficos opengl
void ARDraw(void* param) {
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	drawCameraFrame();
	drawAugmentedScene();
	glFlush();
}

//detectar las marcas con aruco y actualizar la ventana
void detectarMarcador(String ARWindowName) {
	cv::undistort(TheInputImage, TheUndInputImage, TheCameraParams.CameraMatrix, TheCameraParams.Distorsion);
	PPDetector.detect(TheUndInputImage, TheMarkers, TheCameraParams.CameraMatrix, Mat(), TheMarkerSize, false);
	cv::updateWindow(ARWindowName);
}

int main(int argc, char **argv)
{
	try
	{
		String ARWindowName = "Realidad Aumentada (OpenCV+ArUco)";

		//crear una ventana que puede mostrar graficos 3D con OpenGL
		namedWindow(ARWindowName, WINDOW_OPENGL);
		setOpenGlContext(ARWindowName);
		setOpenGlDrawCallback(ARWindowName, ARDraw, NULL);

		//abrir la camara (usar 0) o video (ruta) 
		VideoCapture video("data/video.avi");

		if (!video.isOpened()) { getchar(); return -1; }

		double height = video.get(CAP_PROP_FRAME_HEIGHT);
		double width = video.get(CAP_PROP_FRAME_WIDTH);

		//hacer el tamaño de la ventana igual el tamaño de video
		resizeWindow(ARWindowName, width, height);

		//cargar los parametros de calibacion de la camara
		TheCameraParams.readFromXMLFile("data/intrinsics.yml");

		while (true) {
			video >> TheInputImage;

			if (TheInputImage.empty()) { break; }

			TheCameraParams.resize(TheInputImage.size());
			TheGlWindowSize = TheInputImage.size();

			detectarMarcador(ARWindowName);

			waitKey(33);
		}
	}
	catch (std::exception &ex)
	{
		cout << "Exception :" << ex.what() << endl;
	}
}
