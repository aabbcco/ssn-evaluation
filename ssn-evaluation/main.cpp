
#include <pcl/io/io.h>

#include <pcl/io/pcd_io.h>

#include <pcl/point_cloud.h>

#include<pcl/visualization/pcl_visualizer.h>

#include <pcl/io/vtk_lib_io.h>//loadPolygonFileOBJ所属头文件；
#include<vector>
#include "evaluation.hpp"

#include<boost/program_options.hpp>

#include<fstream>

#include<filesystem>
#include<vector>
#include<functional>

using namespace std;
using namespace pcl;
namespace po = boost::program_options;
using namespace std::filesystem;

using XYZLptr = PointCloud<PointXYZL>::Ptr;

int main(int argc,char** argv)
{
	std::string strs;
	po::options_description desc("options list");
	desc.add_options()
		("folder", po::value<std::string>(&strs)->default_value(""), "a folder")
		("help,h", "c++ code for ssn evaluation");
	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
	}
	catch (...)
	{
		std::cout << "wrong input" << endl;
		return -1;
	}

	if (vm.count("help"))
	{
		std::cout << desc << std::endl;
		return 0;
	}
	po::notify(vm);

	PointCloud<PointXYZRGB>::Ptr pcloud(new PointCloud<PointXYZRGB>);
	vector<float> all = { 0,0 };
	vector<float> usa_all = { 0,0 };
	size_t sz = 0;
	path pat(strs);
	directory_iterator dit(pat);
	ofstream fs("result.csv", ios::out|ios::trunc);
	fs << "filename,trad_acc,neural_acc,trad_acc,trad_usa" << endl;

	for (auto &file:dit)
	{
		cout << file.path().filename();
		io::loadPCDFile(file.path().string(),*pcloud);
		decltype(auto) result =Cal_RGB_Point_Mapping(pcloud,Cal_Achievable_seg_acc<XYZLptr,XYZLptr>);
		auto result1 = Cal_RGB_Point_USA(pcloud);
		cout << result[0] << "\t" << result[1]<<"\t"<<result1[0]<<"\t"<<result1[1] << endl;
		fs << file.path().filename() << "," << result[0] << "," << result[1] << "," << result1[0] << "," << result1[1] << endl;
		all[0] += result[0];
		all[1] += result[1];
		usa_all[0] += result1[0];
		usa_all[1] += result1[1];
		sz++;
	}
	cout << "total:" << all[0] / sz << "," << all[1] / sz << endl << usa_all[0]/sz << "," << usa_all[1]/sz << endl;
	fs.close();

	return 0;

}