#include <mpi.h>

#include <array>
#include <exception>
#include <iomanip>  // std::setw, std::setfill
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef WIN32
#include <direct.h>  // Windows
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "SpatialGMM.hpp"
#include "utils.h"

std::string to_string_with_leading_zeros(int value, int width) {
  std::ostringstream stream;
  stream << std::internal << std::setfill('0') << std::setw(width) << value;
  return stream.str();
}

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  int timestep[2] = {1, 10};
  int dt = 1;

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <DatasetDir>" << std::endl;
    return 1;
  }

  std::string variableName = "Pf";
  std::string datasetName = "Isabel";
  std::string datasetDir = argv[1];
  Configuration config;
  config.setDataPath(datasetDir + datasetName + "/" + variableName + "/");
  config.setDimensions(500, 500, 100);
  config.setBlockSize(32);  // the resolution of each SGMM block
  config.setVariableName(variableName);
  std::string parameterPath = datasetDir + datasetName + "/";
  std::string extensionName = ".bin";

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (int i = timestep[0]; i < timestep[1]; i++) {
    config.setRawDataFilename(
        variableName + to_string_with_leading_zeros(i * dt, 2) + extensionName);
    // building statistical representation of the 3D volume
    SpatialGMM::Representation<float> result = SpatialGMM::DataModeling<float>(
        config.getDimensions(), config.getBlockSize(),
        config.getDataPath() + config.getRawDataFilename(), Endian::Big());

    SpatialGMM::ExportReconstructedVolume<float>(
        config.getDataPath() + "output.raw", result, config.getDimensions(),
        config.getBlockSize());
    std::string folder = parameterPath + "Params/" + variableName +
                         to_string_with_leading_zeros(i * dt, 2) + "/";
#ifdef WIN32
    mkdir(folder.c_str());
#else
    mkdir(folder.c_str(), 0755);
#endif
    SpatialGMM::SaveParameters<float>(
        folder + "params_" + std::to_string(rank) + ".json", result);
  }

  MPI_Finalize();
}
