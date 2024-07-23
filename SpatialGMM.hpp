#pragma once

#include <mpi.h>

#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "utils.h"

#include "GMM3D.hpp"

class Configuration {
public:
	void setDataPath(std::string path) { m_dataPath = path; }
	std::string getDataPath() { return m_dataPath; }
	void setRawDataFilename(std::string filename) { m_rawDataFilename = filename; }
	std::string getRawDataFilename() { return m_rawDataFilename; }
	void setDimensions(int x, int y, int z) { m_dimensions = std::array<int, 3>{x, y, z}; }
	std::array<int, 3> getDimensions() { return m_dimensions; }
	void setBlockSize(int size) { blockSize = size; }
	int getBlockSize() { return blockSize; }
	void setVariableName(std::string varName) { m_variableName = varName; }
	std::string getVariableName() { return m_variableName; }

private:
	std::string m_dataPath;
	std::string m_rawDataFilename;
	std::array<int, 3> m_dimensions;
	int blockSize;
	std::string m_variableName;
};

namespace SpatialGMM {

	template<typename T>
	class Bin {
	public:
		void buildGMMs(std::vector<std::array<float, 3>>& dataset) {
			int maxClusters = 4;
			T bestBIC = std::numeric_limits<T>::max();
			GMM3D<T> gmm;

			for (int n = 1; n <= maxClusters; n++) {
				gmm.fit(n, dataset);
				gmm.setValid();

				T bic = BayesInformationCriteria<T>(gmm, dataset);
				if(std::isnan(bic)) {
					gmm.setInvalid();
					break;
				}
				if (bic < bestBIC) {
					bestBIC = bic;
				}
				else {
					break;
				}
			}
			m_GMM = gmm;
		}

		GMM3D<T> getGMM() { return m_GMM; }
		void setGMM(GMM3D<T> gmm) { m_GMM = gmm; }

		void setProbability(T p) { m_probability = p; }
		T getProbability() { return m_probability; }

		T predict(float x, float y, float z) {
			return m_GMM.predict({ x, y, z });
		}

	private:
		T m_probability;
		GMM3D<T> m_GMM;
	};

	/**
	* @param m_bins: store spatial locations as float numbers
	* @param m_minValue
	* @param m_maxValue
	* @param m_Id: global blockId 
	* @param m_dims: dimension
	* @param m_dataLocal
	*/
	template<typename T>
	class DataBlock {
	public:
		void setblockId(int xId, int yId, int zId) {
			m_Id[0] = xId;
			m_Id[1] = yId;
			m_Id[2] = zId;
		}

		std::array<int, 3> getBlockId() { return m_Id; }

		void setData(std::vector<T>& data) {
			m_dataLocal = data;
		}

		std::vector<T> getData() { return m_dataLocal; }

		// for each bin, build a spatial distribution by GMM
		void buildGMMsPerBin(int blockSize) {
			int numOfBins = m_bins.size();
			for (int binId = 0; binId < numOfBins; binId++) {
				if (m_bins[binId].getProbability() <= 0.0) {
					continue;
				}

				int startX = m_Id[0] * blockSize;
				int startY = m_Id[1] * blockSize;
				int startZ = m_Id[2] * blockSize;
				int endX = startX + m_dims[0];
				int endY = startY + m_dims[1];
				int endZ = startZ + m_dims[2];
				if (endX >= m_dims[0]) {
					endX = m_dims[0] - 1;
				}
				if (endY >= m_dims[1]) {
					endY = m_dims[1] - 1;
				}
				if (endZ >= m_dims[2]) {
					endZ = m_dims[2] - 1;
				}

				T binMin = m_minValue + (m_maxValue - m_minValue) * static_cast<T>(binId) / static_cast<T>(numOfBins);
				T binMax =
					m_minValue + (m_maxValue - m_minValue) * static_cast<T>(binId + 1) / static_cast<T>(numOfBins);
				std::vector<std::array<float, 3>> dataset;
				// get the data points in this bin
				for (int blockZ = 0; blockZ < m_dims[0]; ++blockZ) {
					for (int blockY = 0; blockY < m_dims[1]; ++blockY) {
						for (int blockX = 0; blockX < m_dims[2]; ++blockX) {
							T value = m_dataLocal[blockX + blockY * m_dims[0] + blockZ * m_dims[0] * m_dims[1]];
							if (value >= binMin && value < binMax) {
								//dataset.push_back(
								//	{ {float(blockX + startX), float(blockY + startY), float(blockZ + startZ)} });
								//dataset.push_back(
								//	{ {float(blockX) / float(m_dims[0]), float(blockY) / float(m_dims[1]), float(blockZ) / float(m_dims[2])}});
								dataset.push_back(
									{ float(blockX), float(blockY), float(blockZ) });
							}
							if (value == binMax && binId == numOfBins - 1) {
								//dataset.push_back(
								//	{ {float(blockX + startX), float(blockY + startY), float(blockZ + startZ)} });
								//dataset.push_back(
								//	{ {float(blockX) / float(m_dims[0]), float(blockY) / float(m_dims[1]), float(blockZ) / float(m_dims[2])} });
								dataset.push_back(
									{ float(blockX), float(blockY), float(blockZ) });
							}
						}
					}
				}
				if (dataset.size() > 0) {
					m_bins[binId].buildGMMs(dataset);
				}
			}
		}

		void buildHistogram() {
			// calculate min and max of local data block
			T min = m_dataLocal[0];
			for (int i = 1; i < m_dataLocal.size(); ++i) {
				if (m_dataLocal[i] < min) {
					min = m_dataLocal[i];
				}
			}
			T max = m_dataLocal[0];
			for (int i = 1; i < m_dataLocal.size(); ++i) {
				if (m_dataLocal[i] > max) {
					max = m_dataLocal[i];
				}
			}

			int numOfBins = 128;
			m_bins.resize(numOfBins);

			std::vector<T> histogram(numOfBins);
			// calculate histogram
			for (int i = 0; i < m_dataLocal.size(); ++i) {
				T value = m_dataLocal[i];
				if (value <= min) {
					histogram[0] += 1.0f;
				}
				else if (value >= max) {
					histogram[numOfBins - 1] += 1.0f;
				}
				else {
					// find the bin that contains the value
					for (int j = 0; j < numOfBins; ++j) {
						T binMin = min + (max - min) * static_cast<T>(j) / static_cast<T>(numOfBins);
						T binMax =
							min + (max - min) * (static_cast<T>(j + 1)) / static_cast<T>(numOfBins);
						if (value >= binMin && value < binMax) {
							histogram[j] += 1.0f;
							break;
						}
					}
				}
			}
			for (int i = 0; i < numOfBins; ++i) {
				m_bins[i].setProbability(histogram[i] / static_cast<T>(m_dataLocal.size()));
			}
		}
		void setMinMax(T minVal, T maxVal) {
			m_minValue = minVal;
			m_maxValue = maxVal;
		}

		T getMin() {
			return m_minValue;
		}
		T getMax() {
			return m_maxValue;
		}

		void setDims(int x, int y, int z) {
			m_dims = std::array<int, 3>{ x, y, z };
		}

		std::array<int, 3> getDims() {
			return m_dims;
		}

		std::vector<Bin<float>> getBins() { return m_bins; }

		/**
		* export format
		* - blockId[0], blockId[1], blockId[2]
		* - blockDims[0], blockDims[1], blockDims[2]
		* - numOfBins
		* - minValue
		* - maxValue
		* - histogram
		*		- binId
		*		- probability
		*		- numOfGaussians
		*		- GaussiansList
		*			- weight
		* 		- means
		* 		- covariance(upper triangular)
		*/
		std::vector<T> exportStatistics(){
			std::vector<T> statistics;
			statistics.push_back(m_Id[0]);
			statistics.push_back(m_Id[1]);
			statistics.push_back(m_Id[2]);
			statistics.push_back(m_dims[0]);
			statistics.push_back(m_dims[1]);
			statistics.push_back(m_dims[2]);
			statistics.push_back(m_bins.size());
			statistics.push_back(m_minValue);
			statistics.push_back(m_maxValue);
			for (int i = 0; i < m_bins.size(); ++i) {
				Bin<float> bin = m_bins[i];
				if (bin.getProbability() <= 1e-9) {
					statistics.push_back(-1);
					continue;
				}
				statistics.push_back(i);
				statistics.push_back(bin.getProbability());
				GMM3D<T> gmm = bin.getGMM();
				int numKernels = gmm.numComponents();
				statistics.push_back(numKernels);
				for (int k = 0; k < numKernels; k++) {
					statistics.push_back(gmm.getWeight(k));
				}
				for (int k = 0; k < numKernels; k++) {
					Gaussian3D<T> gaussian = gmm.getGaussian(k);
					std::array<T, 3> mean = gaussian.mean();
					statistics.push_back(mean[0]);
					statistics.push_back(mean[1]);
					statistics.push_back(mean[2]);
				}
				for (int k = 0; k < numKernels; k++) {
					Gaussian3D<T> gaussian = gmm.getGaussian(k);
					std::array<std::array<T, 3>, 3> covariance = gaussian.covariance();
					// export upper triangular matrix
					for (int i = 0; i < 3; ++i) {
						for (int j = i; j < 3; ++j) {
							statistics.push_back(covariance[i][j]);
						}
					}
				}
			}

			return statistics;
		}

		void setBins(std::vector<Bin<float>> bins) {
			m_bins = bins;
		}

	private:
		std::vector<Bin<float>> m_bins;
		T m_minValue;
		T m_maxValue;
		std::array<int, 3> m_Id;
		std::array<int, 3> m_dims;
		std::vector<T> m_dataLocal;
	};

	template<typename T>
	using Representation = std::vector<DataBlock<T>>;

	template<typename T>
	Representation<T> DataModeling(std::array<int, 3> dims, int blockSize, std::string filename, Endian endian) {
		Representation<T> dataBlockRepresentation;

		// Get the number of processes
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);

		// Get the rank of the process
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		int size = dims[0] * dims[1] * dims[2];

		std::ifstream file(filename, std::ios::binary);

		double start_time = MPI_Wtime();

		int numGroups[3] = { dims[0] / blockSize, dims[1] / blockSize,
							dims[2] / blockSize };

		for (int z = 0; z < numGroups[2]; z++) {
			for (int y = 0; y < numGroups[1]; y++) {
				for (int x = 0; x < numGroups[0]; x++) {
					int blockId = x + y * numGroups[0] + z * numGroups[1] * numGroups[0];
					int processId = blockId % world_size;
					if (processId == world_rank) {
						DataBlock<T> block;
						block.setblockId(x, y, z);
						dataBlockRepresentation.push_back(block);
					}
				}
			}
		}

		for (DataBlock<T>& dataBlock : dataBlockRepresentation) {
			std::array<int, 3> blockId = dataBlock.getBlockId();
			int startX = blockId[0] * blockSize;
			int startY = blockId[1] * blockSize;
			int startZ = blockId[2] * blockSize;
			int endX = startX + blockSize;
			int endY = startY + blockSize;
			int endZ = startZ + blockSize;
			if (endX >= dims[0]) {
				endX = dims[0] - 1;
			}
			if (endY >= dims[1]) {
				endY = dims[1] - 1;
			}
			if (endZ >= dims[2]) {
				endZ = dims[2] - 1;
			}

			dataBlock.setDims(endX - startX, endY - startY, endZ - startZ);

			std::vector<T> dataLocal;

			std::array<int, 3> blockDims = { endX - startX, endY - startY,
											endZ - startZ };

			// get local data block
			for (int blockZ = 0; blockZ < blockDims[2]; ++blockZ) {
				for (int blockY = 0; blockY < blockDims[1]; ++blockY) {
					for (int blockX = 0; blockX < blockDims[0]; ++blockX) {
						int id = (startX + blockX + (startY + blockY) * dims[0] +
							(startZ + blockZ) * dims[0] * dims[1]);
						file.seekg(id * sizeof(T), std::ios::beg);
						T val;
						// read a value to buf
						if (file.read(reinterpret_cast<char*>(&val), sizeof(T))) {
							if (endian == Endian::Big) {
								val = reverseByteOrder(val);
							}
							
							dataLocal.push_back(*reinterpret_cast<T*>(&val));
						}
						else {
							std::cout << "Error: read failed" << std::endl;
							return {};
						}
					}
				}
			}

			T minValue = dataLocal[0];
			T maxValue = dataLocal[0];

			for (auto& val : dataLocal) {
				if (val < minValue) {
					minValue = val;
				}
				else if (val > maxValue) {
					maxValue = val;
				}
			}

			dataBlock.setMinMax(minValue, maxValue);

			dataBlock.setData(dataLocal);

			dataBlock.buildHistogram();

			dataBlock.buildGMMsPerBin(blockSize);
		}
		double end_time = MPI_Wtime();

		if (world_rank == 0) {
			file.close();
			std::cout << "Time elapsed: " << end_time - start_time << "s" << std::endl;
		}

		return dataBlockRepresentation;
	}

	/**
	* @brief Estimate the pdf of a point
	* @param x, y, z: coordinates of the point
	* @param binId: id of the histogram bin
	*/
	template<typename T>
	std::vector<float> ValueEstimation(DataBlock<T>& dataBlock, float x, float y, float z) {
		std::vector<Bin<float>> bins = dataBlock.getBins();
		std::vector<float> pdf(bins.size(), 0.0);
		for (int i = 0; i < bins.size(); i++) {

			float Hi = bins[i].getProbability();
			if (Hi > 0.0) {
				pdf[i] = bins[i].predict(x, y, z) * Hi;
			}
		}

		float sumOfProb = 0.0;
		for (auto& val : pdf) {
			sumOfProb += val;
		}

		for (auto& val : pdf) {
			val /= sumOfProb;
		}

		return pdf;
	}

	template<typename T>
	void ExportReconstructedVolume(std::string outputFilename, Representation<T>& dataBlocks, std::array<int, 3> dims, int blockSize) {
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		int size = dims[0] * dims[1] * dims[2];

		if (world_rank == 0) {
			std::ofstream outputFile(outputFilename, std::ios::binary);
			for (int i = 0; i < size; ++i) {
				float val = 0.0f;
				outputFile.write(reinterpret_cast<char*>(&val), sizeof(float));
			}
			outputFile.close();
		}

		std::ofstream outputFile(outputFilename, std::ios::binary);

		for (DataBlock<T>& dataBlock : dataBlocks) {
			std::array<int, 3> blockId = dataBlock.getBlockId();
			std::array<int, 3> blockDims = dataBlock.getDims();
			int startX = blockId[0] * blockSize;
			int startY = blockId[1] * blockSize;
			int startZ = blockId[2] * blockSize;
			int endX = startX + blockDims[0];
			int endY = startY + blockDims[1];
			int endZ = startZ + blockDims[2];
			if (endX >= dims[0]) {
				endX = dims[0] - 1;
			}
			if (endY >= dims[1]) {
				endY = dims[1] - 1;
			}
			if (endZ >= dims[2]) {
				endZ = dims[2] - 1;
			}

			// output data reconstructed to file
			std::vector<float> reconstructedVolume(blockDims[0] * blockDims[1] * blockDims[2]);

			std::vector<Bin<float>> bins = dataBlock.getBins();

			// adjust the histogram
			for (Bin<float>& bin : bins) {
				float probability = bin.getProbability();
				if (probability > 0.0 && !std::isnan(probability)) {
					float factor = 0.0;
					for (int z = 0; z < blockDims[2]; ++z) {
						for (int y = 0; y < blockDims[1]; ++y) {
							for (int x = 0; x < blockDims[0]; ++x) {
								//float vx = static_cast<float>(x + startX);
								//float vy = static_cast<float>(y + startY);
								//float vz = static_cast<float>(z + startZ);
								//float vx = static_cast<float>(x) / static_cast<float>(blockDims[0]);
								//float vy = static_cast<float>(y) / static_cast<float>(blockDims[1]);
								//float vz = static_cast<float>(z) / static_cast<float>(blockDims[2]);
								float vx = static_cast<float>(x);
								float vy = static_cast<float>(y);
								float vz = static_cast<float>(z);

								float probability = bin.predict(vx, vy, vz);
								if (!std::isnan(probability)) {
									factor += probability;
								}
							}
						}
					}
					if (factor > 0.0) {
						bin.setProbability(probability / factor);
					}
					else {
						bin.setProbability(0.0);
					}
				}
			}

			for (int z = 0; z < blockDims[2]; ++z) {
				for (int y = 0; y < blockDims[1]; ++y) {
					for (int x = 0; x < blockDims[0]; ++x) {
						//float vx = static_cast<float>(x + startX);
						//float vy = static_cast<float>(y + startY);
						//float vz = static_cast<float>(z + startZ);
						//float vx = static_cast<float>(x) / static_cast<float>(blockDims[0]);
						//float vy = static_cast<float>(y) / static_cast<float>(blockDims[1]);
						//float vz = static_cast<float>(z) / static_cast<float>(blockDims[2]);
						float vx = static_cast<float>(x);
						float vy = static_cast<float>(y);
						float vz = static_cast<float>(z);

						std::vector<float> pdf = ValueEstimation(dataBlock, vx, vy, vz);

						T minValue = dataBlock.getMin();
						T maxValue = dataBlock.getMax();
						T dx = (maxValue - minValue) / (float)(pdf.size());
						//// pick the most likely bin
						//for (int i = 0; i < pdf.size(); ++i)
						//{
						//	reconstructedVolume[x + y * blockDims[0] + z * blockDims[0] * blockDims[1]] +=
						//		(minValue + static_cast<float>(i) * dx + 0.5 * dx) * pdf[i];
						//}

						int mostLikelyBin = 0;
						for (int i = 0; i < pdf.size(); ++i)
						{
							if (!std::isnan(pdf[i]) && pdf[mostLikelyBin] < pdf[i]) {
								mostLikelyBin = i;
							}
						}

						reconstructedVolume[x + y * blockDims[0] + z * blockDims[0] * blockDims[1]] =
							(minValue + static_cast<float>(mostLikelyBin) * dx);

					}
				}
			}

			// export reconstructed volume
			for (int blockZ = 0; blockZ < blockDims[2]; ++blockZ) {
				for (int blockY = 0; blockY < blockDims[1]; ++blockY) {
					for (int blockX = 0; blockX < blockDims[0]; ++blockX) {
						int id = (startX + blockX + (startY + blockY) * dims[0] +
							(startZ + blockZ) * dims[0] * dims[1]);
						outputFile.seekp(id * sizeof(float), std::ios::beg);
						float val =
							reconstructedVolume[blockX + blockY * blockDims[0] + blockZ * blockDims[0] * blockDims[1]];
						outputFile.write(reinterpret_cast<char*>(&val), sizeof(float));
					}
				}
			}

			
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (world_rank == 0)
			outputFile.close();
		
	}

	/**
	* Save parameters to JSON file.
	*/
	template<typename T>
	void SaveParameters(std::string filename, Representation<T>& data) {
		std::ofstream outputFile(filename);
		outputFile << "[" << std::endl;
		int id = 0;
		int numBlocks = data.size();
		for (DataBlock<T>& dataBlock : data) {
			outputFile << "{";
			std::array<int, 3> blockId = dataBlock.getBlockId();
			outputFile << "\"BlockId\":[" << blockId[0] << "," << blockId[1] << "," << blockId[2] << "]," << std::endl;
			std::vector<Bin<float>> bins = dataBlock.getBins();
			outputFile << "\"H\":[";
			std::vector<int> validBinIds;
			for(int i = 0; i < bins.size(); ++i){
				if (bins[i].getProbability() > 1e-9) {
					validBinIds.push_back(i);
				}
			}

			for (int i = 0; i < validBinIds.size(); i++) {
				int binId = validBinIds[i];
				T probability = bins[binId].getProbability();
				outputFile << "{\"I\":" << i << ",\"V\":" << probability << "}";
				if (i != validBinIds.size() - 1) {
					outputFile << ",";
				}
			}
			outputFile << "],";
			outputFile << "\"D\":["; // spatial GMM per bin
			for (int i = 0; i < validBinIds.size(); i++) {
				int binId = validBinIds[i];
				T probability = bins[binId].getProbability();
				outputFile << "{";
				outputFile << "\"I\":" << i << ",";
				outputFile << "\"V\":{";
				// "w": adjustment factor of SGMM(TODO)
				outputFile << "\"w\": 1.0,";
				outputFile << "\"gmm\":{";
				GMM3D<float> gmm = bins[binId].getGMM();
				outputFile << "\"K\":" << gmm.numComponents() << ",";
				outputFile << "\"g\":[";
				for (int k = 0; k < gmm.numComponents(); k++) {
					outputFile << "{\"w\":" << gmm.getWeight(k) << ",";
					Gaussian3D<T> gaussian = gmm.getGaussian(k);
					std::array<T, 3> mean = gaussian.mean();
					outputFile << "\"m\":[" << mean[0] << "," << mean[1] << "," << mean[2] << "],";
					std::array<std::array<T, 3>, 3> variance = gaussian.covariance();
					outputFile << "\"s\":[" << variance[0][0] << "," << variance[0][1] << "," << variance[0][2]
						<< "," << variance[1][1] << "," << variance[1][2] << "," << variance[2][2] << "]" << std::endl; // upper
					outputFile << "}"; // one gaussian ends
					if (k != gmm.numComponents() - 1) {
						outputFile << ",";
					}
				}
				outputFile << "]"; // gaussians end
				outputFile << "}"; // gmm end
				outputFile << "}"; // V end
				outputFile << "}"; // dataBlock end
				if (i != validBinIds.size() - 1) {
					outputFile << ",";
				}
			}
			outputFile << "]"; // "D" end
			outputFile << "}" << std::endl; // one data block ends
			if(id != numBlocks - 1) {
				outputFile << "," << std::endl;
			}
			id++;
		}
		outputFile << "]" << std::endl;

		outputFile.close();
	}

}