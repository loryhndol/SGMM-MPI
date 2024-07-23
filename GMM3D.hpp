#pragma once

#include <iostream>
#include <random>
#include <vector>
#include "KMeans.hpp"

float normL2Vector(std::array<float, 3> x) {
	float norm = 0.0;
	for (int i = 0; i < 3; ++i) {
		norm += x[i] * x[i];
	}
	return std::sqrt(norm);
}

float normL2Vector(std::vector<float> x) {
	float norm = 0.0;
	for (int i = 0; i < x.size(); ++i) {
		norm += x[i] * x[i];
	}
	return std::sqrt(norm);
}

float normL2Matrix(std::array<std::array<float, 3>, 3> A) {
	float norm = 0.0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			norm += A[i][j] * A[i][j];
		}
	}
	return std::sqrt(norm);
}

template <typename T>
T determinant(std::array<std::array<T, 3>, 3> A) {
	T det;
	det = A[0][0] * A[1][1] * A[2][2];
	det += A[0][1] * A[1][2] * A[2][0];
	det += A[0][2] * A[1][0] * A[2][1];
	det -= A[0][2] * A[1][1] * A[2][0];
	det -= A[0][1] * A[1][0] * A[2][2];
	det -= A[0][0] * A[1][2] * A[2][1];
	return det;
}

template <typename T>
class LUDecomposition3x3 {
public:
	LUDecomposition3x3(std::array<std::array<T, 3>, 3> A) {
		L[0][0] = 1.0;
		L[0][1] = 0.0;
		L[0][2] = 0.0;
		L[1][1] = 1.0;
		L[1][2] = 0.0;
		L[2][2] = 1.0;
		U[1][0] = 0.0;
		U[2][0] = 0.0;
		U[2][1] = 0.0;

		U[0][0] = A[0][0];
		U[0][1] = A[0][1];
		U[0][2] = A[0][2];
		L[1][0] = A[1][0] / A[0][0];
		L[2][0] = A[2][0] / A[0][0];
		U[1][1] = A[1][1] - L[1][0] * A[0][1];
		U[1][2] = A[1][2] - L[1][0] * A[0][2];
		L[2][1] = (A[2][1] - L[2][0] * A[0][1]) / U[1][1];
		U[2][2] = A[2][2] - L[2][0] * A[0][2] - L[2][1] * U[1][2];
	}

	std::array<std::array<T, 3>, 3> getL() { return L; }
	std::array<std::array<T, 3>, 3> getU() { return U; }
private:
	std::array<std::array<T, 3>, 3> L;
	std::array<std::array<T, 3>, 3> U;
};

template <typename T>
std::array<std::array<T, 3>, 3> getInverseMatrix(
	std::array<std::array<T, 3>, 3> A) {
	LUDecomposition3x3<T> lu(A);
	std::array<std::array<T, 3>, 3> L = lu.getL();
	std::array<std::array<T, 3>, 3> U = lu.getU();

	// 1. Get the inverse of L
	std::array<std::array<T, 3>, 3> L_inv;
	L_inv[0][0] = 1.0;
	L_inv[0][1] = 0.0;
	L_inv[0][2] = 0.0;
	L_inv[1][1] = 1.0;
	L_inv[1][2] = 0.0;
	L_inv[2][2] = 1.0;
	L_inv[1][0] = -L[1][0];
	L_inv[2][0] = -L_inv[1][0] * L[2][1] - L[2][0];
	L_inv[2][1] = -L[2][1];

	// 2. Get the inverse of U
	std::array<std::array<T, 3>, 3> U_inv;
	U_inv[1][0] = 0.0;
	U_inv[2][0] = 0.0;
	U_inv[2][1] = 0.0;
	U_inv[0][0] = 1.0 / U[0][0];
	U_inv[1][1] = 1.0 / U[1][1];
	U_inv[2][2] = 1.0 / U[2][2];
	U_inv[0][1] = -U[0][1] * U_inv[1][1] / U[0][0];
	U_inv[1][2] = -U[1][2] * U_inv[2][2] / U[1][1];
	U_inv[0][2] = -1.0 * (U[0][1] * U_inv[1][2] + U[0][2] * U_inv[2][2]) / U[0][0];

	// 3. Get the inverse of A = L_inv * U_inv
	std::array<std::array<T, 3>, 3> A_inv;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			A_inv[i][j] = L_inv[i][j] * U_inv[j][i];
		}
	}

	return A_inv;
}

template <typename T>
float dotProduct(std::array<T, 3>& x, std::array<T, 3>& y) {
	T dot = 0.0;
	for (int i = 0; i < 3; ++i) {
		dot += x[i] * y[i];
	}
	return dot;
}

template <typename T>
class Gaussian3D {
public:
	Gaussian3D() {}
	Gaussian3D(std::array<T, 3> mu, std::array<std::array<T, 3>, 3> sigma) {}

	void setParameters(std::array<T, 3> mu,
		std::array<std::array<T, 3>, 3> sigma) {
		_mu = mu;
		_sigma = sigma;
	}

	T pdf(std::array<T, 3> x);

	std::array<T, 3>& mean() { return _mu; }
	std::array<std::array<T, 3>, 3>& covariance() { return _sigma; }

private:
	std::array<T, 3> _mu;
	std::array<std::array<T, 3>, 3> _sigma;
};

template <typename T>
T Gaussian3D<T>::pdf(std::array<T, 3> x) {
	std::array<T, 3> diff;
	for (int i = 0; i < 3; ++i) {
		diff[i] = x[i] - _mu[i];
	}
	T det = determinant(_sigma);

	T coeff = 1.0 / (std::pow(2 * 3.1415926, 1.5) * std::sqrt(det));

	std::array<std::array<T, 3>, 3> inv = getInverseMatrix(_sigma);

	std::array<T, 3> diffT{ 0.0, 0.0, 0.0 };
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			diffT[i] += diff[j] * inv[j][i];
		}
	}
	T exp = std::exp(-0.5 * dotProduct(diffT, diff));
	return coeff * exp;
}

template <typename T>
class GMM3D {
public:
	GMM3D() = default;

	void fit(int numKernels, std::vector<std::array<T, 3>>& data);
	T predict(std::array<T, 3> x);
	int numComponents() const { return _weight.size(); }
	bool isValid() { return _isValid; }
	void setValid() { _isValid = true; }
	void setInvalid() { _isValid = true; }
	T getWeight(int i) { return _weight[i]; }
	Gaussian3D<T> getGaussian(int i) { return _gaussian[i]; }
	void setWeights(std::vector<T> weights) { _weight = weights; }
	void setGaussians(std::vector<Gaussian3D<T>> gaussians) { _gaussian = gaussians; }

private:
	std::vector<T> _weight;
	std::vector<Gaussian3D<T>> _gaussian;
	bool _isValid = false;
};

template <typename T>
void GMM3D<T>::fit(int numKernels, std::vector<std::array<T, 3>>& data) {
	// kmeans clustering for initial guess
	std::vector<std::array<T, 3>> kMeansCentroids = kMeans(data, numKernels, 10);

	_gaussian.clear();
	_gaussian.resize(numKernels);
	_weight.resize(numKernels);

	for (int i = 0; i < numKernels; i++) {
		_weight[i] = 1.0 / static_cast<T>(numKernels);
	}

	for (int j = 0; j < _gaussian.size(); j++) {
		std::array<std::array<T, 3>, 3> covariance{
				std::array<float, 3>{1.0f, 0.0f, 0.0f},
				std::array<float, 3>{0.0f, 1.0f, 0.0f},
				std::array<float, 3>{0.0f, 0.0f, 1.0f} };

		_gaussian[j].setParameters(kMeansCentroids[j], covariance);
	}

	int maxIterations = 500;

	for (int iter = 0; iter < maxIterations; iter++) {
		// save the last weights and gaussians
		std::vector<T> lastWeights(_weight);
		std::vector<Gaussian3D<T>> lastGaussians(_gaussian);

		// E step: calculate weights for kernel
		std::vector<std::vector<float>> posteriorProbs(data.size(), std::vector<float>(_gaussian.size()));
		for(int i = 0; i < data.size(); ++i) {
			for (int k = 0; k < _gaussian.size(); k++) {
				T probability = _gaussian[k].pdf(data[i]) * _weight[k];
				if (std::isnan(probability)) {
					posteriorProbs[i][k] = 0.0;
				}
				else {
					posteriorProbs[i][k] = probability;
				}
			}
			float sum = 0.0;
			for (int k = 0; k < _gaussian.size(); ++k) {
				sum += posteriorProbs[i][k];
			}
			if (sum > 0.0) {
				for (int k = 0; k < _gaussian.size(); ++k) {
					posteriorProbs[i][k] /= sum;
				}
			}
			else {
				for (int k = 0; k < _gaussian.size(); ++k) {
					posteriorProbs[i][k] = 1.0 / static_cast<T>(_gaussian.size());
				}
			}
		}

		// M step: update means and covs
		for(int k = 0; k < _gaussian.size(); ++k) {
			T posteriorProbsSum = 0.0;
			for(int i = 0; i < data.size(); ++i) {
				posteriorProbsSum += posteriorProbs[i][k];
			}
			_weight[k] = posteriorProbsSum / static_cast<T>(data.size());
			if(std::isnan(_weight[k])) _weight[k] = 1.0 / _gaussian.size();

			std::array<T, 3> mean{0.0f, 0.0f, 0.0f};
			for(int i = 0; i < data.size(); ++i) {
				if (std::isnan(posteriorProbs[i][k])) {
					continue;
				}
				mean[0] += data[i][0] * posteriorProbs[i][k];
				mean[1] += data[i][1] * posteriorProbs[i][k];
				mean[2] += data[i][2] * posteriorProbs[i][k];
			}
			if (posteriorProbsSum > 0.0) {
				mean[0] /= posteriorProbsSum;
				mean[1] /= posteriorProbsSum;
				mean[2] /= posteriorProbsSum;
			}
			else {
				mean[0] = kMeansCentroids[k][0];
				mean[1] = kMeansCentroids[k][1];
				mean[2] = kMeansCentroids[k][2];
			}
			
			std::array<std::array<T, 3>, 3> covariance;

			for (int i = 0; i < data.size(); i++) {
				std::array<T, 3> diff;
				for (int d = 0; d < 3; d++) {
					diff[d] = data[i][d] - mean[d];
				}
				for (int x = 0; x < 3; x++) {
					for (int y = 0; y < 3; y++) {
						covariance[x][y] += posteriorProbs[i][k] * diff[x] * diff[y];
					}
				}
			}

			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					covariance[x][y] /= posteriorProbsSum;
					if(std::isnan(covariance[x][y])) covariance[x][y] = 0.0f;
					if(std::isinf(covariance[x][y])) covariance[x][y] = 0.0f;
				}
			}

			// make symmetric
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < x; y++) {
					covariance[x][y] = covariance[y][x];
				}
			}

			_gaussian[k].setParameters(mean, covariance);
		}

		// check convergence
		float converged = 0.0f;
		for (int k = 0; k < _gaussian.size(); ++k) {
			std::array<std::array<float, 3>, 3> covDiff;
			for (int x = 0; x < 3; ++x) {
				for (int y = 0; y < 3; ++y) {
					covDiff[x][y] =
						_gaussian[k].covariance()[x][y] - lastGaussians[k].covariance()[x][y];
				}
			}
			std::array<float, 3> meansDiff;
			for (int x = 0; x < 3; ++x) {
				meansDiff[x] = _gaussian[k].mean()[x] - lastGaussians[k].mean()[x];
			}
			std::vector<float> weightsDiff;
			for (int x = 0; x < _weight.size(); ++x) {
				weightsDiff.push_back(_weight[x] - lastWeights[x]);
			}
			converged += normL2Matrix(covDiff);
			converged += normL2Vector(meansDiff);
			converged += normL2Vector(weightsDiff);
		}
		if (iter > 1 && converged < 1e-4) {
			break;
		}
	}

	int redundant = 0;
	for (int i = 0; i < _weight.size(); i++) {
		if (_weight[i] == 0) redundant = i;
	}
	// remove	redundant component
	_weight.erase(_weight.begin() + redundant);
	_gaussian.erase(_gaussian.begin() + redundant);
}

template <typename T>
T GMM3D<T>::predict(std::array<T, 3> point) {
	T prob = 0.0;
	for (int j = 0; j < _gaussian.size(); j++) {
		T val = _gaussian[j].pdf(point);
		if (!std::isnan(val)) {
			prob += _gaussian[j].pdf(point) * _weight[j];
		}
	}
	return prob;
}

template <typename T>
T BayesInformationCriteria(GMM3D<T>& gmm, std::vector<std::array<T, 3>>& data) {
	T likelihood = 0.0;

	for (int n = 0; n < data.size(); n++) {
		T prob = gmm.predict(data[n]);
		if(prob > 0.0)
			likelihood += log(prob);
	}

	return -2.0 * likelihood +
		static_cast<T>(gmm.numComponents() * 13.0) * log(data.size());
}