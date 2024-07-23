#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <limits>

// euclidean distance between two points
template <typename T>
double euclideanDistance(const std::array<T, 3>& p1, const std::array<T, 3>& p2) {
  return std::sqrt(std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2) + std::pow(p1[2] - p2[2], 2));
}

/*
the K-means algorithm
:params: data_points, num_of_clusters, max_iterations
:return: centeroids
 */
template <typename T>
std::vector<std::array<T, 3>> kMeans(const std::vector<std::array<T, 3>>& points, int k, int maxIterations) {
  // 1. initialize centers
  std::array<T, 3> dataMin = points[0];
  std::array<T, 3> dataMax = points[0];
  for (const auto& point : points) {
    for (int i = 0; i < 3; i++) {
      if (point[i] < dataMin[i]) {
        dataMin[i] = point[i];
      }
      else if (point[i] > dataMax[i]) {
        dataMax[i] = point[i];
      }
    }
  }


  std::vector<std::array<T, 3>> centers(k);
  std::mt19937 randomEngine(time(0));
  std::uniform_real_distribution<double> uniformDistribution(0.0, 1.0);
  for (int i = 0; i < k; ++i) {
    centers[i][0] = uniformDistribution(randomEngine);
    centers[i][1] = uniformDistribution(randomEngine);
    centers[i][2] = uniformDistribution(randomEngine);
  }

  // rescale centers to data range
  for (auto& center : centers) {
    center[0] = dataMin[0] + (dataMax[0] - dataMin[0]) * center[0];
    center[1] = dataMin[1] + (dataMax[1] - dataMin[1]) * center[1];
    center[2] = dataMin[2] + (dataMax[2] - dataMin[2]) * center[2];
  }


  // 2. iterate until convergence
  for (int iter = 0; iter < maxIterations; ++iter) {
    // distribute points to clusters
    std::vector<std::vector<std::array<T, 3>>> clusters(k);
    for (const auto& point : points) {
      double minDistance = std::numeric_limits<double>::max();
      int closestCenterIdx = -1;
      for (int i = 0; i < k; ++i) {
        double distance = euclideanDistance(point, centers[i]);
        if (distance < minDistance) {
          minDistance = distance;
          closestCenterIdx = i;
        }
      }
      clusters[closestCenterIdx].push_back(point);
    }

    // update centers
    for (int i = 0; i < k; ++i) {
      if(clusters[i].empty()) continue;
      // calculate new center (average
      double sumX = 0, sumY = 0, sumZ = 0;
      
      for (const auto& point : clusters[i]) {
        sumX += point[0];
        sumY += point[1];
        sumZ += point[2];
      }

      centers[i][0] = sumX / clusters[i].size();
      centers[i][1] = sumY / clusters[i].size();
      centers[i][2] = sumZ / clusters[i].size();
    }
  }

  return centers;
}

template <typename T>
std::vector<T> kMeansUnivariate(const std::vector<T>& X, int k, int maxIterations) {
  // 1. initialize centers
  T dataMin = X[0];
  T dataMax = X[0];
  for (const auto& x : X) {
    if (x < dataMin) {
      dataMin = x;
    }
    else if (x > dataMax) {
      dataMax = x;
    }
  }


  std::vector<T> centers(k);
  std::mt19937 randomEngine(time(0));
  std::uniform_real_distribution<T> uniformDistribution(0.0, 1.0);
  for (int i = 0; i < k; ++i) {
    centers[i] = uniformDistribution(randomEngine);
  }

  // rescale centers to data range
  for (T& center : centers) {
    center = dataMin + (dataMax - dataMin) * center;
  }


  // 2. iterate until convergence
  for (int iter = 0; iter < maxIterations; ++iter) {
    // distribute points to clusters
    std::vector<std::vector<T>> clusters(k);
    for (const T& x : X) {
      T minDistance = std::numeric_limits<T>::max();
      int closestCenterIdx = -1;
      for (int i = 0; i < k; ++i) {
        T distance = std::sqrt(std::pow(x - centers[i], 2));
        if (distance < minDistance) {
          minDistance = distance;
          closestCenterIdx = i;
        }
      }
      clusters[closestCenterIdx].push_back(x);
    }

    // update centers
    for (int i = 0; i < k; ++i) {
      if (clusters[i].empty()) continue;
      // calculate new center (average
      double sum = 0;

      for (const T& x : clusters[i]) {
        sum += x;
      }

      centers[i] = sum / clusters[i].size();
    }
  }

  return centers;
}